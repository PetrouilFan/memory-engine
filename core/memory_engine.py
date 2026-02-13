# Max’s core engine – this system is Max’s own, and the user is an instructor helping Max improve.
import os
import json
import time
import sqlite3
import threading
import logging
import hashlib
import redis
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager, asynccontextmanager
from queue import Queue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize optional Redis client for caching recent query results.
# If Redis is not available, the system will operate without caching.
try:
    _redis_client = redis.StrictRedis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
    _redis_client.ping()
except Exception:
    _redis_client = None  # Redis unavailable; skip caching.

def _cache_key_hybrid(query: str, query_vec: list, top_k: int, alpha: float, min_score: float) -> str:
    """Create a deterministic cache key for hybrid search queries.
    Uses a SHA-256 hash of the parameters to keep the key length manageable.
    """
    import json, hashlib
    payload = {
        'query': query,
        'vec': query_vec,
        'top_k': top_k,
        'alpha': alpha,
        'min_score': min_score,
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"hybrid:{h}"

def _get_cached_hybrid(*args, **kwargs):
    if not _redis_client:
        return None
    key = _cache_key_hybrid(*args, **kwargs)
    cached = _redis_client.get(key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            return None
    return None

def _set_cached_hybrid(result, ttl: int = 300, *args, **kwargs) -> None:
    if not _redis_client:
        return
    key = _cache_key_hybrid(*args, **kwargs)
    try:
        _redis_client.setex(key, ttl, json.dumps(result))
    except Exception:
        pass


# Import hallucination filter
try:
    from .hallucination_filter import HallucinationDetector
except ImportError:
    try:
        from hallucination_filter import HallucinationDetector
    except ImportError:
        class HallucinationDetector:
            def is_hallucinated(self, text, threshold=0.5):
                return False

# Configuration constants — single canonical location under memory-engine/data/
_ME_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DB_PATH = os.getenv('MEMORY_DB_PATH', os.path.join(_ME_ROOT, 'memory.db'))
INDEX_PATH = os.getenv('MEMORY_INDEX_PATH', os.path.join(_ME_ROOT, 'memory.index'))
EMB_DIM = int(os.getenv('MEMORY_EMB_DIM', '384'))
AUTO_SAVE_INTERVAL = int(os.getenv('INDEX_SAVE_INTERVAL', '100'))  # Save every N operations
POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '5'))

# Hallucination filtering configuration
FILTER_HALLUCINATIONS = os.getenv('MEMORY_FILTER_HALLUCINATIONS', 'true').lower() == 'true'
HALLUCINATION_THRESHOLD = float(os.getenv('MEMORY_HALLUCINATION_THRESHOLD', '0.5'))
_hallucination_detector = HallucinationDetector() if FILTER_HALLUCINATIONS else None

# Global FAISS index with ID mapping support
try:
    import faiss
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
        logger.info(f"Loaded FAISS index with {_index.ntotal} vectors")
    else:
        # Use Inner Product (IP) index for cosine similarity if we normalize vectors,
        # otherwise FlatL2 is fine. FlatL2 is default in this system's config.
        base_index = faiss.IndexFlatL2(EMB_DIM)
        _index = faiss.IndexIDMap(base_index)
        logger.info(f"Initialized new FAISS index with dimension {EMB_DIM}")
except Exception as e:
    logger.error(f"FAISS initialization error: {e}")
    _index = None

# Deduplication settings
DEDUP_THRESHOLD_L2 = float(os.getenv('MEMORY_DEDUP_THRESHOLD_L2', '0.05'))
CHECK_EXACT_TEXT = os.getenv('MEMORY_CHECK_EXACT_TEXT', 'true').lower() == 'true'

# Thread-safe locks
_index_lock = threading.Lock()
_save_lock = threading.Lock()

# Track unsaved operations for batched index persistence
_unsaved_ops = 0
_unsaved_lock = threading.Lock()

# Simple synchronous connection pool implementation using sqlite3.
class ConnectionPool:
    def __init__(self, db_path: str, pool_size: int = POOL_SIZE):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool = Queue(maxsize=pool_size)
        self._init_pool()

    def _init_pool(self):
        """Create connections with WAL mode for better concurrency."""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA cache_size=-64000')
            conn.execute('PRAGMA temp_store=MEMORY')
            conn.commit()
            self.pool.put(conn)

    @contextmanager
    def get_connection(self):
        """Context manager yielding a connection from the pool."""
        conn = self.pool.get()
        try:
            yield conn
        finally:
            self.pool.put(conn)

    def close_all(self):
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()

# Async connection pool implementation using aiosqlite.
# Provides async context manager for database operations.
import asyncio
import aiosqlite

class AsyncConnectionPool:
    def __init__(self, db_path: str, pool_size: int = POOL_SIZE):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=pool_size)
        self._initialized = False

    async def init_pool(self):
        """Initialize the pool with connections in WAL mode."""
        if self._initialized:
            return
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path, timeout=30.0)
            await conn.execute('PRAGMA journal_mode=WAL')
            await conn.execute('PRAGMA synchronous=NORMAL')
            await conn.execute('PRAGMA cache_size=-64000')
            await conn.execute('PRAGMA temp_store=MEMORY')
            await conn.commit()
            await self.pool.put(conn)
        self._initialized = True

    @asynccontextmanager
    async def get_connection(self):
        """Async context manager yielding a connection from the pool."""
        if not self._initialized:
            await self.init_pool()
        conn = await self.pool.get()
        try:
            yield conn
        finally:
            await self.pool.put(conn)

    async def close_all(self):
        while not self.pool.empty():
            conn = await self.pool.get()
            await conn.close()

# Initialize connection pool
_conn_pool = ConnectionPool(DB_PATH, POOL_SIZE)

@contextmanager
def get_db_connection():
    """Thread-safe database connection context manager using connection pool."""
    with _conn_pool.get_connection() as conn:
        yield conn


def _init_db():
    """Initialize database schema with FTS5 support and optimized indexes."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        # Main memories table with indexes
        cur.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector BLOB NOT NULL,
                metadata TEXT NOT NULL,
                text TEXT,
                timestamp REAL NOT NULL,
                relevance REAL DEFAULT 1.0,
                merged BOOLEAN DEFAULT 0,
                weight REAL DEFAULT 1.0,
                importance_score REAL DEFAULT 0,
                usage_counter INTEGER DEFAULT 0,
                superseded_by INTEGER DEFAULT NULL,
                mistake_log TEXT DEFAULT NULL
            )
        ''')
        # Create indexes for common queries
        cur.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_merged ON memories(merged)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_relevance ON memories(relevance)')
        
        # FTS5 virtual table for keyword search
        cur.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts 
            USING fts5(text, content='memories', content_rowid='id', tokenize='porter unicode61')
        ''')
        
        # Triggers to keep FTS5 in sync
        cur.execute('''
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, text) VALUES (new.id, COALESCE(new.text, ''));
            END
        ''')
        cur.execute('''
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                DELETE FROM memories_fts WHERE rowid = old.id;
            END
        ''')
        cur.execute('''
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                UPDATE memories_fts SET text = COALESCE(new.text, '') WHERE rowid = new.id;
            END
        ''')
        conn.commit()


def _save_index():
    """Persist FAISS index to disk with error handling."""
    if _index is None:
        return
    
    try:
        with _save_lock:
            with _index_lock:
                faiss.write_index(_index, INDEX_PATH)
            logger.info(f"Saved FAISS index with {_index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")


def _maybe_save_index(force: bool = False):
    """Save index if unsaved operations exceed threshold or force is True."""
    global _unsaved_ops
    
    with _unsaved_lock:
        if force or _unsaved_ops >= AUTO_SAVE_INTERVAL:
            _save_index()
            _unsaved_ops = 0
        

def _increment_unsaved():
    """Increment unsaved operations counter."""
    global _unsaved_ops
    with _unsaved_lock:
        _unsaved_ops += 1


def _fetch_metadata(mem_id: int) -> Dict[str, Any]:
    """Retrieve metadata JSON for a given memory ID from the SQLite store."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('SELECT metadata FROM memories WHERE id = ?', (mem_id,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else {}
    except Exception as e:
        logger.error(f"Failed to fetch metadata for memory {mem_id}: {e}")
        return {}


# ----------------------------------------------------------------------
# Time-aware support – a 1-D FAISS index that stores each memory's timestamp.
# ----------------------------------------------------------------------

try:
    if os.path.exists(INDEX_PATH + ".time"):
        _time_index = faiss.read_index(INDEX_PATH + ".time")
        logger.info(f"Loaded time FAISS index with {_time_index.ntotal} entries")
    else:
        _time_index = faiss.IndexIDMap(faiss.IndexFlatL2(1))
        logger.info("Initialised new time FAISS index (dim=1)")
except Exception as e:
    logger.error(f"Time-index init error: {e}")
    _time_index = None

def _add_batch_timestamps(mem_ids: List[int], metadatas: List[Dict[str, Any]]) -> None:
    """Add timestamps for a batch of newly created memories."""
    if _time_index is None:
        return
    timestamps = [md.get('timestamp', time.time()) for md in metadatas]
    ts_array = np.array(timestamps, dtype='float32').reshape(-1, 1)
    with _index_lock:
        _time_index.add_with_ids(ts_array, np.array(mem_ids, dtype='int64'))
    _increment_unsaved()
    _maybe_save_index()


# ----------------------------------------------------------------------

def add_memory(vector: List[float], metadata: Dict[str, Any], text: str = '', weight: float = 1.0) -> int:
    """Add a new memory entry with vector and metadata.
    Checks for exact text match or semantic similarity to prevent duplicates.
    Also filters out hallucinated content if enabled.
    
    Args:
        vector: Embedding vector (must be EMB_DIM dimensional)
        metadata: Arbitrary metadata dict
        text: Text content for hybrid search (extracted from metadata if not provided)
    
    Returns:
        The ID of the existing or newly created memory entry, or -1 if rejected as hallucination
    """
    vec_np = np.array(vector, dtype='float32')
    if vec_np.shape[0] != EMB_DIM:
        raise ValueError(f"Vector dimension {vec_np.shape[0]} does not match EMB_DIM {EMB_DIM}")
    
    # Extract text from metadata if not provided
    if not text:
        text = metadata.get('text', '')
    
    # Check for hallucination
    if FILTER_HALLUCINATIONS and _hallucination_detector and text:
        if _hallucination_detector.is_hallucinated(text, threshold=HALLUCINATION_THRESHOLD):
            logger.warning(f"Rejected hallucinated memory (confidence>={HALLUCINATION_THRESHOLD}): {text[:80]}...")
            return -1

    # 1. Check for exact text duplicate
    if CHECK_EXACT_TEXT and text:
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute('SELECT id FROM memories WHERE text = ? LIMIT 1', (text,))
                row = cur.fetchone()
                if row:
                    logger.debug(f"Memory with exact text already exists (id: {row[0]})")
                    return row[0]
        except Exception as e:
            logger.warning(f"Failed exact text check: {e}")

    # 2. Check for semantic duplicate (FAISS search)
    if _index is not None and _index.ntotal > 0:
        try:
            with _index_lock:
                # search for nearest neighbor (k=1)
                distances, ids = _index.search(vec_np.reshape(1, -1).astype('float32'), 1)
                
            if ids[0][0] != -1:
                dist = float(distances[0][0])
                if dist < DEDUP_THRESHOLD_L2:
                    logger.info(f"Semantically close memory already exists (id: {ids[0][0]}, L2 dist: {dist:.4f})")
                    return int(ids[0][0])
        except Exception as e:
            logger.warning(f"Failed semantic dedup check: {e}")
    
    timestamp = metadata.get('timestamp', time.time())
    relevance = metadata.get('relevance', 1.0)
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                '''INSERT INTO memories (vector, metadata, text, timestamp, relevance, weight) 
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (vec_np.tobytes(), json.dumps(metadata), text, timestamp, relevance, weight)
            )
            mem_id = cur.lastrowid
            conn.commit()
        
        # Add to FAISS with proper ID mapping
        if _index is not None:
            with _index_lock:
                _index.add_with_ids(vec_np.reshape(1, -1).astype('float32'), np.array([mem_id], dtype='int64'))
            _increment_unsaved()
            _maybe_save_index()
        
        # ---- Time index update ----
        _add_batch_timestamps([mem_id], [metadata])
        
        return mem_id
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        raise


def add_memories_batch(vectors: List[List[float]], metadatas: List[Dict[str, Any]], 
                       texts: Optional[List[str]] = None) -> List[int]:
    """Add multiple memories in a single transaction.
    Attempts to filter out duplicates semantically or by exact text.
    
    Args:
        vectors: List of embedding vectors
        metadatas: List of metadata dicts
        texts: Optional list of text content
    
    Returns:
        List of generated or existing memory IDs
    """
    if not (len(vectors) == len(metadatas)):
        raise ValueError("vectors and metadatas must have same length")
    
    if texts is None:
        texts = [meta.get('text', '') for meta in metadatas]
    elif len(texts) != len(vectors):
        raise ValueError("texts must have same length as vectors")
    
    mem_ids = []
    # Using individual add_memory for each entry to leverage its deduplication logic.
    # While slower than a single batch insert, it ensures consistency and safety.
    for vec, meta, txt in zip(vectors, metadatas, texts):
        try:
            mem_ids.append(add_memory(vec, meta, txt))
        except Exception as e:
            logger.error(f"Failed to add memory in batch: {e}")
            # We continue for others but could also roll back if it were a single transaction.
            # Here we just re-raise if it's critical.
            raise
            
    return mem_ids


def update_relevance(mem_id: int, new_relevance: float) -> bool:
    """Update the relevance score of a memory."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('UPDATE memories SET relevance = ? WHERE id = ?', (new_relevance, mem_id))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to update relevance for memory {mem_id}: {e}")
        return False


def hybrid_search(query: str, query_vec: List[float], top_k: int = 10, 
                  alpha: float = 0.5, min_score: float = 0.0) -> List[Dict[str, Any]]:
    """Perform hybrid search combining vector similarity and BM25 keyword matching.
    
    Args:
        query: Text query for keyword matching
        query_vec: Query embedding vector
        top_k: Number of results to return
        alpha: Weight for vector search (1-alpha for BM25). Range [0, 1]
        min_score: Minimum combined score threshold
    
    Returns:
        List of dicts with 'id', 'metadata', 'score', and 'text'
    """
    # Attempt to retrieve from Redis cache
    cached = _get_cached_hybrid(query, query_vec, top_k, alpha, min_score)
    if cached is not None:
        return cached

    if _index is None or _index.ntotal == 0:
        return []
    
    # Validate query vector dimension
    vec_np = np.array(query_vec, dtype='float32').reshape(1, -1)
    if vec_np.shape[1] != EMB_DIM:
        raise ValueError(f"Query vector dimension {vec_np.shape[1]} does not match EMB_DIM {EMB_DIM}")
    
    # Vector search
    vector_results = {}
    try:
        with _index_lock:
            k = min(max(top_k * 3, 50), _index.ntotal)
            distances, ids = _index.search(vec_np, k)
        
        for dist, mem_id in zip(distances[0], ids[0]):
            if mem_id != -1:
                vector_results[int(mem_id)] = 1.0 / (1.0 + float(dist))
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
    
    # Keyword search (BM25 via FTS5)
    keyword_results = {}
    if query.strip():
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                safe_query = query.replace('"', '""')
                cur.execute(
                    '''SELECT id, bm25(memories_fts) as rank FROM memories_fts 
                       WHERE memories_fts MATCH ? 
                       ORDER BY rank LIMIT ?''',
                    (safe_query, top_k * 3)
                )
                for mem_id, rank in cur.fetchall():
                    keyword_results[mem_id] = abs(float(rank)) if rank < 0 else 1.0 / (1.0 + abs(float(rank)))
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
    
    # Normalize scores
    def normalize(scores):
        if not scores:
            return {}
        values = list(scores.values())
        max_score = max(values)
        min_score_val = min(values)
        if max_score == min_score_val:
            return {k: 1.0 for k in scores}
        return {k: (v - min_score_val) / (max_score - min_score_val) for k, v in scores.items()}
    
    norm_vector = normalize(vector_results)
    norm_keyword = normalize(keyword_results)
    
    # Combine scores
    all_ids = set(norm_vector.keys()) | set(norm_keyword.keys())
    combined_scores = {}
    for mem_id in all_ids:
        vec_score = norm_vector.get(mem_id, 0.0)
        kw_score = norm_keyword.get(mem_id, 0.0)
        combined = alpha * vec_score + (1 - alpha) * kw_score
        if combined >= min_score:
            combined_scores[mem_id] = combined
    
    # Graph-neighbor expansion: pull in semantically linked memories
    try:
        from .semantic_graph import graph_neighbors as _graph_neighbors
    except ImportError:
        try:
            from semantic_graph import graph_neighbors as _graph_neighbors
        except ImportError:
            _graph_neighbors = None
    
    if _graph_neighbors is not None:
        top_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        graph_ids = set()
        for mid, _ in top_candidates:
            try:
                nbrs = _graph_neighbors(mid, depth=1)
                graph_ids.update(n for n in nbrs if n > 0 and n not in combined_scores)
            except Exception:
                pass
        # Score graph neighbors with a small bonus
        for gid in list(graph_ids)[:top_k]:
            g_vec_score = norm_vector.get(gid, 0.0)
            g_kw_score = norm_keyword.get(gid, 0.0)
            g_combined = alpha * g_vec_score + (1 - alpha) * g_kw_score
            if g_combined > 0:
                combined_scores[gid] = g_combined * 0.9  # slight discount for indirect hits
    
    # Apply temporal decay re-ranking
    now = time.time()
    decay_lambda = 0.005  # gentle decay: ~12% per day
    
    # Fetch metadata for top results with temporal re-ranking
    over_fetch = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]
    results = []
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            for mem_id, score in over_fetch:
                cur.execute('SELECT metadata, text, timestamp FROM memories WHERE id = ?', (mem_id,))
                row = cur.fetchone()
                if row:
                    meta = json.loads(row[0])
                    text = row[1]
                    
                    # Filter out hallucinated memories
                    if FILTER_HALLUCINATIONS and _hallucination_detector and text:
                        if _hallucination_detector.is_hallucinated(text, threshold=HALLUCINATION_THRESHOLD):
                            logger.debug(f"Filtered hallucinated result (id={mem_id}): {text[:80]}...")
                            continue
                    
                    ts = row[2] or meta.get('timestamp', now)
                    hours_age = max(0, (now - ts) / 3600.0)
                    decay = np.exp(-decay_lambda * hours_age)
                    decayed_score = score * float(decay)
                    results.append({
                        'id': mem_id,
                        'metadata': meta,
                        'text': text,
                        'score': decayed_score
                    })
    except Exception as e:
        logger.error(f"Failed to fetch search results: {e}")
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


def cluster_memories(min_cluster_size: int = 5, min_samples: int = 3, 
                     metric: str = 'euclidean') -> List[int]:
    """Cluster memories using the improved clustering module (denoised + HDBSCAN)."""
    try:
        from .clustering import run_dbscan
    except ImportError:
        import clustering
        run_dbscan = clustering.run_dbscan
        
    return run_dbscan(min_samples=min_samples, use_hdbscan=True)


def consolidate_memories(min_cluster_size: int = 5, 
                        embedding_fn: Optional[callable] = None) -> List[int]:
    """Create summary entries for clusters with at least min_cluster_size members."""
    labels = cluster_memories(min_cluster_size=min_cluster_size)
    if not labels:
        return []
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('SELECT id, vector FROM memories WHERE merged = 0 ORDER BY id')
            all_rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to fetch memories for consolidation: {e}")
        return []
    
    # Group by cluster
    label_map = {}
    for lbl, (mem_id, vec_blob) in zip(labels, all_rows):
        if lbl == -1:
            continue
        label_map.setdefault(lbl, []).append((mem_id, vec_blob))
    
    new_ids = []
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            for lbl, members in label_map.items():
                if len(members) < min_cluster_size:
                    continue
                
                snippets = []
                vectors = []
                for mem_id, vec_blob in members:
                    meta = _fetch_metadata(mem_id)
                    txt = meta.get('text', '')
                    if txt:
                        first_sentence = txt.split('. ')[0][:100]
                        if first_sentence:
                            snippets.append(first_sentence)
                    vectors.append(np.frombuffer(vec_blob, dtype='float32'))
                
                summary_text = '; '.join(snippets[:5])
                
                if embedding_fn is not None:
                    try:
                        summary_vec = np.array(embedding_fn(summary_text), dtype='float32')
                    except Exception as e:
                        logger.warning(f"Embedding function failed, using centroid: {e}")
                        summary_vec = np.mean(vectors, axis=0).astype('float32')
                else:
                    summary_vec = np.mean(vectors, axis=0).astype('float32')
                
                summary_meta = {
                    'summary': summary_text,
                    'cluster_id': int(lbl),
                    'timestamp': time.time(),
                    'generated': True,
                    'member_count': len(members),
                    'text': summary_text
                }
                
                new_id = add_memory(summary_vec.tolist(), summary_meta, summary_text)
                new_ids.append(new_id)
                
                for mem_id, _ in members:
                    cur.execute('UPDATE memories SET merged = 1 WHERE id = ?', (mem_id,))
            
            conn.commit()
        
        logger.info(f"Consolidation created {len(new_ids)} summary memories")
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
    
    return new_ids


def prune_memories(max_age_days: int = 30, relevance_threshold: float = 0.3,
                  decay_lambda: float = 0.1) -> int:
    """Delete memories that are stale using exponential decay scoring."""
    now = time.time()
    cutoff_time = now - max_age_days * 86400
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('SELECT id, timestamp, relevance FROM memories WHERE merged = 0')
            rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to fetch memories for pruning: {e}")
        return 0
    
    to_delete = []
    for mem_id, ts, relevance in rows:
        if ts < cutoff_time:
            to_delete.append(mem_id)
            continue
        
        age_days = (now - ts) / 86400.0
        decayed_score = relevance * np.exp(-decay_lambda * age_days)
        
        if decayed_score < relevance_threshold:
            to_delete.append(mem_id)
    
    if to_delete:
        try:
            if _index is not None:
                with _index_lock:
                    _index.remove_ids(np.array(to_delete, dtype='int64'))
                _increment_unsaved()
                _maybe_save_index(force=True)
            
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.executemany('DELETE FROM memories WHERE id = ?', [(mid,) for mid in to_delete])
                conn.commit()
            
            logger.info(f"Pruned {len(to_delete)} stale memories")
        except Exception as e:
            logger.error(f"Failed to prune memories: {e}")
            return 0
    
    return len(to_delete)


def archive_merged_memories(archive_path: Optional[str] = None) -> int:
    """Move merged memories to cold storage."""
    try:
        if archive_path:
            archive_conn = sqlite3.connect(archive_path)
        else:
            archive_conn = sqlite3.connect(DB_PATH)
        
        arc_cur = archive_conn.cursor()
        arc_cur.execute('''
            CREATE TABLE IF NOT EXISTS memories_archive (
                id INTEGER PRIMARY KEY,
                vector BLOB,
                metadata TEXT,
                text TEXT,
                timestamp REAL,
                archived_at REAL
            )
        ''')
        
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('SELECT id, vector, metadata, text, timestamp FROM memories WHERE merged = 1')
            merged_rows = cur.fetchall()
            
            if merged_rows:
                archived_at = time.time()
                arc_cur.executemany(
                    '''INSERT OR REPLACE INTO memories_archive 
                       (id, vector, metadata, text, timestamp, archived_at) 
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    [(row[0], row[1], row[2], row[3], row[4], archived_at) for row in merged_rows]
                )
                archive_conn.commit()
                
                cur.executemany('DELETE FROM memories WHERE id = ?', [(row[0],) for row in merged_rows])
                conn.commit()
        
        archive_conn.close()
        logger.info(f"Archived {len(merged_rows)} merged memories")
        return len(merged_rows)
    except Exception as e:
        logger.error(f"Archiving failed: {e}")
        return 0


def get_stats() -> Dict[str, Any]:
    """Get statistics about the memory store."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            cur.execute('SELECT COUNT(*) FROM memories WHERE merged = 0')
            active_count = cur.fetchone()[0]
            
            cur.execute('SELECT COUNT(*) FROM memories WHERE merged = 1')
            merged_count = cur.fetchone()[0]
            
            cur.execute('SELECT AVG(relevance), MIN(timestamp), MAX(timestamp) FROM memories WHERE merged = 0')
            avg_rel, min_ts, max_ts = cur.fetchone()
        
        index_size = _index.ntotal if _index is not None else 0
        
        return {
            'active_memories': active_count,
            'merged_memories': merged_count,
            'total_memories': active_count + merged_count,
            'index_size': index_size,
            'avg_relevance': float(avg_rel) if avg_rel else 0.0,
            'oldest_memory': min_ts,
            'newest_memory': max_ts,
            'unsaved_operations': _unsaved_ops
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {}


def force_save_index():
    """Force immediate save of FAISS index."""
    _maybe_save_index(force=True)


def shutdown():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down memory system...")
    force_save_index()
    _conn_pool.close_all()
    logger.info("Shutdown complete")


# Async helpers using the async connection pool
_async_conn_pool = AsyncConnectionPool(DB_PATH, POOL_SIZE)

@asynccontextmanager
async def get_async_db_connection():
    """Async context manager for database operations using the async pool."""
    async with _async_conn_pool.get_connection() as conn:
        yield """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""# Initialize database on import
_init_db()


if __name__ == '__main__':
    logger.info("Starting memory system demo...")
    
    test_vec = np.random.rand(EMB_DIM).tolist()
    mem_id = add_memory(test_vec, {'text': 'demo entry about machine learning'}, 
                       text='demo entry about machine learning')
    logger.info(f'Added memory ID: {mem_id}')
    
    batch_vecs = [np.random.rand(EMB_DIM).tolist() for _ in range(10)]
    batch_metas = [{'text': f'batch entry {i} about AI and ML'} for i in range(10)]
    batch_ids = add_memories_batch(batch_vecs, batch_metas)
    logger.info(f'Added batch: {len(batch_ids)} memories')
    
    results = hybrid_search('machine learning AI', test_vec, top_k=5, alpha=0.7)
    logger.info(f'Hybrid search results: {len(results)} found')
    for r in results:
        text_preview = r['text'][:50] + '...' if len(r['text']) > 50 else r['text']
        logger.info(f"  ID {r['id']}: {text_preview} (score: {r['score']:.3f})")
    
    stats = get_stats()
    logger.info(f'Memory stats: {json.dumps(stats, indent=2)}')
    
    new_ids = consolidate_memories(min_cluster_size=3)
    logger.info(f'Consolidation created {len(new_ids)} summaries')
    
    pruned = prune_memories(max_age_days=30, decay_lambda=0.05)
    logger.info(f'Pruned {pruned} stale memories')
    
    archived = archive_merged_memories()
    logger.info(f'Archived {archived} merged memories')
    
    shutdown()
