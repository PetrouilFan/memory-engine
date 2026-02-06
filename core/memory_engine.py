import os
import json
import time
import sqlite3
import threading
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from queue import Queue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
DB_PATH = os.getenv('MEMORY_DB_PATH', '/root/.openclaw/workspace/data/memory.db')
INDEX_PATH = os.getenv('MEMORY_INDEX_PATH', '/root/.openclaw/workspace/data/memory.index')
EMB_DIM = int(os.getenv('MEMORY_EMB_DIM', '768'))
AUTO_SAVE_INTERVAL = int(os.getenv('INDEX_SAVE_INTERVAL', '100'))  # Save every N operations
POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '5'))

# Global FAISS index with ID mapping support
try:
    import faiss
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
        logger.info(f"Loaded FAISS index with {_index.ntotal} vectors")
    else:
        base_index = faiss.IndexFlatL2(EMB_DIM)
        _index = faiss.IndexIDMap(base_index)
        logger.info(f"Initialized new FAISS index with dimension {EMB_DIM}")
except Exception as e:
    logger.error(f"FAISS initialization error: {e}")
    _index = None

# Thread-safe locks
_index_lock = threading.Lock()
_save_lock = threading.Lock()

# Track unsaved operations for batched index persistence
_unsaved_ops = 0
_unsaved_lock = threading.Lock()

# Simple connection pool implementation
class ConnectionPool:
    def __init__(self, db_path: str, pool_size: int = POOL_SIZE):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool = Queue(maxsize=pool_size)
        self._init_pool()
    
    def _init_pool(self):
        """Initialize connection pool with WAL mode for better concurrency."""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            conn.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging
            conn.execute('PRAGMA synchronous=NORMAL')  # Faster writes
            conn.execute('PRAGMA cache_size=-64000')  # 64MB cache
            conn.execute('PRAGMA temp_store=MEMORY')
            self.pool.put(conn)
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic return."""
        conn = self.pool.get()
        try:
            yield conn
        finally:
            self.pool.put(conn)
    
    def close_all(self):
        """Close all connections in pool."""
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()

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
                merged BOOLEAN DEFAULT 0
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

def add_memory(vector: List[float], metadata: Dict[str, Any], text: str = '') -> int:
    """Add a new memory entry with vector and metadata.
    
    Args:
        vector: Embedding vector (must be EMB_DIM dimensional)
        metadata: Arbitrary metadata dict
        text: Text content for hybrid search (extracted from metadata if not provided)
    
    Returns:
        The auto-generated integer ID
    
    Raises:
        ValueError: If vector dimension doesn't match EMB_DIM
    """
    vec_np = np.array(vector, dtype='float32')
    if vec_np.shape[0] != EMB_DIM:
        raise ValueError(f"Vector dimension {vec_np.shape[0]} does not match EMB_DIM {EMB_DIM}")
    
    # Extract text from metadata if not provided
    if not text:
        text = metadata.get('text', '')
    
    timestamp = metadata.get('timestamp', time.time())
    relevance = metadata.get('relevance', 1.0)
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                '''INSERT INTO memories (vector, metadata, text, timestamp, relevance) 
                   VALUES (?, ?, ?, ?, ?)''',
                (vec_np.tobytes(), json.dumps(metadata), text, timestamp, relevance)
            )
            mem_id = cur.lastrowid
            conn.commit()
        
        # Add to FAISS with proper ID mapping
        if _index is not None:
            with _index_lock:
                _index.add_with_ids(vec_np.reshape(1, -1), np.array([mem_id], dtype='int64'))
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
    """Add multiple memories in a single transaction for better performance.
    
    Args:
        vectors: List of embedding vectors
        metadatas: List of metadata dicts
        texts: Optional list of text content
    
    Returns:
        List of generated memory IDs
    
    Raises:
        ValueError: If input lists have different lengths or vector dimensions are wrong
    """
    if not (len(vectors) == len(metadatas)):
        raise ValueError("vectors and metadatas must have same length")
    
    if texts is None:
        texts = [meta.get('text', '') for meta in metadatas]
    elif len(texts) != len(vectors):
        raise ValueError("texts must have same length as vectors")
    
    # Validate and prepare data
    vecs_np = []
    for vec in vectors:
        vec_np = np.array(vec, dtype='float32')
        if vec_np.shape[0] != EMB_DIM:
            raise ValueError(f"Vector dimension {vec_np.shape[0]} does not match EMB_DIM {EMB_DIM}")
        vecs_np.append(vec_np)
    
    mem_ids = []
    
    try:
        # Batch insert into SQLite
        with get_db_connection() as conn:
            cur = conn.cursor()
            for vec_np, metadata, text in zip(vecs_np, metadatas, texts):
                timestamp = metadata.get('timestamp', time.time())
                relevance = metadata.get('relevance', 1.0)
                
                cur.execute(
                    '''INSERT INTO memories (vector, metadata, text, timestamp, relevance) 
                       VALUES (?, ?, ?, ?, ?)''',
                    (vec_np.tobytes(), json.dumps(metadata), text, timestamp, relevance)
                )
                mem_ids.append(cur.lastrowid)
            conn.commit()
        
        # Batch add to FAISS
        if _index is not None and vecs_np:
            vectors_matrix = np.vstack(vecs_np)
            ids_array = np.array(mem_ids, dtype='int64')
            
            with _index_lock:
                _index.add_with_ids(vectors_matrix, ids_array)
            
            _increment_unsaved()
            _maybe_save_index()
        
        # ---- Time index update ----
        if _time_index is not None:
            timestamps = [md.get('timestamp', time.time()) for md in metadatas]
            ts_array = np.array(timestamps, dtype='float32').reshape(-1, 1)
            with _index_lock:
                _time_index.add_with_ids(ts_array, np.array(mem_ids, dtype='int64'))
            _increment_unsaved()
            _maybe_save_index()
        
        logger.info(f"Added batch of {len(mem_ids)} memories")
        return mem_ids
    except Exception as e:
        logger.error(f"Failed to add memory batch: {e}")
        raise


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
    
    # Fetch metadata for top results
    sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            for mem_id, score in sorted_ids:
                cur.execute('SELECT metadata, text FROM memories WHERE id = ?', (mem_id,))
                row = cur.fetchone()
                if row:
                    results.append({
                        'id': mem_id,
                        'metadata': json.loads(row[0]),
                        'text': row[1],
                        'score': score
                    })
    except Exception as e:
        logger.error(f"Failed to fetch search results: {e}")
    
    return results


def cluster_memories(min_cluster_size: int = 5, min_samples: int = 3, 
                     metric: str = 'euclidean') -> List[int]:
    """Cluster memories using HDBSCAN on their embeddings."""
    try:
        import hdbscan
    except ImportError:
        logger.error("HDBSCAN not available; install with: pip install hdbscan")
        return []
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('SELECT id, vector FROM memories WHERE merged = 0 ORDER BY id')
            rows = cur.fetchall()
        
        if len(rows) < min_cluster_size:
            logger.info(f"Not enough memories ({len(rows)}) for clustering (min: {min_cluster_size})")
            return [-1] * len(rows)
        
        vectors = []
        mem_ids = []
        for mem_id, vec_blob in rows:
            vec = np.frombuffer(vec_blob, dtype='float32')
            vectors.append(vec)
            mem_ids.append(mem_id)
        
        X = np.vstack(vectors)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric
        )
        labels = clusterer.fit_predict(X)
        
        logger.info(f"Clustered {len(rows)} memories into {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        return labels.tolist()
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        return []


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


# Initialize database on import
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
