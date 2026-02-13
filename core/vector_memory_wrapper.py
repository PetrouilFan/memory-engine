"""
vector_memory_wrapper.py — Thin FAISS + SQLite wrapper for the canonical memory DB.

Uses the SAME database and schema as memory_engine.py:
    projects/memory-engine/data/memory.db
        id INTEGER, vector BLOB, metadata TEXT, text TEXT,
        timestamp REAL, relevance REAL, merged BOOLEAN

The FAISS index is rebuilt from the DB vectors on first load (lazy) so there
is exactly ONE source of truth — the SQLite database.
"""
import os
import json
import time
import logging
import subprocess
import threading
import urllib.request
import urllib.error
from typing import List, Dict, Any
import numpy as np
import faiss
import sqlite3

logger = logging.getLogger(__name__)

# ---- Embedding Server Config ----
EMBEDDING_SERVER_HOST = os.getenv('EMBEDDING_SERVER_HOST', 'localhost')
EMBEDDING_SERVER_PORT = int(os.getenv('EMBEDDING_SERVER_PORT', '9999'))
EMBEDDING_SERVER_URL = f"http://{EMBEDDING_SERVER_HOST}:{EMBEDDING_SERVER_PORT}"

# ---- Canonical paths (same default as memory_engine.py) ----
_ME_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DB_PATH = os.getenv('MEMORY_DB_PATH', os.path.join(_ME_DATA, 'memory.db'))
INDEX_PATH = os.getenv('MEMORY_INDEX_PATH', os.path.join(_ME_DATA, 'memory.index'))
EMB_DIM = int(os.getenv('MEMORY_EMB_DIM', '384'))

_lock = threading.Lock()

# ---------- FAISS index ----------

def _build_index_from_db() -> faiss.IndexIDMap2:
    """Build a FAISS inner-product index from all vectors in memory.db."""
    base = faiss.IndexFlatIP(EMB_DIM)
    index = faiss.IndexIDMap2(base)

    if not os.path.exists(DB_PATH):
        logger.info("No memory.db yet — starting with empty index")
        return index

    conn = sqlite3.connect(DB_PATH, timeout=10)
    cur = conn.cursor()
    cur.execute("SELECT id, vector FROM memories ORDER BY id")
    rows = cur.fetchall()
    conn.close()

    ids, vecs = [], []
    for mem_id, vec_blob in rows:
        if vec_blob is None:
            continue
        vec = np.frombuffer(vec_blob, dtype="float32").copy()
        if vec.shape[0] != EMB_DIM:
            continue
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        ids.append(int(mem_id))
        vecs.append(vec)

    if vecs:
        vectors = np.vstack(vecs).astype("float32")
        id_arr = np.array(ids, dtype="int64")
        index.add_with_ids(vectors, id_arr)
        logger.info(f"Built FAISS index: {index.ntotal} vectors from {DB_PATH}")
    else:
        logger.info("No valid vectors found in memory.db")

    return index


def _load_or_build_index() -> faiss.IndexIDMap2:
    """Load persisted index if available, otherwise rebuild from DB."""
    if os.path.exists(INDEX_PATH):
        try:
            raw = faiss.read_index(INDEX_PATH)
            idx = faiss.IndexIDMap2(raw) if not isinstance(raw, faiss.IndexIDMap2) else raw
            logger.info(f"Loaded FAISS index: {idx.ntotal} vectors")
            return idx
        except Exception as e:
            logger.warning(f"Failed to load FAISS index, rebuilding: {e}")
    return _build_index_from_db()


_index = _load_or_build_index()

# Forward declaration for count_memories (defined later in file)
def count_memories() -> int:
    return 0

# Initialize Bloom filter for pre‑filtering IDs based on metadata flag "bloom": true
from .bloom_filter import SimpleBloomFilter

def _load_bloom_filter() -> SimpleBloomFilter:
    """Load all memory IDs into the Bloom filter for fast membership testing.
    This enables pre-filtering to skip expensive vector lookups when needed.
    """
    # Estimate capacity as total memories
    total = count_memories()
    bf = SimpleBloomFilter(capacity=max(total, 1), error_rate=0.01)
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cur = conn.cursor()
        # Load all memory IDs into the bloom filter
        cur.execute("SELECT id FROM memories")
        for (mem_id,) in cur.fetchall():
            bf.add(mem_id)
        conn.close()
        logger.info(f"Loaded Bloom filter with {total} memory IDs")
    except Exception as e:
        logger.warning(f"Failed to build Bloom filter: {e}")
    return bf

_bloom_filter = _load_bloom_filter()


# ---------- Helpers ----------

def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


# ---------- Embedding via API ----------


def denoise_embeddings(
    vectors: np.ndarray,
    remove_top_d: int = 3,
    target_dim: int | None = None,
) -> np.ndarray:
    """Denoise embeddings for better semantic similarity.
    Subtracts the top principal components representing corpus-level bias.
    """
    from sklearn.decomposition import PCA

    vecs = vectors.astype(np.float64)

    # Step 1: mean-center
    mean = vecs.mean(axis=0, keepdims=True)
    vecs = vecs - mean

    # Step 2: all-but-the-top
    d = min(remove_top_d, vecs.shape[1], len(vecs) - 1)
    if d > 0:
        pca_top = PCA(n_components=d, random_state=42)
        pca_top.fit(vecs)
        components = pca_top.components_
        projections = vecs @ components.T @ components
        vecs = vecs - projections

    # Step 3: optional PCA reduction
    if target_dim is not None and vecs.shape[1] > target_dim:
        pca_reduce = PCA(n_components=target_dim, random_state=42)
        vecs = pca_reduce.fit_transform(vecs)

    # Step 4: L2-normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vec_denoised = vecs / (norms + 1e-8)

    return vec_denoised.astype(np.float32)

def get_embedding(text: str) -> List[float]:
    """Generate embedding for given text via the embedding server API.
    Uses the primary model by default; if the text contains non‑ASCII characters
    and a multilingual model is available, it uses that model instead. Falls back
    to a deterministic hash‑seeded vector if embedding generation fails.
    """
    try:
        req = urllib.request.Request(
            EMBEDDING_SERVER_URL,
            data=json.dumps({'text': text[:2000]}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            embedding = data.get('embedding')
            if embedding and len(embedding) == EMB_DIM:
                return embedding
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
        logger.warning(f"Embedding server unavailable ({e}), using fallback")

    # Deterministic fallback so the same text always gets the same vector
    seed = int.from_bytes(text.encode("utf-8"), "little", signed=False) % (2**32)
    rng = np.random.RandomState(seed)
    return rng.randn(EMB_DIM).astype("float32").tolist()


# ---------- CRUD ----------

def add_memory(embedding: List[float], metadata: Dict[str, Any]) -> int:
    """Insert a memory into the canonical DB and FAISS index.

    Compatible with the full memory_engine schema.
    """
    vec = _normalize(np.array(embedding, dtype='float32'))
    vec_blob = vec.astype("float32").tobytes()
    meta_json = json.dumps(metadata)
    text = metadata.get('text', '')
    ts = metadata.get('timestamp', time.time())
    relevance = metadata.get('relevance', 1.0)

    with _lock:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO memories (vector, metadata, text, timestamp, relevance, merged) VALUES (?,?,?,?,?,?)',
            (vec_blob, meta_json, text, ts, relevance, 0),
        )
        mem_id = cur.lastrowid
        conn.commit()
        conn.close()

        _index.add_with_ids(
            vec.reshape(1, -1).astype("float32"),
            np.array([mem_id], dtype='int64'),
        )
    return mem_id


def search_memory(query_embedding: List[float], k: int = 5,
                  decay_rate: float = 0.005, use_graph: bool = True) -> List[Dict]:
    """Return top-k memories by cosine similarity with temporal decay re-ranking
    and optional graph-neighbor expansion.
    
    Pipeline:
      1. FAISS search (over-fetch 3x)
      2. Apply exponential time-decay to raw scores
      3. Optionally expand result set with graph neighbors
      4. Return top-k by decayed score
    """
    import math
    q = _normalize(np.array(query_embedding, dtype='float32')).reshape(1, -1)
    
    # Over-fetch for re-ranking headroom
    fetch_k = min(k * 3, _index.ntotal) if _index.ntotal > 0 else k
    with _lock:
        distances, ids = _index.search(q, fetch_k)

    now = time.time()
    results = []
    seen_ids = set()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cur = conn.cursor()
    
    # Check if importance_score column exists (written by importance_scanner.py)
    _has_imp_col = 'importance_score' in {
        row[1] for row in cur.execute('PRAGMA table_info(memories)').fetchall()
    }
    _imp_sql = (
        'SELECT metadata, text, timestamp, importance_score FROM memories WHERE id = ?'
        if _has_imp_col else
        'SELECT metadata, text, timestamp FROM memories WHERE id = ?'
    )

    for dist, mem_id in zip(distances[0], ids[0]):
        if mem_id == -1:
            continue
        mem_id = int(mem_id)
        # Bloom filter pre‑filter: skip if ID not in filter (shouldn't happen with full load)
        if mem_id not in _bloom_filter:
            continue
        seen_ids.add(mem_id)
        cur.execute(_imp_sql, (mem_id,))
        row = cur.fetchone()
        if row is None:
            continue
        meta = json.loads(row[0]) if row[0] else {}
        text = row[1] or meta.get('text', '')
        ts = row[2] or meta.get('timestamp', now)
        imp_score = float(row[3]) if _has_imp_col and row[3] is not None else None
        
        # Time-decay scoring
        hours_age = max(0, (now - ts) / 3600.0)
        decay = math.exp(-decay_rate * hours_age)
        decayed_score = float(dist) * decay
        
        results.append({
            'id': mem_id,
            'score': decayed_score,
            'raw_score': float(dist),
            'text': text,
            'metadata': meta,
            'importance_score': imp_score,
        })
    
    # Graph-neighbor expansion: boost memories connected to top results
    if use_graph and results:
        try:
            from .semantic_graph import graph_neighbors as _graph_neighbors
        except ImportError:
            try:
                from semantic_graph import graph_neighbors as _graph_neighbors
            except ImportError:
                _graph_neighbors = None
        
        if _graph_neighbors is not None:
            # Get graph neighbors of top-3 results
            top_ids = [r['id'] for r in sorted(results, key=lambda x: x['score'], reverse=True)[:3]]
            graph_candidates = set()
            for tid in top_ids:
                try:
                    nbrs = _graph_neighbors(tid, depth=1)
                    graph_candidates.update(n for n in nbrs if n > 0 and n not in seen_ids)
                except Exception:
                    pass
            
            # Fetch and score graph neighbors
            _g_sql = (
                'SELECT vector, metadata, text, timestamp, importance_score FROM memories WHERE id = ?'
                if _has_imp_col else
                'SELECT vector, metadata, text, timestamp FROM memories WHERE id = ?'
            )
            for gid in list(graph_candidates)[:k]:
                cur.execute(_g_sql, (gid,))
                row = cur.fetchone()
                if row is None:
                    continue
                vec = np.frombuffer(row[0], dtype='float32')
                if vec.shape[0] != EMB_DIM:
                    continue
                g_score = float(np.dot(_normalize(vec), q.flatten()))
                meta = json.loads(row[1]) if row[1] else {}
                text = row[2] or meta.get('text', '')
                ts = row[3] or meta.get('timestamp', now)
                g_imp = float(row[4]) if _has_imp_col and row[4] is not None else None
                hours_age = max(0, (now - ts) / 3600.0)
                decay = math.exp(-decay_rate * hours_age)
                
                results.append({
                    'id': gid,
                    'score': g_score * decay,
                    'raw_score': g_score,
                    'text': text,
                    'metadata': meta,
                    'importance_score': g_imp,
                })
    
    conn.close()
    
    # Sort by decayed score, return top-k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:k]


def delete_memory(mem_id: int) -> bool:
    """Remove a memory by ID from both FAISS and SQLite."""
    try:
        with _lock:
            _index.remove_ids(np.array([mem_id], dtype='int64'))
            conn = sqlite3.connect(DB_PATH, timeout=10)
            conn.execute('DELETE FROM memories WHERE id = ?', (mem_id,))
            conn.commit()
            conn.close()
        return True
    except Exception:
        return False


def update_memory_text(mem_id: int, new_text: str) -> bool:
    """Update just the text content of a memory (vector stays the same)."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute('UPDATE memories SET text = ? WHERE id = ?', (new_text, mem_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error("Failed to update memory %s: %s", mem_id, e)
        return False


def count_memories() -> int:
    """Return the total number of stored memories."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        count = conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def list_memories_by_age(limit: int = 5000) -> List[Dict]:
    """List memories ordered by ID (insertion order)."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        rows = conn.execute(
            'SELECT id, metadata FROM memories ORDER BY id ASC LIMIT ?', (limit,)
        ).fetchall()
        conn.close()
        return [
            {'id': row[0], 'metadata': json.loads(row[1]) if row[1] else {}}
            for row in rows
        ]
    except Exception:
        return []


def list_all_memories_with_text(limit: int = 10000) -> List[Dict]:
    """List memories with their text content for content-based scanning.
    Returns id, text, and metadata for each entry."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        rows = conn.execute(
            'SELECT id, text, metadata FROM memories ORDER BY id ASC LIMIT ?', (limit,)
        ).fetchall()
        conn.close()
        return [
            {
                'id': row[0],
                'text': row[1] or '',
                'metadata': json.loads(row[2]) if row[2] else {},
            }
            for row in rows
        ]
    except Exception:
        return []


def save_index():
    """Persist the FAISS index to disk."""
    try:
        with _lock:
            faiss.write_index(_index, INDEX_PATH)
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")


def rebuild_index():
    """Force rebuild FAISS index from the canonical DB."""
    global _index
    with _lock:
        _index = _build_index_from_db()
    save_index()


import atexit
atexit.register(save_index)

if __name__ == '__main__':
    print(f"DB: {DB_PATH}")
    print(f"Index: {INDEX_PATH}")
    print(f"Memories in DB: {count_memories()}")
    print(f"Vectors in FAISS: {_index.ntotal}")
