import os
import json
import threading
from typing import List, Dict, Any
import numpy as np
import faiss
import sqlite3

# Paths for persistence
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'vector_index.faiss')
DB_PATH = os.path.join(os.path.dirname(__file__), 'metadata.db')

_lock = threading.Lock()

# ---------- SQLite metadata layer ----------

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vector_id INTEGER UNIQUE,
            metadata TEXT
        )
    ''')
    conn.commit()
    conn.close()

_init_db()

# ---------- FAISS index layer ----------

def _load_index(dim: int) -> faiss.IndexIDMap2:
    if os.path.exists(INDEX_PATH):
        raw = faiss.read_index(INDEX_PATH)
        index = faiss.IndexIDMap2(raw)
    else:
        quantizer = faiss.IndexFlatIP(dim)  # inner product works with normalized vectors
        raw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIDMap2(raw)
    return index

# Assume embedding dimension is known (e.g., 768 for nomic-embed-text)
EMB_DIM = 768
_index = _load_index(EMB_DIM)

# ---------- Helper functions ----------

def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def add_memory(embedding: List[float], metadata: Dict[str, Any]):
    """Add a memory to the vector store and persist its metadata.
    Embedding will be L2-normalized for cosine similarity via inner product.
    """
    vec = _normalize(np.array(embedding, dtype='float32')).reshape(1, -1)
    meta_json = json.dumps(metadata)
    with _lock:
        # Insert placeholder row to get a unique ID
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('INSERT INTO memories (metadata) VALUES (?)', (meta_json,))
        mem_id = cur.lastrowid
        conn.commit()
        conn.close()
        # Add to FAISS with the same ID
        _index.add_with_ids(vec, np.array([mem_id], dtype='int64'))
        # Save index to disk
        faiss.write_index(_index.quantizer, INDEX_PATH)
    return mem_id

def search_memory(query_embedding: List[float], k: int = 5):
    """Return top-k memory IDs and similarity scores for a query vector."""
    q = _normalize(np.array(query_embedding, dtype='float32')).reshape(1, -1)
    with _lock:
        distances, ids = _index.search(q, k)
    results = []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for dist, mem_id in zip(distances[0], ids[0]):
        if mem_id == -1:
            continue
        cur.execute('SELECT metadata FROM memories WHERE id = ?', (mem_id,))
        row = cur.fetchone()
        meta = json.loads(row[0]) if row else {}
        results.append({'id': mem_id, 'score': float(dist), 'metadata': meta})
    conn.close()
    return results

def delete_memory(mem_id: int) -> bool:
    """Remove a memory by ID from both FAISS and SQLite."""
    try:
        with _lock:
            _index.remove_ids(np.array([mem_id], dtype='int64'))
            conn = sqlite3.connect(DB_PATH)
            conn.execute('DELETE FROM memories WHERE id = ?', (mem_id,))
            conn.commit()
            conn.close()
            faiss.write_index(_index.quantizer, INDEX_PATH)
        return True
    except Exception:
        return False

def count_memories() -> int:
    """Return the total number of stored memories."""
    try:
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0

def list_memories_by_age(limit: int = 5000) -> List[Dict]:
    """List memories ordered by ID (insertion order) via SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            'SELECT id, metadata FROM memories ORDER BY id ASC LIMIT ?', (limit,)
        ).fetchall()
        conn.close()
        results = []
        for row in rows:
            meta = json.loads(row[1]) if row[1] else {}
            results.append({'id': row[0], 'metadata': meta})
        return results
    except Exception:
        return []

def save_index():
    """Persist the FAISS index to disk (called automatically on exit)."""
    with _lock:
        faiss.write_index(_index.quantizer, INDEX_PATH)

def load_index():
    """Reload the FAISS index from disk (called on module import)."""
    global _index
    with _lock:
        _index = _load_index(EMB_DIM)

# Register atexit handler to auto-save the index when the process exits
import atexit
atexit.register(save_index)

if __name__ == '__main__':
    test_emb = np.random.rand(EMB_DIM).tolist()
    add_memory(test_emb, {'text': 'example', 'timestamp': '2026-02-05'})
    print(search_memory(test_emb))
