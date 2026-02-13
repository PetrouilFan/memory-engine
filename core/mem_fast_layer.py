import sqlite3
import os
import time
import json
import logging
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Canonical paths
_ME_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DB_PATH = os.path.join(_ME_ROOT, 'memory.db')
FAST_INDEX_PATH = os.path.join(_ME_ROOT, 'index_fast.faiss')
SLOW_INDEX_PATH = os.path.join(_ME_ROOT, 'index_slow.faiss')
EMB_DIM = 384

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryLayer:
    def __init__(self):
        self.migrate_schema()
        self.index_fast, self.index_slow = self.load_indexes()
        
    def migrate_schema(self):
        """Add mem_fast table and importance_score if missing."""
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS mem_fast (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                vector BLOB NOT NULL,
                text TEXT,
                timestamp REAL NOT NULL,
                expires_at REAL,
                importance REAL DEFAULT 0,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        ''')
        cur.execute('PRAGMA table_info(memories)')
        cols = [row[1] for row in cur.fetchall()]
        if 'importance_score' not in cols:
            cur.execute('ALTER TABLE memories ADD COLUMN importance_score REAL DEFAULT 0')
        if 'access_counter' not in cols:
            cur.execute('ALTER TABLE memories ADD COLUMN access_counter INTEGER DEFAULT 0')
        conn.commit()
        conn.close()

    def load_indexes(self):
        if os.path.exists(FAST_INDEX_PATH):
            idx_f = faiss.read_index(FAST_INDEX_PATH)
        else:
            idx_f = faiss.IndexIDMap(faiss.IndexFlatIP(EMB_DIM))
            
        if os.path.exists(SLOW_INDEX_PATH):
            idx_s = faiss.read_index(SLOW_INDEX_PATH)
        else:
            idx_s = faiss.IndexIDMap(faiss.IndexFlatIP(EMB_DIM))
        return idx_f, idx_s

    def save_indexes(self):
        faiss.write_index(self.index_fast, FAST_INDEX_PATH)
        faiss.write_index(self.index_slow, SLOW_INDEX_PATH)

    def add_to_fast(self, memory_id: int, vector: List[float], text: str, importance: float = 1.0, ttl_hours: int = 24):
        """Add a memory to the fast layer (mem_fast table + fast index)."""
        vec_np = np.array(vector, dtype='float32')
        # Normalize for IndexFlatIP (cosine similarity)
        norm = np.linalg.norm(vec_np)
        if norm > 1e-6: vec_np = vec_np / norm
        
        now = time.time()
        expires_at = now + (ttl_hours * 3600)
        
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO mem_fast (memory_id, vector, text, timestamp, expires_at, importance)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (memory_id, vec_np.tobytes(), text, now, expires_at, importance))
        fast_id = cur.lastrowid
        conn.commit()
        conn.close()
        
        self.index_fast.add_with_ids(vec_np.reshape(1, -1), np.array([fast_id], dtype='int64'))
        self.save_indexes()
        return fast_id

    def dual_search(self, query_vector: List[float], k_fast: int = 5, k_slow: int = 10) -> List[Dict]:
        """Search both fast and slow indexes and merge results."""
        q_vec = np.array(query_vector, dtype='float32')
        norm = np.linalg.norm(q_vec)
        if norm > 1e-6: q_vec = q_vec / norm
        q_vec = q_vec.reshape(1, -1)
        
        results = []
        seen_memory_ids = set()
        
        # 1. Search Fast Index
        if self.index_fast.ntotal > 0:
            dists, ids = self.index_fast.search(q_vec, min(k_fast, self.index_fast.ntotal))
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            for d, fid in zip(dists[0], ids[0]):
                if fid == -1: continue
                cur.execute('SELECT memory_id, text, importance FROM mem_fast WHERE id = ?', (int(fid),))
                row = cur.fetchone()
                if row:
                    mid = row[0]
                    results.append({
                        'memory_id': mid,
                        'text': row[1],
                        'score': float(d) * 1.2, # 20% boost for fast memory
                        'layer': 'fast',
                        'importance': row[2]
                    })
                    seen_memory_ids.add(mid)
            conn.close()
            
        # 2. Search Slow Index
        if self.index_slow.ntotal > 0:
            dists, ids = self.index_slow.search(q_vec, min(k_slow*2, self.index_slow.ntotal))
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            for d, mid in zip(dists[0], ids[0]):
                if mid == -1: continue
                mid = int(mid)
                if mid in seen_memory_ids: continue
                
                cur.execute('SELECT text, importance_score FROM memories WHERE id = ?', (mid,))
                row = cur.fetchone()
                if row:
                    # Increment access counter for retrieved memory
                    cur.execute('UPDATE memories SET access_counter = access_counter + 1 WHERE id = ?', (mid,))
                    conn.commit()
                    results.append({
                        'memory_id': mid,
                        'text': row[0],
                        'score': float(d),
                        'layer': 'slow',
                        'importance': row[1]
                    })
                    seen_memory_ids.add(mid)
            conn.close()
            
        return sorted(results, key=lambda x: x['score'], reverse=True)[:max(k_fast, k_slow)]

    def maintenance(self):
        """Evict expired fast memories and promote important ones to slow if missing."""
        now = time.time()
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # 1. Evict expired
        cur.execute('SELECT id, memory_id FROM mem_fast WHERE expires_at < ?', (now,))
        expired_rows = cur.fetchall()
        expired_ids = [row[0] for row in expired_rows]
        
        if expired_ids:
            logger.info(f"Evicting {len(expired_ids)} expired memories from fast layer")
            cur.execute(f"DELETE FROM mem_fast WHERE id IN ({','.join(['?']*len(expired_ids))})", expired_ids)
            try:
                self.index_fast.remove_ids(np.array(expired_ids, dtype='int64'))
            except Exception as e:
                logger.warning(f"Failed to remove IDs from fast index: {e}")

        # 2. Sync Slow Index (rebuild it from all memories to ensure consistency)
        cur.execute("SELECT id, vector FROM memories ORDER BY id")
        rows = cur.fetchall()
        
        ids, vecs = [], []
        for mem_id, vec_blob in rows:
            if not vec_blob: continue
            vec = np.frombuffer(vec_blob, dtype="float32").copy()
            if vec.shape[0] != EMB_DIM: continue
            norm = np.linalg.norm(vec)
            if norm > 1e-6: vec = vec / norm
            ids.append(int(mem_id))
            vecs.append(vec)
        
        if vecs:
            new_idx_slow = faiss.IndexIDMap(faiss.IndexFlatIP(EMB_DIM))
            new_idx_slow.add_with_ids(np.vstack(vecs).astype("float32"), np.array(ids, dtype="int64"))
            self.index_slow = new_idx_slow
            logger.info(f"Slow index synced: {self.index_slow.ntotal} vectors")
        
        # 3. Promote high‑usage memories to fast layer (dynamic plasticity)
        cur.execute('SELECT id, vector, text, access_counter FROM memories WHERE access_counter > 100')
        high_rows = cur.fetchall()
        for mem_id, vec_blob, text, counter in high_rows:
            if not vec_blob: continue
            # Check if already in fast layer
            cur.execute('SELECT 1 FROM mem_fast WHERE memory_id = ?', (mem_id,))
            if cur.fetchone():
                continue
            # Use higher importance based on counter
            importance = min(5.0, 1.0 + counter/100.0)
            vec = np.frombuffer(vec_blob, dtype='float32').copy().tolist()
            self.add_to_fast(mem_id, vec, text, importance=importance, ttl_hours=48)
        # End of promotion

if __name__ == "__main__":
    layer = MemoryLayer()
    layer.maintenance()
    logger.info("Memory layer initialized and maintenance performed.")
