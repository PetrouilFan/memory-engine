#!/usr/bin/env python3
"""
holographic_memory.py – Time-decayed semantic memory with deduplication.

Cross-platform compatible (Windows + Linux).
"""
import os
import re
import sys
import math
import json
import logging
import tempfile
import threading
from typing import List, Dict, Optional, Any
from datetime import datetime
from hashlib import sha256
import numpy as np

# Cross-platform file locking
if sys.platform == 'win32':
    import msvcrt
    def _lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    def _unlock_file(f):
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl
    def _lock_file(f):
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def _unlock_file(f):
        fcntl.flock(f, fcntl.LOCK_UN)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HolographicMemory")

try:
    from .vector_memory_wrapper import (
        search_memory, add_memory as vector_add, get_embedding,
        delete_memory, count_memories, list_memories_by_age,
    )
except ImportError:
    try:
        from vector_memory_wrapper import (
            search_memory, add_memory as vector_add, get_embedding,
            delete_memory, count_memories, list_memories_by_age,
        )
    except ImportError:
        # Mocks for standalone testing
        def get_embedding(text): return [0.1]*768
        def vector_add(text, metadata): pass
        def search_memory(vector, k=5): return []
        def delete_memory(mem_id): return False
        def count_memories(): return 0
        def list_memories_by_age(limit=5000): return []


class HolographicMemory:
    PRUNE_INTERVAL = 50  # only check pruning every N writes

    def __init__(self, decay_rate: float = 0.01, max_entries: int = 5000):
        self.decay_lambda = decay_rate
        self.max_entries = max_entries
        self._hash_cache = set()
        self._embedding_cache: Dict[str, Any] = {}  # manual cache (no lru_cache leak)
        self._embedding_cache_max = 256
        self._lock = threading.Lock()
        self._write_count = 0
        # Cross-platform temp directory for lock file
        self._lock_file = os.path.join(tempfile.gettempdir(), "holographic_memory.lock")
        self._initialize_cache()

    def _initialize_cache(self):
        """Warm up hash cache from DB metadata (no zero-vector scan)."""
        try:
            rows = list_memories_by_age(limit=self.max_entries)
            for r in rows:
                h = r.get('metadata', {}).get('hash')
                if h:
                    self._hash_cache.add(h)
        except Exception as e:
            logger.warning(f"Cache initialization failed: {e}")

    def _cached_embedding(self, text: str):
        """Cached embedding lookup. Strips whitespace but preserves case
        to avoid destroying meaning for proper nouns and acronyms."""
        key = text.strip()
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        embedding = get_embedding(key)
        # Evict oldest entry if cache is full (simple FIFO)
        if len(self._embedding_cache) >= self._embedding_cache_max:
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        self._embedding_cache[key] = embedding
        return embedding

    def get_health_status(self) -> Dict[str, Any]:
        """Health check API for monitoring system integrity."""
        stats = self.get_stats()
        return {
            "healthy": stats.get("status") == "active",
            "metrics": stats,
            "timestamp": datetime.now().isoformat()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Returns health metrics and average memory age."""
        try:
            count = count_memories()
            if count == 0:
                return {"status": "empty", "total_entries": 0}
            all_mems = list_memories_by_age(limit=self.max_entries)
            ts = [m.get('metadata', {}).get('timestamp', 0) for m in all_mems]
            ts = [t for t in ts if t > 0]
            return {
                "status": "active",
                "total_entries": count,
                "cache_utilization": len(self._hash_cache),
                "avg_age_hours": round((datetime.now().timestamp() - np.mean(ts)) / 3600, 2) if ts else 0
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _semantic_split(self, text: str) -> List[str]:
        """Improved Regex splitting for semantic preservation."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current_chunk = [], []
        for sent in sentences:
            current_chunk.append(sent)
            if len(current_chunk) >= 3:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Adds memory with cross-process locking and periodic auto-pruning."""
        if not text.strip():
            return
        with open(self._lock_file, 'a+') as f:
            try:
                _lock_file(f)
                chunks = self._semantic_split(text)
                added = 0
                for chunk in chunks:
                    chunk_hash = sha256(chunk.encode()).hexdigest()
                    with self._lock:
                        if chunk_hash in self._hash_cache:
                            continue
                        self._hash_cache.add(chunk_hash)

                    meta = (metadata or {}).copy()
                    meta.update({'hash': chunk_hash, 'timestamp': datetime.now().timestamp()})
                    vector_add(chunk, metadata=meta)
                    added += 1

                # Only prune every PRUNE_INTERVAL writes, not on every call
                if added > 0:
                    self._write_count += added
                    if self._write_count >= self.PRUNE_INTERVAL:
                        self._prune_old_memories()
                        self._write_count = 0

                _unlock_file(f)
            except (BlockingIOError, OSError):
                logger.warning("Lock busy (another process is writing). Skipping.")

    def _prune_old_memories(self):
        """Removes the oldest entries if max_entries is exceeded.
        Uses DB-backed listing instead of zero-vector search.
        Errors on individual deletes are logged but don't abort the loop."""
        count = count_memories()
        if count <= self.max_entries:
            return
        overshoot = count - self.max_entries
        oldest = list_memories_by_age(limit=overshoot + 10)
        deleted = 0
        for entry in oldest:
            if deleted >= overshoot:
                break
            try:
                delete_memory(entry['id'])
                h = entry.get('metadata', {}).get('hash')
                if h:
                    self._hash_cache.discard(h)
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to prune memory {entry['id']}: {e}")
        if deleted:
            logger.info(f"Pruned {deleted} old memories (was {count}, target {self.max_entries})")

    def redshifted_recall(self, query: str, k: int = 5, threshold: float = 0.3, filters: Optional[Dict] = None) -> List[Dict]:
        """Time-decayed semantic recall with metadata filtering."""
        try:
            query_vec = self._cached_embedding(query)
            raw_results = search_memory(query_vec, k=k*5)
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return []
        
        now = datetime.now().timestamp()
        final = []
        for r in raw_results:
            if filters and not all(r.get('metadata', {}).get(key) == val for key, val in filters.items()):
                continue

            base_score = r.get('score', 0)
            ts = r.get('metadata', {}).get('timestamp', now)
            hours_age = (now - ts) / 3600.0
            decay_factor = math.exp(-self.decay_lambda * hours_age)
            decayed_score = base_score * decay_factor
            
            if decayed_score >= threshold:
                final.append({
                    'text': r['text'],
                    'decayed_score': round(decayed_score, 4),
                    'metadata': r['metadata'],
                    'age_hours': round(hours_age, 2)
                })
        return sorted(final, key=lambda x: x['decayed_score'], reverse=True)[:k]
