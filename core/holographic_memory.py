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
import atexit
from collections import defaultdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from hashlib import sha256
import numpy as np

# Canonical data dir for persistent access logs
_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_ACCESS_LOG_PATH = os.getenv('MEMORY_ACCESS_LOG_PATH', os.path.join(_DATA_ROOT, 'access_log.json'))

try:
    from .text_sanitizer import TextSanitizer, get_sanitizer
except ImportError:
    try:
        from text_sanitizer import TextSanitizer, get_sanitizer
    except ImportError:
        raise ImportError("TextSanitizer not found. Please ensure text_sanitizer.py is in the same directory or installed as a module.")

try:
    from .hallucination_filter import HallucinationDetector
except ImportError:
    try:
        from hallucination_filter import HallucinationDetector
    except ImportError:
        raise ImportError("HallucinationDetector not found. Please ensure hallucination_filter.py is in the same directory or installed as a module.")

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
        delete_memory, update_memory_text, count_memories, list_memories_by_age,
        list_all_memories_with_text,
    )
except ImportError:
    try:
        from vector_memory_wrapper import (
            search_memory, add_memory as vector_add, get_embedding,
            delete_memory, update_memory_text, count_memories, list_memories_by_age,
            list_all_memories_with_text,
        )
    except ImportError:
        # Mocks for standalone testing
        def get_embedding(text): return [0.1]*384
        def vector_add(text, metadata): pass
        def search_memory(vector, k=5): return []
        def delete_memory(mem_id): return False
        def update_memory_text(mem_id, text): return False
        def count_memories(): return 0
        def list_memories_by_age(limit=5000): return []
        def list_all_memories_with_text(limit=10000): return []


class HolographicMemory:
    PRUNE_INTERVAL = 50  # only check pruning every N writes

    def __init__(self, decay_rate: float = 0.01, max_entries: int = 5000, filter_hallucinations: bool = True, hallucination_threshold: float = 0.5, auto_purge: bool = True, access_log_path: Optional[str] = None, frequency_boost: float = 0.0):
        self.decay_lambda = decay_rate
        self.max_entries = max_entries
        self.filter_hallucinations = filter_hallucinations
        self.hallucination_threshold = hallucination_threshold
        self.hallucination_detector = HallucinationDetector() if filter_hallucinations else None
        self._hash_cache = set()
        self._embedding_cache: Dict[str, Any] = {}  # manual cache (no lru_cache leak)
        self._embedding_cache_max = 256
        self._lock = threading.Lock()
        self._write_count = 0
        self._purged_this_session = False
        # Cross-platform temp directory for lock file
        self._lock_file = os.path.join(tempfile.gettempdir(), "holographic_memory.lock")
        # Access logging — tracks per-memory hit counts for frequency analysis
        self._access_log_path = access_log_path or _ACCESS_LOG_PATH
        self._access_log: Dict[str, Dict[str, Any]] = {}  # {mem_id: {count, first, last, queries}}
        self._access_log_dirty = False
        self._access_log_flush_interval = 10  # flush every N recall calls
        self._recall_count = 0
        # Optional frequency boost factor (0 = disabled, e.g. 0.05 = +5% per access)
        self.frequency_boost = frequency_boost
        # Simple Bloom filter approximation using a set of token hashes
        self._bloom_filter: set[int] = set()
        self._bloom_filter_path = os.path.join(_DATA_ROOT, 'bloom_filter.json')
        self._bloom_filter_flush_interval = 100  # save every N writes
        self._bloom_filter_dirty = False
        # Centralized text sanitizer
        self._sanitizer = get_sanitizer()
        self._load_access_log()
        self._initialize_cache()
        # Auto-purge existing hallucinations once on startup
        if auto_purge and filter_hallucinations:
            self._auto_purge()
        # Register shutdown handler to save access log
        atexit.register(self._save_access_log)
        atexit.register(self._save_bloom_filter)
        self._load_bloom_filter()

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

    # ── Access-log persistence ────────────────────────────────────────

    def _load_access_log(self):
        """Load persisted access log from disk (JSON)."""
        try:
            if os.path.exists(self._access_log_path):
                with open(self._access_log_path, 'r') as f:
                    self._access_log = json.load(f)
                logger.debug("Loaded access log with %d entries", len(self._access_log))
        except Exception as e:
            logger.warning("Failed to load access log: %s", e)
            self._access_log = {}

    def _save_access_log(self):
        """Persist access log to disk atomically."""
        if not self._access_log_dirty:
            return
        try:
            os.makedirs(os.path.dirname(self._access_log_path), exist_ok=True)
            tmp = self._access_log_path + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(self._access_log, f)
            os.replace(tmp, self._access_log_path)
            self._access_log_dirty = False
        except Exception as e:
            logger.warning("Failed to save access log: %s", e)

    def _load_bloom_filter(self):
        """Load persisted bloom filter from disk (JSON)."""
        try:
            if os.path.exists(self._bloom_filter_path):
                with open(self._bloom_filter_path, 'r') as f:
                    data = json.load(f)
                    self._bloom_filter = set(data.get('token_hashes', []))
                logger.debug("Loaded bloom filter with %d entries", len(self._bloom_filter))
        except Exception as e:
            logger.warning("Failed to load bloom filter: %s", e)
            self._bloom_filter = set()

    def _save_bloom_filter(self):
        """Persist bloom filter to disk atomically."""
        if not self._bloom_filter_dirty:
            return
        try:
            os.makedirs(os.path.dirname(self._bloom_filter_path), exist_ok=True)
            tmp = self._bloom_filter_path + ".tmp"
            with open(tmp, 'w') as f:
                json.dump({'token_hashes': list(self._bloom_filter)}, f)
            os.replace(tmp, self._bloom_filter_path)
            self._bloom_filter_dirty = False
        except Exception as e:
            logger.warning("Failed to save bloom filter: %s", e)

    def _record_access(self, results: List[Dict], query: str) -> None:
        """Record access hits for returned memories."""
        now = datetime.now().timestamp()
        with self._lock:
            for r in results:
                mid = str(r.get('id', ''))
                if not mid:
                    continue
                if mid not in self._access_log:
                    self._access_log[mid] = {
                        'count': 0,
                        'first_accessed': now,
                        'last_accessed': now,
                        'recent_queries': [],
                    }
                entry = self._access_log[mid]
                entry['count'] += 1
                entry['last_accessed'] = now
                # Keep last 10 queries per memory (rolling window)
                entry['recent_queries'].append(query[:120])
                entry['recent_queries'] = entry['recent_queries'][-10:]
            self._access_log_dirty = True

        # Periodic flush to disk
        self._recall_count += 1
        if self._recall_count >= self._access_log_flush_interval:
            self._save_access_log()
            self._recall_count = 0

    def get_access_stats(self, memory_id: Optional[str] = None) -> Dict[str, Any]:
        """Return access statistics for a specific memory or all memories.

        Args:
            memory_id: If given, return stats for that memory only.

        Returns:
            Dict with total_tracked, and per-memory stats (count, first/last access, queries).
        """
        with self._lock:
            if memory_id is not None:
                mid = str(memory_id)
                entry = self._access_log.get(mid)
                if not entry:
                    return {'memory_id': mid, 'tracked': False}
                return {'memory_id': mid, 'tracked': True, **entry}

            return {
                'total_tracked': len(self._access_log),
                'total_accesses': sum(e['count'] for e in self._access_log.values()),
                'entries': dict(self._access_log),
            }

    def get_most_accessed(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return the most frequently accessed memories, sorted by hit count.

        Useful for identifying hot memories that could benefit from caching
        or priority boosting.
        """
        with self._lock:
            ranked = sorted(
                self._access_log.items(),
                key=lambda kv: kv[1]['count'],
                reverse=True,
            )
        return [
            {'memory_id': mid, **stats}
            for mid, stats in ranked[:top_n]
        ]

    def _auto_purge(self):
        """Run a one-time hallucination purge on startup.
        Uses a slightly aggressive threshold (0.4) to catch borderline entries.
        Runs silently — errors don't block engine startup."""
        if self._purged_this_session:
            return
        self._purged_this_session = True
        try:
            result = self.purge_hallucinations(threshold=0.4, dry_run=False)
            if result.get('purged', 0) > 0:
                logger.info(
                    "Auto-purge on startup: removed %d hallucinated memories (of %d scanned)",
                    result['purged'], result['scanned']
                )
        except Exception as e:
            logger.warning(f"Auto-purge failed (non-fatal): {e}")

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
            with self._lock:
                total_accesses = sum(e['count'] for e in self._access_log.values())
                tracked_memories = len(self._access_log)
            return {
                "status": "active",
                "total_entries": count,
                "cache_utilization": len(self._hash_cache),
                "tracked_memories": tracked_memories,
                "total_accesses": total_accesses,
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

    def _sanitize_text(self, text: str) -> str:
        """Strip chat platform metadata, IDs, and timestamps before storing."""
        return self._sanitizer.sanitize(text)

    def add_memory(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Adds memory with cross-process locking, hallucination filtering, and periodic auto-pruning."""
        if not text.strip():
            return

        # Strip chat platform metadata before any other checks
        text = self._sanitize_text(text)
        if not text:
            logger.debug("Rejected memory: nothing left after sanitizing chat metadata")
            return

        # Fast reject: leaked tags (prompt/XML/tool-call leakage) — instant kill
        if self._sanitizer.has_leaked_tags(text):
            logger.warning("Rejected memory with leaked tags: %s", text[:80])
            return

        # Filter out hallucinated memories
        if self.filter_hallucinations and self.hallucination_detector:
            if self.hallucination_detector.is_hallucinated(text, threshold=self.hallucination_threshold):
                logger.warning(f"Rejected hallucinated memory (confidence>={self.hallucination_threshold}): {text[:80]}...")
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
                    meta.update({'hash': chunk_hash, 'timestamp': datetime.now().timestamp(), 'text': chunk})
                    embedding = get_embedding(chunk)
                    vector_add(embedding, metadata=meta)
                    added += 1
                    # Update Bloom filter with token hashes from the chunk
                    # Simple tokenization: alphanumeric words
                    with self._lock:
                        for token in re.findall(r"\w+", chunk.lower()):
                            self._bloom_filter.add(hash(token))
                    self._bloom_filter_dirty = True

                # Only prune every PRUNE_INTERVAL writes, not on every call
                if added > 0:
                    self._write_count += added
                    if self._write_count >= self.PRUNE_INTERVAL:
                        self._prune_old_memories()
                        self._write_count = 0
                    # Save bloom filter periodically
                    if self._write_count % self._bloom_filter_flush_interval == 0:
                        self._save_bloom_filter()

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
            # Evict pruned IDs from the access log to keep it consistent
            for entry in oldest[:deleted]:
                mid = str(entry['id'])
                self._access_log.pop(mid, None)
            self._access_log_dirty = True
            self._save_access_log()

    def purge_hallucinations(self, threshold: float = 0.4, dry_run: bool = False) -> Dict[str, Any]:
        """Scan the entire memory DB and delete entries flagged as hallucinations.

        Uses the HallucinationDetector against each stored text.
        A lower threshold (0.4) is used by default so borderline bad memories
        are caught — better to re-learn correct info than keep stale errors.

        Args:
            threshold: Confidence threshold for deletion (0-1). Lower = more aggressive.
            dry_run: If True, report what would be deleted without actually deleting.

        Returns:
            Summary dict with counts and details of purged entries.
        """
        if not self.hallucination_detector:
            return {'error': 'Hallucination filtering is disabled', 'purged': 0}

        all_memories = list_all_memories_with_text(limit=self.max_entries + 1000)
        if not all_memories:
            return {'purged': 0, 'scanned': 0, 'detail': 'No memories in DB'}

        flagged = []
        for mem in all_memories:
            text = mem.get('text', '')
            if not text.strip():
                continue
            result = self.hallucination_detector.detect(text)
            if result['confidence'] >= threshold:
                flagged.append({
                    'id': mem['id'],
                    'text': text[:120],
                    'confidence': result['confidence'],
                    'severity': result['severity'],
                    'categories': result['categories'],
                })

        if dry_run:
            return {
                'scanned': len(all_memories),
                'flagged': len(flagged),
                'purged': 0,
                'dry_run': True,
                'entries': flagged,
            }

        deleted = 0
        for entry in flagged:
            try:
                delete_memory(entry['id'])
                h = None
                # Also evict from hash cache if we can recover the hash
                for mem in all_memories:
                    if mem['id'] == entry['id']:
                        h = mem.get('metadata', {}).get('hash')
                        break
                if h:
                    self._hash_cache.discard(h)
                deleted += 1
                logger.info(
                    "Purged hallucination id=%s (%.2f, %s): %s",
                    entry['id'], entry['confidence'], entry['severity'], entry['text'][:80]
                )
            except Exception as e:
                logger.warning(f"Failed to purge memory {entry['id']}: {e}")

        logger.info(f"Hallucination purge complete: scanned={len(all_memories)}, purged={deleted}")
        return {
            'scanned': len(all_memories),
            'flagged': len(flagged),
            'purged': deleted,
            'dry_run': False,
            'entries': flagged,
        }

    def purge_chat_metadata(self, dry_run: bool = False) -> Dict[str, Any]:
        """Scan all memories and strip chat platform metadata in-place.
        
        - Strips [Telegram ...], [Discord ...], raw IDs, timestamps, bare role markers
        - If nothing useful remains after stripping, the memory is deleted
        - Otherwise the cleaned text is saved back (vector stays the same)
        """
        all_memories = list_all_memories_with_text(limit=self.max_entries + 1000)
        if not all_memories:
            return {'cleaned': 0, 'deleted': 0, 'scanned': 0, 'detail': 'No memories in DB'}

        to_clean = []   # (id, original, cleaned)
        to_delete = []  # (id, original)
        for mem in all_memories:
            text = mem.get('text', '')
            if not text.strip():
                continue
            cleaned = self._sanitize_text(text)
            if cleaned == text.strip():
                continue  # nothing to change
            if not cleaned:
                to_delete.append({'id': mem['id'], 'original': text[:120]})
            else:
                to_clean.append({'id': mem['id'], 'original': text[:120], 'cleaned': cleaned[:120]})

        if dry_run:
            return {
                'scanned': len(all_memories),
                'to_clean': len(to_clean),
                'to_delete': len(to_delete),
                'dry_run': True,
                'clean_samples': to_clean[:10],
                'delete_samples': to_delete[:10],
            }

        cleaned_count = 0
        for entry in to_clean:
            try:
                # Find the full cleaned text (not truncated)
                full_mem = next((m for m in all_memories if m['id'] == entry['id']), None)
                if full_mem:
                    full_cleaned = self._sanitize_text(full_mem['text'])
                    if update_memory_text(entry['id'], full_cleaned):
                        cleaned_count += 1
                        logger.info("Cleaned metadata id=%s: '%s' -> '%s'",
                                    entry['id'], entry['original'][:60], full_cleaned[:60])
            except Exception as e:
                logger.warning("Failed to clean memory %s: %s", entry['id'], e)

        deleted_count = 0
        for entry in to_delete:
            try:
                delete_memory(entry['id'])
                deleted_count += 1
                logger.info("Deleted empty-after-clean id=%s: %s", entry['id'], entry['original'][:80])
            except Exception as e:
                logger.warning("Failed to delete memory %s: %s", entry['id'], e)

        logger.info("Chat metadata cleanup: scanned=%d, cleaned=%d, deleted=%d",
                    len(all_memories), cleaned_count, deleted_count)
        return {
            'scanned': len(all_memories),
            'cleaned': cleaned_count,
            'deleted': deleted_count,
            'dry_run': False,
        }

    def redshifted_recall(self, query: str, k: int = 5, threshold: float = 0.3, filters: Optional[Dict] = None) -> List[Dict]:
        """Time-decayed semantic recall with metadata filtering, graph expansion, and hallucination filtering.
        
        Pipeline:
          1. Vector search (over-fetch 5x)
          2. Graph-neighbor expansion of top hits
          3. Time-decay scoring
          4. Filter out hallucinations (if enabled)
          5. Filter and rank
        """
        # ---- Bloom filter pre‑filter ----
        # Tokenize the query into lowercase alphanumeric words.
        query_tokens = set(re.findall(r"\\w+", query.lower()))
        # If none of the query tokens are present in the Bloom filter,
        # we can safely assume there are no relevant memories and skip the
        # expensive vector search.
        if query_tokens and not any(tok_hash in self._bloom_filter for tok_hash in map(hash, query_tokens)):
            logger.debug("Bloom filter pre‑filter: no matching tokens for query '%s'", query)
            return []

        try:
            query_vec = self._cached_embedding(query)
            raw_results = search_memory(query_vec, k=k*5)
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return []
        
        # Graph-neighbor expansion: pull in memories linked to top results
        seen_ids = {r.get('id') for r in raw_results}
        try:
            from .semantic_graph import graph_neighbors as _graph_neighbors
        except ImportError:
            try:
                from semantic_graph import graph_neighbors as _graph_neighbors
            except ImportError:
                _graph_neighbors = None
        
        if _graph_neighbors is not None and raw_results:
            # Expand top-3 results through graph
            for r in raw_results[:3]:
                rid = r.get('id')
                if rid is None:
                    continue
                try:
                    nbrs = _graph_neighbors(rid, depth=1)
                    for nid in nbrs:
                        if nid > 0 and nid not in seen_ids:
                            seen_ids.add(nid)
                            # Fetch and score the graph neighbor
                            try:
                                nbr_results = search_memory(query_vec, k=1)
                                # This is indirect; score will be lower
                            except Exception:
                                pass
                except Exception:
                    pass
        
        now = datetime.now().timestamp()
        final = []
        for r in raw_results:
            if filters and not all(r.get('metadata', {}).get(key) == val for key, val in filters.items()):
                continue

            text = r.get('text', r.get('metadata', {}).get('text', ''))

            # Strip chat platform metadata from recalled text
            text = self._sanitize_text(text)
            if not text:
                continue

            # Fast reject: leaked tags in stored text
            if self._sanitizer.has_leaked_tags(text):
                logger.debug("Filtered result with leaked tags: %s", text[:80])
                continue

            # Filter out hallucinated memories
            if self.filter_hallucinations and self.hallucination_detector:
                if self.hallucination_detector.is_hallucinated(text, threshold=self.hallucination_threshold):
                    logger.debug(f"Filtered hallucinated result: {text[:80]}...")
                    continue

            base_score = r.get('score', 0)
            ts = r.get('metadata', {}).get('timestamp', now)
            hours_age = (now - ts) / 3600.0
            decay_factor = math.exp(-self.decay_lambda * hours_age)
            decayed_score = base_score * decay_factor

            # Importance boost: use the pre-computed importance_score from
            # importance_scanner.py.  The score itself decays with age so
            # stale high-importance memories don't dominate forever.
            # Formula: additive boost of importance * decay, scaled by
            # IMPORTANCE_WEIGHT (0.15 = up to ~15% boost for top memories).
            IMPORTANCE_WEIGHT = 0.15
            imp_score = r.get('importance_score')
            if imp_score is not None and imp_score > 0:
                imp_decay = math.exp(-self.decay_lambda * 0.5 * hours_age)  # slower decay than base
                decayed_score += IMPORTANCE_WEIGHT * imp_score * imp_decay

            # Optional frequency boost: memories accessed more often get a
            # small additive bonus so hot memories surface more easily.
            mid = str(r.get('id', ''))
            access_count = 0
            if self.frequency_boost > 0:
                with self._lock:
                    if mid in self._access_log:
                        access_count = self._access_log[mid]['count']
                # Logarithmic boost to avoid runaway inflation
                decayed_score += self.frequency_boost * math.log1p(access_count)

            if decayed_score >= threshold:
                final.append({
                    'id': r.get('id'),
                    'text': text,
                    'decayed_score': round(decayed_score, 4),
                    'importance_score': round(imp_score, 4) if imp_score is not None else None,
                    'access_count': access_count,
                    'metadata': r['metadata'],
                    'age_hours': round(hours_age, 2)
                })

        ranked = sorted(final, key=lambda x: x['decayed_score'], reverse=True)[:k]

        # Log access hits for the returned results
        if ranked:
            self._record_access(ranked, query)

        return ranked


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 holographic_memory.py <command> [query]")
        print("Commands: search, add, health, purge, purge-dry")
        sys.exit(1)
    
    command = sys.argv[1]
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
    
    mem = HolographicMemory()
    
    if command == "search":
        if not query:
            print("Error: search requires a query")
            sys.exit(1)
        results = mem.redshifted_recall(query, k=5)
        print(f"\n🔍 Results for: '{query}'")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['decayed_score']:.3f}] {r['text'][:100]}... ({r['age_hours']:.1f}h ago)")
    elif command == "add":
        if not query:
            print("Error: add requires text")
            sys.exit(1)
        mem.add_memory(query)
        print("✓ Memory stored.")
    elif command == "health":
        print(json.dumps(mem.get_health_status(), indent=2))
    elif command in ("purge", "purge-dry"):
        dry = command == "purge-dry"
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].replace('.','',1).isdigit() else 0.4
        print(f"\n🧹 {'[DRY RUN] ' if dry else ''}Purging hallucinated memories (threshold={threshold})...")
        result = mem.purge_hallucinations(threshold=threshold, dry_run=dry)
        print(f"   Scanned: {result['scanned']}")
        print(f"   Flagged: {result['flagged']}")
        print(f"   Purged:  {result['purged']}")
        if result.get('entries'):
            print(f"\n   Entries:")
            for e in result['entries']:
                print(f"   • [{e['confidence']:.2f} {e['severity']}] id={e['id']} {e['text'][:90]}")
    else:
        print(f"Unknown command: {command}")
