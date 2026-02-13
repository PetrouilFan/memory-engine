# Fast/Slow Memory Layer
"""
Implementation of a two‑tier memory system:

* **Fast tier** – recent, high‑frequency items stored in an in‑memory SQLite table
  ``mem_fast`` and a lightweight FAISS index (e.g. IVF‑Flat) for quick
  approximate nearest‑neighbor (ANN) search.
* **Slow tier** – long‑term archive stored in the existing ``memory.db``
  (FAISS index ``memory.index``) which may be larger and slower but holds
  the full history.

The module provides a thin wrapper around the existing ``memory_engine``
functions, automatically routing inserts to the fast tier and falling
back to the slow tier when the fast tier exceeds a configurable size.
When the fast tier is flushed, its vectors are merged into the slow tier
index.

Typical workflow:

```python
from fast_slow_memory import FastSlowMemory

fsm = FastSlowMemory(max_fast_items=5000)

# Add a document (text, optional metadata)
fsm.add(text="My recent note", meta={"source": "daily"})

# Search – will first hit the fast tier, then the slow tier for completeness
results = fsm.search("quick reminder", k=10)
```

The implementation is deliberately lightweight – it re‑uses the existing
FAISS utilities in ``vector_memory_wrapper.py`` and the SQLite helper in
``memory_engine.py``.  Only the fast tier tables and index are created on
first use.
"""

import os
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Paths – keep them relative to the project root so they work both in
# the sandbox and on the host.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAST_DB_PATH = PROJECT_ROOT / "data" / "mem_fast.db"
FAST_INDEX_PATH = PROJECT_ROOT / "data" / "mem_fast.index"

# Import helpers from the existing memory engine.
# We import lazily to avoid heavy imports when the module is not used.

def _load_helpers():
    from core.memory_engine import add_memory, search_memory
    from core.vector_memory_wrapper import FAISSWrapper
    return add_memory, search_memory, FAISSWrapper


class FastSlowMemory:
    """Two‑tier memory with automatic flushing.

    Parameters
    ----------
    max_fast_items: int
        When the fast SQLite table exceeds this count, it is flushed into the
        slow tier (the main ``memory_engine``) and the fast structures are
        cleared.
    """

    def __init__(self, max_fast_items: int = 5000):
        self.max_fast_items = max_fast_items
        self._ensure_fast_db()
        # Lazy load heavy helpers
        self._add_memory, self._search_memory, self._FAISSWrapper = _load_helpers()
        # Initialise FAISS wrapper for the fast index
        self.fast_faiss = self._FAISSWrapper(str(FAST_INDEX_PATH), create_if_missing=True)

    def _ensure_fast_db(self):
        """Create the fast SQLite DB and ``mem_fast`` table if missing."""
        os.makedirs(FAST_DB_PATH.parent, exist_ok=True)
        conn = sqlite3.connect(FAST_DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS mem_fast (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                meta JSON,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
        conn.close()

    def _fast_count(self) -> int:
        conn = sqlite3.connect(FAST_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM mem_fast")
        (cnt,) = cur.fetchone()
        conn.close()
        return cnt

    def _increment_access(self, row_id: int):
        """Increment the access_count for a given fast tier row."""
        conn = sqlite3.connect(FAST_DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE mem_fast SET access_count = access_count + 1 WHERE id = ?", (row_id,))
        conn.commit()
        conn.close()

    def _maybe_adjust_embedding(self, row_id: int, current_count: int):
        """Placeholder for dynamic embedding size adjustment based on usage.

        In a real system this would re‑embed the text with a larger model or
        higher‑dimensional vector when ``current_count`` exceeds a threshold.
        Here we simply log the event for later implementation.
        """
        THRESHOLD = 100  # Example threshold for high‑frequency items
        if current_count >= THRESHOLD:
            # Re‑embed logic would go here. For now, we just note it.
            print(f"[Plasticity] Row {row_id} reached {current_count} accesses – consider larger embedding.")

    def _flush_fast_to_slow(self):
        """Move all fast records into the main memory engine.

        After flushing, the fast DB and FAISS index are cleared.
        """
        conn = sqlite3.connect(FAST_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, text, meta FROM mem_fast")
        rows = cur.fetchall()
        if not rows:
            conn.close()
            return

        # Insert into the slow tier using the existing ``add_memory`` helper.
        for _id, text, meta_json in rows:
            meta = json.loads(meta_json) if meta_json else {}
            self._add_memory(text=text, meta=meta)  # goes to the main DB + index

        # Clear fast DB and its FAISS index.
        cur.execute("DELETE FROM mem_fast")
        conn.commit()
        conn.close()
        # Re‑create a fresh fast index.
        self.fast_faiss = self._FAISSWrapper(str(FAST_INDEX_PATH), create_if_missing=True)

    def add(self, text: str, meta: Dict[str, Any] = None):
        """Add a document to the fast tier.

        If the fast tier exceeds ``max_fast_items`` after insertion, it is
        flushed automatically.
        """
        # Ensure metadata dict exists and initialize usage counter fields.
        meta = meta or {}
        # We'll store the SQLite row id in metadata for quick updates later.
        meta_json = json.dumps(meta) if meta else None
        conn = sqlite3.connect(FAST_DB_PATH)
        cur = conn.cursor()
+        # Insert with explicit access_count (defaults to 0).
+        cur.execute(
+            "INSERT INTO mem_fast (text, meta, access_count) VALUES (?, ?, 0)",
+            (text, meta_json),
+        )
+        row_id = cur.lastrowid
+        conn.commit()
+        conn.close()
+
+        # Attach the row id to metadata for later plasticity updates.
+        meta["__id"] = row_id
+
+        # Index the vector immediately – we reuse the same embedding pipeline
+        # as the main engine via ``FAISSWrapper.add_text`` (a thin wrapper).
+        self.fast_faiss.add_text(text, meta)

        if self._fast_count() > self.max_fast_items:
            self._flush_fast_to_slow()

    def search(self, query: str, k: int = 10) -> List[Tuple[float, str, Dict]]:
        """Hybrid search across fast and slow tiers.

        Returns a list of ``(score, text, meta)`` sorted by relevance.
        Also updates access counters for fast‑tier hits and triggers
        plasticity checks.
        """
        # Fast tier search
        fast_results = self.fast_faiss.search(query, k)
        # Slow tier search – reuse existing ``search_memory`` which returns a
        # list of dicts with ``score``, ``text`` and ``meta``.
        slow_results = self._search_memory(query, k)

        # Normalise formats for merging and handle usage counters
        merged = []
        for score, txt, meta in fast_results:
            # meta may contain the fast DB row id under "__id"
            row_id = meta.get("__id") if isinstance(meta, dict) else None
            if row_id:
                # Increment access count in the fast DB
                self._increment_access(row_id)
                # Fetch new count to decide on embedding adjustment
                conn = sqlite3.connect(FAST_DB_PATH)
                cur = conn.cursor()
                cur.execute("SELECT access_count FROM mem_fast WHERE id = ?", (row_id,))
                (count,) = cur.fetchone()
                conn.close()
                self._maybe_adjust_embedding(row_id, count)
            merged.append((score, txt, meta))
        for item in slow_results:
            merged.append((item["score"], item["text"], item.get("meta", {})))

        # Sort by score (lower is better for FAISS L2) – we assume both
        # components use comparable scores; if not, this is a simple heuristic.
        merged.sort(key=lambda x: x[0])
        return merged[:k]

    # Convenience helpers ---------------------------------------------------
    def flush(self):
        """Public method to force a flush of the fast tier."""
        self._flush_fast_to_slow()

    def stats(self) -> Dict[str, Any]:
        """Return simple statistics for monitoring purposes."""
        return {
            "fast_items": self._fast_count(),
            "fast_index_size": self.fast_faiss.ntotal,
        }

# Example usage when run as a script
if __name__ == "__main__":
    fsm = FastSlowMemory(max_fast_items=1000)
    fsm.add("Hello world", {"source": "demo"})
    print("Stats:", fsm.stats())
    print("Search results:", fsm.search("hello"))
