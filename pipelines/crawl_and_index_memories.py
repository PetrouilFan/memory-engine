#!/usr/bin/env python3
"""
crawl_and_index_memories.py
---------------------------
Traverse all markdown files and batch-index them via memory_engine.
Uses Ollama HTTP API for embeddings (falls back to zero vectors).
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Allow importing from core/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

# Configuration
MEMORY_ROOT = Path(__file__).resolve().parent.parent  # project root
BATCH_SIZE = 200

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def embed_text(text: str) -> list:
    """Generate an embedding using Ollama's HTTP API with nomic-embed model."""
    try:
        import subprocess
        payload = json.dumps({"model": "nomic-embed", "input": text[:2000]})
        result = subprocess.check_output([
            "curl", "-s", "-X", "POST", "http://localhost:11434/api/embeddings",
            "-d", payload,
            "-H", "Content-Type: application/json"
        ], text=True)
        data = json.loads(result)
        embedding = data.get("embedding") or data.get("embeddings")
        if isinstance(embedding, list):
            return embedding
    except Exception as e:
        log.error(f"Ollama embedding via HTTP failed: {e}")
    from core.memory_engine import EMB_DIM
    return [0.0] * EMB_DIM


def main():
    from core.memory_engine import add_memories_batch, _maybe_save_index

    if not MEMORY_ROOT.is_dir():
        log.error(f"Root folder '{MEMORY_ROOT}' not found.")
        return

    md_files = sorted(MEMORY_ROOT.rglob("*.md"))
    log.info(f"Found {len(md_files)} markdown files under '{MEMORY_ROOT}'.")

    vectors_batch, metadatas_batch, texts_batch = [], [], []
    processed = 0

    for md_path in md_files:
        try:
            text = md_path.read_text(encoding="utf-8")
            metadata = {
                "source_file": str(md_path),
                "mtime": md_path.stat().st_mtime,
                "timestamp": time.time(),
                "text": text[:2000],
            }
            vector = embed_text(text)
            vectors_batch.append(vector)
            metadatas_batch.append(metadata)
            texts_batch.append(text)
            processed += 1
            if processed % BATCH_SIZE == 0:
                added_ids = add_memories_batch(vectors_batch, metadatas_batch, texts_batch)
                log.debug(f"Batch added {len(added_ids)} memories")
                vectors_batch.clear()
                metadatas_batch.clear()
                texts_batch.clear()
                _maybe_save_index(force=True)
        except Exception as e:
            log.exception(f"Failed to index {md_path}: {e}")

    if vectors_batch:
        added_ids = add_memories_batch(vectors_batch, metadatas_batch, texts_batch)
        log.debug(f"Final batch added {len(added_ids)} memories")
        _maybe_save_index(force=True)

    log.info(f"Indexing complete: {processed} files processed.")


if __name__ == "__main__":
    main()
