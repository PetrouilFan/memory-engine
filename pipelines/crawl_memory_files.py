#!/usr/bin/env python3
"""
crawl_memory_files.py
---------------------
Unified memory crawler: traverses markdown files, splits into semantic chunks,
filters out hallucinations and leaked tags, then batch-indexes via memory_engine.

Replaces the old crawl_and_index_memories.py (whole-file, no filtering).

Uses the embedding server API for embeddings (falls back to zero vectors).
"""

import os
import sys
import re
import time
import logging
import json
import urllib.request
from pathlib import Path
from typing import List

# Allow importing from core/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.hallucination_filter import HallucinationDetector
from core.memory_engine import EMB_DIM

EMBEDDING_SERVER_HOST = os.getenv('EMBEDDING_SERVER_HOST', 'localhost')
EMBEDDING_SERVER_PORT = int(os.getenv('EMBEDDING_SERVER_PORT', '9999'))
EMBEDDING_SERVER_URL = f"http://{EMBEDDING_SERVER_HOST}:{EMBEDDING_SERVER_PORT}"

# Configuration
MEMORY_ROOT = Path(__file__).resolve().parent.parent  # project root
MEMORY_DIRS = [
    MEMORY_ROOT,
    Path("/root/.openclaw/workspace/memory"),  # workspace memory directory
]
BATCH_SIZE = 200
MAX_CHUNK_LENGTH = 500   # Max chars per chunk before sentence splitting
MIN_CHUNK_LENGTH = 10    # Min chars for a chunk to be useful

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Shared detector for the crawl run
_detector = HallucinationDetector()


def embed_text(text: str) -> List[float]:
    """Generate embedding via embedding server API (384-dim)."""
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
    except Exception as e:
        log.error(f"Embedding server unavailable ({e}), using fallback")

    return [0.0] * EMB_DIM


# Pre-compiled tag regex — matches any <tag>, <tag attr>, </tag>
_TAG_RE = re.compile(r"</?[a-zA-Z_][\w.-]*(?:\s[^>]*)?>")


def _has_tags(text: str) -> bool:
    """Fast check: does text contain any XML/HTML-style tags?
    These are prompt/tool-call leakage and never belong in memories."""
    return bool(_TAG_RE.search(text))


def is_useful_chunk(chunk: str) -> bool:
    """Check if a chunk contains useful, non-leaked information."""
    chunk = chunk.strip()
    if len(chunk) < MIN_CHUNK_LENGTH:
        return False
    if chunk.startswith("```") or re.match(r"^\s{4,}", chunk):
        return False
    if re.match(r"^[^\w]*$", chunk):
        return False
    # Reject chunks containing any leaked tags
    if _has_tags(chunk):
        return False
    return True


def split_semantically(text: str) -> List[str]:
    """Split text into semantic chunks: paragraphs first, then sentences.
    Filters out irrelevant, tagged, and hallucinated chunks."""
    chunks = []
    paragraphs = re.split(r"\n\s*\n|(?=#+\s)", text.strip())
    for para in paragraphs:
        para = para.strip()
        if not is_useful_chunk(para):
            continue
        if len(para) > MAX_CHUNK_LENGTH:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if is_useful_chunk(sent):
                    chunks.append(sent)
        else:
            chunks.append(para)
    return chunks


def main():
    from core.memory_engine import add_memories_batch, _maybe_save_index

    # Collect markdown files from all configured directories
    all_md_files = []
    for mem_dir in MEMORY_DIRS:
        mem_path = Path(mem_dir)
        if not mem_path.is_dir():
            log.warning(f"Memory directory '{mem_path}' not found, skipping.")
            continue
        md_files = sorted(mem_path.rglob("*.md"))
        all_md_files.extend(md_files)
        log.info(f"Found {len(md_files)} markdown files under '{mem_path}'.")

    if not all_md_files:
        log.error("No markdown files found in any memory directory.")
        return

    vectors_batch, metadatas_batch, texts_batch = [], [], []
    total_chunks = 0
    rejected_tags = 0
    rejected_hallucinations = 0

    for md_path in all_md_files:
        try:
            full_text = md_path.read_text(encoding="utf-8")
            chunks = split_semantically(full_text)
            for idx, chunk in enumerate(chunks):
                # Belt-and-suspenders: reject anything with leaked tags
                if _has_tags(chunk):
                    rejected_tags += 1
                    continue

                # Reject hallucinated content
                if _detector.is_hallucinated(chunk, threshold=0.5):
                    rejected_hallucinations += 1
                    log.debug("Rejected hallucinated chunk from %s: %s", md_path.name, chunk[:80])
                    continue

                metadata = {
                    "source_file": str(md_path),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "mtime": md_path.stat().st_mtime,
                    "timestamp": time.time(),
                    "text_preview": chunk[:200],
                }
                vector = embed_text(chunk)
                vectors_batch.append(vector)
                metadatas_batch.append(metadata)
                texts_batch.append(chunk)
                total_chunks += 1

                if total_chunks % BATCH_SIZE == 0:
                    added_ids = add_memories_batch(vectors_batch, metadatas_batch, texts_batch)
                    log.info(
                        "Batch: %d added (total: %d, rejected: %d tagged, %d hallucinated)",
                        len(added_ids), total_chunks, rejected_tags, rejected_hallucinations,
                    )
                    vectors_batch.clear()
                    metadatas_batch.clear()
                    texts_batch.clear()
                    _maybe_save_index(force=True)
        except Exception as e:
            log.exception(f"Failed to index {md_path}: {e}")

    if vectors_batch:
        added_ids = add_memories_batch(vectors_batch, metadatas_batch, texts_batch)
        log.info("Final batch: %d added (total: %d)", len(added_ids), total_chunks)
        _maybe_save_index(force=True)

    log.info(
        "Indexing complete: %d chunks from %d files. "
        "Rejected: %d tagged, %d hallucinated.",
        total_chunks, len(all_md_files), rejected_tags, rejected_hallucinations,
    )


if __name__ == "__main__":
    main()
