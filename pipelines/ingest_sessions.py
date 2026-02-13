#!/usr/bin/env python3
"""
ingest_sessions.py – Semantically split session files and ingest into memory-engine.

Reads every .md file under a given directory (default: ../../memory/), detects
whether it's a structured session dump or a free-form note, and splits it into
semantically coherent chunks before storing each chunk as a separate memory.

Semantic splitting strategy
---------------------------
**Session files** (have ``# Session:`` header):
  1. Strip the metadata header (session key, ID, source).
  2. Parse conversation into individual turns (user / assistant / system).
  3. Group adjacent turns into *exchanges* (user question + assistant answer).
  4. Merge short exchanges together, split long ones, so each chunk is
     between ~200 and ~1500 characters — large enough for context, small
     enough for precise recall.
  5. Detect topic shifts via time gaps (>30 min between user messages) and
     force a chunk boundary there.

**Free-form notes** (no session header):
  1. Split on markdown headings (``## …``).
  2. Within each section, split on double-newline paragraph boundaries.
  3. Merge tiny paragraphs, split oversized ones.

Every chunk is stored with rich metadata:
  - source_file, session_id, session_source (telegram, etc.)
  - chunk_type (exchange | note | system)
  - timestamp of the first message in the chunk
  - speaker(s) involved
  - sequential chunk_index within the file

Usage
-----
    python ingest_sessions.py                          # default: ../../memory/
    python ingest_sessions.py /path/to/memory/folder
    python ingest_sessions.py --dry-run                # parse only, no DB writes
"""

import os
import sys
import re
import json
import time
import logging
import hashlib
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup – allow running from anywhere
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CORE_DIR = PROJECT_ROOT / "core"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("ingest_sessions")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMB_DIM = int(os.getenv("MEMORY_EMB_DIM", "384"))

EMBEDDING_SERVER_HOST = os.getenv('EMBEDDING_SERVER_HOST', 'localhost')
EMBEDDING_SERVER_PORT = int(os.getenv('EMBEDDING_SERVER_PORT', '9999'))
EMBEDDING_SERVER_URL = f"http://{EMBEDDING_SERVER_HOST}:{EMBEDDING_SERVER_PORT}"

# Chunk-size guardrails (characters)
MIN_CHUNK_LEN = 200
MAX_CHUNK_LEN = 1500
IDEAL_CHUNK_LEN = 800

# Time gap (seconds) that signals a topic shift inside a session
TOPIC_GAP_SECONDS = 30 * 60  # 30 minutes

# ---------------------------------------------------------------------------
# Regex patterns for parsing session files
# ---------------------------------------------------------------------------
RE_SESSION_HEADER = re.compile(r"^#\s+Session:\s+(.+)$", re.MULTILINE)
RE_META_BULLET = re.compile(
    r"^-\s+\*\*(?P<key>[^*]+)\*\*:\s*(?P<val>.+)$", re.MULTILINE
)
RE_CONV_SECTION = re.compile(r"^##\s+Conversation Summary\s*$", re.MULTILINE)
RE_TURN = re.compile(
    r"^(?P<role>user|assistant|A|System):\s*", re.MULTILINE
)
RE_TELEGRAM_TS = re.compile(
    r"\[Telegram\s.*?(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+(?P<tz>GMT[+-]?\d+)"
)
RE_MESSAGE_ID = re.compile(r"^\[message_id:\s*\d+\]\s*$", re.MULTILINE)
RE_REPLY_TAG = re.compile(r"^\[\[reply_to[^\]]*\]\]\s*$", re.MULTILINE)
RE_REACTION = re.compile(r"^\[Reaction added:.*?\]\s*$", re.MULTILINE)
RE_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_text(text: str) -> List[float]:
    """Generate embedding via embedding server API (384-dim).
    Falls back to deterministic hash-seeded random vector."""
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
        log.debug(f"Embedding server unavailable ({e}), using fallback")

    # Deterministic fallback: hash-seeded random vector (reproducible per text)
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    return rng.randn(EMB_DIM).astype("float32").tolist()


# ===================================================================
# Parsing
# ===================================================================

def _clean_noise(text: str) -> str:
    """Remove bot-internal markers that add no semantic value."""
    text = RE_MESSAGE_ID.sub("", text)
    text = RE_REPLY_TAG.sub("", text)
    text = RE_REACTION.sub("", text)
    return text.strip()


def _extract_telegram_timestamp(text: str) -> Optional[float]:
    """Pull the first Telegram-style timestamp from a turn's text."""
    m = RE_TELEGRAM_TS.search(text)
    if not m:
        return None
    try:
        ts_str = m.group("ts")
        tz_str = m.group("tz")
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
        # Parse GMT offset (e.g. GMT+2 → +7200)
        offset_match = re.search(r"([+-]?\d+)", tz_str)
        offset_hours = int(offset_match.group(1)) if offset_match else 0
        epoch = dt.timestamp() - offset_hours * 3600
        return epoch
    except Exception:
        return None


def _parse_session_header(raw: str) -> Tuple[Dict[str, str], str]:
    """Split a session file into (metadata_dict, conversation_body)."""
    meta: Dict[str, str] = {}

    m = RE_SESSION_HEADER.search(raw)
    if m:
        meta["session_date"] = m.group(1).strip()

    for mb in RE_META_BULLET.finditer(raw):
        key = mb.group("key").strip().lower().replace(" ", "_")
        meta[key] = mb.group("val").strip()

    # Everything after "## Conversation Summary"
    conv_match = RE_CONV_SECTION.search(raw)
    if conv_match:
        body = raw[conv_match.end():]
    else:
        # Fallback: skip the first heading block
        lines = raw.split("\n")
        start = 0
        for i, line in enumerate(lines):
            if line.startswith("user:") or line.startswith("assistant:") or line.startswith("A:"):
                start = i
                break
        body = "\n".join(lines[start:])

    return meta, body.strip()


# ---------------------------------------------------------------------------

class Turn:
    """A single conversation turn."""
    __slots__ = ("role", "text", "timestamp")

    def __init__(self, role: str, text: str, timestamp: Optional[float] = None):
        self.role = "assistant" if role == "A" else role.lower()
        self.text = _clean_noise(text)
        self.timestamp = timestamp or _extract_telegram_timestamp(text)

    def __repr__(self):
        preview = self.text[:60].replace("\n", "↵")
        return f"Turn({self.role}, ts={self.timestamp}, {preview!r}…)"


def _parse_turns(body: str) -> List[Turn]:
    """Split conversation body into Turn objects."""
    turns: List[Turn] = []
    # Find all turn boundaries
    markers = list(RE_TURN.finditer(body))
    if not markers:
        # No structured turns – treat entire body as a single chunk
        return [Turn("note", body)]

    for i, marker in enumerate(markers):
        role = marker.group("role")
        start = marker.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(body)
        text = body[start:end].strip()
        if text:
            turns.append(Turn(role, text))
    return turns


# ===================================================================
# Semantic chunking
# ===================================================================

def _chunk_session_turns(turns: List[Turn]) -> List[Dict[str, Any]]:
    """Group turns into semantically coherent exchange chunks.

    Rules:
      - Each exchange starts with a user turn (or system turn).
      - The assistant response(s) that follow are part of the same exchange.
      - A topic-gap (>30 min) between consecutive user turns forces a new chunk.
      - Chunks are merged or split to stay within MIN/MAX_CHUNK_LEN.
    """
    if not turns:
        return []

    # -- Step 1: Build raw exchanges (user + following assistant turns) ------
    exchanges: List[Dict[str, Any]] = []
    current_turns: List[Turn] = []
    current_ts: Optional[float] = None

    def _flush():
        nonlocal current_turns, current_ts
        if not current_turns:
            return
        text_parts = []
        speakers = set()
        for t in current_turns:
            prefix = f"[{t.role}]" if t.role != "note" else ""
            text_parts.append(f"{prefix} {t.text}" if prefix else t.text)
            speakers.add(t.role)
        exchanges.append({
            "text": "\n\n".join(text_parts),
            "timestamp": current_ts,
            "speakers": sorted(speakers),
            "turn_count": len(current_turns),
        })
        current_turns = []
        current_ts = None

    for turn in turns:
        # Detect topic gap
        if turn.role in ("user", "system") and current_ts and turn.timestamp:
            gap = abs(turn.timestamp - current_ts)
            if gap > TOPIC_GAP_SECONDS:
                _flush()

        # Start new exchange on user/system turn (unless accumulating)
        if turn.role in ("user", "system") and current_turns:
            # Only flush if the current exchange already has an assistant reply
            has_reply = any(t.role == "assistant" for t in current_turns)
            if has_reply:
                _flush()

        current_turns.append(turn)
        if turn.timestamp and (current_ts is None or turn.timestamp < current_ts):
            current_ts = turn.timestamp

    _flush()

    # -- Step 2: Merge small exchanges, split large ones --------------------
    merged: List[Dict[str, Any]] = []
    buffer: Optional[Dict[str, Any]] = None

    for ex in exchanges:
        if buffer is None:
            buffer = ex
            continue

        combined_len = len(buffer["text"]) + len(ex["text"])
        if combined_len <= IDEAL_CHUNK_LEN:
            # Merge
            buffer["text"] += "\n\n---\n\n" + ex["text"]
            buffer["speakers"] = sorted(set(buffer["speakers"] + ex["speakers"]))
            buffer["turn_count"] += ex["turn_count"]
            if ex["timestamp"] and (buffer["timestamp"] is None or ex["timestamp"] < buffer["timestamp"]):
                buffer["timestamp"] = ex["timestamp"]
        else:
            merged.append(buffer)
            buffer = ex

    if buffer:
        merged.append(buffer)

    # -- Step 3: Hard-split any chunk that still exceeds MAX_CHUNK_LEN ------
    final: List[Dict[str, Any]] = []
    for chunk in merged:
        text = chunk["text"]
        if len(text) <= MAX_CHUNK_LEN:
            final.append(chunk)
            continue
        # Split on double-newline boundaries
        paragraphs = re.split(r"\n{2,}", text)
        part_buf = ""
        for para in paragraphs:
            if len(part_buf) + len(para) + 2 > MAX_CHUNK_LEN and part_buf:
                final.append({**chunk, "text": part_buf.strip()})
                part_buf = para
            else:
                part_buf += "\n\n" + para if part_buf else para
        if part_buf.strip():
            final.append({**chunk, "text": part_buf.strip()})

    return final


def _chunk_freeform(raw: str, source_file: str) -> List[Dict[str, Any]]:
    """Split free-form markdown notes into semantic chunks by headings and paragraphs."""
    chunks: List[Dict[str, Any]] = []

    # Split on headings
    sections = re.split(r"(?=^#{1,6}\s+)", raw, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= MAX_CHUNK_LEN:
            chunks.append({
                "text": section,
                "timestamp": None,
                "speakers": ["note"],
                "turn_count": 0,
            })
            continue

        # Further split on double-newline
        paragraphs = re.split(r"\n{2,}", section)
        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) + 2 > IDEAL_CHUNK_LEN and buf:
                chunks.append({
                    "text": buf.strip(),
                    "timestamp": None,
                    "speakers": ["note"],
                    "turn_count": 0,
                })
                buf = para
            else:
                buf += "\n\n" + para if buf else para
        if buf.strip():
            chunks.append({
                "text": buf.strip(),
                "timestamp": None,
                "speakers": ["note"],
                "turn_count": 0,
            })

    # Merge tiny chunks
    merged: List[Dict[str, Any]] = []
    for c in chunks:
        if merged and len(merged[-1]["text"]) + len(c["text"]) < MIN_CHUNK_LEN:
            merged[-1]["text"] += "\n\n" + c["text"]
        else:
            merged.append(c)

    return merged


# ===================================================================
# File processing
# ===================================================================

def process_file(filepath: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Parse one .md file and return (chunks, session_meta)."""
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return [], {}

    is_session = bool(RE_SESSION_HEADER.search(raw))

    if is_session:
        meta, body = _parse_session_header(raw)
        turns = _parse_turns(body)
        chunks = _chunk_session_turns(turns)
        for i, c in enumerate(chunks):
            c["chunk_type"] = "exchange"
            c["chunk_index"] = i
        return chunks, meta
    else:
        chunks = _chunk_freeform(raw, str(filepath))
        for i, c in enumerate(chunks):
            c["chunk_type"] = "note"
            c["chunk_index"] = i
        return chunks, {}


# ===================================================================
# Ingestion
# ===================================================================

def ingest_directory(
    memory_dir: Path,
    dry_run: bool = False,
    batch_size: int = 50,
) -> Dict[str, Any]:
    """Walk *memory_dir*, semantically split every .md file, and store chunks."""

    md_files = sorted(memory_dir.rglob("*.md"))
    log.info(f"Found {len(md_files)} markdown files under {memory_dir}")

    if not dry_run:
        from core.memory_engine import add_memories_batch, _maybe_save_index
    
    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "chunks_created": 0,
        "chunks_by_type": {"exchange": 0, "note": 0},
        "errors": [],
    }

    # Dedup: track hashes of already-ingested chunks
    seen_hashes: set = set()

    # Batching accumulators
    batch_vecs: List[List[float]] = []
    batch_metas: List[Dict[str, Any]] = []
    batch_texts: List[str] = []

    def _flush_batch():
        if not batch_vecs or dry_run:
            return
        try:
            ids = add_memories_batch(batch_vecs, batch_metas, batch_texts)
            log.info(f"  → Batch stored {len(ids)} chunks")
        except Exception as e:
            log.error(f"  ✗ Batch insert failed: {e}")
            stats["errors"].append(str(e))
        finally:
            batch_vecs.clear()
            batch_metas.clear()
            batch_texts.clear()

    for filepath in md_files:
        rel_path = filepath.relative_to(memory_dir)
        log.info(f"Processing {rel_path}")

        try:
            chunks, session_meta = process_file(filepath)
        except Exception as e:
            log.error(f"  ✗ Parse error: {e}")
            stats["errors"].append(f"{rel_path}: {e}")
            stats["files_skipped"] += 1
            continue

        if not chunks:
            stats["files_skipped"] += 1
            continue

        for chunk in chunks:
            text = chunk["text"]
            # Dedup by content hash
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Build metadata
            meta = {
                "source_file": str(rel_path),
                "chunk_index": chunk["chunk_index"],
                "chunk_type": chunk["chunk_type"],
                "speakers": chunk.get("speakers", []),
                "turn_count": chunk.get("turn_count", 0),
                "content_hash": h,
                "ingested_at": time.time(),
                "text": text[:2000],  # preview stored in metadata
            }
            if chunk.get("timestamp"):
                meta["timestamp"] = chunk["timestamp"]
            else:
                # Fall back to file mtime
                meta["timestamp"] = filepath.stat().st_mtime

            # Merge session-level metadata
            for k, v in session_meta.items():
                meta[f"session_{k}"] = v

            if dry_run:
                log.info(
                    f"  [DRY] chunk {chunk['chunk_index']:>3d}  "
                    f"type={chunk['chunk_type']:>8s}  "
                    f"len={len(text):>5d}  "
                    f"speakers={chunk.get('speakers', [])}"
                )
                stats["chunks_created"] += 1
                stats["chunks_by_type"][chunk["chunk_type"]] += 1
                continue

            # Embed and accumulate
            vec = embed_text(text)
            batch_vecs.append(vec)
            batch_metas.append(meta)
            batch_texts.append(text)

            stats["chunks_created"] += 1
            stats["chunks_by_type"][chunk["chunk_type"]] += 1

            if len(batch_vecs) >= batch_size:
                _flush_batch()

        stats["files_processed"] += 1

    # Final flush
    _flush_batch()
    if not dry_run:
        try:
            _maybe_save_index(force=True)
        except Exception:
            pass

    return stats


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Semantically split session files and ingest into memory-engine.",
    )
    parser.add_argument(
        "memory_dir",
        nargs="?",
        default=str(Path(__file__).resolve().parent.parent.parent.parent / "memory"),
        help="Path to the directory containing .md session files (default: ../../memory/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and split files but do not write to the database.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of chunks to batch-insert at once (default: 50).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    memory_dir = Path(args.memory_dir).resolve()
    if not memory_dir.is_dir():
        log.error(f"Directory not found: {memory_dir}")
        sys.exit(1)

    log.info(f"Memory directory : {memory_dir}")
    log.info(f"Dry run          : {args.dry_run}")
    log.info(f"Batch size       : {args.batch_size}")
    log.info("")

    t0 = time.time()
    stats = ingest_directory(memory_dir, dry_run=args.dry_run, batch_size=args.batch_size)
    elapsed = time.time() - t0

    log.info("")
    log.info("═" * 50)
    log.info(f"  Files processed : {stats['files_processed']}")
    log.info(f"  Files skipped   : {stats['files_skipped']}")
    log.info(f"  Chunks created  : {stats['chunks_created']}")
    log.info(f"    ↳ exchanges   : {stats['chunks_by_type']['exchange']}")
    log.info(f"    ↳ notes       : {stats['chunks_by_type']['note']}")
    log.info(f"  Errors          : {len(stats['errors'])}")
    log.info(f"  Elapsed         : {elapsed:.1f}s")
    log.info("═" * 50)

    if stats["errors"]:
        log.warning("Errors encountered:")
        for e in stats["errors"][:10]:
            log.warning(f"  • {e}")


if __name__ == "__main__":
    main()
