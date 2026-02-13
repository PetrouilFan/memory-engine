#!/usr/bin/env python3
"""
ingest_workspace.py – Scan all markdown files under a workspace path and
ensure every semantically split chunk is stored in the memory engine.

Unlike ``ingest_sessions.py`` (which targets session logs with conversation
turns), this script targets **all** markdown files in a workspace tree:
config docs, identity files, HOWTOs, task lists, daily memory notes, etc.

Idempotency
-----------
On each run the script builds a content-hash manifest and compares it to
the last-known manifest stored on disk (``<project>/data/workspace_manifest.json``).
Only *new* or *changed* files are re-ingested; deleted files have their
chunks removed from the DB.  This makes it safe to run on every heartbeat.

Semantic splitting
------------------
1. Split on markdown headings (``# / ## / ### …``).
2. Within each section, split on paragraph boundaries (double newline).
3. Merge tiny pieces and hard-split oversized ones so every chunk is
   between ~200 and ~1 500 characters.
4. Code fences are kept intact (never broken mid-fence).

Usage
-----
    python ingest_workspace.py                                  # default path
    python ingest_workspace.py /root/.openclaw/workspace        # explicit
    python ingest_workspace.py --dry-run                        # no DB writes
    python ingest_workspace.py --exclude memory --exclude .git  # skip dirs
"""

from __future__ import annotations

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
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
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
log = logging.getLogger("ingest_workspace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMB_DIM = int(os.getenv("MEMORY_EMB_DIM", "384"))
EMBEDDING_SERVER_HOST = os.getenv('EMBEDDING_SERVER_HOST', 'localhost')
EMBEDDING_SERVER_PORT = int(os.getenv('EMBEDDING_SERVER_PORT', '9999'))
EMBEDDING_SERVER_URL = f"http://{EMBEDDING_SERVER_HOST}:{EMBEDDING_SERVER_PORT}"
DEFAULT_WORKSPACE = "/root/.openclaw/workspace"
MANIFEST_PATH = PROJECT_ROOT / "data" / "workspace_manifest.json"

# Chunk size guardrails (characters)
MIN_CHUNK = 200
MAX_CHUNK = 1500
IDEAL_CHUNK = 800
CHUNK_OVERLAP = 200  # Characters to overlap between chunks for context preservation

# Hallucination filter - disable for workspace docs since they're trusted sources
ENABLE_HALLUCINATION_FILTER = os.getenv("DISABLE_HALLUCINATION_FILTER", "true").lower() in ("1", "true", "yes")

# Try to import nltk for sentence splitting
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass
HAS_NLTK = True
try:
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False

# Directories to skip by default
DEFAULT_EXCLUDES: List[str] = [
    ".git",
    "__pycache__",
    "node_modules",
    ".vscode",
    "projects",       # self-contained – ingested separately if desired
]

# ---------------------------------------------------------------------------
# Embedding helper  (shared pattern with ingest_sessions)
# ---------------------------------------------------------------------------

def embed_text(text: str) -> List[float]:
    """Generate embedding via embedding server API (384-dim).
    Falls back to deterministic hash-seeded random vector so ingestion never stalls."""
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

    # Deterministic fallback
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    return rng.randn(EMB_DIM).astype("float32").tolist()


# ===================================================================
# Manifest – tracks what was already ingested
# ===================================================================

def _load_manifest(path: Path) -> Dict[str, Any]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": {}, "chunk_ids": {}}


def _save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)


# ===================================================================
# File discovery
# ===================================================================

def discover_files(
    root: Path,
    excludes: List[str],
) -> List[Path]:
    """Return sorted list of .md files under *root*, honouring excludes."""
    results: List[Path] = []
    exclude_set = {e.lower() for e in excludes}

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in dirnames if d.lower() not in exclude_set
        ]
        for fn in filenames:
            if fn.lower().endswith(".md"):
                results.append(Path(dirpath) / fn)

    return sorted(results)


def _file_content_hash(path: Path) -> str:
    """SHA-256 of a file's content."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ===================================================================
# Semantic chunking
# ===================================================================

_RE_CODE_FENCE = re.compile(r"^```", re.MULTILINE)


def _protect_code_fences(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace fenced code blocks with placeholders so they aren't split."""
    placeholders: Dict[str, str] = {}
    parts = _RE_CODE_FENCE.split(text)
    rebuilt: List[str] = []
    inside = False
    buf = ""
    for part in text.split("\n"):
        if part.startswith("```"):
            if inside:
                buf += part + "\n"
                ph = f"__CODE_BLOCK_{len(placeholders)}__"
                placeholders[ph] = buf
                rebuilt.append(ph)
                buf = ""
                inside = False
            else:
                inside = True
                buf = part + "\n"
        elif inside:
            buf += part + "\n"
        else:
            rebuilt.append(part)

    # Unclosed fence – flush remaining
    if buf:
        ph = f"__CODE_BLOCK_{len(placeholders)}__"
        placeholders[ph] = buf
        rebuilt.append(ph)

    return "\n".join(rebuilt), placeholders


def _restore_code_fences(text: str, placeholders: Dict[str, str]) -> str:
    for ph, original in placeholders.items():
        text = text.replace(ph, original)
    return text


def semantic_split(raw: str, source_label: str = "") -> List[Dict[str, Any]]:
    """Split a markdown document into semantically coherent chunks.

    Strategy:
      1. Protect code fences from being broken.
      2. Split on headings (``# … `` through ``###### …``).
      3. Within each section, split on paragraph boundaries (``\\n\\n``).
      4. Merge small pieces; hard-split oversized ones.
      5. Restore code fences.

    Returns a list of dicts with keys: text, heading, chunk_index.
    """
    if not raw.strip():
        return []

    protected, code_phs = _protect_code_fences(raw)

    # ---- Split on headings ------------------------------------------------
    heading_re = re.compile(r"(?=^#{1,6}\s+)", re.MULTILINE)
    sections = heading_re.split(protected)

    raw_chunks: List[Dict[str, Any]] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading (if present)
        heading = ""
        hm = re.match(r"^(#{1,6}\s+.+?)$", section, re.MULTILINE)
        if hm:
            heading = hm.group(1).strip()
            # Remove heading from the text to avoid duplication
            section_body = re.sub(r"^#{1,6}\s+.+?\n", "", section, count=1).strip()
        else:
            section_body = section

        # If the section fits, keep it whole
        if len(section) <= MAX_CHUNK:
            # Prepend heading to text for embedding context
            text_with_heading = f"{heading}\n\n{section_body}" if heading else section_body
            raw_chunks.append({"text": text_with_heading, "heading": heading})
            continue

        # Sub-split using sentence tokenization if available
        if HAS_NLTK:
            try:
                sentences = sent_tokenize(section_body)
                # Group sentences into chunks
                buf = ""
                for sent in sentences:
                    candidate = (buf + " " + sent).strip() if buf else sent
                    if len(candidate) > MAX_CHUNK and buf:
                        raw_chunks.append({"text": f"{heading}\n\n{buf}" if heading else buf, "heading": heading})
                        # Start new chunk with overlap from previous
                        overlap_start = max(0, len(buf) - CHUNK_OVERLAP)
                        buf = buf[overlap_start:] + " " + sent if overlap_start else sent
                    else:
                        buf = candidate
                if buf.strip():
                    raw_chunks.append({"text": f"{heading}\n\n{buf}" if heading else buf, "heading": heading})
            except Exception:
                # Fallback to paragraph splitting
                paragraphs = re.split(r"\n{2,}", section_body)
                buf = ""
                for para in paragraphs:
                    candidate = (buf + "\n\n" + para).strip() if buf else para
                    if len(candidate) > MAX_CHUNK and buf:
                        raw_chunks.append({"text": f"{heading}\n\n{buf}" if heading else buf, "heading": heading})
                        buf = para
                    else:
                        buf = candidate
                if buf.strip():
                    raw_chunks.append({"text": f"{heading}\n\n{buf}" if heading else buf, "heading": heading})
        else:
            # Fallback to paragraph splitting
            paragraphs = re.split(r"\n{2,}", section_body)
            buf = ""
            for para in paragraphs:
                candidate = (buf + "\n\n" + para).strip() if buf else para
                if len(candidate) > MAX_CHUNK and buf:
                    raw_chunks.append({"text": f"{heading}\n\n{buf}" if heading else buf, "heading": heading})
                    buf = para
                else:
                    buf = candidate
            if buf.strip():
                raw_chunks.append({"text": f"{heading}\n\n{buf}" if heading else buf, "heading": heading})

    # ---- Merge tiny chunks -------------------------------------------------
    merged: List[Dict[str, Any]] = []
    for c in raw_chunks:
        if merged and len(merged[-1]["text"]) < MIN_CHUNK:
            merged[-1]["text"] += "\n\n" + c["text"]
            if c["heading"] and not merged[-1]["heading"]:
                merged[-1]["heading"] = c["heading"]
        else:
            merged.append(c)

    # Handle trailing runt
    if len(merged) > 1 and len(merged[-1]["text"]) < MIN_CHUNK:
        merged[-2]["text"] += "\n\n" + merged[-1]["text"]
        merged.pop()

    # ---- Hard-split any remaining chunks > MAX_CHUNK -----------------------
    final: List[Dict[str, Any]] = []
    for c in merged:
        text = c["text"]
        if len(text) <= MAX_CHUNK:
            final.append(c)
            continue
        # Split on line boundaries
        lines = text.split("\n")
        buf = ""
        for line in lines:
            if len(buf) + len(line) + 1 > MAX_CHUNK and buf:
                final.append({"text": buf.strip(), "heading": c["heading"]})
                buf = line
            else:
                buf += ("\n" + line) if buf else line
        if buf.strip():
            final.append({"text": buf.strip(), "heading": c["heading"]})

    # ---- Restore code fences & assign indexes ------------------------------
    for i, c in enumerate(final):
        c["text"] = _restore_code_fences(c["text"], code_phs)
        c["chunk_index"] = i

    return final


# ===================================================================
# Ingestion engine
# ===================================================================

def _delete_old_chunks(mem_ids: List[int]) -> int:
    """Remove previously ingested chunks from the DB by ID."""
    if not mem_ids:
        return 0
    try:
        from memory_engine import get_db_connection
        with get_db_connection() as conn:
            cur = conn.cursor()
            placeholders = ",".join("?" * len(mem_ids))
            cur.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", mem_ids)
            conn.commit()
            return cur.rowcount
    except Exception as e:
        log.error(f"Failed to delete old chunks: {e}")
        return 0


def ingest_workspace(
    workspace_dir: Path,
    excludes: List[str],
    dry_run: bool = False,
    batch_size: int = 50,
) -> Dict[str, Any]:
    """Main ingestion loop.  Compares the workspace against the last manifest
    and only processes changed/new files.  Removes chunks for deleted files."""

    manifest = _load_manifest(MANIFEST_PATH)
    prev_files: Dict[str, Any] = manifest.get("files", {})
    chunk_ids_map: Dict[str, List[int]] = manifest.get("chunk_ids", {})

    md_files = discover_files(workspace_dir, excludes)
    log.info(f"Discovered {len(md_files)} markdown files under {workspace_dir}")

    # Compute current hashes
    current_hashes: Dict[str, str] = {}
    for f in md_files:
        rel = str(f.relative_to(workspace_dir))
        current_hashes[rel] = _file_content_hash(f)

    # Determine what changed
    new_files: List[str] = []
    changed_files: List[str] = []
    unchanged_files: List[str] = []
    deleted_files: List[str] = []

    for rel, h in current_hashes.items():
        if rel not in prev_files:
            new_files.append(rel)
        elif prev_files[rel] != h:
            changed_files.append(rel)
        else:
            unchanged_files.append(rel)

    for rel in prev_files:
        if rel not in current_hashes:
            deleted_files.append(rel)

    log.info(f"  New: {len(new_files)}  Changed: {len(changed_files)}  "
             f"Unchanged: {len(unchanged_files)}  Deleted: {len(deleted_files)}")

    if not dry_run:
        from memory_engine import add_memories_batch, _maybe_save_index

    stats: Dict[str, Any] = {
        "files_new": len(new_files),
        "files_changed": len(changed_files),
        "files_unchanged": len(unchanged_files),
        "files_deleted": len(deleted_files),
        "chunks_created": 0,
        "chunks_removed": 0,
        "errors": [],
    }

    # ---- Phase 1: Remove chunks for deleted & changed files ----------------
    files_to_reindex = changed_files + deleted_files
    for rel in files_to_reindex:
        old_ids = chunk_ids_map.pop(rel, [])
        if old_ids and not dry_run:
            removed = _delete_old_chunks(old_ids)
            stats["chunks_removed"] += removed
            log.info(f"  ✗ Removed {removed} old chunks for {rel}")
        elif old_ids:
            log.info(f"  [DRY] Would remove {len(old_ids)} chunks for {rel}")
            stats["chunks_removed"] += len(old_ids)

    # Also remove from prev_files so manifest stays clean
    for rel in deleted_files:
        prev_files.pop(rel, None)

    # ---- Phase 2: Ingest new & changed files -------------------------------
    files_to_ingest = new_files + changed_files

    # Dedup hashes across the entire run
    seen_hashes: Set[str] = set()

    # Batch accumulators
    batch_vecs: List[List[float]] = []
    batch_metas: List[Dict[str, Any]] = []
    batch_texts: List[str] = []
    batch_file_key: Optional[str] = None
    pending_ids: List[int] = []

    def _flush_batch() -> List[int]:
        if not batch_vecs or dry_run:
            return []
        try:
            ids = add_memories_batch(batch_vecs, batch_metas, batch_texts)
            log.info(f"    → Stored {len(ids)} chunks")
            return ids
        except Exception as e:
            log.error(f"    ✗ Batch insert failed: {e}")
            stats["errors"].append(str(e))
            return []
        finally:
            batch_vecs.clear()
            batch_metas.clear()
            batch_texts.clear()

    for rel in files_to_ingest:
        filepath = workspace_dir / rel
        if not filepath.is_file():
            continue

        log.info(f"  Ingesting {rel}")

        try:
            raw = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.error(f"    ✗ Read error: {e}")
            stats["errors"].append(f"{rel}: read error: {e}")
            continue

        chunks = semantic_split(raw, source_label=rel)
        if not chunks:
            log.info(f"    (empty – skipped)")
            prev_files[rel] = current_hashes[rel]
            chunk_ids_map[rel] = []
            continue

        file_chunk_ids: List[int] = []

        for chunk in chunks:
            text = chunk["text"]
            content_hash = hashlib.sha256(text.encode()).hexdigest()

            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            meta = {
                "source_file": rel,
                "source_type": "workspace_doc",
                "heading": chunk.get("heading", ""),
                "chunk_index": chunk["chunk_index"],
                "content_hash": content_hash,
                "ingested_at": time.time(),
                "timestamp": filepath.stat().st_mtime,
                "text": text[:2000],
            }

            if dry_run:
                log.info(
                    f"    [DRY] chunk {chunk['chunk_index']:>2d}  "
                    f"len={len(text):>5d}  "
                    f"heading={chunk.get('heading', '')[:50]}"
                )
                stats["chunks_created"] += 1
                continue

            vec = embed_text(text)
            batch_vecs.append(vec)
            batch_metas.append(meta)
            batch_texts.append(text)
            stats["chunks_created"] += 1

            if len(batch_vecs) >= batch_size:
                ids = _flush_batch()
                file_chunk_ids.extend(ids)

        # Flush remaining
        ids = _flush_batch()
        file_chunk_ids.extend(ids)

        # Update manifest
        prev_files[rel] = current_hashes[rel]
        chunk_ids_map[rel] = file_chunk_ids

    # ---- Phase 3: Save FAISS index & manifest ------------------------------
    if not dry_run:
        try:
            _maybe_save_index(force=True)
        except Exception:
            pass

    manifest = {"files": {**prev_files, **{r: current_hashes[r] for r in unchanged_files}},
                "chunk_ids": chunk_ids_map}
    if not dry_run:
        _save_manifest(MANIFEST_PATH, manifest)
        log.info(f"  Manifest saved to {MANIFEST_PATH}")

    return stats


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest all workspace markdown files into the memory engine.",
    )
    parser.add_argument(
        "workspace_dir",
        nargs="?",
        default=DEFAULT_WORKSPACE,
        help=f"Root of the workspace to scan (default: {DEFAULT_WORKSPACE}).",
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
        "--exclude",
        action="append",
        default=None,
        help="Directory name to exclude (repeatable).  Defaults: "
             + ", ".join(DEFAULT_EXCLUDES),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest every file, ignoring the manifest.",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Remove manifest entries for files that no longer exist in the workspace. "
             "Useful for cleaning up after files/directories are deleted.",
    )
    parser.add_argument(
        "--no-hallucination-filter",
        action="store_true",
        help="Disable hallucination filter for workspace ingestion. "
             "Workspace docs are trusted sources, so filtering is disabled by default.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    excludes = args.exclude if args.exclude else list(DEFAULT_EXCLUDES)
    workspace = Path(args.workspace_dir).resolve()

    if not workspace.is_dir():
        log.error(f"Directory not found: {workspace}")
        sys.exit(1)

    if args.force and MANIFEST_PATH.is_file():
        log.info("--force: deleting existing manifest")
        MANIFEST_PATH.unlink()

    # Disable hallucination filter for workspace ingestion (trusted sources)
    if args.no_hallucination_filter:
        os.environ['MEMORY_FILTER_HALLUCINATIONS'] = 'false'
        log.info("Hallucination filter disabled for workspace ingestion")

    # Handle --prune-missing: remove manifest entries for deleted files
    if args.prune_missing and MANIFEST_PATH.is_file():
        manifest = _load_manifest(MANIFEST_PATH)
        prev_files = manifest.get("files", {})
        chunk_ids_map = manifest.get("chunk_ids", {})
        
        missing_files = []
        for rel in list(prev_files.keys()):
            if not (workspace / rel).exists():
                missing_files.append(rel)
        
        if missing_files:
            log.info(f"--prune-missing: cleaning up {len(missing_files)} deleted file(s)")
            
            # Delete chunks from DB for missing files
            if not args.dry_run:
                all_ids_to_delete = []
                for rel in missing_files:
                    all_ids_to_delete.extend(chunk_ids_map.pop(rel, []))
                    prev_files.pop(rel, None)
                
                if all_ids_to_delete:
                    removed = _delete_old_chunks(all_ids_to_delete)
                    log.info(f"  → Removed {removed} chunks from DB")
            else:
                for rel in missing_files:
                    prev_files.pop(rel, None)
                    chunk_ids_map.pop(rel, [])
                log.info(f"  [DRY] Would remove {len(missing_files)} file entries")
            
            # Save cleaned manifest
            manifest = {"files": prev_files, "chunk_ids": chunk_ids_map}
            if not args.dry_run:
                _save_manifest(MANIFEST_PATH, manifest)
                log.info(f"  Manifest updated and saved")
            
            log.info("--prune-missing: done")
            sys.exit(0)
        
        log.info("--prune-missing: no missing files found")

    log.info(f"Workspace        : {workspace}")
    log.info(f"Excludes         : {excludes}")
    log.info(f"Dry run          : {args.dry_run}")
    log.info(f"Batch size       : {args.batch_size}")
    log.info("")

    t0 = time.time()
    stats = ingest_workspace(
        workspace,
        excludes=excludes,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0

    log.info("")
    log.info("═" * 50)
    log.info(f"  New files       : {stats['files_new']}")
    log.info(f"  Changed files   : {stats['files_changed']}")
    log.info(f"  Unchanged files : {stats['files_unchanged']}")
    log.info(f"  Deleted files   : {stats['files_deleted']}")
    log.info(f"  Chunks created  : {stats['chunks_created']}")
    log.info(f"  Chunks removed  : {stats['chunks_removed']}")
    log.info(f"  Errors          : {len(stats['errors'])}")
    log.info(f"  Elapsed         : {elapsed:.1f}s")
    log.info("═" * 50)

    if stats["errors"]:
        log.warning("Errors encountered:")
        for e in stats["errors"][:10]:
            log.warning(f"  • {e}")


if __name__ == "__main__":
    main()
