#!/usr/bin/env python3
"""
Import all markdown memory files (memory/*.md) into the memory store.

Usage:
    python3 import_memories.py
"""

import os
import sys
import json
import glob
import time
import re
import numpy as np

# Allow importing from core/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

try:
    import core.memory_engine as memory_engine
except ImportError:
    import memory_engine

EMB_DIM = int(os.getenv("MEMORY_EMB_DIM", "768"))


def dummy_embedding() -> list:
    """Return a random float vector of the correct dimension."""
    return np.random.rand(EMB_DIM).astype("float32").tolist()


def import_memories(memory_dir: str = None):
    if memory_dir is None:
        memory_dir = os.path.join(PROJECT_ROOT, "memory")

    mem_files = sorted(glob.glob(os.path.join(memory_dir, "*.md")))
    if not mem_files:
        print("⚡ No memory files found")
        return

    added = 0
    session_map = {}
    timestamp_regex = re.compile(r"^\[?\d{4}-\d{2}-\d{2}.*?\]?\s*")

    for path in mem_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except Exception as e:
            print(f"⚠️  Failed to read {path}: {e}")
            continue

        if not raw_text.strip():
            continue

        # Split on timestamp lines
        chunks = []
        current_chunk = []
        for line in raw_text.splitlines():
            if timestamp_regex.match(line):
                if current_chunk:
                    chunks.append("\n".join(current_chunk).strip())
                    current_chunk = []
                current_chunk.append(line.strip())
            else:
                current_chunk.append(line.strip())
        if current_chunk:
            chunks.append("\n".join(current_chunk).strip())

        if len(chunks) == 1:
            chunks = [c.strip() for c in raw_text.split("\n\n") if c.strip()]

        file_mem_ids = []
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            metadata = {
                "source_file": os.path.relpath(path, PROJECT_ROOT),
                "imported_at": time.time(),
                "chunk_index": i,
                "preview": chunk[:200],
            }
            try:
                mem_id = memory_engine.add_memory(
                    vector=dummy_embedding(),
                    metadata=metadata,
                    text=chunk,
                )
                print(f"✅ Added chunk {i} from {path!r} as memory ID {mem_id}")
                added += 1
                file_mem_ids.append(mem_id)
            except Exception as e:
                print(f"❌ Error adding chunk {i} from {path!r}: {e}")
        session_map[os.path.relpath(path, PROJECT_ROOT)] = file_mem_ids

    try:
        map_path = os.path.join(PROJECT_ROOT, "session_to_memories.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(session_map, f, indent=2)
        print(f"🗂️  Session-memory map written to {map_path}")
    except Exception as e:
        print(f"⚠️  Failed to write session map: {e}")

    print(f"🚀 Finished – {added}/{len(mem_files)} chunks imported.")


if __name__ == "__main__":
    start = time.time()
    import_memories()
    print(f"⏱️  Took {time.time() - start:.2f}s")
