#!/usr/bin/env python3
"""Export all holographic memories to a JSON file."""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from core.holographic_memory import HolographicMemory

store_path = Path(__file__).parent.parent / "data" / "holo_store.json"
raw_entries = []
if store_path.is_file():
    try:
        raw_entries = json.loads(store_path.read_text())
    except Exception as e:
        print(f"Failed to read {store_path}: {e}")

mem = HolographicMemory()
for entry in raw_entries:
    text = entry.get("text", "")
    meta = entry.get("metadata", {})
    mem.add_memory(text, meta)

export = mem.export_all()
output_path = Path(__file__).parent.parent / "data" / "memory_export.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(export, ensure_ascii=False, indent=2))
print(f"Exported {len(export)} memory entries to {output_path}")
