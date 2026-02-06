"""Crawl markdown files and index them via HolographicMemory."""
import os
from pathlib import Path
import sys

# Allow importing from core/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from core.holographic_memory import HolographicMemory


def crawl_and_index(root: Path):
    """Recursively find .md files and store them via HolographicMemory."""
    md_files = list(root.rglob('*.md'))
    print(f'Found {len(md_files)} markdown files under {root}')
    hm = HolographicMemory()
    added = 0

    for md_path in md_files:
        try:
            content = md_path.read_text(encoding='utf-8')
            if not content.strip():
                continue
            hm.add_memory(content, metadata={
                'source_path': str(md_path),
                'file_mtime': os.path.getmtime(md_path)
            })
            added += 1
            print(f'→ Indexed {md_path}')
        except Exception as e:
            print(f'⚠️  Failed {md_path}: {e}')

    print(f'Finished – {added} files processed.')

if __name__ == '__main__':
    workspace_root = Path(__file__).parent
    crawl_and_index(workspace_root)
