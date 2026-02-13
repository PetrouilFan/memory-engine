import json
import os
from pathlib import Path

CACHE_FILE = Path('.memory_slice_index.json')
CACHE = {}


def load_cache():
    global CACHE
    if CACHE_FILE.is_file():
        try:
            CACHE = json.loads(CACHE_FILE.read_text())
        except Exception:
            CACHE = {}
    else:
        CACHE = {}


def save_cache():
    CACHE_FILE.write_text(json.dumps(CACHE, indent=2))


def index_file(path):
    """Scan a Python file and store line ranges for each top-level class/function."""
    lines = Path(path).read_text().splitlines()
    entries = {}
    name = None
    start = None
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith('def ') or stripped.startswith('class '):
            if name:
                entries[name] = (start, i - 1)
            name = stripped.split('(')[0].split()[1].split(':')[0]
            start = i
    if name:
        entries[name] = (start, len(lines))
    CACHE[path] = entries
    save_cache()


def get_location(path, name):
    load_cache()
    if path not in CACHE:
        index_file(path)
    return CACHE.get(path, {}).get(name)


def read_slice(path, start, end, pad=20):
    """Return lines start..end (1-based inclusive)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{path} not found")
    lines = p.read_text().splitlines()
    total = len(lines)
    s = max(1, start)
    e = min(total, end)
    while e - s + 1 < (end - start + 1) and (s > 1 or e < total):
        if s > 1:
            s = max(1, s - pad)
        if e < total:
            e = min(total, e + pad)
    return "\n".join(lines[s-1:e])


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Read a slice of a file')
    parser.add_argument('path', help='File path')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--range', nargs=2, metavar=('START', 'END'), type=int)
    group.add_argument('--symbol', help='Function or class name to read')
    parser.add_argument('--pad', type=int, default=20)
    args = parser.parse_args()
    if args.range:
        start, end = args.range
        print(read_slice(args.path, start, end, pad=args.pad))
    else:
        loc = get_location(args.path, args.symbol)
        if not loc:
            index_file(args.path)
            loc = get_location(args.path, args.symbol)
        if not loc:
            raise ValueError(f"Symbol {args.symbol} not found in {args.path}")
        start, end = loc
        print(read_slice(args.path, start, end, pad=args.pad))


if __name__ == '__main__':
    main()
