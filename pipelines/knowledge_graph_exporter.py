# Knowledge-Graph Exporter
# Scans MEMORY.md and daily logs to extract simple entity-relationship triples.
# Output is a CSV file with columns: subject, predicate, object, source_file.

import os
import re
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_FILE = PROJECT_ROOT / "MEMORY.md"
MEMORY_DIR = PROJECT_ROOT / "memory"
OUTPUT = PROJECT_ROOT / "data" / "knowledge_graph.csv"

TRIPLE_REGEX = re.compile(r"(?P<subj>\w+)\s+(?P<pred>\w+)\s+(?P<obj>\w+)")


def extract_triples(text, source):
    triples = []
    for line in text.splitlines():
        m = TRIPLE_REGEX.search(line)
        if m:
            triples.append((m.group('subj'), m.group('pred'), m.group('obj'), source))
    return triples


def gather_sources():
    sources = []
    if MEMORY_FILE.exists():
        sources.append((MEMORY_FILE.read_text(encoding='utf-8'), "MEMORY.md"))
    if MEMORY_DIR.is_dir():
        for fname in sorted(os.listdir(MEMORY_DIR)):
            if fname.endswith('.md'):
                path = MEMORY_DIR / fname
                sources.append((path.read_text(encoding='utf-8'), f"memory/{fname}"))
    return sources


def build_graph():
    all_triples = []
    for text, src in gather_sources():
        all_triples.extend(extract_triples(text, src))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["subject", "predicate", "object", "source_file"])
        for row in all_triples:
            writer.writerow(row)
    print(f"Knowledge graph written to {OUTPUT} ({len(all_triples)} triples)")


if __name__ == "__main__":
    build_graph()
