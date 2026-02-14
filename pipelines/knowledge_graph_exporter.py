# Knowledge-Graph Exporter
# Scans MEMORY.md and daily logs to extract entity-relationship triples.
# Output is a CSV file with columns: subject, predicate, object, source_file.

import os
import re
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MEMORY_FILE = Path("/root/.openclaw/workspace/MEMORY.md")
MEMORY_DIR = Path("/root/.openclaw/workspace/memory")
OUTPUT = PROJECT_ROOT / "projects/memory-engine/data" / "knowledge_graph.csv"

ACTION_VERBS = [
    'added', 'created', 'fixed', 'updated', 'implemented', 'removed',
    'patched', 'disabled', 'installed', 'wired', 'configured',
    'improved', 'refactored', 'enhanced', 'resolved', 'replaced'
]

def clean_object(text):
    text = text.strip()
    text = re.sub(r'^[-*]\s*', '', text)
    text = re.sub(r'^(\d+\.)\s*', '', text)
    text = text.strip('.,;:')
    return text

def extract_from_line(line, source, triples):
    line = line.strip()
    if not line or line.startswith('#'):
        return
    
    line_clean = re.sub(r'^\d{4}-\d{2}-\d{2}:\s*', '', line)
    
    for verb in ACTION_VERBS:
        pattern = rf'^{verb}\s+(.+)$'
        m = re.match(pattern, line_clean, re.IGNORECASE)
        if m:
            obj = clean_object(m.group(1))
            if obj and len(obj) > 1:
                obj = obj[:100]
                triples.append(('agent', verb, obj, source))
            return


def extract_triples(text, source):
    triples = []
    for line in text.splitlines():
        line = line.strip()
        
        if line.startswith('- ') or line.startswith('* '):
            content = line[2:]
            if content.startswith('- ') or content.startswith('* '):
                content = content[2:]
            extract_from_line(content, source, triples)
        elif line.startswith('  - ') or line.startswith('  * '):
            content = line[4:]
            extract_from_line(content, source, triples)
        else:
            extract_from_line(line, source, triples)
    
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
