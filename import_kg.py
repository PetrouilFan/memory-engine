#!/usr/bin/env python3
"""
Import knowledge graph triples from CSV into kg.db

Supports the enhanced schema with:
- subject, predicate, object
- subject_type, object_type (agent, skill, tool, service, file, user, cron, api, other)
- source_file
- confidence score
"""
import csv
import sqlite3
import sys
from pathlib import Path
import importlib.util

CSV_FILE = Path("/root/.openclaw/workspace/projects/memory-engine/data/knowledge_graph.csv")
DB_FILE = Path("/root/.openclaw/workspace/projects/memory-engine/data/kg.db")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.knowledge_graph_layer import _get_conn, add_triples_batch, get_stats

def import_csv():
    """Import triples from CSV file into kg.db."""
    if not CSV_FILE.exists():
        print(f"CSV file not found: {CSV_FILE}")
        sys.exit(1)
    
    conn = _get_conn()
    
    triples = []
    skipped = 0
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row.get('subject', '').strip()
            predicate = row.get('predicate', '').strip()
            obj = row.get('object', '').strip()
            subject_type = row.get('subject_type', 'other').strip() or 'other'
            object_type = row.get('object_type', 'other').strip() or 'other'
            source_file = row.get('source_file', '').strip()
            
            try:
                confidence = float(row.get('confidence', 1.0))
            except:
                confidence = 1.0
            
            if subject and predicate and obj:
                triples.append({
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj,
                    'subject_type': subject_type,
                    'object_type': object_type,
                    'source_file': source_file,
                    'confidence': confidence
                })
            else:
                skipped += 1
    
    conn.execute("DELETE FROM triples")
    conn.commit()
    
    batch_size = 100
    for i in range(0, len(triples), batch_size):
        batch = triples[i:i + batch_size]
        for t in batch:
            try:
                conn.execute(
                    """INSERT INTO triples 
                       (subject, predicate, object, subject_type, object_type, source_file, confidence) 
                       VALUES (?,?,?,?,?,?,?)""",
                    (t['subject'], t['predicate'], t['object'], 
                     t['subject_type'], t['object_type'], t['source_file'], t['confidence'])
                )
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        print(f"Imported {min(i + batch_size, len(triples))}/{len(triples)} triples...")
    
    conn.close()
    
    print(f"\nImported {len(triples)} triples into {DB_FILE}")
    if skipped:
        print(f"Skipped {skipped} invalid rows")
    
    stats = get_stats()
    print(f"\nKnowledge Graph Statistics:")
    print(f"  Total triples: {stats['total_triples']}")
    print(f"  Subject types: {stats['subject_types']}")
    print(f"  Object types: {stats['object_types']}")
    print(f"  Top predicates: {dict(list(stats['top_predicates'].items())[:10])}")

if __name__ == "__main__":
    import_csv()
