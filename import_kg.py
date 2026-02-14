#!/usr/bin/env python3
import csv
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_FILE = Path("/root/.openclaw/workspace/projects/memory-engine/data/knowledge_graph.csv")
DB_FILE = Path("/root/.openclaw/workspace/projects/memory-engine/data/kg.db")

def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS triples (
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            PRIMARY KEY (subject, predicate, object)
        )
    ''')
    return conn

def import_csv():
    if not CSV_FILE.exists():
        print(f"CSV file not found: {CSV_FILE}")
        sys.exit(1)
    
    conn = get_conn()
    count = 0
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row.get('subject', '').strip()
            predicate = row.get('predicate', '').strip()
            obj = row.get('object', '').strip()
            
            if subject and predicate and obj:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO triples (subject, predicate, object) VALUES (?,?,?)",
                        (subject, predicate, obj)
                    )
                    count += 1
                except Exception as e:
                    print(f"Error inserting ({subject}, {predicate}, {obj}): {e}")
    
    conn.commit()
    conn.close()
    print(f"Imported {count} triples into {DB_FILE}")

if __name__ == "__main__":
    import_csv()
