"""
knowledge_graph_layer.py – SQLite knowledge-graph helper.

Stores triples (subject, predicate, object) in a local SQLite DB (kg.db).
"""
import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple

DB_PATH = Path(__file__).with_name('kg.db')

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS triples (
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            PRIMARY KEY (subject, predicate, object)
        )
    ''')
    return conn

def add_triple(subject: str, predicate: str, obj: str):
    """Insert a new triple. Duplicate triples are ignored."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO triples (subject, predicate, object) VALUES (?,?,?)",
            (subject, predicate, obj)
        )
        conn.commit()

def query(subject: Optional[str] = None, predicate: Optional[str] = None,
          obj: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """Return matching (subject, predicate, object) triples."""
    sql = "SELECT subject, predicate, object FROM triples WHERE 1=1"
    params = []
    if subject is not None:
        sql += " AND subject = ?"
        params.append(subject)
    if predicate is not None:
        sql += " AND predicate = ?"
        params.append(predicate)
    if obj is not None:
        sql += " AND object = ?"
        params.append(obj)
    with _get_conn() as conn:
        cur = conn.execute(sql, params)
        return cur.fetchall()

def relational_query(question: str) -> List[str]:
    """Naive NL parser for questions like 'what agents are following X?'."""
    lowered = question.lower()
    if "following" in lowered:
        parts = lowered.split("following")
        target = parts[1].strip().strip('?')
        rows = query(predicate="follows", obj=target)
        return [subj for subj, _, _ in rows]
    return []

if __name__ == "__main__":
    add_triple("agentA", "follows", "agentB")
    print("All triples:", query())
    print("Who follows agentB?", relational_query("What agents are following agentB?"))
