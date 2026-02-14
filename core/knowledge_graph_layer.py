"""
knowledge_graph_layer.py – SQLite knowledge-graph helper.

Stores triples (subject, predicate, object) with entity types in a local SQLite DB (kg.db).

Schema:
- subject, predicate, object: core triple
- subject_type: 'agent', 'skill', 'tool', 'service', 'file', 'user', 'cron', 'api', 'other'
- object_type: same as above
- source_file: where the triple came from
- confidence: 0.0-1.0 (how confident we are in this triple)
"""
import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

DB_PATH = Path(__file__).resolve().parent.parent / 'data' / 'kg.db'

ENTITY_TYPES = ['agent', 'skill', 'tool', 'service', 'file', 'user', 'cron', 'api', 'other']

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            subject_type TEXT DEFAULT 'other',
            object_type TEXT DEFAULT 'other',
            source_file TEXT,
            confidence REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(subject, predicate, object)
        )
    ''')
    
    conn.execute('CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_predicate ON triples(predicate)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_object ON triples(object)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_subject_type ON triples(subject_type)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_object_type ON triples(object_type)')
    
    return conn

def detect_entity_type(text: str) -> str:
    """Detect entity type from text content."""
    text_lower = text.lower()
    
    if text_lower in ['agent', 'system', 'max', 'user', 'assistant']:
        return 'agent'
    
    if 'skill' in text_lower or text.startswith('skills/'):
        return 'skill'
    
    if '.py' in text or '.sh' in text or text.startswith('tools/') or text.startswith('scripts/'):
        return 'tool'
    
    if any(svc in text_lower for svc in ['postgres', 'redis', 'nextcloud', 'ollama', 'gateway', 'stt', 'embedding', 'tts']):
        return 'service'
    
    if 'api_key' in text_lower or '_api_key' in text_lower:
        return 'api'
    
    if 'cron' in text_lower or text.startswith('cron'):
        return 'cron'
    
    if any(ext in text for ext in ['.py', '.md', '.json', '.sh', '.txt']):
        return 'file'
    
    return 'other'

def add_triple(subject: str, predicate: str, obj: str, 
               subject_type: Optional[str] = None,
               object_type: Optional[str] = None,
               source_file: str = None,
               confidence: float = 1.0):
    """Insert a new triple. Duplicate triples are ignored."""
    if subject_type is None:
        subject_type = detect_entity_type(subject)
    if object_type is None:
        object_type = detect_entity_type(obj)
    
    with _get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO triples 
               (subject, predicate, object, subject_type, object_type, source_file, confidence) 
               VALUES (?,?,?,?,?,?,?)""",
            (subject, predicate, obj, subject_type, object_type, source_file, confidence)
        )
        conn.commit()

def add_triples_batch(triples: List[Dict[str, Any]]):
    """Batch insert triples for efficiency."""
    with _get_conn() as conn:
        for t in triples:
            subject = t.get('subject', '')
            predicate = t.get('predicate', '')
            obj = t.get('object', '')
            subject_type = t.get('subject_type') or detect_entity_type(subject)
            object_type = t.get('object_type') or detect_entity_type(obj)
            source_file = t.get('source_file')
            confidence = t.get('confidence', 1.0)
            
            conn.execute(
                """INSERT OR IGNORE INTO triples 
                   (subject, predicate, object, subject_type, object_type, source_file, confidence) 
                   VALUES (?,?,?,?,?,?,?)""",
                (subject, predicate, obj, subject_type, object_type, source_file, confidence)
            )
        conn.commit()

def query(subject: Optional[str] = None, predicate: Optional[str] = None,
          obj: Optional[str] = None, 
          subject_type: Optional[str] = None,
          object_type: Optional[str] = None) -> List[Tuple]:
    """Return matching triples with full details."""
    sql = "SELECT id, subject, predicate, object, subject_type, object_type, source_file, confidence FROM triples WHERE 1=1"
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
    if subject_type is not None:
        sql += " AND subject_type = ?"
        params.append(subject_type)
    if object_type is not None:
        sql += " AND object_type = ?"
        params.append(object_type)
        
    with _get_conn() as conn:
        cur = conn.execute(sql, params)
        return cur.fetchall()

def get_stats() -> Dict[str, Any]:
    """Get knowledge graph statistics."""
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        
        subject_counts = dict(conn.execute(
            "SELECT subject_type, COUNT(*) FROM triples GROUP BY subject_type"
        ).fetchall())
        
        object_counts = dict(conn.execute(
            "SELECT object_type, COUNT(*) FROM triples GROUP BY object_type"
        ).fetchall())
        
        pred_counts = dict(conn.execute(
            "SELECT predicate, COUNT(*) FROM triples GROUP BY predicate ORDER BY COUNT(*) DESC LIMIT 20"
        ).fetchall())
        
        return {
            'total_triples': total,
            'subject_types': subject_counts,
            'object_types': object_counts,
            'top_predicates': pred_counts
        }

def relational_query(question: str) -> List[str]:
    """Naive NL parser for questions like 'what agents are following X?'."""
    lowered = question.lower()
    if "following" in lowered:
        parts = lowered.split("following")
        target = parts[1].strip().strip('?')
        rows = query(predicate="follows", obj=target)
        return [subj for _, _, _, _, _, _, _, _ in rows]
    return []

def query_by_entity(entity: str, entity_type: str = None, limit: int = 20) -> List[Dict]:
    """Find all triples involving an entity (as subject or object)."""
    with _get_conn() as conn:
        cur = conn.execute("""
            SELECT subject, predicate, object, subject_type, object_type, source_file, confidence
            FROM triples 
            WHERE subject = ? OR object = ?
            ORDER BY confidence DESC, subject, predicate
            LIMIT ?
        """, (entity, entity, limit))
        return _rows_to_dicts(cur.fetchall())

def query_related(entity: str, depth: int = 1, limit: int = 50) -> Dict[str, List]:
    """Find related entities up to a certain depth."""
    related = {'direct': [], 'entities': set()}
    
    direct = query_by_entity(entity, limit=limit)
    related['direct'] = direct
    
    for t in direct:
        if t['subject'] != entity:
            related['entities'].add(t['subject'])
        if t['object'] != entity:
            related['entities'].add(t['object'])
    
    related['entities'] = list(related['entities'])[:limit]
    return related

def query_by_type(entity_type: str, as_subject: bool = True, limit: int = 50) -> List[Dict]:
    """Find all triples where entity is of a specific type."""
    with _get_conn() as conn:
        if as_subject:
            cur = conn.execute("""
                SELECT subject, predicate, object, subject_type, object_type, source_file, confidence
                FROM triples 
                WHERE subject_type = ?
                ORDER BY confidence DESC
                LIMIT ?
            """, (entity_type, limit))
        else:
            cur = conn.execute("""
                SELECT subject, predicate, object, subject_type, object_type, source_file, confidence
                FROM triples 
                WHERE object_type = ?
                ORDER BY confidence DESC
                LIMIT ?
            """, (entity_type, limit))
        return _rows_to_dicts(cur.fetchall())

def query_by_predicate(predicate: str, limit: int = 50) -> List[Dict]:
    """Find all triples with a specific predicate."""
    with _get_conn() as conn:
        cur = conn.execute("""
            SELECT subject, predicate, object, subject_type, object_type, source_file, confidence
            FROM triples 
            WHERE predicate = ?
            ORDER BY confidence DESC
            LIMIT ?
        """, (predicate, limit))
        return _rows_to_dicts(cur.fetchall())

def search_by_keyword(keyword: str, limit: int = 50) -> List[Dict]:
    """Search for triples containing a keyword in subject or object."""
    with _get_conn() as conn:
        pattern = f"%{keyword}%"
        cur = conn.execute("""
            SELECT subject, predicate, object, subject_type, object_type, source_file, confidence
            FROM triples 
            WHERE subject LIKE ? OR object LIKE ? OR predicate LIKE ?
            ORDER BY confidence DESC
            LIMIT ?
        """, (pattern, pattern, pattern, limit))
        return _rows_to_dicts(cur.fetchall())

def _rows_to_dicts(rows: List[Tuple]) -> List[Dict]:
    """Convert database rows to list of dicts."""
    result = []
    for row in rows:
        result.append({
            'subject': row[0],
            'predicate': row[1],
            'object': row[2],
            'subject_type': row[3],
            'object_type': row[4],
            'source_file': row[5],
            'confidence': row[6]
        })
    return result

if __name__ == "__main__":
    add_triple("agent", "uses", "browser_skill", subject_type="agent", object_type="skill")
    print("All triples:", query())
    print("Stats:", get_stats())
