"""
semantic_graph.py – lightweight hierarchical semantic graph built on top of memory_engine.

Stores concept nodes and edges in a dedicated SQLite table `mem_graph_edges`.
Concept nodes use negative IDs to distinguish from memory IDs.
"""

import re
import json
import logging
import sqlite3
from typing import List, Set, Optional
from contextlib import contextmanager
from collections import deque

try:
    from .memory_engine import get_db_connection, _increment_unsaved, _maybe_save_index, _index_lock
    from .vector_memory_wrapper import denoise_embeddings, EMB_DIM
except ImportError:
    from memory_engine import get_db_connection, _increment_unsaved, _maybe_save_index, _index_lock
    from vector_memory_wrapper import denoise_embeddings, EMB_DIM

import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Edge table setup
# ----------------------------------------------------------------------

def _ensure_edge_table():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS mem_graph_edges (
                src_id INTEGER NOT NULL,
                dst_id INTEGER NOT NULL,
                weight REAL DEFAULT 1.0,
                type TEXT NOT NULL,
                UNIQUE(src_id, dst_id, type)
            )
        ''')
        conn.commit()

_ensure_edge_table()

def _get_or_create_concept_id(concept: str) -> int:
    """Return a stable negative ID for *concept*."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT src_id FROM mem_graph_edges WHERE dst_id = ? AND type = 'concept'",
            (concept,)
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute('SELECT MIN(src_id) FROM mem_graph_edges WHERE src_id < 0')
        min_neg = cur.fetchone()[0]
        new_id = -1 if min_neg is None else min_neg - 1
        cur.execute(
            "INSERT INTO mem_graph_edges (src_id, dst_id, weight, type) VALUES (?, ?, 1.0, 'concept')",
            (new_id, concept)
        )
        conn.commit()
        return new_id

def _add_edge(src: int, dst: int, weight: float = 1.0, edge_type: str = 'contains') -> None:
    """Insert a directed edge (src -> dst)."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO mem_graph_edges (src_id, dst_id, weight, type) VALUES (?, ?, ?, ?)",
            (src, dst, weight, edge_type)
        )
        conn.commit()
    _increment_unsaved()
    _maybe_save_index()

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def index_memory(mem_id: int, text: str) -> None:
    """Extract simple terms from *text* and link the memory to those concept nodes."""
    tokens = set(t.lower() for t in re.findall(r"\b\w{4,}\b", text))
    for token in tokens:
        concept_id = _get_or_create_concept_id(token)
        _add_edge(mem_id, concept_id, edge_type='contains')
        _add_edge(concept_id, mem_id, edge_type='instance')


def link_semantic_neighbors(k: int = 8, temporal_weight: float = 0.15, source_boost: float = 0.05) -> None:
    """
    Advanced relationship indexing.
    Computes k-NN using denoised embeddings, temporal proximity, and source files.
    This mirrors the 'high quality' clustering found in the visualizer.
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, vector, metadata, timestamp FROM memories WHERE merged = 0")
        rows = cur.fetchall()

    if len(rows) < 2:
        return

    ids = []
    vectors = []
    timestamps = []
    sources = []
    
    for mid, v_blob, meta_json, ts in rows:
        vec = np.frombuffer(v_blob, dtype='float32')
        if vec.shape[0] != EMB_DIM: continue
        meta = json.loads(meta_json) if meta_json else {}
        
        ids.append(mid)
        vectors.append(vec)
        timestamps.append(ts)
        sources.append(meta.get('source_file', meta.get('source', '')))

    X = np.vstack(vectors)
    # 1. Denoise (all-but-the-top)
    X_denoised = denoise_embeddings(X, remove_top_d=3, target_dim=min(256, len(X)-1))
    
    # 2. Base k-NN
    fetch_k = min(k * 2, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=fetch_k + 1, metric='cosine', algorithm='brute').fit(X_denoised)
    distances, indices = nbrs.kneighbors(X_denoised)

    # 3. Blend and add edges
    tau = 43200.0 # 12h half-life
    
    for i in range(len(ids)):
        src_id = ids[i]
        for j_idx, dist in zip(indices[i][1:], distances[i][1:]):
            dst_id = ids[j_idx]
            cos_sim = max(0.0, 1.0 - float(dist))
            
            # Temporal boost
            dt = abs(timestamps[i] - timestamps[j_idx])
            t_sim = float(np.exp(-(dt**2) / (2 * tau**2)))
            
            # Source boost
            s_boost = source_boost if sources[i] and sources[i] == sources[j_idx] else 0.0
            
            blended = (1.0 - temporal_weight) * cos_sim + temporal_weight * t_sim + s_boost
            
            if blended > 0.4: # Quality threshold
                _add_edge(src_id, dst_id, weight=round(blended, 3), edge_type='neighbor')


def graph_neighbors(node_id: int, depth: int = 1) -> Set[int]:
    """BFS walk from *node_id* up to *depth* hops."""
    visited: Set[int] = {node_id}
    frontier: Set[int] = {node_id}
    for _ in range(depth):
        next_frontier: Set[int] = set()
        with get_db_connection() as conn:
            cur = conn.cursor()
            for nid in frontier:
                cur.execute('SELECT dst_id FROM mem_graph_edges WHERE src_id = ?', (nid,))
                for (dst,) in cur.fetchall():
                    if dst not in visited:
                        visited.add(dst)
                        next_frontier.add(dst)
        if not next_frontier:
            break
        frontier = next_frontier
    visited.remove(node_id)
    return visited

def graph_path(src_id: int, dst_id: int, max_hops: int = 3) -> Optional[List[int]]:
    """Return a simple BFS path from *src_id* to *dst_id*."""
    queue = deque([(src_id, [src_id])])
    visited = {src_id}
    while queue:
        current, path = queue.popleft()
        if len(path) > max_hops + 1:
            continue
        if current == dst_id:
            return path
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute('SELECT dst_id FROM mem_graph_edges WHERE src_id = ?', (current,))
            for (nbr,) in cur.fetchall():
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, path + [nbr]))
    return None

# ----------------------------------------------------------------------
# Auto-index hook: monkey-patches memory_engine.add_memory
# ----------------------------------------------------------------------

try:
    from .memory_engine import add_memory as _orig_add_memory
    import importlib
    from . import memory_engine as _me_module
except ImportError:
    from memory_engine import add_memory as _orig_add_memory
    import memory_engine as _me_module

def _patched_add_memory(vector, metadata, text=''):
    mem_id = _orig_add_memory(vector, metadata, text)
    effective_text = text or metadata.get('text', '')
    if effective_text:
        index_memory(mem_id, effective_text)
    return mem_id

_me_module.add_memory = _patched_add_memory
logger.info('Semantic graph hook installed – new memories will be indexed automatically.')
