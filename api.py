"""FastAPI wrapper exposing the memory_engine functionality via a REST API.

The core engine lives in :pymod:`projects.memory_engine.core.memory_engine` and
provides the following high‑level operations that are useful for external
clients:

* ``add_memory`` – add a single memory entry (vector, metadata, optional text)
* ``add_memories_batch`` – add multiple entries in one request
* ``hybrid_search`` – perform a hybrid vector + keyword search
* ``get_stats`` – retrieve basic statistics about the store
* ``force_save_index`` – trigger immediate index persistence (admin endpoint)
* ``shutdown`` – cleanly close resources (admin endpoint)

Only a small, stable surface is exposed – internal helper functions remain
private.  The API expects JSON payloads and returns JSON responses with clear
status codes.  Errors from the core library are caught and returned as HTTP
400/500 with a ``detail`` field.

Running the server:

```
python -m uvicorn projects.memory-engine.api:app --host 0.0.0.0 --port 8000
```

The module can also be started directly via ``python projects/memory-engine/api.py``.
"""

from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import uvicorn
import logging

# Import core engine functions
from .core.memory_engine import (
    add_memory,
    add_memories_batch,
    hybrid_search,
    get_stats,
    force_save_index,
    shutdown,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Memory Engine API", version="1.0.0")

# ---------------------------------------------------------------------------
# Pydantic models for request bodies
# ---------------------------------------------------------------------------

class AddMemoryRequest(BaseModel):
    vector: List[float] = Field(..., description="Embedding vector (length must match MEMORY_EMB_DIM)")
    metadata: Dict[str, Any] = Field(..., description="Arbitrary metadata for the memory")
    text: Optional[str] = Field(None, description="Optional textual content for keyword search")

    @validator("vector")
    def vector_length(cls, v):
        # The core engine validates dimension, but we give a helpful early error.
        if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("vector must be a list of numbers")
        return v

class AddBatchRequest(BaseModel):
    vectors: List[List[float]] = Field(..., description="List of embedding vectors")
    metadatas: List[Dict[str, Any]] = Field(..., description="Corresponding metadata dicts")
    texts: Optional[List[Optional[str]]] = Field(
        None,
        description="Optional list of texts. If omitted, text is taken from each metadata entry",
    )

    @validator("vectors")
    def vectors_structure(cls, v):
        if not all(isinstance(vec, list) for vec in v):
            raise ValueError("each vector must be a list of numbers")
        return v

    @validator("metadatas")
    def metas_structure(cls, v):
        if not all(isinstance(m, dict) for m in v):
            raise ValueError("each metadata entry must be a dict")
        return v

    @validator("texts")
    def texts_length(cls, v, values):
        if v is not None:
            if len(v) != len(values.get("vectors", [])):
                raise ValueError("texts length must match vectors length")
        return v

class SearchRequest(BaseModel):
    query: str = Field(..., description="Keyword query for BM25 part")
    query_vector: List[float] = Field(..., description="Embedding vector for the query")
    top_k: int = Field(10, ge=1, le=100, description="Maximum number of results")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight for vector similarity (1-alpha for keyword)")
    min_score: float = Field(0.0, ge=0.0, description="Minimum combined score to return")

    @validator("query_vector")
    def vector_check(cls, v):
        if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("query_vector must be a list of numbers")
        return v

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/add_memory", response_model=Dict[str, int])
def api_add_memory(payload: AddMemoryRequest = Body(...)):
    """Add a single memory entry.

    Returns the memory ID. If the entry is rejected as hallucination the engine
    returns ``-1`` – we forward that value unchanged.
    """
    try:
        mem_id = add_memory(payload.vector, payload.metadata, payload.text or "")
        return {"id": mem_id}
    except Exception as e:
        logger.exception("add_memory failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/add_memories_batch", response_model=Dict[str, List[int]])
def api_add_memories_batch(payload: AddBatchRequest = Body(...)):
    """Add multiple memories in one request.

    Returns a list of IDs corresponding to the provided vectors.
    """
    try:
        ids = add_memories_batch(payload.vectors, payload.metadatas, payload.texts)
        return {"ids": ids}
    except Exception as e:
        logger.exception("add_memories_batch failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/search", response_model=List[Dict[str, Any]])
def api_search(payload: SearchRequest = Body(...)):
    """Hybrid search returning a list of memory entries.

    Each result contains ``id``, ``metadata``, ``text`` and ``score``.
    """
    try:
        results = hybrid_search(
            query=payload.query,
            query_vec=payload.query_vector,
            top_k=payload.top_k,
            alpha=payload.alpha,
            min_score=payload.min_score,
        )
        return results
    except Exception as e:
        logger.exception("hybrid_search failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/search-kg", response_model=Dict[str, Any])
def api_search_kg(payload: SearchRequest = Body(...)):
    """Hybrid search with Knowledge Graph context.
    
    Returns memory results plus KG context that can be injected into LLM prompts.
    
    Response contains:
    - memory_results: standard hybrid search results
    - kg_context: knowledge graph triples relevant to query
    - combined_context: formatted string for LLM prompt injection
    """
    try:
        from .core.kg_recall import kg_enhanced_search, get_kg_context
        
        memory_results = hybrid_search(
            query=payload.query,
            query_vec=payload.query_vector,
            top_k=payload.top_k,
            alpha=payload.alpha,
            min_score=payload.min_score,
        )
        
        enhanced = kg_enhanced_search(
            query=payload.query,
            memory_results=memory_results,
            max_kg_triples=payload.top_k
        )
        
        return enhanced
    except ImportError as e:
        logger.warning(f"KG recall not available: {e}")
        return {"error": "Knowledge graph not available", "memory_results": memory_results}
    except Exception as e:
        logger.exception("search-kg failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/kg-stats", response_model=Dict[str, Any])
def api_kg_stats():
    """Return knowledge graph statistics."""
    try:
        from .core.knowledge_graph_layer import get_stats
        return get_stats()
    except Exception as e:
        logger.exception("kg_stats failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=Dict[str, Any])
def api_stats():
    """Return basic statistics about the memory store."""
    try:
        return get_stats()
    except Exception as e:
        logger.exception("get_stats failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Admin endpoints – protected by a simple token for now (environment variable).
# ---------------------------------------------------------------------------

import os
ADMIN_TOKEN = os.getenv("MEMORY_API_ADMIN_TOKEN")  # Set via environment variable for admin access.

def _check_admin(token: Optional[str]):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")


@app.post("/admin/force_save")
def admin_force_save(token: Optional[str] = Query(None, description="Admin token")):
    _check_admin(token)
    try:
        force_save_index()
        return {"status": "index saved"}
    except Exception as e:
        logger.exception("force_save_index failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/shutdown")
def admin_shutdown(token: Optional[str] = Query(None, description="Admin token")):
    _check_admin(token)
    try:
        shutdown()
        return {"status": "shutdown complete"}
    except Exception as e:
        logger.exception("shutdown failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point for ``python -m uvicorn`` or direct execution.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    host = os.getenv("MEMORY_API_HOST", "0.0.0.0")
    port = int(os.getenv("MEMORY_API_PORT", "8000"))
    uvicorn.run("projects.memory-engine.api:app", host=host, port=port, log_level="info")
