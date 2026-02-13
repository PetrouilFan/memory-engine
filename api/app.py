"""FastAPI wrapper for the memory_engine module.
Provides HTTP endpoints to add memories, perform hybrid search, and retrieve stats.

Production-ready with:
- CORS middleware
- Rate limiting
- Structured logging with correlation IDs
- Global exception handlers
- Security headers
"""

import os
import uuid
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Header, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, field_validator
from pydantic.types import conint
import uvicorn

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False

from ..core.memory_engine import (
    add_memory,
    add_memories_batch,
    hybrid_search,
    get_stats,
    prune_memories,
    consolidate_memories,
    update_relevance,
    force_save_index,
)

MEMORY_EMB_DIM = int(os.getenv("MEMORY_EMB_DIM", "384"))
MEMORY_API_ADMIN_TOKEN = os.getenv("MEMORY_API_ADMIN_TOKEN", "")
MAX_REQUEST_SIZE = 10 * 1024 * 1024
MAX_BATCH_SIZE = 1000
MAX_TOP_K = 1000

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

if SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None


class Settings:
    """Application settings using pydantic-settings pattern."""
    MEMORY_EMB_DIM: int = 384
    MEMORY_API_ADMIN_TOKEN: str = ""
    MAX_REQUEST_SIZE: int = 10 * 1024 * 1024
    MAX_BATCH_SIZE: int = 1000
    CORS_ORIGINS: List[str] = ["*"]
    DEBUG: bool = False
    RATE_LIMIT_PER_MINUTE: int = 60


settings = Settings()


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
    timestamp: str


def generate_request_id() -> str:
    return str(uuid.uuid4())


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add correlation ID to each request for tracing."""
    request_id = request.headers.get("X-Request-ID", generate_request_id())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    logger.info(
        f"method={request.method} path={request.url.path} "
        f"status={response.status_code} duration={process_time:.3f}s "
        f"request_id={request_id}"
    )
    
    return response


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.middleware("http")
async def check_request_size(request: Request, call_next):
    """Limit request body size."""
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_SIZE:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request too large",
                    "detail": f"Maximum size is {MAX_REQUEST_SIZE} bytes"
                }
            )
    response = await call_next(request)
    return response


if SLOWAPI_AVAILABLE:
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "detail": str(exc),
                "retry_after": getattr(exc, "retry_after", 60)
            }
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_vector_dimension(vector: List[float]) -> None:
    if len(vector) != MEMORY_EMB_DIM:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Vector dimension must be {MEMORY_EMB_DIM}, got {len(vector)}"
        )


async def verify_admin_token(x_admin_token: Optional[str] = Header(None)) -> str:
    if not MEMORY_API_ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin token not configured on server"
        )
    if x_admin_token != MEMORY_API_ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token"
        )
    return x_admin_token


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper error response."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions - log full error, return sanitized response."""
    request_id = getattr(request.state, "request_id", None)
    
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {exc}",
        exc_info=True,
        extra={"request_id": request_id}
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred" if not DEBUG_MODE else str(exc),
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Memory Engine API started with EMB_DIM={MEMORY_EMB_DIM}")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    logger.info(f"CORS origins: {CORS_ORIGINS}")
    yield
    logger.info("Memory Engine API shutting down")


app = FastAPI(
    title="Memory Engine API",
    version="1.0",
    lifespan=lifespan,
    docs_url="/docs" if DEBUG_MODE else None,
    redoc_url="/redoc" if DEBUG_MODE else None,
    openapi_url="/openapi.json" if DEBUG_MODE else None,
)


class AddMemoryRequest(BaseModel):
    vector: List[float] = Field(..., description=f"Embedding vector (length must match {MEMORY_EMB_DIM})")
    metadata: Dict[str, Any] = Field(..., description="Arbitrary metadata for the memory")
    text: Optional[str] = Field(None, description="Optional raw text for keyword search")
    weight: Optional[float] = Field(1.0, ge=0.0, le=10.0, description="Weight of the memory entry")

    @field_validator("vector")
    @classmethod
    def validate_vector_dim(cls, v: List[float]) -> List[float]:
        if len(v) != MEMORY_EMB_DIM:
            raise ValueError(f"Vector dimension must be {MEMORY_EMB_DIM}, got {len(v)}")
        return v


class AddMemoryResponse(BaseModel):
    memory_id: int


class ValidationErrorResponse(BaseModel):
    detail: str


if SLOWAPI_AVAILABLE:
    @app.post("/add_memory")
    @limiter.limit("60/minute")
    async def api_add_memory(request: Request, req: AddMemoryRequest):
        try:
            mem_id = add_memory(req.vector, req.metadata, text=req.text or "", weight=req.weight or 1.0)
            if mem_id == -1:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Memory rejected as hallucination")
            return {"memory_id": mem_id}
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Runtime error: {str(e)}")
else:
    @app.post("/add_memory", response_model=AddMemoryResponse)
    def api_add_memory(req: AddMemoryRequest):
        try:
            mem_id = add_memory(req.vector, req.metadata, text=req.text or "", weight=req.weight or 1.0)
            if mem_id == -1:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Memory rejected as hallucination")
            return {"memory_id": mem_id}
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Runtime error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in add_memory: {type(e).__name__}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


class AddBatchRequest(BaseModel):
    vectors: List[List[float]] = Field(..., max_length=MAX_BATCH_SIZE)
    metadatas: List[Dict[str, Any]]
    texts: Optional[List[str]] = None

    @field_validator("vectors")
    @classmethod
    def validate_vectors_dim(cls, v: List[List[float]]) -> List[List[float]]:
        if not v:
            return v
        dim = len(v[0])
        if dim != MEMORY_EMB_DIM:
            raise ValueError(f"Vector dimensions must be {MEMORY_EMB_DIM}, got {dim}")
        for vec in v:
            if len(vec) != MEMORY_EMB_DIM:
                raise ValueError(f"Vector dimension must be {MEMORY_EMB_DIM}, got {len(vec)}")
        return v

    @field_validator("metadatas")
    @classmethod
    def validate_metadatas_length(cls, v: List[Dict[str, Any]], info) -> List[Dict[str, Any]]:
        vectors = info.data.get("vectors", [])
        if vectors and len(v) != len(vectors):
            raise ValueError("metadatas length must match vectors length")
        return v


class AddBatchResponse(BaseModel):
    memory_ids: List[int]


@app.post("/add_memories_batch", response_model=AddBatchResponse)
def api_add_memories_batch(req: AddBatchRequest):
    try:
        ids = add_memories_batch(req.vectors, req.metadatas, texts=req.texts)
        return {"memory_ids": ids}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Runtime error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in add_memories_batch: {type(e).__name__}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    query_vec: List[float]
    top_k: int = Field(10, ge=1, le=MAX_TOP_K)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    min_score: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("query_vec")
    @classmethod
    def validate_query_vec_dim(cls, v: List[float]) -> List[float]:
        if len(v) != MEMORY_EMB_DIM:
            raise ValueError(f"Query vector dimension must be {MEMORY_EMB_DIM}, got {len(v)}")
        return v


class SearchResult(BaseModel):
    id: int
    metadata: Dict[str, Any]
    text: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


@app.post("/search", response_model=SearchResponse)
def api_search(req: SearchRequest):
    try:
        results = hybrid_search(
            query=req.query,
            query_vec=req.query_vec,
            top_k=req.top_k,
            alpha=req.alpha,
            min_score=req.min_score,
        )
        return {"results": results}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Runtime error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in search: {type(e).__name__}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@app.get("/stats")
def api_stats():
    try:
        return get_stats()
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Runtime error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in stats: {type(e).__name__}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


class PruneRequest(BaseModel):
    max_age_days: int = Field(30, ge=1, le=365)
    relevance_threshold: float = Field(0.3, ge=0.0, le=1.0)
    decay_lambda: float = Field(0.1, ge=0.0, le=1.0)


class PruneResponse(BaseModel):
    pruned: int


@app.post("/prune", response_model=PruneResponse)
def api_prune(req: PruneRequest):
    try:
        count = prune_memories(
            max_age_days=req.max_age_days,
            relevance_threshold=req.relevance_threshold,
            decay_lambda=req.decay_lambda,
        )
        return {"pruned": count}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Runtime error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in prune: {type(e).__name__}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


class ConsolidateRequest(BaseModel):
    min_cluster_size: int = Field(5, ge=2, le=100)
    embedding_fn: Optional[str] = None


class ConsolidateResponse(BaseModel):
    created_ids: List[int]


@app.post("/consolidate", response_model=ConsolidateResponse)
def api_consolidate(req: ConsolidateRequest):
    try:
        ids = consolidate_memories(min_cluster_size=req.min_cluster_size)
        return {"created_ids": ids}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Runtime error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in consolidate: {type(e).__name__}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


class UpdateRelevanceRequest(BaseModel):
    memory_id: int = Field(..., gt=0)
    new_relevance: float = Field(..., ge=0.0, le=1.0)


class UpdateRelevanceResponse(BaseModel):
    updated: bool


@app.post("/update_relevance", response_model=UpdateRelevanceResponse)
def api_update_relevance(req: UpdateRelevanceRequest):
    try:
        ok = update_relevance(req.memory_id, req.new_relevance)
        if not ok:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
        return {"updated": ok}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Runtime error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_relevance: {type(e).__name__}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@app.get("/list_memories")
def api_list_memories(offset: conint(ge=0) = 0, limit: conint(ge=1, le=1000) = 100):
    from ..core.vector_memory_wrapper import list_memories_by_age
    try:
        all_memories = list_memories_by_age(limit=10000)
        total = len(all_memories)
        memories = all_memories[offset:offset + limit]
        return {
            "memories": memories,
            "total": total,
            "offset": offset,
            "limit": limit
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Runtime error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in list_memories: {type(e).__name__}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@app.post("/admin/force_save")
async def api_force_save(x_admin_token: Optional[str] = Header(None)):
    try:
        verify_admin_token(x_admin_token)
    except HTTPException:
        raise
    try:
        force_save_index()
        return {"status": "index saved"}
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Runtime error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in force_save: {type(e).__name__}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@app.post("/admin/shutdown")
async def api_shutdown(x_admin_token: Optional[str] = Header(None)):
    try:
        verify_admin_token(x_admin_token)
    except HTTPException:
        raise
    import asyncio
    asyncio.create_task(shutdown_server())
    return {"status": "shutting down"}


async def shutdown_server():
    await asyncio.sleep(1)
    import sys
    sys.exit(0)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
