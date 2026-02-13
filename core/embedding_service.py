"""
embedding_service.py - CPU-optimized embedding generation

Optimized for CPU with limited RAM (6GB):
- ONNX runtime for faster inference
- Configurable batch sizes
- Memory-efficient processing
- Caching support
"""
import os
import logging
from typing import List, Optional
import hashlib

logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))  # CPU-optimized
USE_ONNX = os.getenv("EMBEDDING_USE_ONNX", "false").lower() == "true"
CACHE_EMBEDDINGS = os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"

# Embedding cache
_embedding_cache: dict = {}
MAX_CACHE_SIZE = 1000


def _get_cache_key(text: str) -> str:
    """Generate cache key for text."""
    return hashlib.md5(text.strip().encode()).hexdigest()


def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text string.
    
    Uses ONNX if enabled for faster CPU inference.
    """
    text = text.strip()
    if not text:
        return [0.0] * 384
    
    # Check cache
    if CACHE_EMBEDDINGS:
        key = _get_cache_key(text)
        if key in _embedding_cache:
            return _embedding_cache[key]
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use ONNX for faster CPU inference if available
        if USE_ONNX:
            try:
                model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    backend="onnx",
                    device_kwargs={"onnx": {"execution_providers": ["CPUExecutionProvider"]}}
                )
            except Exception as e:
                logger.warning(f"ONNX not available, falling back: {e}")
                model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        else:
            model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        
        # Encode with optimized batch size
        embedding = model.encode(
            [text],
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
        
        # Cache result
        if CACHE_EMBEDDINGS:
            if len(_embedding_cache) >= MAX_CACHE_SIZE:
                # Simple FIFO eviction
                first_key = next(iter(_embedding_cache))
                del _embedding_cache[first_key]
            _embedding_cache[key] = embedding
        
        return embedding
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return [0.0] * 384


def get_embeddings_batch(texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
    """Get embeddings for multiple texts efficiently.
    
    Batches multiple texts together for faster processing.
    """
    if not texts:
        return []
    
    # Filter empty and check cache
    valid_texts = []
    valid_indices = []
    results = [None] * len(texts)
    
    batch_size = batch_size or BATCH_SIZE
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Determine model based on ONNX setting
        if USE_ONNX:
            try:
                model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    backend="onnx",
                    device_kwargs={"onnx": {"execution_providers": ["CPUExecutionProvider"]}}
                )
            except Exception:
                model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        else:
            model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        
        # Check cache for each text
        texts_to_encode = []
        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                results[i] = [0.0] * 384
                continue
                
            if CACHE_EMBEDDINGS:
                key = _get_cache_key(text)
                if key in _embedding_cache:
                    results[i] = _embedding_cache[key]
                    continue
            
            texts_to_encode.append((i, text))
        
        if texts_to_encode:
            # Encode in batches for memory efficiency
            texts_only = [t[1] for t in texts_to_encode]
            
            # Process in smaller batches to limit RAM usage
            embeddings = []
            for start in range(0, len(texts_only), batch_size):
                batch = texts_only[start:start + batch_size]
                batch_embeddings = model.encode(
                    batch,
                    batch_size=len(batch),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                embeddings.extend(batch_embeddings.tolist())
            
            # Store results and cache
            for idx, (i, text) in enumerate(texts_to_encode):
                emb = embeddings[idx]
                results[i] = emb
                
                if CACHE_EMBEDDINGS:
                    key = _get_cache_key(text)
                    if len(_embedding_cache) >= MAX_CACHE_SIZE:
                        first_key = next(iter(_embedding_cache))
                        del _embedding_cache[first_key]
                    _embedding_cache[key] = emb
        
        # Fill any remaining None values
        for i, r in enumerate(results):
            if r is None:
                results[i] = [0.0] * 384
        
        return results
        
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}")
        return [[0.0] * 384 for _ in texts]


def clear_cache():
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache = {}
    logger.info("Embedding cache cleared")


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "size": len(_embedding_cache),
        "max_size": MAX_CACHE_SIZE,
        "enabled": CACHE_EMBEDDINGS,
    }
