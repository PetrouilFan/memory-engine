#!/usr/bin/env python3
"""Redis caching utilities for Memory Engine."""
import os
import json
import logging
from typing import Optional, Any
import hashlib

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))


class CacheManager:
    """Redis-based cache manager for memory queries."""
    
    def __init__(self, redis_url: str = REDIS_URL, ttl: int = CACHE_TTL):
        self.redis_url = redis_url
        self.ttl = ttl
        self._client = None
        self._enabled = False
        
    @property
    def client(self):
        """Lazy-load Redis client."""
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self.redis_url)
                self._client.ping()
                self._enabled = True
                logger.info(f"Redis cache enabled: {self.redis_url}")
            except Exception as e:
                logger.warning(f"Redis unavailable, caching disabled: {e}")
                self._enabled = False
        return self._client
    
    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return f"memory_engine:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._enabled:
            return None
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self._enabled:
            return False
        try:
            self.client.setex(key, ttl or self.ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._enabled:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with given prefix."""
        if not self._enabled:
            return 0
        try:
            pattern = f"memory_engine:{prefix}:*"
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                return self.client.delete(*keys)
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
        return 0
    
    def get_cached_search(self, query: str, top_k: int = 5) -> Optional[list]:
        """Get cached search results."""
        key = self._make_key("search", query, top_k)
        return self.get(key)
    
    def set_cached_search(self, query: str, top_k: int, results: list) -> bool:
        """Cache search results."""
        key = self._make_key("search", query, top_k)
        return self.set(key, results)
    
    def invalidate_memories(self):
        """Invalidate all memory-related cache."""
        return self.clear_prefix("search")


_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get global cache manager instance."""
    global _cache
    if _cache is None:
        _cache = CacheManager()
    return _cache
