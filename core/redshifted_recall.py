#!/usr/bin/env python3
"""CLI/API wrapper for time-decayed semantic recall."""
from typing import List, Dict, Any, Optional

try:
    from .holographic_memory import HolographicMemory
except ImportError:
    from holographic_memory import HolographicMemory

import sys
import json
from datetime import datetime

# Singleton engine instance
_mem_engine = HolographicMemory(decay_rate=0.01, max_entries=5000)

import json
import hashlib

# Attempt to connect to a local Redis server for caching recent query results.
try:
    import redis
    _redis_client = redis.StrictRedis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
    # Test connection
    _redis_client.ping()
except Exception:
    _redis_client = None  # Redis not available; fallback to no caching.

def _cache_key(query: str, k: int, threshold: float, filters: Optional[Dict]) -> str:
    """Generate a deterministic cache key for a query.
    Uses a hash to keep key length reasonable.
    """
    base = f"{query}|{k}|{threshold}|{json.dumps(filters, sort_keys=True) if filters else ''}"
    h = hashlib.sha256(base.encode()).hexdigest()
    return f"recall:{h}"

def _get_cached(query: str, k: int, threshold: float, filters: Optional[Dict]) -> List[Dict[str, Any]]:
    if not _redis_client:
        return None
    key = _cache_key(query, k, threshold, filters)
    cached = _redis_client.get(key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            return None
    return None

def _set_cached(query: str, k: int, threshold: float, filters: Optional[Dict], result: List[Dict[str, Any]], ttl: int = 300) -> None:
    if not _redis_client:
        return
    key = _cache_key(query, k, threshold, filters)
    try:
        _redis_client.setex(key, ttl, json.dumps(result))
    except Exception:
        pass

def redshifted_recall(query: str, k: int = 5, threshold: float = 0.3, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """API for recall. Supports time-weighted relevance and metadata filtering.
    Default threshold matches HolographicMemory.redshifted_recall (0.3).
    Implements Redis caching for recent queries.
    """
    # Check cache first
    cached = _get_cached(query, k, threshold, filters)
    if cached is not None:
        return cached
    # Compute result
    result = _mem_engine.redshifted_recall(query, k=k, threshold=threshold, filters=filters)
    # Store in cache for future calls (default TTL 5 minutes)
    _set_cached(query, k, threshold, filters, result)
    return result

def get_health():
    """Returns real-time health and metrics for monitoring dashboards."""
    return _mem_engine.get_health_status()

def purge_hallucinations(threshold: float = 0.4, dry_run: bool = False):
    """Scan all stored memories and delete those flagged as hallucinations."""
    return _mem_engine.purge_hallucinations(threshold=threshold, dry_run=dry_run)

def add_memory(text: str, metadata: Dict = None) -> None:
    """Stores new memories safely with semantic integrity."""
    _mem_engine.add_memory(text, metadata)

def purge_chat_metadata(dry_run: bool = False):
    """Scan all memories and delete those that are purely chat platform metadata."""
    return _mem_engine.purge_chat_metadata(dry_run=dry_run)

def get_access_stats(memory_id: Optional[str] = None) -> Dict[str, Any]:
    """Return per-memory or global access statistics."""
    return _mem_engine.get_access_stats(memory_id=memory_id)

def get_most_accessed(top_n: int = 10) -> List[Dict[str, Any]]:
    """Return the top-N most frequently recalled memories."""
    return _mem_engine.get_most_accessed(top_n=top_n)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 redshifted_recall.py [add/recall/health/purge/purge-dry/purge-meta/purge-meta-dry] [text/threshold]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "health":
        print(json.dumps(get_health(), indent=2))
    elif mode == "add":
        add_memory(" ".join(sys.argv[2:]))
        print("✓ Memory stored.")
    elif mode == "recall":
        query_str = " ".join(sys.argv[2:])
        results = redshifted_recall(query_str)
        print(f"\n🔍 Results for: '{query_str}'")
        if not results:
            print("  (no results found)")
        for i, r in enumerate(results, 1):
            imp = r.get('importance_score')
            imp_tag = f" imp={imp}" if imp is not None else ""
            text = r.get('text', r.get('metadata', {}).get('text', ''))
            score = r.get('decayed_score', r.get('score', 0))
            age = r.get('age_hours', '?')
            print(f"{i}. [{score}]{imp_tag} {text[:100]}... ({age}h ago)")
    elif mode in ("purge", "purge-dry"):
        dry = mode == "purge-dry"
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].replace('.','',1).isdigit() else 0.4
        print(f"\n🧹 {'[DRY RUN] ' if dry else ''}Purging hallucinated memories (threshold={threshold})...")
        result = purge_hallucinations(threshold=threshold, dry_run=dry)
        print(json.dumps(result, indent=2))
    elif mode in ("purge-meta", "purge-meta-dry"):
        dry = mode == "purge-meta-dry"
        print(f"\n🧹 {'[DRY RUN] ' if dry else ''}Purging chat platform metadata from memories...")
        result = purge_chat_metadata(dry_run=dry)
        print(json.dumps(result, indent=2))
    elif mode == "access-stats":
        mid = sys.argv[2] if len(sys.argv) > 2 else None
        stats = get_access_stats(memory_id=mid)
        print(json.dumps(stats, indent=2, default=str))
    elif mode == "most-accessed":
        top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        results = get_most_accessed(top_n=top_n)
        if not results:
            print("No access data recorded yet.")
        else:
            print(f"\n📊 Top {len(results)} most accessed memories:")
            for i, r in enumerate(results, 1):
                last = r.get('last_accessed', 0)
                age_h = round((datetime.now().timestamp() - last) / 3600, 1) if last else '?'
                print(f"  {i}. id={r['memory_id']}  hits={r['count']}  last={age_h}h ago")
                if r.get('recent_queries'):
                    print(f"     recent queries: {r['recent_queries'][-3:]}")
    else:
        print(f"Unknown mode: {mode}")
        print("Available: add, recall, health, purge, purge-dry, purge-meta, purge-meta-dry, access-stats, most-accessed")
