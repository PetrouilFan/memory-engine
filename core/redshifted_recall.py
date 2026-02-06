#!/usr/bin/env python3
"""CLI/API wrapper for time-decayed semantic recall."""
from typing import List, Dict, Any, Optional

try:
    from .holographic_memory import HolographicMemory
except ImportError:
    from holographic_memory import HolographicMemory

import sys
import json

# Singleton engine instance
_mem_engine = HolographicMemory(decay_rate=0.01, max_entries=5000)

def redshifted_recall(query: str, k: int = 5, threshold: float = 0.3, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """API for recall. Supports time-weighted relevance and metadata filtering.
    Default threshold matches HolographicMemory.redshifted_recall (0.3)."""
    return _mem_engine.redshifted_recall(query, k=k, threshold=threshold, filters=filters)

def get_health():
    """Returns real-time health and metrics for monitoring dashboards."""
    return _mem_engine.get_health_status()

def add_memory(text: str, metadata: Dict = None) -> None:
    """Stores new memories safely with semantic integrity."""
    _mem_engine.add_memory(text, metadata)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 redshifted_recall.py [add/recall/health] 'text'")
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
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['decayed_score']}] {r['text'][:100]}... ({r['age_hours']}h ago)")
