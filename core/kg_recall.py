#!/usr/bin/env python3
"""
kg_recall.py - Knowledge Graph Enhanced Recall

Combines semantic memory search with knowledge graph context.
 KG results are returned as a separate "context" section rather than 
expanding the query text.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("kg_recall")

try:
    from .knowledge_graph_layer import (
        query_by_entity,
        query_related,
        query_by_type,
        query_by_predicate,
        search_by_keyword,
        get_stats
    )
except ImportError:
    from knowledge_graph_layer import (
        query_by_entity,
        query_related,
        query_by_type,
        query_by_predicate,
        search_by_keyword,
        get_stats
    )

ENTITY_PATTERNS = [
    r'skills?/([a-zA-Z0-9_-]+)',
    r'([a-zA-Z0-9_-]+)_skill',
    r'tools?/([a-zA-Z0-9_]+\.py)',
    r'scripts?/([a-zA-Z0-9_]+\.sh)',
    r'(postgres|redis|nextcloud|ollama|gateway)',
    r'(OPENAI_API_KEY|GROQ_API_KEY|GEMINI_API_KEY|PERPLEXITY_API_KEY)',
]

SKILL_KEYWORDS = [
    'skill', 'browser', 'tts', 'stt', 'healthcheck', 'opencode', 
    'perplexity', 'command', 'tool', 'script'
]

def extract_entities(query: str) -> List[str]:
    """Extract potential entities from a query."""
    entities = []
    
    for pattern in ENTITY_PATTERNS:
        matches = re.findall(pattern, query, re.IGNORECASE)
        entities.extend(matches)
    
    for keyword in SKILL_KEYWORDS:
        if keyword.lower() in query.lower():
            entities.append(keyword)
    
    entities = list(set(e.lower() for e in entities if e))
    return entities

def get_kg_context(query: str, max_triples: int = 10) -> Dict[str, Any]:
    """
    Get knowledge graph context for a query.
    
    Returns a dict with:
    - kg_found: bool - whether KG had relevant results
    - entities: list - entities extracted from query
    - triples: list - relevant KG triples
    - summary: str - human-readable summary
    """
    entities = extract_entities(query)
    
    if not entities:
        return {
            'kg_found': False,
            'entities': [],
            'triples': [],
            'summary': 'No entities detected in query'
        }
    
    all_triples = []
    seen = set()
    
    for entity in entities[:5]:
        try:
            results = query_by_entity(entity, limit=max_triples)
            for t in results:
                key = (t['subject'], t['predicate'], t['object'])
                if key not in seen:
                    seen.add(key)
                    all_triples.append(t)
        except Exception as e:
            logger.warning(f"KG query failed for entity {entity}: {e}")
    
    all_triples = all_triples[:max_triples]
    
    if not all_triples:
        return {
            'kg_found': False,
            'entities': entities,
            'triples': [],
            'summary': f'No KG entries found for: {", ".join(entities)}'
        }
    
    subject_types = {}
    predicates = {}
    for t in all_triples:
        st = t.get('subject_type', 'other')
        subject_types[st] = subject_types.get(st, 0) + 1
        predicates[t['predicate']] = predicates.get(t['predicate'], 0) + 1
    
    summary_parts = []
    if subject_types:
        summary_parts.append(f"Types: {', '.join(f'{k}({v})' for k,v in subject_types.items())}")
    if predicates:
        summary_parts.append(f"Relations: {', '.join(predicates.keys())}")
    
    return {
        'kg_found': True,
        'entities': entities,
        'triples': all_triples,
        'summary': '; '.join(summary_parts) if summary_parts else f'Found {len(all_triples)} related entries'
    }

def format_kg_context_for_prompt(kg_context: Dict) -> str:
    """
    Format KG context for injection into LLM prompt.
    """
    if not kg_context.get('kg_found'):
        return ""
    
    triples = kg_context.get('triples', [])
    if not triples:
        return ""
    
    lines = ["\n### Relevant Knowledge Graph Context:\n"]
    
    for t in triples[:8]:
        lines.append(f"- {t['subject']} → {t['predicate']} → {t['object']}")
    
    return "\n".join(lines)

def kg_enhanced_search(query: str, memory_results: List[Dict], 
                       max_kg_triples: int = 10) -> Dict[str, Any]:
    """
    Combine semantic memory search with knowledge graph context.
    
    Args:
        query: Original user query
        memory_results: Results from hybrid_search or similar
        max_kg_triples: Max KG triples to retrieve
        
    Returns:
        Dict with:
        - memory_results: original memory results
        - kg_context: KG context dict
        - combined_context: formatted string for LLM prompt
    """
    kg_context = get_kg_context(query, max_triples=max_kg_triples)
    combined_context = format_kg_context_for_prompt(kg_context)
    
    return {
        'query': query,
        'memory_results': memory_results,
        'kg_context': kg_context,
        'combined_context': combined_context,
        'has_kg_context': kg_context.get('kg_found', False)
    }

def suggest_related_queries(query: str, max_suggestions: int = 3) -> List[str]:
    """Suggest related queries based on KG entities."""
    entities = extract_entities(query)
    suggestions = []
    
    for entity in entities[:3]:
        try:
            related = query_related(entity, depth=1, limit=5)
            for t in related.get('direct', [])[:2]:
                if t['subject'] != entity:
                    suggestions.append(f"What is {t['subject']}?")
                if t['object'] != entity:
                    suggestions.append(f"How does {t['object']} work?")
        except:
            pass
    
    return suggestions[:max_suggestions]

if __name__ == "__main__":
    import sys
    
    query = sys.argv[1] if len(sys.argv) > 1 else "browser skill"
    
    print(f"Query: {query}")
    print(f"Entities: {extract_entities(query)}")
    
    context = get_kg_context(query)
    print(f"\nKG Context:")
    print(f"  Found: {context['kg_found']}")
    print(f"  Summary: {context['summary']}")
    print(f"  Triples: {len(context['triples'])}")
    
    for t in context['triples'][:5]:
        print(f"    {t['subject']} → {t['predicate']} → {t['object']}")
