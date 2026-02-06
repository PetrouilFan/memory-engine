# Hierarchical Summaries Pipeline
# topic -> domain -> life summary generation using clustering and LLM summarization.

import json
import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

try:
    from core.holographic_memory import HolographicMemory
    _hm = HolographicMemory()
    def add_memory(text, metadata=None):
        _hm.add_memory(text, metadata)
    def redshifted_recall(query):
        return _hm.redshifted_recall(query)
except ImportError:
    def add_memory(text, metadata=None): pass
    def redshifted_recall(query): return []


def cluster_texts(texts: List[str], n_clusters: int = 5) -> Dict[int, List[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(X)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(km.labels_):
        clusters[label].append(texts[idx])
    return clusters


def llm_summarize(texts: List[str], prompt: str) -> str:
    """Placeholder LLM summarizer – replace with actual API call."""
    combined = "\n\n".join(texts)
    return f"Summary based on prompt '{prompt}': {combined[:200]}..."


def generate_topic_summaries(documents: List[Dict[str, Any]]) -> Dict[str, str]:
    topics = {}
    for doc in documents:
        topics.setdefault(doc['topic'], []).append(doc['content'])
    topic_summaries = {}
    for topic, texts in topics.items():
        summary = llm_summarize(texts, f"Summarize the topic: {topic}")
        topic_summaries[topic] = summary
    return topic_summaries


def generate_domain_summary(topic_summaries: Dict[str, str]) -> str:
    summaries = list(topic_summaries.values())
    clusters = cluster_texts(summaries, n_clusters=3)
    domain_parts = []
    for cid, texts in clusters.items():
        domain_summary = llm_summarize(texts, f"Summarize domain cluster {cid}")
        domain_parts.append(domain_summary)
    return "\n\n".join(domain_parts)


def generate_life_summary(domain_summary: str) -> str:
    return llm_summarize([domain_summary], "Create an overall life summary")


def run_pipeline(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    topic_summaries = generate_topic_summaries(documents)
    domain_summary = generate_domain_summary(topic_summaries)
    life_summary = generate_life_summary(domain_summary)
    result = {
        "topic_summaries": topic_summaries,
        "domain_summary": domain_summary,
        "life_summary": life_summary,
    }
    add_memory(json.dumps(result), metadata={"type": "hierarchical_summary"})
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hierarchical_summaries.py <documents.jsonl>")
        sys.exit(1)
    docs = []
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    out = run_pipeline(docs)
    print(json.dumps(out, indent=2, ensure_ascii=False))
