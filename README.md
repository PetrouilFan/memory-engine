# Memory Engine

A modular vector-memory system with time-decayed recall, hybrid search, semantic graphs, and knowledge-graph support.

## Project Structure

```
memory-engine/
├── core/                          # Core modules (import from here)
│   ├── memory_engine.py           # Primary store: FAISS + SQLite + FTS5 hybrid search
│   ├── vector_memory_wrapper.py   # Lightweight FAISS + SQLite vector store
│   ├── holographic_memory.py      # Time-decayed semantic memory with dedup (cross-platform)
│   ├── redshifted_recall.py       # CLI/API wrapper for holographic recall
│   ├── semantic_graph.py          # Concept graph layer on top of memory_engine
│   ├── clustering.py              # DBSCAN/HDBSCAN clustering on vectors
│   ├── knowledge_graph_layer.py   # Triple-store (subject/predicate/object)
│   └── postgres_support.py        # Optional PostgreSQL backend via SQLAlchemy
│
├── pipelines/                     # Data ingestion & export scripts
│   ├── crawl_memory_files.py      # Index .md files via HolographicMemory
│   ├── crawl_and_index_memories.py  # Batch index .md files via memory_engine
│   ├── import_memories.py         # Chunk-based importer for memory/*.md
│   ├── export_memory.py           # Export holographic memories to JSON
│   ├── knowledge_graph_exporter.py  # Extract triples from .md files → CSV
│   └── hierarchical_summaries.py  # Topic → domain → life summary pipeline
│
├── tools/                         # Utilities
│   └── memory_slice.py            # File-slicing utility with symbol lookup
│
├── evaluation/                    # Benchmarking & telemetry
│   ├── run_evaluation.py          # Precision@k eval against gold queries
│   ├── telemetry_aggregation.py   # Stats logger
│   ├── gold_queries.json          # Benchmark query set
│   └── README.md
│
├── tests/                         # Tests
│   └── test_memory.py             # Concurrency stress test
│
├── data/                          # Generated data files (gitignored)
├── requirements.txt
└── README.md
```

## Architecture

There are two parallel storage stacks:

1. **`memory_engine.py`** (primary) — Full-featured: FAISS + SQLite with FTS5, WAL, connection pooling, hybrid vector+BM25 search, HDBSCAN clustering, consolidation, pruning, archival, and a time-aware FAISS sub-index.

2. **`vector_memory_wrapper.py` → `holographic_memory.py`** — Lighter: HNSW FAISS index with cosine similarity, time-decayed recall ("redshifted recall"), SHA-256 deduplication, semantic chunking.

`semantic_graph.py` bridges the two by monkey-patching `memory_engine.add_memory` to auto-index new memories into a concept graph.

## Quick Start

```bash
pip install -r requirements.txt

# Import markdown files
python pipelines/import_memories.py

# Query with time-decay
python core/redshifted_recall.py recall "what happened yesterday"

# Run evaluation
python evaluation/run_evaluation.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MEMORY_DB_PATH` | `/root/.openclaw/workspace/memory.db` | SQLite database path |
| `MEMORY_INDEX_PATH` | `/root/.openclaw/workspace/memory.index` | FAISS index path |
| `MEMORY_EMB_DIM` | `768` | Embedding dimension |
| `INDEX_SAVE_INTERVAL` | `100` | Auto-save after N operations |
| `DB_POOL_SIZE` | `5` | SQLite connection pool size |
| `POSTGRES_URL` | *(unset)* | PostgreSQL connection string (optional) |

## Cross-Platform

`holographic_memory.py` uses `msvcrt` on Windows and `fcntl` on Linux for file locking. Lock files are placed in the system temp directory.
