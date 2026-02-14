# 🧠 Memory Engine

**A persistent, intelligent memory system for AI agents.** Built for [OpenClaw](https://github.com/openclaw), Memory Engine gives your AI agent long-term memory with time-decayed recall, hallucination filtering, and semantic search — so it remembers what matters, forgets what's noise, and never makes up facts from broken memories.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🤔 Why Memory Engine?

AI agents are stateless by default. Every conversation starts from zero. This is a problem when your agent needs to:

- **Remember past conversations** across sessions, days, or weeks
- **Recall relevant context** automatically when the user asks a question
- **Avoid repeating mistakes** — if something failed before, don't suggest it again
- **Build knowledge over time** — accumulate facts, preferences, and decisions

Memory Engine solves all of this. It stores memories as vectors (embeddings), applies time-decay so recent memories rank higher, deduplicates content, and — critically — **filters out hallucinations** so your agent's memory stays clean and trustworthy.

### Key Differentiators

| Feature | Memory Engine | Naive Vector Store |
|---------|:---:|:---:|
| Hybrid search (vector + keyword) | ✅ | ❌ |
| Time-decayed recall | ✅ | ❌ |
| Hallucination filtering | ✅ | ❌ |
| Deduplication (SHA-256) | ✅ | ❌ |
| Semantic graph linking | ✅ | ❌ |
| Knowledge graph (triples) | ✅ | ❌ |
| Memory consolidation & pruning | ✅ | ❌ |
| OpenClaw integration | ✅ | ❌ |

---

## ✨ Features

- **🛡️ Hallucination Filtering** — Automatically detects and rejects unreliable memories (error messages, speculative guesses, terminal artifacts) before they pollute your agent's knowledge
- **⏳ Time-Decayed Recall ("Redshifted Recall")** — Recent memories rank higher; old memories naturally fade unless frequently accessed
- **🔍 Hybrid Search** — Combined FAISS vector similarity + BM25 keyword matching for the best of both worlds
- **🕸️ Semantic Graphs** — Automatically cross-links related memories into a navigable concept graph
- **📐 Knowledge Graphs** — Triple-store (subject → predicate → object) for structured fact storage
- **⚡ FAISS Indexing** — Fast approximate nearest-neighbor search with HNSW indices
- **🧹 Memory Hygiene** — Deduplication, consolidation, pruning, and archival pipelines
- **🌉 Memory Bridge** — Drop-in proxy that injects relevant memories into any OpenAI-compatible API call
- **📊 Evaluation Harness** — Precision@k benchmarking against gold-standard query sets

---

## 🕸️ Knowledge Graph

The Knowledge Graph stores structured triples (subject → predicate → object) for reasoning about relationships between skills, tools, services, and entities.

### Features

- **Entity Types**: Automatically detects types (skill, tool, service, file, api, cron)
- **Rich Relationships**: Uses, depends_on, has_command, created, fixed, installed, etc.
- **Multi-Source Extraction**: Extracts from MEMORY.md, daily logs, TOOLS.md, skills, TASKS.md
- **Memory Integration**: KG context is injected into LLM prompts via memory_bridge
- **Visualization**: Interactive graph view at `/knowledge-graph` (dashboard)

### Usage

```bash
# Query knowledge graph
python3 core/kg_recall.py "browser skill"

# Update knowledge graph
python3 update_kg.py
```

### API

```bash
# Search with KG context
curl -X POST http://localhost:8000/search-kg \
  -H "Content-Type: application/json" \
  -d '{"query": "how does browser work", "top_k": 5}'
```

### Data Sources

The exporter extracts triples from:
- `MEMORY.md` - Structured bullet points
- `memory/*.md` - Daily session logs  
- `TOOLS.md` - Tool definitions
- `skills/*/SKILL.md` - Skill commands
- `TASKS.md` - Task definitions

---

## 📁 Project Structure

```
memory-engine/
├── core/                              # Core modules
│   ├── memory_engine.py               # Primary store: FAISS + SQLite + FTS5 hybrid search
│   ├── holographic_memory.py          # Time-decayed semantic memory with dedup
│   ├── vector_memory_wrapper.py       # Thin FAISS + SQLite vector wrapper
│   ├── redshifted_recall.py           # CLI/API for time-decayed recall
│   ├── hallucination_filter.py        # Hallucination detection & filtering
│   ├── semantic_graph.py              # Concept graph layer
│   ├── clustering.py                  # DBSCAN/HDBSCAN clustering
│   ├── knowledge_graph_layer.py        # Triple-store (subject/predicate/object)
│   ├── kg_recall.py                   # Knowledge graph enhanced context
│   ├── mem_fast_layer.py              # Fast-path memory layer
│   └── postgres_support.py            # Optional PostgreSQL backend
│
├── pipelines/                         # Data ingestion & maintenance
│   ├── ingest_workspace.py            # Scan & index all workspace markdown files
│   ├── ingest_sessions.py             # Ingest conversation session logs
│   ├── crawl_memory_files.py          # Index .md files via HolographicMemory
│   ├── crawl_and_index_memories.py    # Batch index .md files via memory_engine
│   ├── import_memories.py             # Chunk-based importer for memory/*.md
│   ├── export_memory.py               # Export memories to JSON
│   ├── cleanup_hallucinations.py      # Scan & remove hallucinated memories
│   ├── knowledge_graph_exporter.py    # Extract triples from .md → CSV
│   └── hierarchical_summaries.py      # Topic → domain → life summary pipeline
│
├── tools/                             # Utilities
│   └── memory_slice.py                # File-slicing utility with symbol lookup
│
├── update_kg.py                       # Periodic KG sync script
│
├── evaluation/                        # Benchmarking & telemetry
│   ├── run_evaluation.py              # Precision@k eval against gold queries
│   ├── telemetry_aggregation.py       # Stats logger
│   └── gold_queries.json              # Benchmark query set
│
├── tests/                             # Test suite
│   ├── test_memory.py                 # Concurrency stress test
│   └── test_hallucination_filter.py   # Hallucination detection tests
│
├── api.py                             # FastAPI REST API
├── memory_bridge.py                   # OpenAI-compatible memory proxy
├── data/                              # Runtime data (gitignored)
├── requirements.txt
└── README.md
```

---

## 🏗️ Architecture

Memory Engine has two parallel storage stacks that serve different use cases:

### Stack 1: Full-Featured Engine (`memory_engine.py`)

The primary store — FAISS + SQLite with FTS5 full-text search, WAL mode, connection pooling, hybrid vector + BM25 search, HDBSCAN clustering, consolidation, pruning, archival, and a time-aware FAISS sub-index. Hallucination filtering is applied on both add and retrieval.

### Stack 2: Holographic Memory (`holographic_memory.py`)

A lighter, time-aware store — HNSW FAISS index with cosine similarity, exponential time-decay ("redshifted recall"), SHA-256 deduplication, and semantic chunking. Designed for fast, recency-biased retrieval. Also includes hallucination filtering.

### How They Connect

```
                    ┌─────────────────────┐
                    │   semantic_graph.py  │  ← Concept graph overlay
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
   ┌──────────▼──────────┐    │    ┌───────────▼───────────┐
   │   memory_engine.py  │    │    │ holographic_memory.py  │
   │  (FAISS+SQLite+FTS) │    │    │ (HNSW+time-decay)      │
   └──────────┬──────────┘    │    └───────────┬───────────┘
              │               │                │
              │    ┌──────────▼──────────┐     │
              └───►│vector_memory_wrapper│◄────┘
                   │   (shared FAISS+DB) │
                   └─────────────────────┘
```

The **Memory Bridge** (`memory_bridge.py`) sits in front of any OpenAI-compatible API and automatically injects relevant recalled memories into the prompt — making any LLM memory-aware without changing its code.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/memory-engine.git
cd memory-engine
pip install -r requirements.txt
```

### 2. Index Your Memories

```bash
# Ingest all markdown files from a workspace
python pipelines/ingest_workspace.py /path/to/your/workspace

# Or import specific memory files
python pipelines/import_memories.py
```

### 3. Query with Time-Decay

```bash
# Recall memories with recency-biased ranking
python core/redshifted_recall.py recall "what happened yesterday"

# Check system health
python core/redshifted_recall.py health
```

### 4. Clean Up Hallucinations

```bash
# Scan for hallucinated memories (report only)
python pipelines/cleanup_hallucinations.py --scan --threshold 0.5

# Dry run — see what would be deleted
python pipelines/cleanup_hallucinations.py --remove --threshold 0.5

# Actually remove (with confirmation)
python pipelines/cleanup_hallucinations.py --remove --confirm --threshold 0.5
```

### 5. Run the REST API

```bash
# Start the FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 6. Run Evaluation

```bash
python evaluation/run_evaluation.py
```

---

## 🛡️ Hallucination Filtering

Both storage stacks include automatic hallucination detection and filtering — a critical feature for any AI memory system, since LLMs routinely generate memories containing errors, speculation, or terminal noise.

### How It Works

Each memory is scanned against 30+ regex patterns across multiple categories:

| Category | What It Catches | Examples |
|----------|----------------|----------|
| **Terminal Artifacts** | ANSI escapes, sysinfo banners, neofetch output | `\x1b[32m`, `OS: Ubuntu` |
| **Fabrication Patterns** | False success claims | `✅ deployed successfully` |
| **Error/Failure Indicators** | Remembered errors that aren't facts | `failed to start daemon` |
| **Uncertainty Patterns** | Guesses treated as knowledge | `I think`, `maybe`, `probably` |
| **Placeholder Content** | Unfinished templates | `your-gateway.example.com` |

### Confidence Scoring

```
score = min(0.9, (match_count × 0.2) + (category_count × 0.15))
```

- **Critical** (≥ 0.95): 3+ categories or 5+ pattern matches
- **High** (≥ 0.85): 2+ categories or 3+ pattern matches
- **Medium** (≥ 0.6): 2+ pattern matches
- **Low** (≥ 0.4): 1 pattern match

Memories scoring above the threshold (default `0.5`) are rejected on add and filtered on retrieval.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `MEMORY_FILTER_HALLUCINATIONS` | `true` | Enable hallucination filtering |
| `MEMORY_HALLUCINATION_THRESHOLD` | `0.5` | Confidence threshold (0-1); memories scoring ≥ threshold are filtered |

See [HALLUCINATION_FILTERING.md](HALLUCINATION_FILTERING.md) for the full guide on detection categories, tuning thresholds, and batch cleanup.

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MEMORY_DB_PATH` | `data/memory.db` | SQLite database path |
| `MEMORY_INDEX_PATH` | `data/memory.index` | FAISS index path |
| `MEMORY_EMB_DIM` | `384` | Embedding dimension |
| `INDEX_SAVE_INTERVAL` | `100` | Auto-save FAISS index after N operations |
| `DB_POOL_SIZE` | `5` | SQLite connection pool size |
| `MEMORY_FILTER_HALLUCINATIONS` | `true` | Enable hallucination filtering |
| `MEMORY_HALLUCINATION_THRESHOLD` | `0.5` | Hallucination confidence threshold (0–1) |
| `MEMORY_API_ADMIN_TOKEN` | *(unset)* | Token for admin API endpoints |
| `ENABLE_MULTILINGUAL_EMBEDDINGS` | `false` | Enable optional multilingual embeddings (set to `true` to use multilingual model) |
| `MULTILINGUAL_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Model name for multilingual embeddings |


Copy `.env.example` to `.env` and fill in your values.

---

## 🌉 Memory Bridge (OpenAI-Compatible Proxy)

The Memory Bridge (`memory_bridge.py`) is a FastAPI server that acts as a drop-in proxy for any OpenAI-compatible API. It intercepts incoming chat requests, performs a time-decayed memory recall, filters out hallucinated results, and injects relevant context into the prompt — all transparently.

```bash
# Start the bridge
python memory_bridge.py

# Point your client to http://localhost:19192 instead of the upstream API
# All requests are proxied through with automatic memory augmentation
```

This is how Memory Engine integrates with OpenClaw's agent loop — the agent's requests pass through the bridge, getting enriched with relevant memories before reaching the LLM.

---

## 🐾 Using with OpenClaw

Memory Engine is designed as a **project** inside an [OpenClaw](https://github.com/openclaw) workspace. To use it:

### Setup

1. **Clone into your OpenClaw workspace:**
   ```bash
   cd ~/.openclaw/workspace/projects/
   git clone https://github.com/YOUR_USERNAME/memory-engine.git
   ```

2. **Install dependencies:**
   ```bash
   cd memory-engine
   pip install -r requirements.txt
   ```

3. **Configure environment** (copy and edit):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and paths
   ```

### How OpenClaw Uses It

- **Workspace Ingestion** — On each heartbeat, `ingest_workspace.py` scans all workspace markdown files, computes content hashes, and only re-indexes changed or new files. Deleted files are cleaned up automatically.
- **Session Ingestion** — `ingest_sessions.py` parses conversation session logs, splits them into semantic chunks, and indexes them as memories.
- **Passive Recall** — The Memory Bridge intercepts every LLM call, recalls the top-3 relevant memories by decayed score, and injects them as `[MEM]` blocks into the prompt.
- **Hallucination Hygiene** — Memories generated from model errors, speculative outputs, or terminal noise are automatically filtered before storage.
- **Knowledge Graph** — Structured facts (triples) can be extracted from markdown files and queried for relationship-based reasoning.

### Recommended Pipelines

```bash
# Full workspace re-index (idempotent, safe to run on every heartbeat)
python pipelines/ingest_workspace.py

# Ingest conversation session logs
python pipelines/ingest_sessions.py

# Generate hierarchical summaries (topic → domain → life)
python pipelines/hierarchical_summaries.py

# Export knowledge graph triples to CSV
python pipelines/knowledge_graph_exporter.py
```

---

## 🧪 Running Tests

```bash
# Run all tests
bash run_tests.sh

# Run specific test suites
python tests/test_hallucination_filter.py
python tests/test_memory.py

# Run evaluation benchmarks
python evaluation/run_evaluation.py
```

---

## 📊 REST API

The FastAPI server (`api.py`) exposes a clean REST interface:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/add_memory` | POST | Add a single memory (vector + metadata + optional text) |
| `/add_memories_batch` | POST | Batch add multiple memories |
| `/search` | POST | Hybrid vector + keyword search |
| `/search-kg` | POST | Hybrid search + Knowledge Graph context |
| `/kg-stats` | GET | Knowledge graph statistics |
| `/stats` | GET | Memory store statistics |
| `/admin/force_save` | POST | Force-save FAISS index (requires admin token) |
| `/admin/shutdown` | POST | Clean shutdown (requires admin token) |

```bash
# Example: search memories
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "meeting notes", "query_vector": [...], "top_k": 5}'
```

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
