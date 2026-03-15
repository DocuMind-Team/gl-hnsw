# gl-hnsw

[![tag](https://img.shields.io/github/v/tag/DocuMind-Team/gl-hnsw?label=version)](https://github.com/DocuMind-Team/gl-hnsw/tags)
[![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![tests](https://img.shields.io/badge/tests-pytest-success)](./tests)
[![docs](https://img.shields.io/badge/docs-architecture%20%26%20benchmark-informational)](./docs)

`gl-hnsw` is an agent-centric retrieval system that combines dense HNSW search with an offline logic-overlay graph.
The system uses offline agents for document profiling, candidate discovery, relation judging, review, and memory curation,
then serves queries from the built index and graph without invoking agents online.

## Highlights

- Offline `agent-centric` indexing: agents participate in `profile -> scout -> judge -> review -> memory curate`.
- Online query path stays local: `query embedding -> HNSW -> sparse supplement -> graph expansion -> rerank`.
- Document-level retrieval with persisted HNSW indices, graph edges, and memory stores.
- Deterministic `stub` mode for local development and live `openai_compatible` mode for remote models.
- Multi-dataset evaluation harness covering demo corpora and BEIR-style samples.

## Documentation

- [Architecture Guide](./docs/architecture.md)
- [Agent Runtime Guide](./docs/agent-runtime.md)
- [Benchmark Report](./docs/benchmark-report.md)
- [Release Notes v0.1](./docs/release-v0.1.md)

## Quick Start

### 1. Install

```bash
pip install -e .[dev]
```

### 2. Build demo artifacts

```bash
python scripts/build_embeddings.py
python scripts/build_hnsw.py
python scripts/profile_docs.py
python scripts/discover_edges.py
python scripts/evaluate.py
```

### 3. Run the API

```bash
python scripts/serve_retrieval.py
```

## Runtime Modes

By default the repository runs in deterministic `stub` provider mode so the full demo works without external model credentials.

To switch to a real OpenAI-compatible backend, set:

- `GL_HNSW_PROVIDER_KIND=openai_compatible`
- `GL_HNSW_BASE_URL=...`
- `GL_HNSW_API_KEY=...`
- `GL_HNSW_CHAT_MODEL=...`
- `GL_HNSW_EMBEDDING_MODEL=...`

For the current project setup, the live path typically uses:

- remote chat / agent model: `mimo-v2-flash`
- local embedding model: `bge-m3`

## Core Workflow

```text
raw corpus
  -> document normalization
  -> embeddings + HNSW
  -> offline agent indexing
  -> logic overlay graph + memory
  -> online retrieval and reranking
```

## Repository Layout

- `configs/`: YAML configuration for providers, HNSW, agents, and retrieval.
- `data/raw/`: source corpus inputs.
- `data/demo/`: demo queries, qrels, and gold logic edges.
- `docs/`: architecture, benchmark, and release documentation.
- `scripts/`: runnable pipeline and evaluation entry points.
- `src/hnsw_logic/`: application package.
- `tests/`: unit and integration tests.

## Versioning

Current published milestone:

- `v0.1`: initial public baseline with offline agent-centric indexing, benchmark documentation, and GitHub release packaging.
