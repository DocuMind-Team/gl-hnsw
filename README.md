# gl-hnsw

`gl-hnsw` is a document-level retrieval system that combines:

- HNSW for geometric nearest-neighbor search
- a logic overlay graph for one-hop query-conditioned expansion
- DeepAgents-backed task runners for profiling, discovery, judging, and memory updates

## Quick start

1. Install dependencies:

```bash
pip install -e .[dev]
```

2. Build demo artifacts:

```bash
python scripts/build_embeddings.py
python scripts/build_hnsw.py
python scripts/profile_docs.py
python scripts/discover_edges.py
python scripts/evaluate.py
```

3. Start the API:

```bash
python scripts/serve_retrieval.py
```

By default the repository runs in deterministic `stub` provider mode so the full demo works without external model credentials. To switch to a real OpenAI-compatible backend, set:

- `GL_HNSW_PROVIDER_KIND=openai_compatible`
- `GL_HNSW_BASE_URL=...`
- `GL_HNSW_API_KEY=...`
- `GL_HNSW_CHAT_MODEL=...`
- `GL_HNSW_EMBEDDING_MODEL=...`

## Repository layout

- `configs/`: YAML configuration
- `data/raw/`: source corpus
- `data/demo/`: demo queries, qrels, and gold logic edges
- `src/hnsw_logic/`: application package
- `scripts/`: runnable entry points
- `tests/`: unit and integration tests
