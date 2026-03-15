# Release Notes: v0.1

## Overview

`v0.1` is the first packaged version of `gl-hnsw`.
This release establishes the full offline indexing and online retrieval stack:

- document normalization and embedding generation
- HNSW dense indexing
- offline agent-centric graph construction
- logic-overlay retrieval with one-hop expansion
- benchmark and architecture documentation

The release is designed to be runnable in deterministic local mode and extensible to live remote-model mode.

## Included Components

### Offline indexing

- corpus ingestion from `data/raw/`
- processed document store in `data/processed/docs.jsonl`
- embedding build pipeline
- HNSW index build and persistence
- offline agent chain for:
  - document profiling
  - candidate scouting
  - relation judging
  - edge reviewing
  - memory curation

### Online retrieval

- dense HNSW baseline retrieval
- sparse supplemental retrieval
- graph-aware one-hop expansion
- fusion and reranking
- FastAPI service endpoints

### Evaluation and docs

- demo evaluation pipeline
- BEIR-style sample benchmarking harness
- architecture documentation
- benchmark documentation

## Delivery Scope

This version intentionally keeps the main agent contribution in the offline indexing phase.
Online query execution does not invoke agents.
Instead, online retrieval consumes the graph and memory artifacts built offline.

## Supported modes

### Stub mode

Default local mode for deterministic development and testing.

### Live mode

OpenAI-compatible remote model support for offline agent execution.
Typical project setup:

- chat / agent model: `mimo-v2-flash`
- embedding model: local `bge-m3`

## Key engineering decisions

- document-level indexing instead of chunk-level indexing
- HNSW used only for geometric retrieval, not as a mutable logic graph
- one-hop graph expansion only
- file-based graph and memory persistence
- SQLite-backed job registry
- offline-first agent design

## Benchmark snapshot

This release includes benchmark coverage across multiple corpora, including:

- in-domain demo corpus
- `scifact` sample
- `nfcorpus` sample
- `arguana` sample

Representative current results are documented in [Benchmark Report](./benchmark-report.md).

## Known limitations

- offline live indexing remains significantly slower than stub mode
- cross-dataset uplift is positive but not yet uniform across every metric and dataset size
- graph quality is still sensitive to reviewer utility ranking and bridge selection
- current logic expansion remains limited to one hop

## Recommended next steps

- strengthen utility-first offline reviewer ranking
- expand multi-dataset benchmark coverage
- improve duplicate-bridge suppression
- optimize live offline indexing latency
