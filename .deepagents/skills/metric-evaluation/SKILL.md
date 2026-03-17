---
name: metric-evaluation
description: Evaluate candidate and review bundles from a retrieval-utility perspective. Use when ranking edges by likely recall, rank gain, novelty, and drift risk instead of semantic plausibility alone.
---

# Metric Evaluation

Use this skill when a reviewer or supervisor must decide whether a candidate edge is
worth keeping for retrieval.

## Workflow

1. Read reviewed utility, kept count, and graph coverage signals.
2. Prefer edges that improve access to new lexical or semantic surfaces.
3. Penalize edges that only restate already-covered content.
4. Make utility decisions explicit and auditable.

## Output expectations

- Report coverage gain and drift risk in concise terms.
- Keep decisions tied to retrieval quality, not only semantic correctness.
