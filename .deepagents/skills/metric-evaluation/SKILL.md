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
4. For comparison edges, score topic consistency separately from stance contrast.
5. Make utility decisions explicit and auditable.

## Output expectations

- Report coverage gain and drift risk in concise terms.
- Keep decisions tied to retrieval quality, not only semantic correctness.
- Mention when a candidate is “same-side duplication” versus “same-topic contrast”.

## Recommended tools

- `evaluate_anchor_utility`
- `read_candidate_bundle`
- `read_judgment_bundle`
- `read_counterevidence_bundle`
