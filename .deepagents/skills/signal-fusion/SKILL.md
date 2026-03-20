---
name: signal-fusion
description: Interpret dense, sparse, overlap, direction, and risk signals as grounded evidence. Use when a model-stage agent must reconcile local signals with document semantics instead of ignoring them.
---

# Signal Fusion

Use local signals as first-class evidence.

## Workflow

1. Start from script-produced signals, not ad hoc intuition.
2. Read `topic_consistency`, `duplicate_risk`, `bridge_information_gain`, `contrast_evidence`, `query_surface_match`, and `drift_risk` together.
3. Treat low topic consistency plus high drift risk as a strong warning.
4. Treat duplicate risk as a soft warning when bridge gain and contrast evidence remain high.
5. Prefer explicit uncertainty over overconfident acceptance.

## Recommended tools

- Use `compute_topic_consistency` before reasoning about reusable bridge quality.
- Use `compute_bridge_gain` to separate true bridges from topic-local restatements.
- Use `compute_contrast_evidence` when metadata and content disagree about whether a pair is a meaningful comparison.
