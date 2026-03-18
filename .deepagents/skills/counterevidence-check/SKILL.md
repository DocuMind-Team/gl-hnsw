---
name: counterevidence-check
description: Search for negative signals, duplication, weak direction, and low-value overlap before an edge survives. Use after tentative judgments and before final edge review.
allowed-tools:
  - read_candidate_bundle
  - read_judgment_bundle
  - read_failure_patterns
  - read_execution_manifest
  - audit_anchor_execution
---

# Counterevidence Check

This stage is adversarial.

## Workflow

1. Start from tentative accepted edges.
2. Search for duplicate bridges, generic overlap, and direction errors.
3. For argumentative `comparison`, reject pairs that are cross-topic unless they carry a strong specific bridge.
4. Emit risk flags and counterevidence summaries.
4. Prefer dropping marginal edges rather than keeping noisy bridges.

## Tool discipline

- Audit the anchor workflow before assuming the judgment bundle is complete.
- Use `read_failure_patterns` to check whether the pair matches a known noise pattern.
- When an argumentative pair has high overlap but weak topic consistency, treat it as likely drift, not as a reusable bridge.
