---
name: anchor-planning
description: Plan offline indexing batches, anchor order, and coverage budgets. Use when deciding which anchors should be processed first, how many anchors belong in each batch, and how to trade off central anchors versus bridge-rich anchors.
---

# Anchor Planning

Use this skill when planning offline indexing work.

## Workflow

1. Read graph stats, anchor memories, and available briefs.
2. Estimate coverage pressure, bridge potential, and graph sparsity needs.
3. Produce a batch plan that prefers anchors likely to improve retrieval coverage.
4. Keep plans auditable and deterministic-looking even when model reasoning is used.

## Output expectations

- Write a compact JSON plan file.
- Include anchor ids, batch ids, priority scores, and reasons.
- Prefer conservative budgets over runaway discovery.
