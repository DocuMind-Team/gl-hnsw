---
name: graph-hygiene
description: Maintain graph sparsity, diversity, symmetry policy, and duplicate suppression. Use when deciding whether accepted edges should actually be written into the persisted overlay graph.
---

# Graph Hygiene

Guard the graph against low-value growth.

## Rules

- Keep the graph sparse.
- Prefer diverse edge families.
- Mirror only explicitly symmetric relations.
- Reject duplicate or near-duplicate bridges unless they add clear value.
