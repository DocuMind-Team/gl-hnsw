---
name: relation-judging
description: Judge candidate pairs into semantic and canonical relations using grounded signals. Use when deciding whether a pair should remain a durable retrieval edge candidate.
---

# Relation Judging

Judge pairs with precision-first reasoning.

## Requirements

- Read local signals before free-form reasoning.
- Prefer abstaining over weak acceptance.
- Separate `semantic_relation_label` from `canonical_relation`.
- Always include uncertainty and risk-aware reasoning.
