---
name: candidate-expansion
description: Expand offline candidate sets using dense neighbors, sparse matches, entities, claims, and memory hints. Use when an anchor needs a high-recall but still utility-aware candidate bundle.
---

# Candidate Expansion

Use this skill to build candidate bundles for offline discovery.

## Workflow

1. Start from dense neighbors and lexical matches.
2. Add entity, claim, and memory-based candidates when they add new retrieval surfaces.
3. Prefer diversity over many near-duplicates.
4. Write a candidate bundle with reasons and signal summaries.
