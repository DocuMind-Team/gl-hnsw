---
name: candidate-expansion
description: Expand offline candidate sets using dense neighbors, sparse matches, entities, claims, and memory hints. Use when an anchor needs a high-recall but still utility-aware candidate bundle.
---

# Candidate Expansion

Use this skill to build candidate bundles for offline discovery.

## Workflow

1. Start from dense neighbors and lexical matches.
2. Add entity, claim, and memory-based candidates only when they add new retrieval surfaces or bridge access.
3. In argumentative corpora, prioritize same-topic contrastive candidates before high-overlap same-side candidates.
4. Prefer diversity over many near-duplicates and record why each candidate was kept.
5. Write a candidate bundle with reasons, signal summaries, and explicit bridge hypotheses.

## Recommended tools

- Use `read_anchor_dossier` first so expansion stays tied to the anchor dossier.
- Use `get_hnsw_neighbors` and `search_summaries` as the primary recall sources.
- Use `lookup_entities` only to add missing but high-value bridge candidates.
- Use `read_doc_brief` to verify same-topic alignment before promoting argumentative comparisons.
- If structured family identifiers or sibling-document families are available, treat them as higher-confidence same-topic cues than broad policy vocabulary alone.
