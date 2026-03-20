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

## Workflow

1. Read the anchor dossier and the full candidate bundle before judging any pair.
2. Use local signals as the primary evidence and text spans as supporting evidence.
3. For `comparison`, verify topic consistency before rewarding stance contrast.
4. When a structured topic family or sibling-document family is available, treat it as a stronger same-topic cue than broad policy overlap.
5. If a pair is plausible but low utility, keep `accepted=false` and explain the retrieval risk.

## Output expectations

- Emit explicit `utility_score`, `uncertainty`, and `risk_flags`.
- Distinguish “same topic but opposite stance” from “different topic with generic argumentative overlap”.
- Prefer same-family contrast pairs over cross-family analogies when both seem plausible.

## Recommended tools

- `read_anchor_dossier`
- `read_candidate_bundle`
