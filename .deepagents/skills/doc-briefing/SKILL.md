---
name: doc-briefing
description: Build concise document dossiers and structured briefs. Use when a document must be normalized into title, summary, entities, claims, relation hints, and indexing-facing metadata.
---

# Doc Briefing

Create compact, grounded dossiers for offline indexing.

## Workflow

1. Read the normalized document view, starting from title, excerpt, top terms, and topic hints.
2. Produce concise summary, entities, claims, and relation hints.
3. Preserve only evidence grounded in the document.
4. For sensitive or provocative text, paraphrase neutrally unless the exact wording is essential for stance or retrieval utility.
5. Write the dossier as structured JSON.

## Recommended tools

- `read_doc_full`
- `read_doc_brief`
- `audit_anchor_execution`

## Tool discipline

- Prefer excerpt-first reading before expanding to the full document.
- Avoid copying long raw passages into stage artifacts when a neutral paraphrase preserves meaning.
