---
name: memory-summarization
description: Summarize accepted and rejected edge patterns into reusable offline indexing learnings. Use when turning stage outputs into controlled updates for AGENTS.md and skill references.
---

# Memory Summarization

Summarize learnings conservatively.

## Workflow

1. Separate stable lessons from one-off incidents.
2. Store failure patterns in concise reusable form.
3. Only propose updates inside allowed memory zones.
4. Materialize the stage output by calling `execute_memory_summarization`.

## Tool Discipline

- Read the review bundle before summarizing learnings.
- Use `execute_memory_summarization` to produce the canonical memory artifact.
- Treat the stage as incomplete until `indexing/memory/<anchor_doc_id>.json` exists.
- Do not replace the bundle with freeform notes or a narrative-only response.
