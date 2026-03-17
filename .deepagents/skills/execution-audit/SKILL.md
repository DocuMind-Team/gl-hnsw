---
name: execution-audit
description: Audit per-anchor execution state, required artifacts, and stage completeness. Use when deciding whether an offline indexing task actually finished, which stage is missing, and whether fallback or recovery is required.
---

# Execution Audit

Use this skill whenever a main agent, checker, or reviewer needs to verify that an
anchor workflow is complete.

## Workflow

1. Read the execution manifest before assuming any stage succeeded.
2. Audit dossier, candidate, judgment, check, review, and memory bundles against
   the stage contract.
3. Report the next missing stage, retry pressure, and whether fallback is required.
4. Prefer explicit audit output over optimistic natural-language assumptions.

## Tool discipline

- Use `audit_anchor_execution` to determine the authoritative workflow state.
- Use `read_execution_manifest` for details such as retry counts and notes.
- Do not infer completion from partial workspace files alone.
