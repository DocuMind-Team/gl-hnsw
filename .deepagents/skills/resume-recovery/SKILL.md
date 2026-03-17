---
name: resume-recovery
description: Resume interrupted offline indexing tasks from the first missing stage. Use when bundles already exist, retries have occurred, or a supervisor must recover without repeating completed work.
---

# Resume Recovery

Use this skill when an anchor workflow has partial outputs and must resume safely.

## Workflow

1. Audit the current execution state before taking any action.
2. Resume from the first missing stage only.
3. Reuse existing bundles whenever they satisfy the stage contract.
4. Escalate to fallback only after the retry budget is exhausted or audit shows no progress.

## Guard rails

- Never delete completed bundle files during recovery.
- Never rerun a finished stage unless its artifact is missing or corrupt.
- Treat the execution manifest as the primary recovery ledger.
