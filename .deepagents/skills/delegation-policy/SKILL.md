---
name: delegation-policy
description: Constrain supervisor delegation, retry budgets, and stage ordering for offline indexing. Use when the main agent must decide which subagent to call next, how many task rounds remain, and when to stop and fall back.
---

# Delegation Policy

Use this skill to keep task delegation bounded and auditable.

## Workflow

1. Audit the anchor before delegating any work.
2. Delegate only the next missing stage, never the whole workflow blindly.
3. Respect iteration caps and task-slot budgets.
4. If audit shows no progress across retries, stop delegating and signal fallback.
5. After every delegated task, re-audit and explicitly record whether the next stage is now unblocked.

## Guard rails

- Never skip required stage order.
- Never continue once the retry budget is exhausted.
- Prefer a deterministic fallback over speculative extra delegation.
- Never assume a stage succeeded until `audit_anchor_execution` confirms it.

## Recommended tools

- `read_execution_manifest`
- `audit_anchor_execution`
