# Stage Contract

The offline indexing workflow uses these ordered stages:

1. `dossiers`
2. `candidates`
3. `judgments`
4. `checks` (only when counterevidence is enabled)
5. `reviews`
6. `memory`

A stage is considered complete only when its bundle file exists in the workspace.

`ready_for_commit` means all pre-commit stages are complete:

- `dossiers`
- `candidates`
- `judgments`
- `checks` when enabled
- `reviews`

`workflow_complete` means `ready_for_commit` plus the `memory` bundle exists.

If the next missing stage has already exhausted its retry budget, the workflow
should be marked as incomplete for supervisor escalation.
