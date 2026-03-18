# Task Policy

Supervisor policy:

- Audit first.
- Delegate one missing stage at a time.
- Re-audit after each delegated task.
- Stop after the configured iteration cap.
- Use task slots conservatively; prioritize stable completion over maximal fan-out.
- Treat the execution manifest as authoritative state.
- Never move to a later stage until the current stage artifact exists and passes audit.
