# Resume Policy

Resume policy is conservative:

- If a bundle exists and matches the expected stage, reuse it.
- If a stage is missing, run only that stage.
- If a stage has failed repeatedly and hit the iteration cap, stop delegating and
  hand control back to the supervisor for escalation.
- Record each recovery attempt in the execution manifest notes.
