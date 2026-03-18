# Retrieval Utility Heuristics

Utility review should answer:

- Does this edge expose new query terms or claim surfaces?
- Does it connect otherwise separate semantic clusters?
- Does it avoid duplicating existing high-confidence neighbors?
- Is the risk of topic drift lower than the expected retrieval gain?

High utility usually combines:

- non-trivial reviewed utility
- bridge vocabulary or claim novelty
- low duplicate pressure
- low contradiction or hygiene risk

Argumentative comparison utility:

- high when the pair stays on the same topic but presents opposing positions
- low when the pair is only broadly argumentative or policy-related
- low when stance contrast exists but topic-specific bridge terms are absent
