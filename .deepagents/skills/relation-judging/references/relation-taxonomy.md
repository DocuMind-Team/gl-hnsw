# Relation Taxonomy

Canonical relations:

- `supporting_evidence`
- `implementation_detail`
- `same_concept`
- `comparison`
- `prerequisite`
- `none`

Additional guidance:

- `comparison` is only durable when the pair shares a topic, claim family, or specific bridge surface.
- Opposite stance alone is not enough for `comparison`.
- If documents expose a structured sibling/topic-family key, treat family agreement as a strong same-topic signal.
- Use `none` when the pair is broadly related but unlikely to improve retrieval.
- When a pair is high-signal but still ambiguous, prefer `accepted=false` with an explicit uncertainty note over forcing a canonical relation.
