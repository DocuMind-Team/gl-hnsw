# Signal Semantics

- `topic_consistency`: Whether the anchor and candidate belong to the same topic cluster or family.
- `query_surface_match`: Whether the pair exposes terms that can be matched directly by a user query.
- `drift_risk`: The likelihood that following this pair will pull retrieval away from the intended topic.
- `uncertainty_hint`: A conservative estimate of how ambiguous the pair still is after local signal extraction.
