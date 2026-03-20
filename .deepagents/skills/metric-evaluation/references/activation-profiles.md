# Activation Profiles

- `topic_signature`: Topic-cluster or topic-family signals that summarize where the edge is useful.
- `query_surface_terms`: Surface terms that can be matched directly at query time.
- `edge_use_cases`: Typical information-need patterns that benefit from activating the edge.
- `activation_prior`: The offline prior that estimates how often the edge should be activated at all.
- `negative_patterns`: Query patterns that should suppress activation because they are likely to drift.
- keep activation profiles generic enough to transfer across corpora, but specific enough to block broad topic drift.
