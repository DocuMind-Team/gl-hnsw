# Signal Fusion

When a local signal bundle is provided:

- read `dense_score`, `sparse_score`, `mention_score`, `forward_reference_score`, `direction_score`, `stage_pair`, and `risk_flags` first
- treat the local signals as grounded evidence, not as optional hints
- only override a strong local signal when the document evidence clearly contradicts it
- prefer abstaining when `utility_score` is low or `risk_flags` indicate topic drift or weak direction
