# Review Rubric

- keep if edge is durable and retrieval-useful
- drop if edge is weakly useful, redundant, or drift-prone
- lower confidence when checker finds unresolved risk
- argumentative comparison requires same-topic consistency, not just opposite stance
- prefer one precise contrast bridge to several broad debate neighbors
- duplicate-only risk should not defeat a same-topic contrast bridge when the bridge is reusable and high utility
- use `select_edge_budget.py` to combine utility, novelty, and activation prior into a compact ranking signal
- activation profiles should describe likely query use cases and known failure modes, not dataset-specific shortcuts
