# Edge Utility

Judge not only whether two documents are semantically related, but whether the edge is useful for retrieval.

- high utility: the candidate adds a mechanism, dependency, or strong supporting constraint that can improve recall
- low utility: the pair is topically related but mostly restates, co-occurs, or belongs to the same cluster without adding retrieval value
- if utility is low, set the canonical relation to `none` and explain why
