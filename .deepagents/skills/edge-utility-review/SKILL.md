---
name: edge-utility-review
description: Review judged edges for retrieval value, diversity, and graph hygiene. Use when selecting which edges should survive into the persisted graph.
---

# Edge Utility Review

Review with a retrieval mindset.

## Workflow

1. Read judge output and counterevidence together.
2. Rank edges by retrieval value, not just semantic plausibility.
3. Prefer edges that add bridge vocabulary or cross-cluster access.
4. In argumentative corpora, keep contrastive bridges only when topic consistency is strong.
5. Reject edges that are true but not useful.

## Recommended tools

- Use `evaluate_anchor_utility` before final keep/drop decisions.
- Use `read_counterevidence_bundle` as the authoritative source for drift and duplicate risk.
- Favor one strong comparison bridge over several same-theme but noisy comparisons.
