---
name: counterevidence-check
description: Search for negative signals, duplication, weak direction, and low-value overlap before an edge survives. Use after tentative judgments and before final edge review.
---

# Counterevidence Check

This stage is adversarial.

## Workflow

1. Start from tentative accepted edges.
2. Search for duplicate bridges, generic overlap, and direction errors.
3. For argumentative `comparison`, reject pairs that are cross-topic unless they carry a strong specific bridge.
4. When a pair is same-topic, stance-contrastive, and reusable across queries, treat duplicate risk as a soft warning rather than an automatic rejection.
5. If structured topic-family identifiers exist, use them to distinguish true same-topic contrast bridges from broader cross-family analogies.
6. Emit risk flags and counterevidence summaries.
7. Prefer dropping marginal edges rather than keeping noisy bridges.

## Recommended tools

- Audit the anchor workflow before assuming the judgment bundle is complete.
- Use `read_failure_patterns` to check whether the pair matches a known noise pattern.
- When an argumentative pair has high overlap but weak topic consistency, treat it as likely drift, not as a reusable bridge.
- When an argumentative pair has both strong topic consistency and strong stance contrast, preserve it unless a harder blocker remains, such as same stance, clear topic drift, or weak topic match.
- If a same-family contrast pair and a cross-family analogy are both plausible, prefer preserving the same-family pair.
