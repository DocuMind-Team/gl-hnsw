from __future__ import annotations

import re
from typing import Any


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _brief_terms(brief: dict[str, Any]) -> set[str]:
    parts = [
        brief.get("title", ""),
        brief.get("summary", ""),
        " ".join(brief.get("claims", []) or []),
        " ".join(brief.get("keywords", []) or []),
        " ".join(brief.get("relation_hints", []) or []),
    ]
    return _tokens(" ".join(part for part in parts if part))


def main(payload: dict[str, Any]) -> dict[str, Any]:
    anchor = payload.get("anchor", {}) or {}
    candidate = payload.get("candidate", {}) or {}
    local = payload.get("local_signals", {}) or {}

    anchor_terms = _brief_terms(anchor)
    candidate_terms = _brief_terms(candidate)
    overlap = anchor_terms & candidate_terms
    union = anchor_terms | candidate_terms

    family_match = 1.0 if anchor.get("metadata", {}).get("topic_family") and anchor.get("metadata", {}).get("topic_family") == candidate.get("metadata", {}).get("topic_family") else 0.0
    cluster_match = 1.0 if anchor.get("metadata", {}).get("topic_cluster") and anchor.get("metadata", {}).get("topic_cluster") == candidate.get("metadata", {}).get("topic_cluster") else 0.0
    lexical = len(overlap) / max(1, min(len(union), 12))
    title_overlap = len(_tokens(str(anchor.get("title", ""))) & _tokens(str(candidate.get("title", ""))))
    query_surface = min(1.0, 0.55 * lexical + 0.25 * family_match + 0.2 * cluster_match)
    topic_consistency = min(1.0, 0.45 * lexical + 0.25 * family_match + 0.2 * cluster_match + 0.1 * min(title_overlap, 2))
    drift_risk = max(0.0, 0.65 - topic_consistency)
    uncertainty_hint = max(0.0, 0.5 - lexical)
    if float(local.get("topic_alignment", 0.0) or 0.0) >= 1.0:
        topic_consistency = max(topic_consistency, 0.72)
        drift_risk = min(drift_risk, 0.22)
    notes: list[str] = []
    if family_match:
        notes.append("shared_topic_family")
    if cluster_match:
        notes.append("shared_topic_cluster")
    if overlap:
        notes.append("lexical_topic_overlap")
    return {
        "topic_consistency": round(topic_consistency, 6),
        "query_surface_match": round(query_surface, 6),
        "drift_risk": round(drift_risk, 6),
        "uncertainty_hint": round(uncertainty_hint, 6),
        "notes": notes[:6],
    }
