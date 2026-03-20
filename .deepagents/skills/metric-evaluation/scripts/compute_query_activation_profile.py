from __future__ import annotations

import re
from typing import Any


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]


def _brief_dict(brief: Any) -> dict[str, Any]:
    if isinstance(brief, dict):
        return brief
    if hasattr(brief, "__dict__"):
        return dict(getattr(brief, "__dict__", {}))
    return {"title": str(brief), "summary": "", "claims": [], "keywords": [], "relation_hints": [], "metadata": {}}


def _top_terms(brief: dict[str, Any], limit: int = 12) -> list[str]:
    brief = _brief_dict(brief)
    seen: list[str] = []
    for token in _tokens(
        " ".join(
            [
                str(brief.get("title", "")),
                str(brief.get("summary", "")),
                " ".join(brief.get("claims", []) or []),
                " ".join(brief.get("keywords", []) or []),
                " ".join(brief.get("relation_hints", []) or []),
            ]
        )
    ):
        if token in seen:
            continue
        seen.append(token)
        if len(seen) >= limit:
            break
    return seen


def main(payload: dict[str, Any]) -> dict[str, Any]:
    anchor = _brief_dict(payload.get("anchor", {}) or {})
    candidate = _brief_dict(payload.get("candidate", {}) or {})
    relation_type = str(payload.get("relation_type", "")).strip() or "supporting_evidence"
    local = payload.get("local_signals", {}) or {}
    verdict = payload.get("verdict", {}) or {}

    anchor_terms = _top_terms(anchor, limit=10)
    candidate_terms = _top_terms(candidate, limit=12)
    relation_hints = [str(item) for item in candidate.get("relation_hints", [])][:6]
    use_cases = [relation_type]
    if relation_type == "comparison":
        use_cases.append("same-topic-contrast")
    elif relation_type == "same_concept":
        use_cases.append("concept-bridge")
    elif relation_type == "implementation_detail":
        use_cases.append("mechanism-detail")
    elif relation_type == "supporting_evidence":
        use_cases.append("claim-support")
    topic_signature = sorted(
        {
            *(anchor_terms[:4]),
            *(candidate_terms[:4]),
            str(anchor.get("metadata", {}).get("topic_family", "")),
            str(anchor.get("metadata", {}).get("topic_cluster", "")),
            str(candidate.get("metadata", {}).get("topic_family", "")),
            str(candidate.get("metadata", {}).get("topic_cluster", "")),
        }
        - {""}
    )[:10]
    activation_prior = min(
        1.0,
        0.38 * float(local.get("utility_score", verdict.get("utility_score", 0.0)) or 0.0)
        + 0.32 * float(local.get("bridge_information_gain", local.get("bridge_gain", 0.0)) or 0.0)
        + 0.18 * float(local.get("topic_consistency", 0.0) or 0.0)
        + 0.12 * float(local.get("query_surface_match", 0.0) or 0.0),
    )
    drift_risk = max(
        float(local.get("drift_risk", 0.0) or 0.0),
        1.0 - min(1.0, float(local.get("topic_consistency", 0.0) or 0.0) + 0.2),
    )
    negative_patterns = []
    if drift_risk >= 0.4:
        negative_patterns.append("topic_drift")
    if float(local.get("duplicate_risk", 0.0) or 0.0) >= 0.55:
        negative_patterns.append("near_duplicate")
    return {
        "topic_signature": topic_signature,
        "query_surface_terms": sorted({*anchor_terms[:5], *candidate_terms[:7], *relation_hints})[:12],
        "edge_use_cases": use_cases,
        "drift_risk": round(drift_risk, 6),
        "activation_prior": round(activation_prior, 6),
        "negative_patterns": negative_patterns,
    }
