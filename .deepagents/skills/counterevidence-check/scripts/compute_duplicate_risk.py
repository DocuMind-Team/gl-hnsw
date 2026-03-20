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
    ]
    return _tokens(" ".join(part for part in parts if part))


def main(payload: dict[str, Any]) -> dict[str, Any]:
    anchor = payload.get("anchor", {}) or {}
    candidate = payload.get("candidate", {}) or {}
    local = payload.get("local_signals", {}) or {}

    anchor_terms = _brief_terms(anchor)
    candidate_terms = _brief_terms(candidate)
    overlap = len(anchor_terms & candidate_terms)
    union = len(anchor_terms | candidate_terms)
    lexical_similarity = overlap / max(1, union)
    duplicate_risk = min(1.0, 0.65 * lexical_similarity + 0.35 * float(local.get("content_overlap_score", 0.0) or 0.0))
    if anchor.get("metadata", {}).get("topic_family") and anchor.get("metadata", {}).get("topic_family") == candidate.get("metadata", {}).get("topic_family"):
        duplicate_risk *= 0.92
    drift_risk = max(0.0, duplicate_risk - 0.28)
    uncertainty_hint = max(0.0, 0.35 - abs(0.5 - duplicate_risk))
    notes: list[str] = []
    if duplicate_risk >= 0.55:
        notes.append("high_overlap_duplicate_risk")
    if overlap >= 4:
        notes.append("shared_surface_terms")
    return {
        "duplicate_risk": round(min(1.0, duplicate_risk), 6),
        "drift_risk": round(drift_risk, 6),
        "uncertainty_hint": round(uncertainty_hint, 6),
        "notes": notes[:6],
    }
