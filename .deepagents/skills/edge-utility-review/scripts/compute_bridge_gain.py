from __future__ import annotations

import re
from typing import Any


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def main(payload: dict[str, Any]) -> dict[str, Any]:
    anchor = payload.get("anchor", {}) or {}
    candidate = payload.get("candidate", {}) or {}
    local = payload.get("local_signals", {}) or {}

    anchor_terms = _tokens(
        " ".join(
            [
                str(anchor.get("title", "")),
                str(anchor.get("summary", "")),
                " ".join(anchor.get("claims", []) or []),
                " ".join(anchor.get("keywords", []) or []),
            ]
        )
    )
    candidate_terms = _tokens(
        " ".join(
            [
                str(candidate.get("title", "")),
                str(candidate.get("summary", "")),
                " ".join(candidate.get("claims", []) or []),
                " ".join(candidate.get("keywords", []) or []),
                " ".join(candidate.get("relation_hints", []) or []),
            ]
        )
    )
    novel_terms = candidate_terms - anchor_terms
    overlap_terms = candidate_terms & anchor_terms
    bridge_gain = min(
        1.0,
        0.5 * min(len(novel_terms) / max(1, min(len(candidate_terms), 8)), 1.0)
        + 0.25 * float(local.get("mention_score", 0.0) or 0.0)
        + 0.25 * float(local.get("dense_score", 0.0) or 0.0),
    )
    query_surface_match = min(1.0, 0.55 * bridge_gain + 0.45 * min(len(overlap_terms) / max(1, min(len(anchor_terms), 6)), 1.0))
    drift_risk = max(0.0, 0.42 - bridge_gain)
    notes = []
    if novel_terms:
        notes.append("adds_novel_surface_terms")
    if overlap_terms:
        notes.append("retains_anchor_overlap")
    return {
        "bridge_information_gain": round(bridge_gain, 6),
        "query_surface_match": round(query_surface_match, 6),
        "drift_risk": round(drift_risk, 6),
        "uncertainty_hint": round(max(0.0, 0.4 - bridge_gain), 6),
        "notes": notes[:6],
        "novel_terms": sorted(novel_terms)[:12],
    }
