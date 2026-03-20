from __future__ import annotations

import re
from typing import Any


CONTRAST_CUES = {
    "however",
    "instead",
    "oppose",
    "opposes",
    "opposed",
    "against",
    "contrast",
    "contrary",
    "alternative",
    "critic",
    "disagree",
    "disagrees",
}


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def main(payload: dict[str, Any]) -> dict[str, Any]:
    anchor = payload.get("anchor", {}) or {}
    candidate = payload.get("candidate", {}) or {}
    local = payload.get("local_signals", {}) or {}
    verdict = payload.get("verdict", {}) or {}

    anchor_stance = str(anchor.get("metadata", {}).get("stance", "")).lower()
    candidate_stance = str(candidate.get("metadata", {}).get("stance", "")).lower()
    stance_contrast = 1.0 if anchor_stance and candidate_stance and anchor_stance != candidate_stance else 0.0
    verdict_text = " ".join(
        [
            str(verdict.get("decision_reason", "")),
            str(verdict.get("rationale", "")),
            " ".join(verdict.get("contradiction_flags", []) or []),
        ]
    ).lower()
    cue_hits = len(_tokens(verdict_text) & CONTRAST_CUES)
    lexical_contrast = min(1.0, cue_hits / 2.0)
    local_contrast = float(local.get("stance_contrast", 0.0) or 0.0)
    contrast_evidence = min(1.0, 0.45 * stance_contrast + 0.25 * lexical_contrast + 0.3 * min(local_contrast, 1.0))
    return {
        "contrast_evidence": round(contrast_evidence, 6),
        "query_surface_match": round(min(1.0, 0.4 + 0.6 * contrast_evidence) if contrast_evidence > 0 else 0.0, 6),
        "uncertainty_hint": round(max(0.0, 0.45 - contrast_evidence), 6),
        "notes": ["explicit_contrast_signal"] if contrast_evidence >= 0.55 else [],
    }
