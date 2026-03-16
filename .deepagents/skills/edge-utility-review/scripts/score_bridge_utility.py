"""Reference helper for review-stage utility ranking."""

from __future__ import annotations


def rank_score(utility: float, confidence: float, risk_penalty: float) -> float:
    return round(max(0.0, utility * confidence * (1.0 - risk_penalty)), 6)
