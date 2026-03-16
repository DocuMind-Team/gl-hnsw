"""Reference helper for bridge utility scoring."""

from __future__ import annotations


def bridge_score(confidence: float, utility: float) -> float:
    return round(max(0.0, min(1.0, 0.6 * confidence + 0.4 * utility)), 6)
