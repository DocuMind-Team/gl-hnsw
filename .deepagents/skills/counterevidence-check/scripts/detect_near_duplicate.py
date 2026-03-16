"""Reference helper for near-duplicate detection."""

from __future__ import annotations


def duplicate_penalty(title_overlap: float, bridge_gain: float) -> float:
    return round(max(0.0, min(1.0, 0.7 * title_overlap + 0.3 * max(0.0, 1.0 - bridge_gain))), 6)
