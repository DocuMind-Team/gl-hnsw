"""Reference helper for graph hygiene duplicate checks."""

from __future__ import annotations


def should_drop(title_overlap: float, utility_gain: float) -> bool:
    return title_overlap >= 0.72 and utility_gain < 0.48
