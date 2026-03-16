"""Reference helper for signal-aware utility scoring."""

from __future__ import annotations


def fuse(local_support: float, utility: float, direction: float) -> float:
    return round(max(0.0, min(1.0, 0.45 * local_support + 0.4 * utility + 0.15 * direction)), 6)
