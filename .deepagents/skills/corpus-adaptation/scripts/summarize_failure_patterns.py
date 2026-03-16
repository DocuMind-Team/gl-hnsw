"""Reference helper for summarizing corpus adaptation failures."""

from __future__ import annotations


def summarize(patterns: list[str]) -> list[str]:
    return sorted(set(patterns))
