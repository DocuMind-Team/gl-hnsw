"""Reference helper for memory summarization."""

from __future__ import annotations


def compact(lines: list[str]) -> list[str]:
    return [line for line in sorted(set(lines)) if line.strip()][:20]
