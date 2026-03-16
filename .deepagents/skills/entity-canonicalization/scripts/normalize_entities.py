"""Reference helper for entity normalization."""

from __future__ import annotations


def normalize(items: list[str]) -> list[str]:
    return sorted({item.strip().lower() for item in items if item.strip()})
