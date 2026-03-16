"""Reference helper for merging candidate rows."""

from __future__ import annotations


def merge_ids(rows: list[str]) -> list[str]:
    return list(dict.fromkeys(rows))
