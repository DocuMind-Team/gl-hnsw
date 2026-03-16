"""Reference helper for controlled memory updates."""

from __future__ import annotations


def prepare_updates(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip()]
