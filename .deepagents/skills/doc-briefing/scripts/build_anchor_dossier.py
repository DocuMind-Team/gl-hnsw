"""Reference helper for writing anchor dossier payloads."""

from __future__ import annotations


def build_payload(doc_id: str, title: str) -> dict[str, str]:
    return {"doc_id": doc_id, "title": title}
