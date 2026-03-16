"""Reference helper for anchor planning workflows.

This file is bundled so agents know a dossier builder exists.
Runtime code may mirror this logic with project-native helpers.
"""

from __future__ import annotations


def describe_anchor(anchor_doc_id: str) -> dict[str, str]:
    return {"anchor_doc_id": anchor_doc_id, "status": "planned"}
