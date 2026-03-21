from __future__ import annotations

from pathlib import Path

from hnsw_logic.domain.models import DocBrief
from hnsw_logic.domain.serialization import read_json, write_json


class BriefStore:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, doc_id: str) -> Path:
        return self.root / f"{doc_id}.json"

    def write(self, brief: DocBrief) -> None:
        write_json(self.path_for(brief.doc_id), brief)

    def read(self, doc_id: str) -> DocBrief | None:
        payload = read_json(self.path_for(doc_id))
        if not payload:
            return None
        return DocBrief(**payload)

    def all(self) -> list[DocBrief]:
        briefs: list[DocBrief] = []
        for path in sorted(self.root.glob("*.json")):
            payload = read_json(path)
            if payload:
                briefs.append(DocBrief(**payload))
        return briefs
