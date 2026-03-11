from __future__ import annotations

from pathlib import Path

from hnsw_logic.core.models import AnchorMemory
from hnsw_logic.core.utils import read_json, write_json


class AnchorMemoryStore:
    def __init__(self, root: Path):
        self.root = root

    def path_for(self, doc_id: str) -> Path:
        return self.root / f"{doc_id}.json"

    def read(self, doc_id: str) -> AnchorMemory:
        payload = read_json(self.path_for(doc_id))
        if payload:
            return AnchorMemory(**payload)
        return AnchorMemory(anchor_doc_id=doc_id)

    def write(self, memory: AnchorMemory) -> None:
        write_json(self.path_for(memory.anchor_doc_id), memory)
