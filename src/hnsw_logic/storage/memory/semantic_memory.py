from __future__ import annotations

from pathlib import Path

from hnsw_logic.domain.models import GlobalSemanticMemory
from hnsw_logic.domain.serialization import read_json, write_json


class SemanticMemoryStore:
    def __init__(self, entity_path: Path, relation_path: Path, rejection_path: Path):
        self.entity_path = entity_path
        self.relation_path = relation_path
        self.rejection_path = rejection_path

    def read(self) -> GlobalSemanticMemory:
        entity_payload = read_json(self.entity_path, default={}) or {}
        relation_payload = read_json(self.relation_path, default={}) or {}
        rejection_payload = read_json(self.rejection_path, default={}) or {}
        return GlobalSemanticMemory(
            canonical_entities=entity_payload.get("canonical_entities", {}),
            aliases=entity_payload.get("aliases", {}),
            relation_patterns=relation_payload.get("relation_patterns", {}),
            rejection_patterns=rejection_payload.get("rejection_patterns", {}),
        )

    def write(self, memory: GlobalSemanticMemory) -> None:
        write_json(
            self.entity_path,
            {"canonical_entities": memory.canonical_entities, "aliases": memory.aliases},
        )
        write_json(self.relation_path, {"relation_patterns": memory.relation_patterns})
        write_json(self.rejection_path, {"rejection_patterns": memory.rejection_patterns})
