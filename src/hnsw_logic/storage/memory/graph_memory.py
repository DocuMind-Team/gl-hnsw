from __future__ import annotations

from pathlib import Path

from hnsw_logic.domain.serialization import read_json, write_json


class GraphMemoryStore:
    def __init__(self, stats_path: Path):
        self.stats_path = stats_path

    def read(self) -> dict:
        return read_json(self.stats_path, default={"accepted_edges": 0, "last_revalidated_at": None}) or {}

    def write(self, payload: dict) -> None:
        write_json(self.stats_path, payload)
