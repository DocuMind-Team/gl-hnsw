from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import orjson

from hnsw_logic.domain.models import LogicEdge
from hnsw_logic.domain.serialization import ensure_dir, read_jsonl, to_jsonable


class GraphStore:
    def __init__(self, path: Path):
        self.path = path
        self._edges_by_src: dict[str, list[LogicEdge]] = defaultdict(list)
        self.reload()

    @staticmethod
    def _edge_key(edge: LogicEdge) -> tuple[str, str, str]:
        return edge.src_doc_id, edge.dst_doc_id, edge.relation_type

    @staticmethod
    def _edge_priority(edge: LogicEdge) -> tuple[float, float, str]:
        utility = float(getattr(edge, "utility_score", 0.0) or 0.0)
        return (0.65 * float(edge.confidence) + 0.35 * utility, utility, edge.dst_doc_id)

    def _persist_edges(self) -> None:
        ensure_dir(self.path.parent)
        rows = [to_jsonable(edge) for edge in self.all_edges()]
        with self.path.open("wb") as handle:
            for row in rows:
                handle.write(orjson.dumps(row))
                handle.write(b"\n")

    def reload(self) -> None:
        self._edges_by_src = defaultdict(list)
        by_key: dict[tuple[str, str, str], LogicEdge] = {}
        for row in read_jsonl(self.path):
            edge = LogicEdge(**row)
            key = self._edge_key(edge)
            current = by_key.get(key)
            if current is None or self._edge_priority(edge) > self._edge_priority(current):
                by_key[key] = edge
        for edge in by_key.values():
            self._edges_by_src[edge.src_doc_id].append(edge)

    def add_edges(self, edges: list[LogicEdge]) -> None:
        if not edges:
            return
        by_key = {
            self._edge_key(edge): edge
            for edge in self.all_edges()
        }
        for edge in edges:
            key = self._edge_key(edge)
            current = by_key.get(key)
            if current is None or self._edge_priority(edge) > self._edge_priority(current):
                by_key[key] = edge
        self._edges_by_src = defaultdict(list)
        for edge in by_key.values():
            self._edges_by_src[edge.src_doc_id].append(edge)
        self._persist_edges()

    def get_out_edges(self, doc_id: str) -> list[LogicEdge]:
        return sorted(
            self._edges_by_src.get(doc_id, []),
            key=lambda edge: (-(0.65 * edge.confidence + 0.35 * getattr(edge, "utility_score", 0.0)), -getattr(edge, "utility_score", 0.0), edge.dst_doc_id),
        )

    def has_edges(self) -> bool:
        return any(self._edges_by_src.values())

    def edge_count(self) -> int:
        return sum(len(values) for values in self._edges_by_src.values())

    def all_edges(self) -> list[LogicEdge]:
        edges: list[LogicEdge] = []
        for values in self._edges_by_src.values():
            edges.extend(values)
        return edges
