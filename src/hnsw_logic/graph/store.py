from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from hnsw_logic.core.models import LogicEdge
from hnsw_logic.core.utils import append_jsonl, read_jsonl


class GraphStore:
    def __init__(self, path: Path):
        self.path = path
        self._edges_by_src: dict[str, list[LogicEdge]] = defaultdict(list)
        self.reload()

    def reload(self) -> None:
        self._edges_by_src = defaultdict(list)
        for row in read_jsonl(self.path):
            edge = LogicEdge(**row)
            self._edges_by_src[edge.src_doc_id].append(edge)

    def add_edges(self, edges: list[LogicEdge]) -> None:
        if not edges:
            return
        append_jsonl(self.path, edges)
        for edge in edges:
            self._edges_by_src[edge.src_doc_id].append(edge)

    def get_out_edges(self, doc_id: str) -> list[LogicEdge]:
        return sorted(self._edges_by_src.get(doc_id, []), key=lambda edge: (-edge.confidence, edge.dst_doc_id))

    def all_edges(self) -> list[LogicEdge]:
        edges: list[LogicEdge] = []
        for values in self._edges_by_src.values():
            edges.extend(values)
        return edges
