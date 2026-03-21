from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from hnsw_logic.config.schema import HnswConfig
from hnsw_logic.domain.serialization import read_json


@dataclass(slots=True)
class Neighbor:
    doc_id: str
    score: float
    rank: int


class HnswSearcher:
    def __init__(self, config: HnswConfig, index_path: Path, meta_path: Path):
        self.config = config
        self.index_path = index_path
        self.meta_path = meta_path
        self._index = None
        self._doc_ids: list[str] = []
        self._vectors: np.ndarray | None = None

    def load(self) -> None:
        if self._index is not None:
            return
        if not self.meta_path.exists():
            raise FileNotFoundError(f"HNSW metadata file not found: {self.meta_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"HNSW index file not found: {self.index_path}")
        payload = read_json(self.meta_path, default={})
        self._doc_ids = payload.get("doc_ids", [])
        import hnswlib

        metric = cast(Literal["l2", "ip", "cosine"], self.config.metric)
        index = hnswlib.Index(space=metric, dim=self.config.vector_dim)
        index.load_index(str(self.index_path))
        index.set_ef(self.config.ef_search)
        self._index = index

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Neighbor]:
        if top_k <= 0:
            return []
        self.load()
        if not self._doc_ids:
            return []
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Query vector must be a 1D array")
        if vector.shape[0] != self.config.vector_dim:
            raise ValueError(
                f"Query vector dim {vector.shape[0]} does not match index dim {self.config.vector_dim}"
            )
        if self._index is None:
            raise RuntimeError("HNSW index failed to load")
        k = min(top_k, len(self._doc_ids))
        labels, distances = self._index.knn_query(np.asarray([vector], dtype=np.float32), k=k)
        results: list[Neighbor] = []
        for rank, (label, distance) in enumerate(zip(labels[0], distances[0], strict=True), start=1):
            score = 1.0 - float(distance)
            results.append(Neighbor(doc_id=self._doc_ids[int(label)], score=score, rank=rank))
        return results
