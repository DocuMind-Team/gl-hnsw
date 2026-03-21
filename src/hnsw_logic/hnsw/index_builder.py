from __future__ import annotations

from pathlib import Path

import numpy as np

from hnsw_logic.config.schema import HnswConfig
from hnsw_logic.core.utils import ensure_dir, write_json


class HnswIndexBuilder:
    def __init__(self, config: HnswConfig, index_path: Path, meta_path: Path):
        self.config = config
        self.index_path = index_path
        self.meta_path = meta_path

    def build(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        if not doc_ids:
            raise ValueError("Cannot build HNSW index for an empty corpus")
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        if len(doc_ids) != vectors.shape[0]:
            raise ValueError(
                f"Embedding row count {vectors.shape[0]} does not match doc id count {len(doc_ids)}"
            )
        if vectors.shape[1] != self.config.vector_dim:
            raise ValueError(f"Embedding dim {vectors.shape[1]} does not match config dim {self.config.vector_dim}")
        ensure_dir(self.index_path.parent)
        import hnswlib

        index = hnswlib.Index(space=self.config.metric, dim=self.config.vector_dim)
        index.init_index(max_elements=len(doc_ids), ef_construction=self.config.ef_construction, M=self.config.m)
        index.add_items(vectors, np.arange(len(doc_ids)))
        index.set_ef(self.config.ef_search)
        index.save_index(str(self.index_path))
        write_json(self.meta_path, {"doc_ids": doc_ids})
