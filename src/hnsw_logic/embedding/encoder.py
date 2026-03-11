from __future__ import annotations

from pathlib import Path

import numpy as np

from hnsw_logic.core.models import DocRecord
from hnsw_logic.core.utils import read_json, write_json
from hnsw_logic.embedding.provider import ProviderBase


class EmbeddingEncoder:
    def __init__(self, provider: ProviderBase, output_path: Path):
        self.provider = provider
        self.output_path = output_path

    def build(self, docs: list[DocRecord]) -> np.ndarray:
        embeddings = self.provider.embed_texts([f"{doc.title}\n{doc.text}" for doc in docs])
        write_json(
            self.output_path,
            {
                "doc_ids": [doc.doc_id for doc in docs],
                "vectors": embeddings.tolist(),
                "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            },
        )
        return embeddings

    def load(self) -> tuple[list[str], np.ndarray]:
        payload = read_json(self.output_path, default={})
        doc_ids = payload.get("doc_ids", [])
        vectors = np.asarray(payload.get("vectors", []), dtype=np.float32)
        return doc_ids, vectors
