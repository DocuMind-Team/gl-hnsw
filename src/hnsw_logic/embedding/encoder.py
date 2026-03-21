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
        if embeddings.ndim != 2:
            raise ValueError("Embedding provider must return a 2D array")
        if len(embeddings) != len(docs):
            raise ValueError(
                f"Embedding provider returned {len(embeddings)} vectors for {len(docs)} documents"
            )
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
        doc_ids_raw = payload.get("doc_ids", [])
        vectors_raw = payload.get("vectors", [])
        if not isinstance(doc_ids_raw, list):
            raise ValueError(f"Embedding payload at {self.output_path} has invalid `doc_ids`")
        if not isinstance(vectors_raw, list):
            raise ValueError(f"Embedding payload at {self.output_path} has invalid `vectors`")
        doc_ids = [str(doc_id) for doc_id in doc_ids_raw]
        vectors = np.asarray(vectors_raw, dtype=np.float32)
        if vectors.size == 0:
            vectors = np.empty((0, 0), dtype=np.float32)
        elif vectors.ndim != 2:
            raise ValueError(f"Embedding payload at {self.output_path} must contain a 2D `vectors` array")
        if len(doc_ids) != len(vectors):
            raise ValueError(
                f"Embedding payload at {self.output_path} has {len(doc_ids)} doc ids but {len(vectors)} vectors"
            )
        return doc_ids, vectors
