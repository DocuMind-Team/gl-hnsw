from __future__ import annotations

import numpy as np
import pytest

from hnsw_logic.config.schema import HnswConfig
from hnsw_logic.core.models import DocRecord
from hnsw_logic.core.utils import write_json
from hnsw_logic.docs.loader import DocumentLoader
from hnsw_logic.docs.preprocessor import DocumentPreprocessor
from hnsw_logic.embedding.encoder import EmbeddingEncoder
from hnsw_logic.hnsw.index_builder import HnswIndexBuilder
from hnsw_logic.hnsw.searcher import HnswSearcher


class _StubProvider:
    def __init__(self, vectors: np.ndarray):
        self._vectors = vectors

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self._vectors


def test_document_loader_defaults_title_and_copies_metadata(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "docs.jsonl").write_text(
        '{"doc_id":"alpha-doc","text":"Hello world","metadata":{"topic":"demo"}}\n',
        encoding="utf-8",
    )

    docs = DocumentLoader(raw_dir).load()

    assert docs[0].title == "Alpha Doc"
    assert docs[0].metadata == {"topic": "demo"}
    assert docs[0].metadata is not None


def test_document_loader_raises_on_missing_text(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "docs.jsonl").write_text('{"doc_id":"alpha-doc","title":"Alpha"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Missing `text`"):
        DocumentLoader(raw_dir).load()


def test_document_preprocessor_copies_metadata():
    doc = DocRecord(doc_id="a", title="  Title  ", text="line1 \n line2", metadata={"topic": "demo"})

    normalized = DocumentPreprocessor().normalize([doc])[0]

    assert normalized.title == "Title"
    assert normalized.text == "line1 line2"
    assert normalized.metadata == {"topic": "demo"}
    assert normalized.metadata is not doc.metadata


def test_embedding_encoder_load_validates_payload_shape(tmp_path):
    output_path = tmp_path / "embeddings.json"
    encoder = EmbeddingEncoder(_StubProvider(np.zeros((0, 0), dtype=np.float32)), output_path)

    write_json(output_path, {"doc_ids": ["a", "b"], "vectors": [[0.1, 0.2]]})

    with pytest.raises(ValueError, match="doc ids but 1 vectors"):
        encoder.load()


def test_embedding_encoder_load_returns_empty_matrix_for_missing_vectors(tmp_path):
    output_path = tmp_path / "embeddings.json"
    encoder = EmbeddingEncoder(_StubProvider(np.zeros((0, 0), dtype=np.float32)), output_path)

    write_json(output_path, {"doc_ids": [], "vectors": []})

    doc_ids, vectors = encoder.load()
    assert doc_ids == []
    assert vectors.shape == (0, 0)


def test_embedding_encoder_build_validates_provider_output_row_count(tmp_path):
    output_path = tmp_path / "embeddings.json"
    encoder = EmbeddingEncoder(
        _StubProvider(np.asarray([[1.0, 0.0]], dtype=np.float32)),
        output_path,
    )

    with pytest.raises(ValueError, match="returned 1 vectors for 2 documents"):
        encoder.build(
            [
                DocRecord(doc_id="a", title="A", text="alpha"),
                DocRecord(doc_id="b", title="B", text="beta"),
            ]
        )


def test_hnsw_index_builder_rejects_empty_corpus(tmp_path):
    builder = HnswIndexBuilder(HnswConfig(vector_dim=2), tmp_path / "docs.bin", tmp_path / "docs_meta.json")

    with pytest.raises(ValueError, match="empty corpus"):
        builder.build([], np.empty((0, 2), dtype=np.float32))


def test_hnsw_index_builder_rejects_length_mismatch(tmp_path):
    builder = HnswIndexBuilder(HnswConfig(vector_dim=2), tmp_path / "docs.bin", tmp_path / "docs_meta.json")

    with pytest.raises(ValueError, match="does not match doc id count"):
        builder.build(["a"], np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))


def test_hnsw_searcher_validates_missing_index_files(tmp_path):
    searcher = HnswSearcher(HnswConfig(vector_dim=2), tmp_path / "docs.bin", tmp_path / "docs_meta.json")

    with pytest.raises(FileNotFoundError, match="metadata file not found"):
        searcher.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)


def test_hnsw_searcher_handles_zero_top_k_and_dimension_errors(tmp_path):
    config = HnswConfig(vector_dim=2)
    builder = HnswIndexBuilder(config, tmp_path / "docs.bin", tmp_path / "docs_meta.json")
    builder.build(
        ["a", "b"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    searcher = HnswSearcher(config, tmp_path / "docs.bin", tmp_path / "docs_meta.json")

    assert searcher.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=0) == []

    with pytest.raises(ValueError, match="does not match index dim"):
        searcher.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)
