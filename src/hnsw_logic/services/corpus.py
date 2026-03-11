from __future__ import annotations

from pathlib import Path

from hnsw_logic.core.models import DocRecord
from hnsw_logic.core.utils import append_jsonl, read_jsonl
from hnsw_logic.docs.loader import DocumentLoader
from hnsw_logic.docs.preprocessor import DocumentPreprocessor


class CorpusStore:
    def __init__(self, raw_dir: Path, processed_path: Path):
        self.loader = DocumentLoader(raw_dir)
        self.preprocessor = DocumentPreprocessor()
        self.processed_path = processed_path

    def ingest(self) -> list[DocRecord]:
        docs = self.preprocessor.normalize(self.loader.load())
        if self.processed_path.exists():
            self.processed_path.unlink()
        append_jsonl(self.processed_path, docs)
        return docs

    def read_processed(self) -> list[DocRecord]:
        return [DocRecord(**row) for row in read_jsonl(self.processed_path)]
