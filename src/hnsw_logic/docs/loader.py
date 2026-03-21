from __future__ import annotations

from pathlib import Path

from hnsw_logic.core.models import DocRecord
from hnsw_logic.core.utils import read_jsonl


class DocumentLoader:
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def load(self) -> list[DocRecord]:
        docs: list[DocRecord] = []
        for path in sorted(self.raw_dir.glob("*")):
            if path.suffix == ".jsonl":
                docs.extend(self._load_jsonl(path))
            elif path.suffix in {".md", ".txt"}:
                docs.append(
                    DocRecord(
                        doc_id=path.stem,
                        title=path.stem.replace("-", " ").title(),
                        text=path.read_text(encoding="utf-8"),
                        metadata={"source_path": str(path.relative_to(self.raw_dir))},
                    )
                )
        return docs

    def _load_jsonl(self, path: Path) -> list[DocRecord]:
        rows = read_jsonl(path)
        docs: list[DocRecord] = []
        for line_no, row in enumerate(rows, start=1):
            doc_id = str(row.get("doc_id", "")).strip()
            text = str(row.get("text", "")).strip()
            if not doc_id:
                raise ValueError(f"Missing `doc_id` in {path}:{line_no}")
            if not text:
                raise ValueError(f"Missing `text` in {path}:{line_no} for doc `{doc_id}`")
            title = str(row.get("title", "")).strip() or doc_id.replace("-", " ").title()
            metadata = row.get("metadata", {})
            docs.append(
                DocRecord(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    metadata=dict(metadata) if isinstance(metadata, dict) else {"raw_metadata": metadata},
                )
            )
        return docs
