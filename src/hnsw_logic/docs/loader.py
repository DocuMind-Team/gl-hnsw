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
        return [
            DocRecord(
                doc_id=row["doc_id"],
                title=row["title"],
                text=row["text"],
                metadata=row.get("metadata", {}),
            )
            for row in rows
        ]
