from __future__ import annotations

import re

from hnsw_logic.core.models import DocRecord


SPACE_RE = re.compile(r"\s+")


class DocumentPreprocessor:
    def normalize(self, docs: list[DocRecord]) -> list[DocRecord]:
        normalized: list[DocRecord] = []
        for doc in docs:
            normalized.append(
                DocRecord(
                    doc_id=doc.doc_id,
                    title=doc.title.strip(),
                    text=SPACE_RE.sub(" ", doc.text).strip(),
                    metadata=doc.metadata,
                )
            )
        return normalized
