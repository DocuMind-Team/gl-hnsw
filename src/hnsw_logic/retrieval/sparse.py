from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log

from hnsw_logic.core.facets import build_search_views
from hnsw_logic.core.models import DocBrief, DocRecord
from hnsw_logic.core.utils import tokenize


@dataclass(slots=True)
class SparseHit:
    doc_id: str
    score: float


class SparseRetriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_index: dict[str, Counter[str]] = {}
        self._doc_length: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avg_doc_length = 0.0

    def _doc_terms(self, brief: DocBrief, record: DocRecord | None = None) -> list[str]:
        views = build_search_views(brief)
        parts = [
            views["title"],
            views["title"],
            views["summary"],
            views["claims"],
            views["relation"],
        ]
        if record is not None:
            parts.extend([record.title, record.title, record.text])
        text = " ".join(part for part in parts if part)
        return [token for token in tokenize(text) if len(token) > 2]

    def build(self, briefs: list[DocBrief], records: dict[str, DocRecord] | None = None) -> None:
        self._doc_index = {}
        self._doc_length = {}
        document_frequency: Counter[str] = Counter()
        for brief in briefs:
            terms = self._doc_terms(brief, (records or {}).get(brief.doc_id))
            counts = Counter(terms)
            self._doc_index[brief.doc_id] = counts
            self._doc_length[brief.doc_id] = sum(counts.values())
            for token in counts:
                document_frequency[token] += 1
        doc_count = max(1, len(briefs))
        self._avg_doc_length = sum(self._doc_length.values()) / doc_count
        self._idf = {
            token: log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> list[SparseHit]:
        query_terms = [token for token in tokenize(query) if len(token) > 2]
        if not query_terms or not self._doc_index:
            return []
        scores: list[tuple[float, str]] = []
        for doc_id, counts in self._doc_index.items():
            doc_len = self._doc_length[doc_id]
            score = 0.0
            for token in query_terms:
                tf = counts.get(token, 0)
                if tf <= 0:
                    continue
                idf = self._idf.get(token, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(self._avg_doc_length, 1.0))
                score += idf * (tf * (self.k1 + 1.0)) / max(denom, 1e-6)
            if score > 0.0:
                scores.append((score, doc_id))
        if not scores:
            return []
        scores.sort(key=lambda item: (-item[0], item[1]))
        top = scores[:top_k]
        max_score = top[0][0]
        if max_score <= 0.0:
            return []
        return [SparseHit(doc_id=doc_id, score=score / max_score) for score, doc_id in top]
