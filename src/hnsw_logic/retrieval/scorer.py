from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hnsw_logic.config.schema import RetrievalConfig
from hnsw_logic.core.models import DocBrief, LogicEdge
from hnsw_logic.core.utils import cosine
from hnsw_logic.embedding.provider import ProviderBase


@dataclass(slots=True)
class ExpandedCandidate:
    doc_id: str
    source_doc_id: str
    edge: LogicEdge
    seed_score: float
    edge_match: float
    target_rel_score: float


class RetrievalScorer:
    def __init__(self, provider: ProviderBase, retrieval_config: RetrievalConfig):
        self.provider = provider
        self.alpha = retrieval_config.fusion.alpha
        self.beta = retrieval_config.fusion.beta
        self._brief_embedding_cache: dict[str, np.ndarray] = {}
        self._edge_embedding_cache: dict[str, np.ndarray] = {}

    def encode_query(self, query: str) -> np.ndarray:
        return self.provider.embed_texts([query])[0]

    def score_target(self, query_emb: np.ndarray, brief: DocBrief) -> float:
        target_emb = self._brief_embedding_cache.get(brief.doc_id)
        if target_emb is None:
            target_emb = self.provider.embed_texts([f"{brief.title}\n{brief.summary}"])[0]
            self._brief_embedding_cache[brief.doc_id] = target_emb
        return cosine(query_emb, target_emb)

    def edge_embedding(self, edge: LogicEdge) -> np.ndarray:
        edge_emb = self._edge_embedding_cache.get(edge.edge_card_text)
        if edge_emb is None:
            edge_emb = self.provider.embed_texts([edge.edge_card_text])[0]
            self._edge_embedding_cache[edge.edge_card_text] = edge_emb
        return edge_emb

    def rank(
        self,
        query: str,
        query_emb: np.ndarray,
        seeds: dict[str, tuple[float, str]],
        expanded: list[ExpandedCandidate],
        briefs: dict[str, DocBrief],
        top_k: int,
    ) -> list[dict]:
        merged: dict[str, dict] = {}
        for doc_id, (score_h, source_kind) in seeds.items():
            merged[doc_id] = {
                "doc_id": doc_id,
                "title": briefs[doc_id].title,
                "geometric_score": score_h,
                "logical_score": 0.0,
                "source_kind": source_kind,
                "via_edge": None,
                "summary": briefs[doc_id].summary,
            }
        for candidate in expanded:
            logic_score = candidate.seed_score * candidate.edge.confidence * candidate.edge_match * candidate.target_rel_score
            row = merged.setdefault(
                candidate.doc_id,
                {
                    "doc_id": candidate.doc_id,
                    "title": briefs[candidate.doc_id].title,
                    "geometric_score": 0.0,
                    "logical_score": 0.0,
                    "source_kind": "logic",
                    "via_edge": f"{candidate.source_doc_id}->{candidate.doc_id}",
                    "summary": briefs[candidate.doc_id].summary,
                },
            )
            if logic_score > row["logical_score"]:
                row["logical_score"] = logic_score
                row["via_edge"] = f"{candidate.source_doc_id}->{candidate.doc_id}"
                if row["source_kind"] == "geometric":
                    row["source_kind"] = "hybrid"

        ranked = []
        for row in merged.values():
            logic_weight = self.beta
            if row["source_kind"] == "logic":
                logic_weight *= 0.2
            elif row["source_kind"] == "hybrid":
                logic_weight *= 0.35
            row["final_score"] = self.alpha * row["geometric_score"] + logic_weight * row["logical_score"]
            ranked.append(row)
        ranked.sort(key=lambda item: (-item["final_score"], item["doc_id"]))
        for rank, row in enumerate(ranked[:top_k], start=1):
            row["rank"] = rank
        return ranked[:top_k]
