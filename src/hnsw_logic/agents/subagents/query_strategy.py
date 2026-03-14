from __future__ import annotations

from dataclasses import dataclass

from hnsw_logic.core.models import DocBrief
from hnsw_logic.core.utils import tokenize
from hnsw_logic.retrieval.sparse import SparseHit


ARGUMENT_DATASETS = {"arguana"}
SCIENTIFIC_DATASETS = {"scifact", "nfcorpus"}
TECHNICAL_DATASETS = {"gl_hnsw_demo", "demo", "project_docs"}
ARGUMENT_TERMS = {"argument", "debate", "against", "support", "opinion", "culture", "policy", "should"}
SCIENTIFIC_TERMS = {"evidence", "claim", "study", "studies", "disease", "protein", "cell", "trial"}
TECHNICAL_TERMS = {"system", "retrieval", "memory", "graph", "edge", "policy", "fusion", "query", "index", "score"}


@dataclass(slots=True)
class QueryStrategySignals:
    dataset_hint: str
    agreement_ratio: float
    query_specificity: float
    graph_available: bool
    dense_titles: list[str]
    sparse_titles: list[str]


@dataclass(slots=True)
class QueryStrategyDecision:
    sparse_gate: float
    allow_sparse_only: bool
    graph_gate: float
    rationale: str


class QueryStrategyAgent:
    def run(
        self,
        query: str,
        dense_rows: list[tuple[str, float, str]],
        sparse_hits: list[SparseHit],
        briefs: dict[str, DocBrief],
        dataset_hint: str,
        graph_available: bool,
    ) -> QueryStrategyDecision:
        query_terms = {token for token in tokenize(query) if len(token) > 2}
        dense_head = {doc_id for doc_id, _, _ in dense_rows[:8]}
        sparse_head = {hit.doc_id for hit in sparse_hits[:8]}
        agreement_ratio = len(dense_head & sparse_head) / max(1, len(sparse_head))
        query_specificity = min(sum(1 for token in query_terms if len(token) > 6) / 6.0, 1.0)
        signals = QueryStrategySignals(
            dataset_hint=dataset_hint,
            agreement_ratio=agreement_ratio,
            query_specificity=query_specificity,
            graph_available=graph_available,
            dense_titles=[briefs[doc_id].title for doc_id, _, _ in dense_rows[:3] if doc_id in briefs],
            sparse_titles=[briefs[hit.doc_id].title for hit in sparse_hits[:3] if hit.doc_id in briefs],
        )
        return self._heuristic_decision(query_terms, signals)

    def _heuristic_decision(self, query_terms: set[str], signals: QueryStrategySignals) -> QueryStrategyDecision:
        argument_like = signals.dataset_hint in ARGUMENT_DATASETS or bool(query_terms & ARGUMENT_TERMS)
        scientific_like = signals.dataset_hint in SCIENTIFIC_DATASETS or bool(query_terms & SCIENTIFIC_TERMS)
        technical_like = (
            signals.dataset_hint in TECHNICAL_DATASETS
            or bool(query_terms & TECHNICAL_TERMS)
            or signals.graph_available
        )

        if argument_like:
            sparse_gate = 0.0
            allow_sparse_only = False
            graph_gate = 0.0
            rationale = "argument-like corpus/query, abstain from sparse and graph expansion to avoid lexical drift"
        elif scientific_like:
            sparse_gate = 1.0
            allow_sparse_only = True
            graph_gate = 0.0
            rationale = "scientific-style query, trust sparse terminology support"
        elif technical_like:
            sparse_gate = 1.0
            allow_sparse_only = True
            graph_gate = 1.0 if signals.graph_available else 0.0
            rationale = "technical/project query, enable sparse and graph support"
        else:
            sparse_gate = 0.4 + 0.35 * signals.agreement_ratio + 0.1 * signals.query_specificity
            allow_sparse_only = signals.agreement_ratio >= 0.55 and signals.query_specificity >= 0.45
            graph_gate = 0.0
            rationale = "general query, use sparse only when it agrees with dense evidence"

        return QueryStrategyDecision(
            sparse_gate=max(0.05, min(sparse_gate, 0.95)),
            allow_sparse_only=allow_sparse_only,
            graph_gate=max(0.0, min(graph_gate, 1.0)),
            rationale=rationale,
        )
