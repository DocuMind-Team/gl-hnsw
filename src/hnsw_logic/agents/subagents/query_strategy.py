from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hnsw_logic.core.models import DocBrief
from hnsw_logic.core.utils import append_jsonl, tokenize
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
class QueryCandidateSignal:
    doc_id: str
    title: str
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float
    sparse_score: float
    query_alignment: float
    structure_alignment: float
    raw_coverage: float
    is_novel: bool


@dataclass(slots=True)
class QueryStrategyDecision:
    sparse_gate: float
    allow_sparse_only: bool
    graph_gate: float
    rationale: str
    mode: str = "balanced"
    sparse_boost: float = 1.0
    novelty_bias: float = 1.0
    uncertainty: float = 0.0


class QueryStrategyAgent:
    def __init__(self, provider=None, trace_path: Path | None = None):
        self.provider = provider
        self.trace_path = Path(trace_path) if trace_path else None
        self._cache: dict[tuple[str, str, bool], QueryStrategyDecision] = {}

    def run(
        self,
        query: str,
        dense_rows: list[tuple[str, float, str]],
        sparse_hits: list[SparseHit],
        briefs: dict[str, DocBrief],
        dataset_hint: str,
        graph_available: bool,
        *,
        scorer=None,
        raw_tokens_by_id: dict[str, set[str]] | None = None,
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
        candidates = self._candidate_signals(
            query=query,
            dense_rows=dense_rows,
            sparse_hits=sparse_hits,
            briefs=briefs,
            scorer=scorer,
            raw_tokens_by_id=raw_tokens_by_id or {},
        )
        heuristic = self._heuristic_decision(query_terms, signals, candidates)
        cache_key = (dataset_hint, query.strip().lower(), graph_available)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        payload = {
            "query_preview": " ".join(query.split())[:320],
            "query_terms": sorted(query_terms)[:14],
            "query_length": len(query_terms),
            "signals": asdict(signals),
            "top_candidates": [asdict(candidate) for candidate in candidates[:6]],
        }
        decision = heuristic
        source = "heuristic"
        if self.provider is not None and not self._should_skip_remote(query_terms, signals, heuristic):
            remote_payload = self.provider.plan_query_strategy(payload)
            if remote_payload:
                decision = self._decision_from_payload(remote_payload, heuristic)
                source = "remote"
        elif self.provider is not None:
            source = "heuristic_fast_path"
        self._cache[cache_key] = decision
        self._write_trace(query, payload, decision, source)
        return decision

    def _candidate_signals(
        self,
        *,
        query: str,
        dense_rows: list[tuple[str, float, str]],
        sparse_hits: list[SparseHit],
        briefs: dict[str, DocBrief],
        scorer,
        raw_tokens_by_id: dict[str, set[str]],
    ) -> list[QueryCandidateSignal]:
        dense_rank_map = {doc_id: rank for rank, (doc_id, _, _) in enumerate(dense_rows, start=1)}
        dense_score_map = {doc_id: score for doc_id, score, _ in dense_rows}
        sparse_rank_map = {hit.doc_id: rank for rank, hit in enumerate(sparse_hits, start=1)}
        sparse_score_map = {hit.doc_id: hit.score for hit in sparse_hits}
        candidate_ids: list[str] = []
        for doc_id, _, _ in dense_rows[:6]:
            if doc_id in briefs and doc_id not in candidate_ids:
                candidate_ids.append(doc_id)
        for hit in sparse_hits[:6]:
            if hit.doc_id in briefs and hit.doc_id not in candidate_ids:
                candidate_ids.append(hit.doc_id)
        query_tokens = {token for token in tokenize(query) if len(token) > 2}
        rows: list[QueryCandidateSignal] = []
        for doc_id in candidate_ids:
            brief = briefs[doc_id]
            raw_tokens = raw_tokens_by_id.get(doc_id, set())
            raw_coverage = (
                min(len(query_tokens & raw_tokens) / max(1, min(len(query_tokens), 6)), 1.0)
                if query_tokens
                else 0.0
            )
            query_alignment = scorer.query_alignment(query, brief) if scorer is not None else 0.0
            structure_alignment = scorer.structure_alignment(query, brief) if scorer is not None else 0.0
            rows.append(
                QueryCandidateSignal(
                    doc_id=doc_id,
                    title=brief.title,
                    dense_rank=dense_rank_map.get(doc_id),
                    sparse_rank=sparse_rank_map.get(doc_id),
                    dense_score=float(dense_score_map.get(doc_id, 0.0)),
                    sparse_score=float(sparse_score_map.get(doc_id, 0.0)),
                    query_alignment=float(query_alignment),
                    structure_alignment=float(structure_alignment),
                    raw_coverage=float(raw_coverage),
                    is_novel=doc_id not in dense_rank_map or dense_rank_map.get(doc_id, 999) > 4,
                )
            )
        rows.sort(
            key=lambda item: (
                -max(item.query_alignment, item.raw_coverage, item.sparse_score),
                item.dense_rank or 999,
                item.doc_id,
            )
        )
        return rows

    def _decision_from_payload(self, payload: dict[str, Any], fallback: QueryStrategyDecision) -> QueryStrategyDecision:
        try:
            sparse_gate = max(0.0, min(float(payload.get("sparse_gate", fallback.sparse_gate)), 1.0))
            graph_gate = max(0.0, min(float(payload.get("graph_gate", fallback.graph_gate)), 1.0))
            sparse_boost = max(0.0, min(float(payload.get("sparse_boost", fallback.sparse_boost)), 1.25))
            novelty_bias = max(0.0, min(float(payload.get("novelty_bias", fallback.novelty_bias)), 1.25))
            mode = str(payload.get("mode", fallback.mode) or fallback.mode)
            return QueryStrategyDecision(
                sparse_gate=sparse_gate,
                allow_sparse_only=bool(payload.get("allow_sparse_only", fallback.allow_sparse_only)),
                graph_gate=graph_gate,
                rationale=str(payload.get("reason", fallback.rationale))[:240],
                mode=mode,
                sparse_boost=sparse_boost,
                novelty_bias=novelty_bias,
                uncertainty=max(0.0, min(float(payload.get("uncertainty", fallback.uncertainty)), 1.0)),
            )
        except Exception:
            return fallback

    def _heuristic_decision(
        self,
        query_terms: set[str],
        signals: QueryStrategySignals,
        candidates: list[QueryCandidateSignal],
    ) -> QueryStrategyDecision:
        argument_like = signals.dataset_hint in ARGUMENT_DATASETS or bool(query_terms & ARGUMENT_TERMS)
        scientific_like = signals.dataset_hint in SCIENTIFIC_DATASETS or bool(query_terms & SCIENTIFIC_TERMS)
        technical_like = (
            signals.dataset_hint in TECHNICAL_DATASETS
            or bool(query_terms & TECHNICAL_TERMS)
            or signals.graph_available
        )
        best_candidate = candidates[0] if candidates else None
        best_novel = next((candidate for candidate in candidates if candidate.is_novel), None)

        if argument_like:
            return QueryStrategyDecision(
                sparse_gate=0.0,
                allow_sparse_only=False,
                graph_gate=0.0,
                rationale="argument-like corpus/query, abstain from sparse and graph expansion to avoid lexical drift",
                mode="dense_only",
                sparse_boost=0.0,
                novelty_bias=0.0,
                uncertainty=0.1,
            )
        if scientific_like:
            return QueryStrategyDecision(
                sparse_gate=0.92 if signals.agreement_ratio >= 0.25 or (best_novel and best_novel.raw_coverage >= 0.45) else 0.7,
                allow_sparse_only=True,
                graph_gate=0.0,
                rationale="scientific-style query, allow sparse support when terminology is explicit",
                mode="dense_plus_sparse",
                sparse_boost=1.08,
                novelty_bias=0.95,
                uncertainty=0.18,
            )
        if technical_like:
            return QueryStrategyDecision(
                sparse_gate=0.95,
                allow_sparse_only=True,
                graph_gate=1.0 if signals.graph_available else 0.0,
                rationale="technical/project query, enable sparse and graph support",
                mode="dense_sparse_graph" if signals.graph_available else "dense_plus_sparse",
                sparse_boost=1.0,
                novelty_bias=1.0,
                uncertainty=0.14,
            )

        sparse_gate = 0.28 + 0.42 * signals.agreement_ratio + 0.12 * signals.query_specificity
        if best_candidate is not None:
            sparse_gate += 0.1 * max(best_candidate.query_alignment, best_candidate.raw_coverage)
        allow_sparse_only = bool(
            best_novel
            and best_novel.raw_coverage >= 0.5
            and max(best_novel.query_alignment, best_novel.sparse_score) >= 0.55
        )
        return QueryStrategyDecision(
            sparse_gate=max(0.0, min(sparse_gate, 0.88)),
            allow_sparse_only=allow_sparse_only,
            graph_gate=0.0,
            rationale="general query, use sparse only when it agrees with dense evidence or offers strong novel coverage",
            mode="balanced",
            sparse_boost=0.95,
            novelty_bias=0.85,
            uncertainty=0.3,
        )

    def _should_skip_remote(
        self,
        query_terms: set[str],
        signals: QueryStrategySignals,
        heuristic: QueryStrategyDecision,
    ) -> bool:
        argument_like = signals.dataset_hint in ARGUMENT_DATASETS or bool(query_terms & ARGUMENT_TERMS)
        technical_like = signals.dataset_hint in TECHNICAL_DATASETS or bool(query_terms & TECHNICAL_TERMS)
        if argument_like and heuristic.mode == "dense_only":
            return True
        if technical_like and heuristic.mode == "dense_sparse_graph" and signals.agreement_ratio >= 0.5:
            return True
        return False

    def _write_trace(self, query: str, payload: dict[str, Any], decision: QueryStrategyDecision, source: str) -> None:
        if self.trace_path is None:
            return
        append_jsonl(
            self.trace_path,
            [
                {
                    "query": query,
                    "source": source,
                    "payload": payload,
                    "decision": asdict(decision),
                }
            ],
        )
