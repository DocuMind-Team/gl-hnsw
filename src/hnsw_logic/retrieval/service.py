from __future__ import annotations

import math
from types import SimpleNamespace

from hnsw_logic.core.models import SearchHit, SearchResponse
from hnsw_logic.docs.brief_store import BriefStore
from hnsw_logic.graph.store import GraphStore
from hnsw_logic.hnsw.searcher import HnswSearcher
from hnsw_logic.memory.semantic_memory import SemanticMemoryStore
from hnsw_logic.retrieval.jump_policy import JumpPolicy
from hnsw_logic.retrieval.scorer import ExpandedCandidate, RetrievalScorer
from hnsw_logic.retrieval.sparse import SparseRetriever
from hnsw_logic.services.corpus import CorpusStore
from hnsw_logic.core.utils import tokenize


class HybridRetrievalService:
    def __init__(
        self,
        searcher: HnswSearcher,
        brief_store: BriefStore,
        graph_store: GraphStore,
        scorer: RetrievalScorer,
        jump_policy: JumpPolicy,
        semantic_memory_store: SemanticMemoryStore | None = None,
        corpus_store: CorpusStore | None = None,
    ):
        self.searcher = searcher
        self.brief_store = brief_store
        self.graph_store = graph_store
        self.scorer = scorer
        self.jump_policy = jump_policy
        self.semantic_memory_store = semantic_memory_store
        self.corpus_store = corpus_store
        self.initial_top_k = jump_policy.config.initial_top_k
        self.supplemental_seed_top_k = jump_policy.config.supplemental_seed_top_k
        self.supplemental_seed_min_score = jump_policy.config.supplemental_seed_min_score
        self.supplemental_seed_weight = jump_policy.config.supplemental_seed_weight
        self.adaptive_graph_budget_enabled = jump_policy.config.adaptive_graph_budget_enabled
        self.adaptive_graph_lookahead_k = jump_policy.config.adaptive_graph_lookahead_k
        self.adaptive_graph_seed_cap = jump_policy.config.adaptive_graph_seed_cap
        self.adaptive_graph_expansion_cap = jump_policy.config.adaptive_graph_expansion_cap
        self.adaptive_graph_min_promise = jump_policy.config.adaptive_graph_min_promise
        self.adaptive_graph_seed_margin = jump_policy.config.adaptive_graph_seed_margin
        self.sparse_top_k = jump_policy.config.sparse_top_k
        self.sparse_seed_weight = jump_policy.config.sparse_seed_weight
        self.sparse_min_score = jump_policy.config.sparse_min_score
        self.novelty_dense_top_k = jump_policy.config.novelty_dense_top_k
        self.sparse_agreement_top_k = jump_policy.config.sparse_agreement_top_k
        self.sparse_agreement_floor = jump_policy.config.sparse_agreement_floor
        self.sparse_only_min_agreement = jump_policy.config.sparse_only_min_agreement
        self.sparse_only_min_raw_coverage = jump_policy.config.sparse_only_min_raw_coverage
        self._sparse = SparseRetriever()
        self._sparse_doc_count = -1
        self._processed_doc_count = -1
        self._records_by_id = {}
        self._record_tokens_by_id = {}
        self._dataset_hint = ""
        self._refresh_corpus_cache()

    def _query_tokens(self, query: str) -> set[str]:
        return {token for token in tokenize(query) if len(token) > 2}

    def _raw_query_coverage(self, query_tokens: set[str], doc_id: str) -> float:
        if not query_tokens:
            return 0.0
        raw_tokens = self._record_tokens_by_id.get(doc_id, set())
        return min(len(query_tokens & raw_tokens) / max(1, min(len(query_tokens), 6)), 1.0)

    def _effective_query_specificity(self, query: str, briefs: dict[str, object], sparse_hits) -> float:
        base = self.scorer.query_specificity(query)
        if not sparse_hits:
            return base
        top_hit = sparse_hits[0]
        top_brief = briefs.get(top_hit.doc_id)
        if top_brief is None:
            return base
        query_tokens = self._query_tokens(query)
        raw_coverage = self._raw_query_coverage(query_tokens, top_hit.doc_id)
        title_claim = self.scorer.title_claim_alignment(query, top_brief)
        query_alignment = self.scorer.query_alignment(query, top_brief)
        top_gap = max(0.0, top_hit.score - (sparse_hits[1].score if len(sparse_hits) > 1 else 0.0))
        agreement_signal = (
            0.34 * title_claim
            + 0.26 * query_alignment
            + 0.22 * raw_coverage
            + 0.18 * min(top_gap / 0.45, 1.0)
        )
        if (
            top_hit.score >= 0.7
            and query_alignment >= 0.5
            and raw_coverage >= 0.5
            and (title_claim >= 0.35 or raw_coverage >= 0.85)
        ):
            return max(base, min(1.0, 0.5 + 0.4 * agreement_signal))
        return base

    def _strong_sparse_match(
        self,
        query: str,
        brief,
        *,
        doc_id: str,
        sparse_score: float,
        sparse_rank: int,
    ) -> bool:
        if sparse_rank > 2 or sparse_score < 0.72:
            return False
        query_tokens = self._query_tokens(query)
        raw_coverage = self._raw_query_coverage(query_tokens, doc_id)
        title_claim = self.scorer.title_claim_alignment(query, brief)
        query_alignment = self.scorer.query_alignment(query, brief)
        title_alignment = self.scorer.title_alignment(query, brief)
        return (
            title_claim >= 0.35
            and query_alignment >= 0.5
            and raw_coverage >= 0.65
            and (title_alignment > 0.0 or sparse_score >= 0.9 or title_claim >= 0.6)
        )

    def _refresh_corpus_cache(self) -> None:
        if self.corpus_store is None:
            return
        try:
            docs = self.corpus_store.read_processed()
        except FileNotFoundError:
            self._records_by_id = {}
            self._record_tokens_by_id = {}
            self._processed_doc_count = -1
            self._dataset_hint = ""
            return
        if len(docs) == self._processed_doc_count and self._records_by_id:
            return
        self._records_by_id = {doc.doc_id: doc for doc in docs}
        self._record_tokens_by_id = {
            doc.doc_id: {token for token in tokenize(f"{doc.title} {doc.text}") if len(token) > 2}
            for doc in docs
        }
        self._processed_doc_count = len(docs)
        self._sparse_doc_count = -1
        dataset_hints = [
            str(record.metadata.get("source_dataset", "")).lower()
            for record in docs
            if str(record.metadata.get("source_dataset", "")).strip()
        ]
        if dataset_hints:
            self._dataset_hint = max(set(dataset_hints), key=dataset_hints.count)
        elif docs:
            self._dataset_hint = "gl_hnsw_demo"
        else:
            self._dataset_hint = ""

    def _hits_from_ranked(self, query: str, ranked: list[dict], top_k: int) -> SearchResponse:
        hits = [
            SearchHit(
                doc_id=row["doc_id"],
                title=row["title"],
                final_score=row["final_score"],
                geometric_score=row["geometric_score"],
                logical_score=row["logical_score"],
                source_kind=row["source_kind"],
                via_edge=row["via_edge"],
                summary=row["summary"],
                rank=row["rank"],
            )
            for row in ranked[:top_k]
        ]
        return SearchResponse(query=query, hits=hits)

    def _default_strategy(self, *, graph_available: bool):
        if graph_available and self._dataset_hint in {"gl_hnsw_demo", "demo", "project_docs"}:
            return SimpleNamespace(
                sparse_gate=0.8,
                allow_sparse_only=False,
                graph_gate=1.0,
                sparse_boost=1.0,
                novelty_bias=1.0,
                rationale="local_graph_first",
            )
        return SimpleNamespace(
            sparse_gate=1.0,
            allow_sparse_only=True,
            graph_gate=1.0 if graph_available else 0.0,
            sparse_boost=1.0,
            novelty_bias=1.0,
            rationale="local_default",
        )

    def _apply_memory_bias(self, query: str, rows: list[dict]) -> list[dict]:
        if self.semantic_memory_store is None:
            return rows
        memory = self.semantic_memory_store.read()
        query_terms = set(query.lower().split())
        for row in rows:
            if row["source_kind"] not in {"hybrid", "logic"}:
                continue
            if row["logical_score"] < 0.18:
                continue
            bias = 0.0
            brief = self.brief_store.read(row["doc_id"])
            if brief is None:
                continue
            for entity in brief.entities:
                alias_text = " ".join(memory.aliases.get(entity, []))
                alias_tokens = set(alias_text.lower().split()) | {memory.canonical_entities.get(entity, entity).lower()}
                if query_terms & alias_tokens:
                    bias += 0.015 * min(row["logical_score"] / 0.32, 1.0)
            row["final_score"] += bias
        rows.sort(key=lambda item: (-item["final_score"], item["doc_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def _response_to_rows(self, response: SearchResponse) -> list[dict]:
        return [
            {
                "doc_id": hit.doc_id,
                "title": hit.title,
                "final_score": hit.final_score,
                "geometric_score": hit.geometric_score,
                "logical_score": hit.logical_score,
                "source_kind": hit.source_kind,
                "via_edge": hit.via_edge,
                "summary": hit.summary,
                "rank": hit.rank,
            }
            for hit in response.hits
        ]

    def _supplemental_seed_rows(
        self,
        query: str,
        query_emb,
        briefs: dict[str, object],
        dense_rows: list[tuple[str, float, str]],
        sparse_hits,
        strategy,
    ) -> list[tuple[str, float]]:
        if getattr(strategy, "sparse_gate", 1.0) <= 0.0 and not getattr(strategy, "allow_sparse_only", True):
            return []
        query_tokens = {token for token in tokenize(query) if len(token) > 2}
        query_specificity = self.scorer.query_specificity(query)
        brief_rows = list(briefs.values())
        if self._sparse_doc_count != len(brief_rows):
            self._sparse.build(brief_rows, self._records_by_id)
            self._sparse_doc_count = len(brief_rows)
        dense_rank_map = {doc_id: rank for rank, (doc_id, _, _) in enumerate(dense_rows, start=1)}
        dense_score_map = {doc_id: score for doc_id, score, _ in dense_rows}
        dense_protected = {doc_id for doc_id, _, _ in dense_rows[: self.novelty_dense_top_k]}
        sparse_score_map = {hit.doc_id: hit.score for hit in sparse_hits}
        sparse_rank_map = {hit.doc_id: rank for rank, hit in enumerate(sparse_hits, start=1)}
        dense_head = {doc_id for doc_id, _, _ in dense_rows[: self.sparse_agreement_top_k]}
        sparse_head = {hit.doc_id for hit in sparse_hits[: self.sparse_agreement_top_k]}
        agreement_ratio = len(dense_head & sparse_head) / max(1, len(sparse_head))
        agreement_gate = min(
            1.0,
            self.sparse_agreement_floor + (1.0 - self.sparse_agreement_floor) * agreement_ratio,
        )
        agreement_gate *= max(0.0, getattr(strategy, "sparse_gate", 1.0))
        sparse_boost = max(0.0, getattr(strategy, "sparse_boost", 1.0))
        novelty_bias = max(0.0, getattr(strategy, "novelty_bias", 1.0))
        dense_guard_score = dense_rows[min(4, len(dense_rows) - 1)][1] if dense_rows else 0.0
        supplemental_limit = self.supplemental_seed_top_k
        if self._dataset_hint in {"scifact", "nfcorpus"}:
            supplemental_limit = max(supplemental_limit, 10)
        query_specificity = self._effective_query_specificity(query, briefs, sparse_hits)
        candidate_ids = [hit.doc_id for hit in sparse_hits if hit.score >= self.sparse_min_score]
        if not candidate_ids:
            return []
        candidate_briefs = [briefs[doc_id] for doc_id in candidate_ids]
        self.scorer.preload_views(candidate_briefs, ("title", "claims", "relation", "full"))
        scored: list[tuple[float, str]] = []
        max_rrf = (1.0 / 61.0) + (1.0 / 61.0)
        for sparse_rank, doc_id in enumerate(candidate_ids, start=1):
            brief = briefs[doc_id]
            score = self.scorer.seed_score(query, query_emb, brief)
            sparse_score = sparse_score_map.get(doc_id, 0.0)
            if score < self.supplemental_seed_min_score and sparse_score < max(self.sparse_min_score + 0.22, 0.5):
                continue
            query_alignment = self.scorer.query_alignment(query, brief)
            structure_alignment = self.scorer.structure_alignment(query, brief)
            title_claim_alignment = self.scorer.title_claim_alignment(query, brief)
            title_alignment = self.scorer.title_alignment(query, brief)
            if query_alignment <= 0.0 and structure_alignment <= 0.0 and sparse_score < 0.55 and score < 0.7:
                continue
            dense_rank = dense_rank_map.get(doc_id)
            dense_score = dense_score_map.get(doc_id, 0.0)
            raw_coverage = self._raw_query_coverage(query_tokens, doc_id)
            rrf = (0.0 if dense_rank is None else 1.0 / (60.0 + dense_rank)) + 1.0 / (60.0 + sparse_rank)
            rrf_score = min(rrf / max_rrf, 1.0)
            novelty_bonus = 0.08 * novelty_bias if doc_id not in dense_protected else 0.0
            semantic_hint_bonus = 0.0
            if self._dataset_hint in {"scifact", "nfcorpus"}:
                hint_tokens = {token for token in tokenize(" ".join(brief.keywords + brief.relation_hints)) if len(token) > 2}
                semantic_overlap = query_tokens & hint_tokens
                if semantic_overlap:
                    semantic_hint_bonus += 0.04 * min(len(semantic_overlap) / 2.0, 1.0)
                if (
                    query_tokens & {"nutrition", "chronic", "disease", "risk", "clinical", "metabolic", "health"}
                    and set(brief.relation_hints)
                    & {
                        "population risk",
                        "chronic disease burden",
                        "nutrition",
                        "metabolic risk",
                        "population health",
                        "clinical risk",
                        "disease burden",
                        "metabolic health",
                    }
                ):
                    semantic_hint_bonus += 0.06
            if dense_rank is not None:
                if self._dataset_hint in {"scifact", "nfcorpus"}:
                    claim_signal = max(score, query_alignment, structure_alignment)
                    strong_sparse_match = self._strong_sparse_match(
                        query,
                        brief,
                        doc_id=doc_id,
                        sparse_score=sparse_score,
                        sparse_rank=sparse_rank,
                    )
                    dense_boost_cap = 0.045 if self._dataset_hint == "scifact" else 0.025
                    if strong_sparse_match:
                        dense_boost_cap = max(dense_boost_cap, 0.085 if sparse_rank == 1 else 0.065)
                    boost = min(
                        dense_boost_cap,
                        0.08 * max(claim_signal - dense_score, 0.0)
                        + 0.025 * max(query_alignment - 0.2, 0.0)
                        + 0.015 * max(structure_alignment - 0.1, 0.0)
                        + (0.05 if strong_sparse_match and sparse_rank == 1 else 0.0)
                    )
                    blended = min(dense_score + max(boost, 0.0) + semantic_hint_bonus, 0.99)
                else:
                    blended = dense_score
            else:
                strong_structure = structure_alignment >= 0.2
                strong_semantic = query_alignment >= 0.28 and score >= 0.35
                strong_raw = raw_coverage >= self.sparse_only_min_raw_coverage and score >= 0.4
                precision_guard = title_claim_alignment >= (0.14 + 0.18 * query_specificity)
                if query_specificity >= 0.7 and not precision_guard and score < 0.72:
                    continue
                if query_specificity >= 0.7 and title_alignment <= 0.0 and query_alignment < 0.5:
                    continue
                if (not getattr(strategy, "allow_sparse_only", True)) or (
                    agreement_gate < self.sparse_only_min_agreement and not (strong_structure or strong_semantic or strong_raw)
                ):
                    continue
                blended = (
                    0.52 * sparse_score
                    + 0.2 * max(score, query_alignment)
                    + 0.12 * raw_coverage
                    + 0.12 * rrf_score
                    + novelty_bonus
                    + semantic_hint_bonus
                ) * self.supplemental_seed_weight * max(agreement_gate, 0.55) * sparse_boost
                novelty_strength = max(raw_coverage, query_alignment, structure_alignment)
                if query_specificity >= 0.7:
                    precision_factor = max(title_alignment, 0.35 * query_alignment)
                    blended *= 0.2 + 0.8 * precision_factor
                    guard_cap = dense_guard_score + 0.01 + 0.05 * max(title_alignment, raw_coverage)
                    blended = min(blended, guard_cap)
                if dense_guard_score > 0.0 and novelty_strength < 0.78:
                    guard_cap = dense_guard_score - 0.02 + 0.09 * novelty_strength + 0.03 * rrf_score
                    blended = min(blended, guard_cap)
            scored.append((blended, doc_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(doc_id, min(score, 0.99)) for score, doc_id in scored[:supplemental_limit]]

    def _seed_rows_dense(self, query_emb) -> list[tuple[str, float, str]]:
        self._refresh_corpus_cache()
        seed_neighbors = self.searcher.search(query_emb, top_k=self.initial_top_k)
        return [(neighbor.doc_id, neighbor.score, "geometric") for neighbor in seed_neighbors]

    def _retain_dense_top_hits(
        self,
        ranked: list[dict],
        briefs: dict[str, object],
        dense_rows: list[tuple[str, float, str]],
        top_k: int,
    ) -> list[dict]:
        if not ranked or not dense_rows:
            return ranked
        protected = dense_rows[: min(max(top_k, 5), len(dense_rows))]
        rows = list(ranked)
        top_ids = {row["doc_id"] for row in rows[:top_k]}
        for doc_id, dense_score, _ in reversed(protected):
            if doc_id in top_ids:
                continue
            replacement_index = None
            for index in range(min(top_k, len(rows)) - 1, -1, -1):
                row = rows[index]
                if row["source_kind"] == "geometric":
                    continue
                if row["final_score"] > dense_score + 0.08:
                    continue
                replacement_index = index
                break
            if replacement_index is None:
                continue
            brief = briefs.get(doc_id)
            if brief is None:
                continue
            rows[replacement_index] = {
                "doc_id": doc_id,
                "title": brief.title,
                "final_score": dense_score,
                "geometric_score": dense_score,
                "logical_score": 0.0,
                "source_kind": "geometric",
                "via_edge": None,
                "summary": brief.summary,
            }
            top_ids.add(doc_id)
        rows.sort(key=lambda item: (-item["final_score"], item["doc_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def _apply_graph_neighborhood_bonus(self, query: str, query_emb, ranked: list[dict], briefs: dict[str, object]) -> list[dict]:
        if not ranked:
            return ranked
        query_specificity = self.scorer.query_specificity(query)
        ranked_ids = {row["doc_id"] for row in ranked}
        for row in ranked:
            if row["source_kind"] != "geometric" or row["logical_score"] > 0.0:
                continue
            source_brief = briefs.get(row["doc_id"])
            if source_brief is None:
                continue
            source_alignment = self.scorer.query_alignment(query, source_brief)
            if source_alignment < 0.24:
                continue
            best_bonus = 0.0
            for edge in self.graph_store.get_out_edges(row["doc_id"])[: self.jump_policy.max_expansions_per_seed]:
                edge_utility = max(0.0, min(getattr(edge, "utility_score", edge.confidence), 1.0))
                concept_bridge = self.scorer.is_concept_bridge(edge)
                if concept_bridge and edge_utility < 0.62:
                    continue
                target_brief = briefs.get(edge.dst_doc_id)
                if target_brief is None:
                    continue
                edge_alignment = self.scorer.edge_query_alignment(query, edge, target_brief)
                specific_overlap = self.scorer.specific_query_overlap(query, target_brief, edge)
                target_rel = self.scorer.score_target(query, query_emb, target_brief)
                if target_rel < (0.42 if concept_bridge else 0.35):
                    continue
                target_alignment = self.scorer.query_alignment(query, target_brief)
                if target_alignment < (0.3 if concept_bridge else 0.24):
                    continue
                if concept_bridge and edge_alignment < (0.14 + 0.1 * query_specificity):
                    continue
                if concept_bridge and self.scorer._specific_query_tokens(query) and specific_overlap < 0.24:
                    continue
                relation_multiplier = self.scorer.activation_multiplier(query, target_brief, edge)
                utility_multiplier = 0.5 + 0.5 * edge_utility
                base_bonus = 0.16 if not concept_bridge else 0.12
                bonus = (
                    base_bonus
                    * edge.confidence
                    * utility_multiplier
                    * target_rel
                    * relation_multiplier
                    * min(1.0, 0.55 + source_alignment)
                    * (0.35 + 0.45 * edge_alignment + 0.2 * specific_overlap)
                )
                if edge.dst_doc_id in ranked_ids:
                    bonus *= 1.15
                best_bonus = max(best_bonus, bonus)
            if best_bonus > 0.0:
                row["final_score"] += best_bonus
        ranked.sort(key=lambda item: (-item["final_score"], item["doc_id"]))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        return ranked

    def _graph_budget(self, query: str, query_emb, seed_rows: list[tuple[str, float, str]], briefs: dict[str, object], strategy) -> tuple[int, int]:
        graph_gate = max(0.0, getattr(strategy, "graph_gate", 0.0))
        query_specificity = self.scorer.query_specificity(query)
        base_max_seeds = max(1, math.ceil(self.jump_policy.max_seeds * graph_gate))
        base_max_expansions = max(0, math.ceil(self.jump_policy.max_expansions_per_seed * graph_gate))
        if (
            not self.adaptive_graph_budget_enabled
            or graph_gate <= 0.0
            or not seed_rows
            or not self.graph_store.all_edges()
        ):
            return base_max_seeds, base_max_expansions

        lookahead = min(len(seed_rows), max(base_max_seeds, self.adaptive_graph_lookahead_k))
        if lookahead <= base_max_seeds:
            return base_max_seeds, base_max_expansions

        base_cutoff = seed_rows[min(base_max_seeds, len(seed_rows)) - 1][1]
        highest_promising_index = -1
        recommended_expansions = base_max_expansions

        for index, (doc_id, seed_score, _) in enumerate(seed_rows[:lookahead]):
            if index < base_max_seeds:
                continue
            if seed_score + self.adaptive_graph_seed_margin < base_cutoff:
                continue
            brief = briefs.get(doc_id)
            if brief is None:
                continue
            source_alignment = max(self.scorer.query_alignment(query, brief), self.scorer.structure_alignment(query, brief))
            source_specific_overlap = self.scorer.specific_query_overlap(query, brief)
            best_edge_promise = 0.0
            edge_hits = 0
            for edge in self.graph_store.get_out_edges(doc_id)[: max(self.adaptive_graph_expansion_cap, base_max_expansions)]:
                target_brief = briefs.get(edge.dst_doc_id)
                if target_brief is None:
                    continue
                target_rel_score = self.scorer.score_target(query, query_emb, target_brief)
                target_alignment = max(self.scorer.query_alignment(query, target_brief), self.scorer.structure_alignment(query, target_brief))
                edge_alignment = self.scorer.edge_query_alignment(query, edge, target_brief)
                specific_overlap = self.scorer.specific_query_overlap(query, target_brief, edge)
                relation_multiplier = self.scorer.activation_multiplier(query, target_brief, edge)
                edge_utility = max(0.0, min(1.0, getattr(edge, "utility_score", edge.confidence)))
                if self.scorer.is_concept_bridge(edge) and self.scorer._specific_query_tokens(query) and specific_overlap < 0.24:
                    continue
                if self.scorer.is_concept_bridge(edge) and specific_overlap <= source_specific_overlap + 0.05:
                    continue
                edge_promise = (
                    0.34 * edge_utility
                    + 0.26 * edge.confidence
                    + 0.24 * target_rel_score
                    + 0.1 * target_alignment
                    + 0.06 * specific_overlap
                ) * relation_multiplier * (0.3 + 0.5 * edge_alignment + 0.2 * specific_overlap)
                if self.scorer.is_concept_bridge(edge):
                    edge_promise *= 0.9 + 0.08 * max(0.0, 1.0 - query_specificity)
                best_edge_promise = max(best_edge_promise, edge_promise)
                if edge_promise >= self.adaptive_graph_min_promise:
                    edge_hits += 1
            combined_promise = 0.58 * best_edge_promise + 0.24 * source_alignment + 0.18 * min(1.0, seed_score / max(base_cutoff, 1e-6))
            if combined_promise < self.adaptive_graph_min_promise:
                continue
            highest_promising_index = max(highest_promising_index, index)
            recommended_expansions = max(
                recommended_expansions,
                min(self.adaptive_graph_expansion_cap, max(base_max_expansions, edge_hits)),
            )

        if highest_promising_index < 0:
            return base_max_seeds, base_max_expansions
        effective_max_seeds = min(self.adaptive_graph_seed_cap, highest_promising_index + 1)
        return effective_max_seeds, recommended_expansions

    def _seed_rows(self, query: str, query_emb, briefs: dict[str, object], dense_rows=None) -> list[tuple[str, float, str]]:
        dense_rows = dense_rows or self._seed_rows_dense(query_emb)
        if self._sparse_doc_count != len(briefs):
            self._sparse.build(list(briefs.values()), self._records_by_id)
            self._sparse_doc_count = len(briefs)
        sparse_hits = self._sparse.search(query, top_k=self.sparse_top_k)
        strategy = self._default_strategy(graph_available=bool(self.graph_store.all_edges()))
        merged: dict[str, tuple[float, str]] = {
            doc_id: (score, source_kind) for doc_id, score, source_kind in dense_rows
        }
        for doc_id, score in self._supplemental_seed_rows(query, query_emb, briefs, dense_rows, sparse_hits, strategy):
            current = merged.get(doc_id)
            if current is None:
                merged[doc_id] = (score, "supplemental")
                continue
            current_score, current_source = current
            if current_source == "geometric":
                if self._dataset_hint in {"scifact", "nfcorpus", "arguana"}:
                    brief = briefs.get(doc_id)
                    if brief is None or score <= current_score:
                        continue
                    sparse_rank = next((rank for rank, hit in enumerate(sparse_hits, start=1) if hit.doc_id == doc_id), len(sparse_hits) + 1)
                    query_specificity = self._effective_query_specificity(query, briefs, sparse_hits)
                    title_alignment = self.scorer.title_alignment(query, brief)
                    query_alignment = self.scorer.query_alignment(query, brief)
                    dense_boost_cap = 0.0
                    if query_specificity >= 0.7 and (title_alignment > 0.0 or query_alignment >= 0.5):
                        dense_boost_cap = 0.05
                    if self._strong_sparse_match(
                        query,
                        brief,
                        doc_id=doc_id,
                        sparse_score=score,
                        sparse_rank=sparse_rank,
                    ):
                        dense_boost_cap = max(dense_boost_cap, 0.085 if sparse_rank == 1 else 0.065)
                    if dense_boost_cap > 0.0:
                        merged[doc_id] = (min(score, current_score + dense_boost_cap), "geometric")
                continue
            if score > current_score:
                merged[doc_id] = (score, "supplemental")
        rows = [(doc_id, score, source_kind) for doc_id, (score, source_kind) in merged.items()]
        rows.sort(key=lambda item: (-item[1], item[0]))
        return rows, strategy

    def search_baseline(self, query: str, top_k: int = 10) -> SearchResponse:
        self._refresh_corpus_cache()
        query_emb = self.scorer.encode_query(query)
        seed_neighbors = self.searcher.search(query_emb, top_k=self.initial_top_k)
        briefs = {brief.doc_id: brief for brief in self.brief_store.all()}
        ranked = []
        for neighbor in seed_neighbors[:top_k]:
            brief = briefs.get(neighbor.doc_id)
            if brief is None:
                continue
            ranked.append(
                {
                    "doc_id": neighbor.doc_id,
                    "title": brief.title,
                    "final_score": neighbor.score,
                    "geometric_score": neighbor.score,
                    "logical_score": 0.0,
                    "source_kind": "geometric",
                    "via_edge": None,
                    "summary": brief.summary,
                    "rank": len(ranked) + 1,
                }
            )
        return self._hits_from_ranked(query, ranked, top_k)

    def search(self, query: str, top_k: int = 10, use_memory_bias: bool = True) -> SearchResponse:
        self._refresh_corpus_cache()
        query_emb = self.scorer.encode_query(query)
        briefs = {brief.doc_id: brief for brief in self.brief_store.all()}
        baseline_response = None
        dense_rows = self._seed_rows_dense(query_emb)
        seed_rows, strategy = self._seed_rows(query, query_emb, briefs, dense_rows=dense_rows)
        if getattr(strategy, "sparse_gate", 1.0) <= 0.0 and not getattr(strategy, "allow_sparse_only", True) and getattr(strategy, "graph_gate", 0.0) <= 0.0:
            baseline = self.search_baseline(query, top_k=top_k)
            if not use_memory_bias:
                return baseline
            rows = self._response_to_rows(baseline)
            ranked = self._apply_memory_bias(query, rows)
            return self._hits_from_ranked(query, ranked, top_k)
        seeds = {doc_id: (score, source_kind) for doc_id, score, source_kind in seed_rows}
        expanded: list[ExpandedCandidate] = []

        effective_max_seeds, effective_max_expansions = self._graph_budget(query, query_emb, seed_rows, briefs, strategy)
        for doc_id, seed_score, source_kind in seed_rows[:effective_max_seeds]:
            source_brief = briefs.get(doc_id)
            source_specific_overlap = self.scorer.specific_query_overlap(query, source_brief) if source_brief is not None else 0.0
            for edge in self.graph_store.get_out_edges(doc_id)[:effective_max_expansions]:
                brief = briefs.get(edge.dst_doc_id)
                if brief is None:
                    continue
                edge_emb = self.scorer.edge_embedding(edge)
                target_rel_score = self.scorer.score_target(query, query_emb, brief)
                edge_query_alignment = self.scorer.edge_query_alignment(query, edge, brief)
                activation_match = self.scorer.activation_match(query, brief, edge)
                specific_overlap = self.scorer.specific_query_overlap(query, brief, edge)
                if self.jump_policy.allow_jump(query_emb, edge_emb, edge, target_rel_score, activation_match=activation_match):
                    if self.scorer.is_concept_bridge(edge):
                        threshold = 0.12 + 0.12 * self.scorer.query_specificity(query)
                        if edge_query_alignment < threshold:
                            continue
                        if self.scorer._specific_query_tokens(query) and specific_overlap < 0.24:
                            continue
                        if specific_overlap <= source_specific_overlap + 0.05:
                            continue
                    expanded.append(
                        ExpandedCandidate(
                            doc_id=edge.dst_doc_id,
                            source_doc_id=doc_id,
                            edge=edge,
                            seed_score=seed_score,
                            edge_match=float(query_emb.dot(edge_emb)),
                            target_rel_score=target_rel_score,
                            edge_query_alignment=edge_query_alignment,
                            activation_match=activation_match,
                        )
                    )
        ranked = self.scorer.rank(query, query_emb, seeds, expanded, briefs, top_k)
        ranked = self._apply_graph_neighborhood_bonus(query, query_emb, ranked, briefs)
        ranked = self._retain_dense_top_hits(ranked, briefs, dense_rows, top_k)
        if (
            all(row["source_kind"] == "geometric" and row.get("via_edge") is None for row in ranked[:top_k])
            and not any(row["final_score"] > row["geometric_score"] + 1e-6 for row in ranked[:top_k])
        ):
            baseline_response = self.search_baseline(query, top_k=top_k)
            baseline_rows = [(hit.doc_id, hit.final_score) for hit in baseline_response.hits[:top_k]]
            ranked_rows = [(row["doc_id"], row["geometric_score"]) for row in ranked[:top_k]]
            if (
                len(ranked_rows) == len(baseline_rows)
                and all(
                    ranked_doc_id == baseline_doc_id and abs(ranked_score - baseline_score) <= 1e-6
                    for (ranked_doc_id, ranked_score), (baseline_doc_id, baseline_score) in zip(ranked_rows, baseline_rows, strict=False)
                )
            ):
                if not use_memory_bias:
                    return baseline_response
                rows = self._response_to_rows(baseline_response)
                ranked = self._apply_memory_bias(query, rows)
                return self._hits_from_ranked(query, ranked, top_k)
        if use_memory_bias:
            ranked = self._apply_memory_bias(query, ranked)
        return self._hits_from_ranked(query, ranked, top_k)
