from __future__ import annotations

import math

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
        query_strategy_agent=None,
    ):
        self.searcher = searcher
        self.brief_store = brief_store
        self.graph_store = graph_store
        self.scorer = scorer
        self.jump_policy = jump_policy
        self.semantic_memory_store = semantic_memory_store
        self.corpus_store = corpus_store
        self.query_strategy_agent = query_strategy_agent
        self.initial_top_k = jump_policy.config.initial_top_k
        self.supplemental_seed_top_k = jump_policy.config.supplemental_seed_top_k
        self.supplemental_seed_min_score = jump_policy.config.supplemental_seed_min_score
        self.supplemental_seed_weight = jump_policy.config.supplemental_seed_weight
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

    def _apply_memory_bias(self, query: str, rows: list[dict]) -> list[dict]:
        if self.semantic_memory_store is None:
            return rows
        memory = self.semantic_memory_store.read()
        query_terms = set(query.lower().split())
        for row in rows:
            if row["source_kind"] in {"geometric", "supplemental"}:
                continue
            bias = 0.0
            brief = self.brief_store.read(row["doc_id"])
            if brief is None:
                continue
            for entity in brief.entities:
                alias_text = " ".join(memory.aliases.get(entity, []))
                alias_tokens = set(alias_text.lower().split()) | {memory.canonical_entities.get(entity, entity).lower()}
                if query_terms & alias_tokens:
                    bias += 0.05
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
        brief_rows = list(briefs.values())
        if self._sparse_doc_count != len(brief_rows):
            self._sparse.build(brief_rows, self._records_by_id)
            self._sparse_doc_count = len(brief_rows)
        dense_rank_map = {doc_id: rank for rank, (doc_id, _, _) in enumerate(dense_rows, start=1)}
        dense_score_map = {doc_id: score for doc_id, score, _ in dense_rows}
        dense_protected = {doc_id for doc_id, _, _ in dense_rows[: self.novelty_dense_top_k]}
        sparse_score_map = {hit.doc_id: hit.score for hit in sparse_hits}
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
            if query_alignment <= 0.0 and structure_alignment <= 0.0 and sparse_score < 0.55 and score < 0.7:
                continue
            dense_rank = dense_rank_map.get(doc_id)
            dense_score = dense_score_map.get(doc_id, 0.0)
            raw_tokens = self._record_tokens_by_id.get(doc_id, set())
            raw_coverage = (
                min(len(query_tokens & raw_tokens) / max(1, min(len(query_tokens), 6)), 1.0)
                if query_tokens
                else 0.0
            )
            rrf = (0.0 if dense_rank is None else 1.0 / (60.0 + dense_rank)) + 1.0 / (60.0 + sparse_rank)
            rrf_score = min(rrf / max_rrf, 1.0)
            novelty_bonus = 0.08 * novelty_bias if doc_id not in dense_protected else 0.0
            if dense_rank is not None:
                blended = dense_score
            else:
                strong_structure = structure_alignment >= 0.2
                strong_semantic = query_alignment >= 0.28 and score >= 0.35
                strong_raw = raw_coverage >= self.sparse_only_min_raw_coverage and score >= 0.4
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
                ) * self.supplemental_seed_weight * max(agreement_gate, 0.55) * sparse_boost
                novelty_strength = max(raw_coverage, query_alignment, structure_alignment)
                if dense_guard_score > 0.0 and novelty_strength < 0.78:
                    guard_cap = dense_guard_score - 0.02 + 0.09 * novelty_strength + 0.03 * rrf_score
                    blended = min(blended, guard_cap)
            scored.append((blended, doc_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(doc_id, min(score, 0.99)) for score, doc_id in scored[: self.supplemental_seed_top_k]]

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
                target_brief = briefs.get(edge.dst_doc_id)
                if target_brief is None:
                    continue
                target_rel = self.scorer.score_target(query, query_emb, target_brief)
                if target_rel < 0.35:
                    continue
                target_alignment = self.scorer.query_alignment(query, target_brief)
                if target_alignment < 0.24:
                    continue
                relation_multiplier = self.scorer.relation_query_multiplier(query, target_brief, edge)
                bonus = 0.16 * edge.confidence * target_rel * relation_multiplier * min(1.0, 0.55 + source_alignment)
                if edge.dst_doc_id in ranked_ids:
                    bonus *= 1.15
                best_bonus = max(best_bonus, bonus)
            if best_bonus > 0.0:
                row["final_score"] += best_bonus
        ranked.sort(key=lambda item: (-item["final_score"], item["doc_id"]))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        return ranked

    def _query_strategy(self, query: str, dense_rows: list[tuple[str, float, str]], sparse_hits, briefs: dict[str, object]):
        graph_available = bool(self.graph_store.all_edges())
        if self.query_strategy_agent is None:
            if graph_available and self._dataset_hint in {"gl_hnsw_demo", "demo", "project_docs"}:
                return type(
                    "Strategy",
                    (),
                    {
                        "sparse_gate": 0.8,
                        "allow_sparse_only": False,
                        "graph_gate": 1.0,
                        "rationale": "default_graph_first",
                    },
                )()
            return type(
                "Strategy",
                (),
                {"sparse_gate": 1.0, "allow_sparse_only": True, "graph_gate": 1.0 if graph_available else 0.0, "rationale": "default"},
            )()
        return self.query_strategy_agent.run(
            query=query,
            dense_rows=dense_rows,
            sparse_hits=sparse_hits,
            briefs=briefs,
            dataset_hint=self._dataset_hint,
            graph_available=graph_available,
            scorer=self.scorer,
            raw_tokens_by_id=self._record_tokens_by_id,
        )

    def _seed_rows(self, query: str, query_emb, briefs: dict[str, object], dense_rows=None) -> list[tuple[str, float, str]]:
        dense_rows = dense_rows or self._seed_rows_dense(query_emb)
        if self._sparse_doc_count != len(briefs):
            self._sparse.build(list(briefs.values()), self._records_by_id)
            self._sparse_doc_count = len(briefs)
        sparse_hits = self._sparse.search(query, top_k=self.sparse_top_k)
        strategy = self._query_strategy(query, dense_rows, sparse_hits, briefs)
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

        effective_max_seeds = max(1, math.ceil(self.jump_policy.max_seeds * max(0.0, getattr(strategy, "graph_gate", 0.0))))
        effective_max_expansions = max(
            0,
            math.ceil(self.jump_policy.max_expansions_per_seed * max(0.0, getattr(strategy, "graph_gate", 0.0))),
        )
        for doc_id, seed_score, source_kind in seed_rows[:effective_max_seeds]:
            for edge in self.graph_store.get_out_edges(doc_id)[:effective_max_expansions]:
                brief = briefs.get(edge.dst_doc_id)
                if brief is None:
                    continue
                edge_emb = self.scorer.edge_embedding(edge)
                target_rel_score = self.scorer.score_target(query, query_emb, brief)
                if self.jump_policy.allow_jump(query_emb, edge_emb, edge, target_rel_score):
                    expanded.append(
                        ExpandedCandidate(
                            doc_id=edge.dst_doc_id,
                            source_doc_id=doc_id,
                            edge=edge,
                            seed_score=seed_score,
                            edge_match=float(query_emb.dot(edge_emb)),
                            target_rel_score=target_rel_score,
                        )
                    )
        ranked = self.scorer.rank(query, query_emb, seeds, expanded, briefs, top_k)
        ranked = self._apply_graph_neighborhood_bonus(query, query_emb, ranked, briefs)
        ranked = self._retain_dense_top_hits(ranked, briefs, dense_rows, top_k)
        if use_memory_bias:
            ranked = self._apply_memory_bias(query, ranked)
        return self._hits_from_ranked(query, ranked, top_k)
