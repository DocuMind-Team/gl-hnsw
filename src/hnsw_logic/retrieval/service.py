from __future__ import annotations

from hnsw_logic.core.models import SearchHit, SearchResponse
from hnsw_logic.docs.brief_store import BriefStore
from hnsw_logic.graph.store import GraphStore
from hnsw_logic.hnsw.searcher import HnswSearcher
from hnsw_logic.memory.semantic_memory import SemanticMemoryStore
from hnsw_logic.retrieval.jump_policy import JumpPolicy
from hnsw_logic.retrieval.scorer import ExpandedCandidate, RetrievalScorer
from hnsw_logic.retrieval.sparse import SparseRetriever
from hnsw_logic.services.corpus import CorpusStore


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
        self.sparse_top_k = jump_policy.config.sparse_top_k
        self.sparse_seed_weight = jump_policy.config.sparse_seed_weight
        self.sparse_min_score = jump_policy.config.sparse_min_score
        self.novelty_dense_top_k = jump_policy.config.novelty_dense_top_k
        self._sparse = SparseRetriever()
        self._sparse_doc_count = -1
        self._records_by_id = {}
        if self.corpus_store is not None:
            try:
                self._records_by_id = {doc.doc_id: doc for doc in self.corpus_store.read_processed()}
            except FileNotFoundError:
                self._records_by_id = {}

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

    def _supplemental_seed_rows(
        self,
        query: str,
        query_emb,
        briefs: dict[str, object],
        dense_rows: list[tuple[str, float, str]],
    ) -> list[tuple[str, float]]:
        brief_rows = list(briefs.values())
        if self._sparse_doc_count != len(brief_rows):
            self._sparse.build(brief_rows, self._records_by_id)
            self._sparse_doc_count = len(brief_rows)
        dense_rank_map = {doc_id: rank for rank, (doc_id, _, _) in enumerate(dense_rows, start=1)}
        dense_score_map = {doc_id: score for doc_id, score, _ in dense_rows}
        dense_protected = {doc_id for doc_id, _, _ in dense_rows[: self.novelty_dense_top_k]}
        sparse_hits = self._sparse.search(query, top_k=self.sparse_top_k)
        sparse_score_map = {hit.doc_id: hit.score for hit in sparse_hits}
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
            rrf = (0.0 if dense_rank is None else 1.0 / (60.0 + dense_rank)) + 1.0 / (60.0 + sparse_rank)
            rrf_score = min(rrf / max_rrf, 1.0)
            novelty_bonus = 0.08 if doc_id not in dense_protected else 0.0
            if dense_rank is not None:
                boost = (
                    0.18 * max(0.0, sparse_score - 0.45)
                    + 0.08 * query_alignment
                    + 0.05 * rrf_score
                    + novelty_bonus
                )
                blended = min(dense_score + boost, 0.99)
            else:
                blended = (
                    0.52 * sparse_score
                    + 0.28 * max(score, query_alignment)
                    + 0.12 * rrf_score
                    + novelty_bonus
                ) * self.supplemental_seed_weight
            scored.append((blended, doc_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(doc_id, min(score, 0.99)) for score, doc_id in scored[: self.supplemental_seed_top_k]]

    def _seed_rows_dense(self, query_emb) -> list[tuple[str, float, str]]:
        seed_neighbors = self.searcher.search(query_emb, top_k=self.initial_top_k)
        return [(neighbor.doc_id, neighbor.score, "geometric") for neighbor in seed_neighbors]

    def _seed_rows(self, query: str, query_emb, briefs: dict[str, object]) -> list[tuple[str, float, str]]:
        dense_rows = self._seed_rows_dense(query_emb)
        merged: dict[str, tuple[float, str]] = {
            doc_id: (score, source_kind) for doc_id, score, source_kind in dense_rows
        }
        for doc_id, score in self._supplemental_seed_rows(query, query_emb, briefs, dense_rows):
            current = merged.get(doc_id)
            if current is None or score > current[0]:
                merged[doc_id] = (score, "supplemental")
        rows = [(doc_id, score, source_kind) for doc_id, (score, source_kind) in merged.items()]
        rows.sort(key=lambda item: (-item[1], item[0]))
        return rows

    def search_baseline(self, query: str, top_k: int = 10) -> SearchResponse:
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
        query_emb = self.scorer.encode_query(query)
        briefs = {brief.doc_id: brief for brief in self.brief_store.all()}
        seed_rows = self._seed_rows(query, query_emb, briefs)
        seeds = {doc_id: (score, source_kind) for doc_id, score, source_kind in seed_rows}
        expanded: list[ExpandedCandidate] = []

        for doc_id, seed_score, source_kind in seed_rows[: self.jump_policy.max_seeds]:
            for edge in self.graph_store.get_out_edges(doc_id)[: self.jump_policy.max_expansions_per_seed]:
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
        if use_memory_bias:
            ranked = self._apply_memory_bias(query, ranked)
        return self._hits_from_ranked(query, ranked, top_k)
