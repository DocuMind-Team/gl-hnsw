from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hnsw_logic.config.schema import RetrievalConfig
from hnsw_logic.core.facets import build_search_views
from hnsw_logic.core.models import DocBrief, LogicEdge
from hnsw_logic.core.utils import cosine, tokenize
from hnsw_logic.embedding.provider import ProviderBase


@dataclass(slots=True)
class ExpandedCandidate:
    doc_id: str
    source_doc_id: str
    edge: LogicEdge
    seed_score: float
    edge_match: float
    target_rel_score: float
    edge_query_alignment: float
    activation_match: float


class RetrievalScorer:
    def __init__(self, provider: ProviderBase, retrieval_config: RetrievalConfig):
        self.provider = provider
        self.alpha = retrieval_config.fusion.alpha
        self.beta = retrieval_config.fusion.beta
        self._brief_embedding_cache: dict[tuple[str, str], np.ndarray] = {}
        self._edge_embedding_cache: dict[str, np.ndarray] = {}
        self._view_cache: dict[str, dict[str, str]] = {}

    def _brief_views(self, brief: DocBrief) -> dict[str, str]:
        cached = self._view_cache.get(brief.doc_id)
        if cached is not None:
            return cached
        views = build_search_views(brief)
        self._view_cache[brief.doc_id] = views
        return views

    def _view_embedding(self, brief: DocBrief, view_name: str) -> np.ndarray:
        key = (brief.doc_id, view_name)
        cached = self._brief_embedding_cache.get(key)
        if cached is not None:
            return cached
        views = self._brief_views(brief)
        text = views.get(view_name, "") or views.get("full", "")
        cached = self.provider.embed_texts([text])[0]
        self._brief_embedding_cache[key] = cached
        return cached

    def preload_views(self, briefs: list[DocBrief], view_names: tuple[str, ...]) -> None:
        for view_name in view_names:
            missing = [brief for brief in briefs if (brief.doc_id, view_name) not in self._brief_embedding_cache]
            if not missing:
                continue
            texts = []
            for brief in missing:
                views = self._brief_views(brief)
                texts.append(views.get(view_name, "") or views.get("full", ""))
            embeddings = self.provider.embed_texts(texts)
            for brief, embedding in zip(missing, embeddings):
                self._brief_embedding_cache[(brief.doc_id, view_name)] = embedding

    def _query_tokens(self, query: str) -> set[str]:
        return {token for token in tokenize(query) if len(token) > 2}

    def _specific_query_tokens(self, query: str) -> set[str]:
        query_tokens = self._query_tokens(query)
        specific = {
            token
            for token in query_tokens
            if len(token) >= 7 or any(char.isdigit() for char in token)
        }
        if specific:
            return specific
        if len(query_tokens) <= 3:
            return {token for token in query_tokens if len(token) >= 5}
        return set()

    def query_specificity(self, query: str) -> float:
        query_tokens = list(self._query_tokens(query))
        if not query_tokens:
            return 0.0
        token_count = len(query_tokens)
        if token_count <= 1:
            short_query = 1.0
        elif token_count == 2:
            short_query = 0.8
        elif token_count == 3:
            short_query = 0.55
        else:
            short_query = 0.3
        content_rich = min(sum(1 for token in query_tokens if len(token) >= 5) / len(query_tokens), 1.0)
        alpha_numeric = min(
            sum(1 for token in query_tokens if any(char.isdigit() for char in token)) / len(query_tokens),
            1.0,
        )
        return min(1.0, 0.85 * short_query + 0.1 * content_rich + 0.05 * alpha_numeric)

    def encode_query(self, query: str) -> np.ndarray:
        return self.provider.embed_texts([query])[0]

    def query_alignment(self, query: str, brief: DocBrief) -> float:
        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return 0.0
        views = self._brief_views(brief)
        title_tokens = {token for token in tokenize(brief.title) if len(token) > 2}
        relation_tokens = {token for token in tokenize(views["relation"]) if len(token) > 2}
        claim_tokens = {token for token in tokenize(views["claims"]) if len(token) > 2}
        summary_tokens = {token for token in tokenize(brief.summary) if len(token) > 2}
        structure_tokens = {token for token in tokenize(views["structure"]) if len(token) > 2}
        overlap = query_tokens & (title_tokens | relation_tokens | claim_tokens | summary_tokens | structure_tokens)
        if not overlap:
            return 0.0
        title_overlap = len(query_tokens & title_tokens)
        relation_overlap = len(query_tokens & relation_tokens)
        claim_overlap = len(query_tokens & claim_tokens)
        summary_overlap = len(query_tokens & summary_tokens)
        structure_overlap = len(query_tokens & structure_tokens)
        score = (
            0.34 * min(title_overlap / max(1, min(len(query_tokens), 3)), 1.0)
            + 0.24 * min(relation_overlap / max(1, min(len(query_tokens), 4)), 1.0)
            + 0.18 * min(claim_overlap / max(1, min(len(query_tokens), 4)), 1.0)
            + 0.16 * min(summary_overlap / max(1, min(len(query_tokens), 4)), 1.0)
            + 0.08 * min(structure_overlap / max(1, min(len(query_tokens), 3)), 1.0)
        )
        return min(score, 1.0)

    def structure_alignment(self, query: str, brief: DocBrief) -> float:
        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return 0.0
        structure_tokens = {token for token in tokenize(self._brief_views(brief)["structure"]) if len(token) > 2}
        if not structure_tokens:
            return 0.0
        overlap = query_tokens & structure_tokens
        if not overlap:
            return 0.0
        return min(len(overlap) / max(1, min(len(query_tokens), 3)), 1.0)

    def title_claim_alignment(self, query: str, brief: DocBrief) -> float:
        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return 0.0
        views = self._brief_views(brief)
        title_tokens = {token for token in tokenize(brief.title) if len(token) > 2}
        claim_tokens = {token for token in tokenize(views["claims"]) if len(token) > 2}
        title_overlap = len(query_tokens & title_tokens)
        claim_overlap = len(query_tokens & claim_tokens)
        score = (
            0.62 * min(title_overlap / max(1, min(len(query_tokens), 2)), 1.0)
            + 0.38 * min(claim_overlap / max(1, min(len(query_tokens), 3)), 1.0)
        )
        return min(score, 1.0)

    def title_alignment(self, query: str, brief: DocBrief) -> float:
        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return 0.0
        title_tokens = {token for token in tokenize(brief.title) if len(token) > 2}
        overlap = query_tokens & title_tokens
        if not overlap:
            return 0.0
        return min(len(overlap) / max(1, min(len(query_tokens), 2)), 1.0)

    def seed_score(self, query: str, query_emb: np.ndarray, brief: DocBrief) -> float:
        title_emb = self._view_embedding(brief, "title")
        relation_emb = self._view_embedding(brief, "relation")
        claims_emb = self._view_embedding(brief, "claims")
        full_emb = self._view_embedding(brief, "full")
        dense_score = max(
            0.78 * cosine(query_emb, title_emb),
            0.92 * cosine(query_emb, relation_emb),
            0.88 * cosine(query_emb, claims_emb),
            cosine(query_emb, full_emb),
        )
        lexical_alignment = self.query_alignment(query, brief)
        structure_alignment = self.structure_alignment(query, brief)
        return 0.58 * dense_score + 0.28 * lexical_alignment + 0.14 * structure_alignment

    def score_target(self, query: str, query_emb: np.ndarray, brief: DocBrief) -> float:
        title_emb = self._view_embedding(brief, "title")
        summary_emb = self._view_embedding(brief, "summary")
        claims_emb = self._view_embedding(brief, "claims")
        relation_emb = self._view_embedding(brief, "relation")
        full_emb = self._view_embedding(brief, "full")
        dense_score = max(
            0.84 * cosine(query_emb, title_emb),
            0.94 * cosine(query_emb, summary_emb),
            cosine(query_emb, claims_emb),
            0.96 * cosine(query_emb, relation_emb),
            cosine(query_emb, full_emb),
        )
        lexical_alignment = self.query_alignment(query, brief)
        structure_alignment = self.structure_alignment(query, brief)
        return 0.68 * dense_score + 0.22 * lexical_alignment + 0.1 * structure_alignment

    @staticmethod
    def _edge_activation_profile(edge: LogicEdge) -> dict[str, object]:
        profile = getattr(edge, "activation_profile", {}) or {}
        return profile if isinstance(profile, dict) else {}

    def edge_use_cases(self, edge: LogicEdge) -> set[str]:
        profile = self._edge_activation_profile(edge)
        use_cases = {str(item).strip().lower() for item in profile.get("edge_use_cases", []) or [] if str(item).strip()}
        if not use_cases and edge.relation_type == "same_concept":
            use_cases.add("concept-bridge")
        if not use_cases and edge.relation_type == "comparison":
            use_cases.add("same-topic-contrast")
        return use_cases

    def is_concept_bridge(self, edge: LogicEdge) -> bool:
        return "concept-bridge" in self.edge_use_cases(edge)

    def edge_activation_prior(self, edge: LogicEdge) -> float:
        profile = self._edge_activation_profile(edge)
        prior = profile.get("activation_prior")
        if prior is None:
            return max(0.0, min(getattr(edge, "utility_score", edge.confidence), 1.0))
        return max(0.0, min(float(prior), 1.0))

    def edge_drift_risk(self, edge: LogicEdge) -> float:
        profile = self._edge_activation_profile(edge)
        drift = profile.get("drift_risk")
        if drift is None:
            return 0.0
        return max(0.0, min(float(drift), 1.0))

    def activation_match(self, query: str, brief: DocBrief, edge: LogicEdge) -> float:
        profile = self._edge_activation_profile(edge)
        if not profile:
            return 0.0
        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return 0.0
        surface_terms = {token for token in tokenize(" ".join(profile.get("query_surface_terms", []) or [])) if len(token) > 2}
        topic_terms = {token for token in tokenize(" ".join(profile.get("topic_signature", []) or [])) if len(token) > 2}
        use_cases = {str(item).strip().lower() for item in profile.get("edge_use_cases", []) or [] if str(item).strip()}
        negative_patterns = {str(item).strip().lower() for item in profile.get("negative_patterns", []) or [] if str(item).strip()}
        surface_overlap = min(len(query_tokens & surface_terms) / max(1, min(len(query_tokens), 4)), 1.0) if surface_terms else 0.0
        topic_overlap = min(len(query_tokens & topic_terms) / max(1, min(len(query_tokens), 3)), 1.0) if topic_terms else 0.0
        lexical_alignment = self.query_alignment(query, brief)
        title_claim = self.title_claim_alignment(query, brief)
        use_case_bonus = 0.0
        if "same-topic-contrast" in use_cases and query_tokens & {"contrast", "compare", "comparison", "versus", "against", "debate"}:
            use_case_bonus = 0.16
        elif "claim-support" in use_cases and query_tokens & {"evidence", "support", "claim", "study", "finding"}:
            use_case_bonus = 0.12
        elif "concept-bridge" in use_cases and query_tokens & {"concept", "overview", "family", "related"}:
            use_case_bonus = 0.1
        elif "mechanism-detail" in use_cases and query_tokens & {"how", "mechanism", "detail", "implementation", "score", "fusion"}:
            use_case_bonus = 0.12
        score = (
            0.3 * surface_overlap
            + 0.2 * topic_overlap
            + 0.3 * lexical_alignment
            + 0.2 * title_claim
            + use_case_bonus
        )
        if "topic_drift" in negative_patterns:
            score *= 0.82
        if "near_duplicate" in negative_patterns and surface_overlap < 0.3:
            score *= 0.9
        return max(0.0, min(score, 1.0))

    def activation_multiplier(self, query: str, brief: DocBrief, edge: LogicEdge) -> float:
        activation_match = self.activation_match(query, brief, edge)
        activation_prior = self.edge_activation_prior(edge)
        drift_penalty = 1.0 - 0.35 * self.edge_drift_risk(edge)
        if not self._edge_activation_profile(edge):
            activation_match = max(
                activation_match,
                0.55 * self.query_alignment(query, brief) + 0.45 * self.title_claim_alignment(query, brief),
            )
        return max(0.15, min(1.1, (0.35 + 0.4 * activation_match + 0.25 * activation_prior) * drift_penalty))

    def edge_embedding(self, edge: LogicEdge) -> np.ndarray:
        edge_emb = self._edge_embedding_cache.get(edge.edge_card_text)
        if edge_emb is None:
            edge_emb = self.provider.embed_texts([edge.edge_card_text])[0]
            self._edge_embedding_cache[edge.edge_card_text] = edge_emb
        return edge_emb

    def edge_query_alignment(self, query: str, edge: LogicEdge, target_brief: DocBrief) -> float:
        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return 0.0
        edge_tokens = {token for token in tokenize(edge.edge_card_text) if len(token) > 2}
        evidence_tokens = {
            token
            for span in edge.evidence_spans
            for token in tokenize(span)
            if len(token) > 2
        }
        title_claim = self.title_claim_alignment(query, target_brief)
        structural = self.structure_alignment(query, target_brief)
        edge_overlap = len(query_tokens & edge_tokens)
        evidence_overlap = len(query_tokens & evidence_tokens)
        score = (
            0.28 * min(edge_overlap / max(1, min(len(query_tokens), 2)), 1.0)
            + 0.22 * min(evidence_overlap / max(1, min(len(query_tokens), 2)), 1.0)
            + 0.34 * title_claim
            + 0.16 * structural
        )
        return min(score, 1.0)

    def specific_query_overlap(self, query: str, brief: DocBrief, edge: LogicEdge | None = None) -> float:
        specific_tokens = self._specific_query_tokens(query)
        if not specific_tokens:
            return 0.0
        views = self._brief_views(brief)
        brief_tokens = {
            token
            for token in tokenize(" ".join([brief.title, brief.summary, views["claims"], views["relation"], views["structure"]]))
            if len(token) > 2
        }
        edge_tokens: set[str] = set()
        if edge is not None:
            edge_tokens = {
                token
                for token in tokenize(edge.edge_card_text + " " + " ".join(edge.evidence_spans))
                if len(token) > 2
            }
        overlap = specific_tokens & (brief_tokens | edge_tokens)
        if not overlap:
            return 0.0
        title_tokens = {token for token in tokenize(brief.title) if len(token) > 2}
        title_overlap = specific_tokens & title_tokens
        score = (
            0.7 * min(len(overlap) / max(1, len(specific_tokens)), 1.0)
            + 0.3 * min(len(title_overlap) / max(1, len(specific_tokens)), 1.0)
        )
        return min(score, 1.0)

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
                "edge_relation": None,
                "edge_utility": 0.0,
                "edge": None,
                "edge_activation_match": 0.0,
            }
        for candidate in expanded:
            target_brief = briefs[candidate.doc_id]
            relation_multiplier = self.activation_multiplier(query, target_brief, candidate.edge)
            edge_utility = max(0.0, min(getattr(candidate.edge, "utility_score", candidate.edge.confidence), 1.0))
            activation_prior = self.edge_activation_prior(candidate.edge)
            drift_penalty = 1.0 - 0.3 * self.edge_drift_risk(candidate.edge)
            utility_multiplier = 0.5 + 0.5 * edge_utility
            edge_activation_match = max(candidate.activation_match, self.activation_match(query, target_brief, candidate.edge))
            edge_alignment = max(candidate.edge_query_alignment, self.query_alignment(query, target_brief), edge_activation_match)
            logic_score = (
                candidate.seed_score
                * candidate.edge.confidence
                * (0.42 + 0.58 * utility_multiplier)
                * (0.38 + 0.62 * min(candidate.edge_match, 1.0))
                * (0.38 + 0.62 * min(candidate.target_rel_score, 1.0))
                * (0.35 + 0.65 * relation_multiplier)
                * (0.35 + 0.65 * edge_alignment)
                * (0.45 + 0.55 * activation_prior)
                * drift_penalty
            )
            row = merged.setdefault(
                candidate.doc_id,
                {
                    "doc_id": candidate.doc_id,
                    "title": target_brief.title,
                    "geometric_score": 0.0,
                    "logical_score": 0.0,
                    "source_kind": "logic",
                    "via_edge": f"{candidate.source_doc_id}->{candidate.doc_id}",
                    "summary": target_brief.summary,
                    "edge_relation": None,
                    "edge_utility": 0.0,
                    "edge": None,
                    "edge_activation_match": 0.0,
                },
            )
            if logic_score > row["logical_score"]:
                row["logical_score"] = logic_score
                row["via_edge"] = f"{candidate.source_doc_id}->{candidate.doc_id}"
                row["edge_relation"] = candidate.edge.relation_type
                row["edge_utility"] = edge_utility
                row["edge"] = candidate.edge
                row["edge_activation_match"] = edge_activation_match
                if row["source_kind"] == "geometric":
                    row["source_kind"] = "hybrid"

        ranked = []
        for row in merged.values():
            logic_weight = self.beta
            edge_utility = max(0.0, min(float(row.get("edge_utility", 0.0)), 1.0))
            edge = row.get("edge")
            activation_prior = self.edge_activation_prior(edge) if edge is not None else 0.0
            edge_activation_match = max(0.0, min(float(row.get("edge_activation_match", 0.0)), 1.0))
            drift_penalty = 1.0 - 0.25 * self.edge_drift_risk(edge) if edge is not None else 1.0
            if row["source_kind"] == "logic":
                logic_weight *= 0.55 + 0.6 * edge_utility + 0.45 * activation_prior + 0.65 * edge_activation_match
            elif row["source_kind"] == "hybrid":
                logic_weight *= 0.42 + 0.48 * edge_utility + 0.32 * activation_prior + 0.28 * edge_activation_match
            logic_weight *= drift_penalty
            activation_bonus = 0.0
            if edge is not None:
                activation_bonus = 0.3 * activation_prior * edge_activation_match * (0.6 + 0.4 * edge_utility)
                if row["source_kind"] == "logic":
                    activation_bonus += 0.22 * max(0.0, edge_activation_match - 0.55)
            row["final_score"] = self.alpha * row["geometric_score"] + logic_weight * row["logical_score"] + activation_bonus
            ranked.append(row)
        ranked.sort(key=lambda item: (-item["final_score"], item["doc_id"]))
        for rank, row in enumerate(ranked[:top_k], start=1):
            row["rank"] = rank
        return ranked[:top_k]
