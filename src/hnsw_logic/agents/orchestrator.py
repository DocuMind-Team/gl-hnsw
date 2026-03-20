from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from hnsw_logic.agents.tools.skill_signals import SkillSignalRuntime
from hnsw_logic.config.schema import RelationQualityConfig, RetrievalConfig
from hnsw_logic.core.constants import DEFAULT_TIMESTAMP, RELATION_TYPES
from hnsw_logic.core.models import DocBrief, DocRecord, LogicEdge
from hnsw_logic.core.utils import cosine, tokenize
from hnsw_logic.embedding.provider import CandidateProposal, JudgeSignals


ROLE_WORDS = {"role", "roles", "subagent", "subagents", "profiler", "scout", "judge", "curator"}
LISTING_WORDS = {"include", "includes", "including", "role", "roles", "subagent", "subagents"}
DETAIL_CUES = {
    "detail", "formula", "score", "scores", "policy", "fusion", "registry", "storage", "backend",
    "worker", "workers", "report", "metrics", "tracked", "path", "weighted", "combine", "combines",
    "routing", "route", "persistent", "expansion", "gate", "gates",
}
SUPPORT_CUES = {"allow", "allows", "govern", "governs", "control", "controls", "later", "after", "provide", "provides", "enable", "enables", "enter", "enters", "gate", "gates"}
ORDER_CUES = {"before", "after", "first", "then", "prior", "precedes", "depends"}
SPECIFIC_ROLE_TERMS = {"profiler", "scout", "judge", "curator"}
SERVICE_SURFACE_TERMS = {"service", "fastapi", "api", "endpoint", "endpoints", "health", "inspection", "public"}
FOUNDATIONAL_TERMS = {"similarity", "metric", "metrics", "benchmark", "heuristic", "heuristics"}
HIGH_SPECIFICITY_TITLE_TERMS = {"fusion", "policy", "registry", "backend", "profiler", "scout", "judge", "curator", "revalidation"}
MEDIUM_SPECIFICITY_TITLE_TERMS = {"subagents", "memory", "workers", "jobs"}
LOW_SPECIFICITY_TITLE_TERMS = {"retrieval", "graph", "overview", "service", "metrics", "reporting", "dataset", "similarity", "layers", "parameters", "path"}
METHOD_TITLE_TERMS = {
    "comparison", "comparative", "dosimetry", "estimate", "estimates", "estimating", "imaging",
    "measurement", "measurements", "method", "methods", "protocol", "protocols", "review", "reviews",
    "risk", "risks", "volume",
}
OUTCOME_TITLE_TERMS = {
    "association", "associated", "benefit", "benefits", "decrease", "decreases", "effect", "effects",
    "efficacy", "improve", "improves", "outcome", "outcomes", "reduce", "reduces", "response",
    "responses", "risk", "risks", "therapy", "therapies", "treatment", "treatments",
}
CONTENT_STOPWORDS = {
    "this", "that", "with", "from", "into", "through", "used", "using", "only", "when", "then", "than",
    "their", "there", "about", "project", "system", "document", "documents", "candidate", "candidates",
    "results", "result", "where", "while", "without", "after", "before", "under", "specific", "initial",
}
DETAIL_FAMILIES = (
    ("hybrid", {"hybrid", "retrieval", "ranker", "fusion", "score", "scores", "formula", "weighted", "alpha", "beta", "geometric", "logical", "seed", "target"}),
    ("logic", {"logic", "logical", "overlay", "graph", "jump", "policy", "edge", "confidence", "relevance", "expansion", "candidate", "recall", "one-hop"}),
    ("insert", {"insert", "insertion", "hierarchy", "hierarchical", "walk", "walking", "walks", "neighbor", "selection", "heuristic", "heuristics", "points", "traversal"}),
    ("agent", {"deepagents", "subagents", "profiler", "scout", "judge", "curator", "planning", "filesystem", "shell", "memory", "backend", "storage", "persistent", "route", "routing", "semantic", "procedural", "anchor"}),
    ("ops", {"job", "jobs", "worker", "workers", "registry", "sqlite", "build", "revalidation", "queue", "payload", "state", "timestamps", "messages"}),
)
GRAPH_STAGE_TOKENS = {"logic", "overlay", "graph", "sidecar", "relations", "recall"}
POLICY_STAGE_TOKENS = {"jump", "policy", "confidence", "relevance", "target", "seed", "one-hop", "expansion"}
FUSION_STAGE_TOKENS = {"fusion", "ranker", "alpha", "beta", "score", "scores", "weighted", "path", "formula"}
ROLE_STAGE_TOKENS = {"subagents", "role", "roles", "profiler", "scout", "judge", "curator"}
MEMORY_STAGE_TOKENS = {"memory", "memories", "backend", "storage", "persistent", "semantic", "procedural", "anchor"}
OVERVIEW_STAGE_TOKENS = {"hybrid", "retrieval", "deepagents", "overview", "jobs", "workers", "background"}
REGISTRY_STAGE_TOKENS = {"sqlite", "registry", "queue", "payload", "messages", "timestamps"}
SERVICE_STAGE_TOKENS = {"service", "fastapi", "api", "endpoint", "endpoints", "health", "inspection", "public", "search"}
REVALIDATION_STAGE_TOKENS = {"revalidation", "revalidate", "stale", "corpus"}
DISCOVERY_TERMS = {
    "logic", "hybrid", "jump", "candidate", "fusion", "subagent", "subagents", "memory",
    "profiler", "scout", "judge", "curator", "worker", "workers", "job", "jobs", "registry",
    "report", "benchmark", "metrics", "revalidate", "revalidation",
}
ARGUMENT_STAGE_TOKENS = {"argument", "debate", "claim", "claims", "position", "positions", "policy", "counterargument"}
ARGUMENT_COMPARISON_CUES = {"argument", "debate", "contrast", "versus", "vs", "counter", "counterargument", "opposing", "alternative", "position"}
EVAL_METRIC_STAGE_TOKENS = {"ann", "metric", "metrics", "recall", "mrr", "ndcg", "latency", "benchmark", "benchmarking"}
EVAL_REPORT_STAGE_TOKENS = {"report", "reporting", "compare", "comparison", "edge", "precision", "include", "includes"}
SCIENTIFIC_BRIDGE_TERMS = {
    "claim", "evidence", "study", "nutrition", "diet", "dietary", "metabolic", "risk", "risks",
    "burden", "mortality", "exposure", "disease", "diseases", "health", "population", "susceptibility",
}
CLINICAL_BRIDGE_TERMS = {
    "clinical", "condition", "treatment", "therapy", "diagnosis", "symptom", "patient", "patients",
    "disease", "risk", "metabolic", "nutrition", "outcome", "outcomes",
}
GENERIC_TITLE_BRIDGE_TOKENS = {
    "analysis", "approach", "benefit", "benefits", "cancer", "cell", "cells", "child", "children",
    "clinical", "development", "disease", "diseases", "effect", "effects", "exposure", "fatty",
    "health", "human", "humans", "induced", "infant", "infants", "interaction", "maintenance",
    "multiple", "nutrition", "outcome", "outcomes", "patient", "patients", "protein", "regulates",
    "regulation", "response", "responses", "review", "risk", "risks", "service", "signaling",
    "study", "system", "systems", "treatment", "treatments", "visual",
}


@dataclass(slots=True)
class CandidateAssessment:
    candidate_doc_id: str
    accepted: bool
    reject_reason: str
    score: float
    local_support: float
    evidence_quality: float
    relation_type: str
    confidence: float
    edge: LogicEdge | None = None


@dataclass(slots=True)
class LogicOrchestrator:
    doc_profiler: object
    corpus_scout: object
    relation_judge: object
    memory_curator: object
    edge_reviewer: object | None = None
    deepagent: object | None = None
    retrieval_config: RetrievalConfig | None = None
    _embedding_cache: dict[str, Any] | None = None
    _signal_runtime: SkillSignalRuntime | None = None

    def __post_init__(self):
        if self._embedding_cache is None:
            self._embedding_cache = {}
        if self.retrieval_config is None:
            self.retrieval_config = RetrievalConfig()
        if self._signal_runtime is None:
            self._signal_runtime = SkillSignalRuntime()

    def _provider(self):
        return getattr(self.doc_profiler, "provider", None) or getattr(self.relation_judge, "provider", None)

    def _edge_quality(self):
        return self.retrieval_config.edge_quality

    def _embed_brief(self, brief: DocBrief):
        key = brief.doc_id
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        provider = self._provider()
        if provider is None:
            return None
        vector = provider.embed_texts([f"{brief.title}\n{brief.summary}"])[0]
        self._embedding_cache[key] = vector
        return vector

    def _brief_terms(self, brief: DocBrief) -> set[str]:
        return set(brief.keywords + brief.entities + brief.relation_hints + tokenize(brief.title))

    def _brief_text(self, brief: DocBrief) -> str:
        return " ".join([brief.title, brief.summary, *brief.claims]).lower()

    def _embed_text_cached(self, text: str):
        key = f"text::{text}"
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        provider = self._provider()
        if provider is None:
            return None
        vector = provider.embed_texts([text])[0]
        self._embedding_cache[key] = vector
        return vector

    def _surrogate_query_terms(self, brief: DocBrief) -> list[str]:
        prioritized = [
            *tokenize(brief.title),
            *tokenize(" ".join(brief.entities)),
            *tokenize(" ".join(brief.keywords)),
            *tokenize(" ".join(brief.relation_hints)),
            *tokenize(" ".join(brief.claims[:2])),
        ]
        terms: list[str] = []
        seen: set[str] = set()
        for token in prioritized:
            if len(token) <= 3 or token in CONTENT_STOPWORDS:
                continue
            normalized = self._normalize_token(token)
            if normalized in seen:
                continue
            seen.add(normalized)
            terms.append(token)
            if len(terms) >= 10:
                break
        return terms

    def _bridge_information_gain(self, anchor: DocBrief, candidate: DocBrief) -> float:
        cache_key = f"bridge::{anchor.doc_id}::{candidate.doc_id}"
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return float(cached)

        anchor_terms = set(self._surrogate_query_terms(anchor))
        candidate_terms = set(self._surrogate_query_terms(candidate))
        normalized_anchor_terms = self._normalized_tokens(anchor_terms)
        normalized_candidate_terms = self._normalized_tokens(candidate_terms)
        shared_terms = normalized_anchor_terms & normalized_candidate_terms
        novel_terms = normalized_candidate_terms - normalized_anchor_terms
        shared_ratio = min(len(shared_terms) / max(1, min(len(normalized_anchor_terms), len(normalized_candidate_terms), 5)), 1.0)
        novel_ratio = min(len(novel_terms) / max(1, min(len(normalized_candidate_terms), 5)), 1.0)
        balanced_novelty = max(0.0, 1.0 - abs(novel_ratio - 0.35) / 0.45)

        anchor_vec = self._embed_brief(anchor)
        candidate_vec = self._embed_brief(candidate)
        query_texts = [anchor.title]
        surrogate_terms = self._surrogate_query_terms(anchor)
        if surrogate_terms:
            query_texts.append(" ".join(surrogate_terms[:8]))
        alignments: list[tuple[float, float]] = []
        for query_text in query_texts:
            query_vec = self._embed_text_cached(query_text)
            if query_vec is None or anchor_vec is None or candidate_vec is None:
                continue
            alignments.append((max(cosine(query_vec, anchor_vec), 0.0), max(cosine(query_vec, candidate_vec), 0.0)))
        if alignments:
            anchor_alignment = sum(item[0] for item in alignments) / len(alignments)
            candidate_alignment = sum(item[1] for item in alignments) / len(alignments)
        else:
            anchor_alignment = 0.55 * shared_ratio + 0.25 * max(self._title_specificity_score(anchor), 0.0)
            candidate_alignment = 0.55 * shared_ratio + 0.25 * max(self._title_specificity_score(candidate), 0.0)
        bridge_coverage = min(anchor_alignment, candidate_alignment)
        bridge_stability = max(0.0, 1.0 - max(anchor_alignment - candidate_alignment, 0.0))
        specificity_gain = min(max(self._title_specificity_score(candidate), 0.0) / 1.2, 1.0)
        gain = (
            0.38 * bridge_coverage
            + 0.24 * shared_ratio
            + 0.2 * balanced_novelty
            + 0.1 * bridge_stability
            + 0.08 * specificity_gain
        )
        gain = max(0.0, min(gain, 1.0))
        self._embedding_cache[cache_key] = gain
        return gain

    def _near_duplicate_penalty(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float] | None = None) -> float:
        cache_key = f"duplicate::{anchor.doc_id}::{candidate.doc_id}"
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return float(cached)
        title_a = self._normalized_tokens({token for token in tokenize(anchor.title) if len(token) > 2})
        title_b = self._normalized_tokens({token for token in tokenize(candidate.title) if len(token) > 2})
        title_overlap = len(title_a & title_b) / max(1, len(title_a | title_b))
        exact_title_match = 1.0 if " ".join(sorted(title_a)) == " ".join(sorted(title_b)) and title_a else 0.0
        if metrics is None:
            metrics = self._candidate_metrics(anchor, candidate)
        bridge_gain = self._bridge_information_gain(anchor, candidate)
        penalty = 0.0
        if title_overlap >= 0.72:
            penalty += 0.22
        if exact_title_match >= 1.0:
            penalty += 0.18
        if metrics["content_overlap_score"] >= 0.9 and bridge_gain < 0.52:
            penalty += 0.16
        if metrics["mention_score"] >= 0.45 and metrics["novel_term_ratio"] >= 0.9 and bridge_gain < 0.48:
            penalty += 0.08
        contrastive_reuse = self._contrastive_argument_reuse(anchor, candidate, metrics)
        if contrastive_reuse >= 0.56:
            penalty = max(0.0, penalty - (0.18 + 0.14 * contrastive_reuse))
        penalty = max(0.0, min(penalty, 0.55))
        self._embedding_cache[cache_key] = penalty
        return penalty

    def _is_effective_duplicate(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float] | None = None) -> bool:
        title_a = self._normalized_tokens({token for token in tokenize(anchor.title) if len(token) > 2})
        title_b = self._normalized_tokens({token for token in tokenize(candidate.title) if len(token) > 2})
        if not title_a or not title_b:
            return False
        exact_title_match = " ".join(sorted(title_a)) == " ".join(sorted(title_b))
        if not exact_title_match:
            return False
        if metrics is None:
            metrics = self._candidate_metrics(anchor, candidate)
        if self._contrastive_argument_reuse(anchor, candidate, metrics) >= 0.6:
            return False
        return metrics["content_overlap_score"] >= 0.85 and metrics["dense_score"] >= 0.7

    def _source_dataset(self, brief: DocBrief) -> str:
        return str(brief.metadata.get("source_dataset", "")).lower()

    def _topic_cluster(self, brief: DocBrief) -> str:
        return str(brief.metadata.get("topic_cluster", "")).lower()

    def _topic_family(self, brief: DocBrief) -> str:
        return str(brief.metadata.get("topic_family", "")).lower()

    def _stance(self, brief: DocBrief) -> str:
        return str(brief.metadata.get("stance", "")).lower()

    def _is_argumentative_doc(self, brief: DocBrief) -> bool:
        if self._source_dataset(brief) == "arguana":
            return True
        topic = str(brief.metadata.get("topic", "")).lower()
        if topic == "argument":
            return True
        content_terms = self._content_terms(brief)
        return len(content_terms & ARGUMENT_STAGE_TOKENS) >= 2 and "comparison" in set(brief.relation_hints)

    def _is_argumentative_pair(self, anchor: DocBrief, candidate: DocBrief) -> bool:
        return self._is_argumentative_doc(anchor) and self._is_argumentative_doc(candidate)

    def _content_terms(self, brief: DocBrief) -> set[str]:
        tokens = {
            *tokenize(brief.title),
            *tokenize(brief.summary),
            *tokenize(" ".join(brief.claims)),
            *tokenize(" ".join(brief.keywords)),
            *tokenize(" ".join(brief.entities)),
            *tokenize(" ".join(brief.relation_hints)),
        }
        return {token for token in tokens if len(token) > 3 and token not in CONTENT_STOPWORDS}

    def _normalize_token(self, token: str) -> str:
        if len(token) <= 4:
            return token
        if token.endswith("ies") and len(token) > 5:
            return token[:-3] + "y"
        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    def _normalized_tokens(self, tokens: set[str]) -> set[str]:
        return {self._normalize_token(token) for token in tokens}

    def _dataset_edge_signal(self, brief: DocBrief) -> float:
        content_terms = self._content_terms(brief)
        hint_terms = {token for token in tokenize(" ".join(brief.keywords + brief.relation_hints + brief.entities)) if len(token) > 2}
        terms = content_terms | hint_terms
        scientific_score = min(len(terms & SCIENTIFIC_BRIDGE_TERMS) / 4.0, 1.0)
        clinical_score = min(len(terms & CLINICAL_BRIDGE_TERMS) / 4.0, 1.0)
        argument_score = 0.0
        if self._topic_cluster(brief):
            argument_score += 0.45
        if self._stance(brief):
            argument_score += 0.3
        if content_terms & ARGUMENT_COMPARISON_CUES:
            argument_score += 0.25
        return min(max(scientific_score, clinical_score, argument_score), 1.0)

    def _is_evidence_like_doc(self, brief: DocBrief) -> bool:
        return self._doc_stage(brief) in {"scientific_evidence", "clinical_passage"}

    def _supports_same_concept_pair(self, anchor: DocBrief, candidate: DocBrief) -> bool:
        return self._is_evidence_like_doc(anchor) and self._doc_stage(candidate) == self._doc_stage(anchor)

    def _has_same_concept_signature(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float]) -> bool:
        if not self._supports_same_concept_pair(anchor, candidate):
            return False
        specific_title_bridge = self._specific_title_bridge_score(anchor, candidate)
        evidence_bridge_strength = self._evidence_bridge_strength(anchor, candidate, metrics)
        return (
            metrics["family_bridge_score"] >= 0.45
            or metrics["shared_dominant_family"] >= 1.0
            or metrics["entity_overlap"] >= 1.0
            or specific_title_bridge >= 1.0
            or (
                evidence_bridge_strength >= 0.58
                and metrics["title_overlap"] >= 2.0
                and metrics["topic_alignment"] >= 1.0
            )
        )

    def _high_utility_same_concept_bridge(
        self,
        anchor: DocBrief,
        candidate: DocBrief,
        metrics: dict[str, float],
        *,
        blended_support: float,
        blended_utility: float,
    ) -> bool:
        if not self._supports_same_concept_pair(anchor, candidate):
            return False
        if metrics["topic_drift"] >= 1.0:
            return False
        return (
            metrics["dense_score"] >= 0.75
            and blended_support >= 0.7
            and blended_utility >= 0.72
            and metrics["topic_alignment"] >= 1.0
            and (
                metrics["title_overlap"] >= 2.0
                or metrics["mention_score"] >= 0.32
                or self._specific_title_bridge_score(anchor, candidate) >= 0.45
            )
        )

    def _same_concept_methodology_penalty(self, anchor: DocBrief, candidate: DocBrief) -> float:
        anchor_stage = self._doc_stage(anchor)
        if anchor_stage not in {"scientific_evidence", "clinical_passage"} or self._doc_stage(candidate) != anchor_stage:
            return 0.0
        anchor_title_tokens = {token for token in tokenize(anchor.title) if len(token) > 2}
        candidate_title_tokens = {token for token in tokenize(candidate.title) if len(token) > 2}
        candidate_method_terms = candidate_title_tokens & METHOD_TITLE_TERMS
        if not candidate_method_terms:
            return 0.0
        anchor_method_terms = anchor_title_tokens & METHOD_TITLE_TERMS
        anchor_outcome_terms = anchor_title_tokens & OUTCOME_TITLE_TERMS
        candidate_outcome_terms = candidate_title_tokens & OUTCOME_TITLE_TERMS
        shared_focus = len((anchor_title_tokens - METHOD_TITLE_TERMS) & (candidate_title_tokens - METHOD_TITLE_TERMS))
        penalty = 0.0
        if candidate_method_terms - anchor_method_terms:
            penalty += 0.38
        if not candidate_outcome_terms:
            penalty += 0.18
        if anchor_outcome_terms and not (anchor_outcome_terms & candidate_outcome_terms):
            penalty += 0.12
        if shared_focus <= 1:
            penalty += 0.12
        return min(penalty, 0.8)

    def _family_bridge_score(self, anchor_terms: set[str], candidate_terms: set[str]) -> float:
        best = 0.0
        for _, family in DETAIL_FAMILIES:
            anchor_hits = len(anchor_terms & family)
            candidate_hits = len(candidate_terms & family)
            if anchor_hits == 0 or candidate_hits == 0:
                continue
            best = max(best, min(anchor_hits, candidate_hits) / 4.0)
        return min(best, 1.0)

    def _dominant_family(self, content_terms: set[str]) -> str:
        best_name = ""
        best_hits = 0
        for name, family in DETAIL_FAMILIES:
            hits = len(content_terms & family)
            if hits > best_hits:
                best_name = name
                best_hits = hits
        return best_name if best_hits > 0 else ""

    def _reference_score(self, source: DocBrief, target: DocBrief) -> float:
        source_text = self._brief_text(source)
        target_title = target.title.lower()
        title_tokens = {token for token in tokenize(target.title) if len(token) > 3}
        score = 0.0
        if target_title and target_title in source_text:
            score += 0.55
        source_tokens = set(tokenize(source.summary + " " + " ".join(source.claims) + " " + " ".join(source.relation_hints)))
        score += 0.08 * len(title_tokens & source_tokens)
        return min(score, 1.0)

    def _title_specificity_score(self, brief: DocBrief) -> float:
        tokens = set(tokenize(brief.title))
        score = 0.0
        score += 0.55 * len(tokens & HIGH_SPECIFICITY_TITLE_TERMS)
        score += 0.25 * len(tokens & MEDIUM_SPECIFICITY_TITLE_TERMS)
        score -= 0.3 * len(tokens & LOW_SPECIFICITY_TITLE_TERMS)
        return max(-0.6, min(score, 1.2))

    def _specific_title_terms(self, brief: DocBrief) -> set[str]:
        tokens = {self._normalize_token(token) for token in tokenize(brief.title) if len(token) > 3}
        return {
            token
            for token in tokens
            if len(token) >= 7 and token not in GENERIC_TITLE_BRIDGE_TOKENS
        }

    def _specific_title_bridge_score(self, anchor: DocBrief, candidate: DocBrief) -> float:
        shared = self._specific_title_terms(anchor) & self._specific_title_terms(candidate)
        if not shared:
            return 0.0
        return min(len(shared) / 2.0, 1.0)

    def _argument_bridge_strength(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float]) -> float:
        if not self._is_argumentative_pair(anchor, candidate):
            return 0.0
        topic_consistency = self._argument_topic_consistency(anchor, candidate, metrics)
        specific_title_bridge = self._specific_title_bridge_score(anchor, candidate)
        return max(
            0.0,
            min(
                1.0,
                0.26 * topic_consistency
                + 0.24 * metrics["stance_contrast"]
                + 0.18 * metrics["content_overlap_score"]
                + 0.14 * metrics["overlap_score"]
                + 0.16 * specific_title_bridge,
            ),
        )

    def _argument_topic_consistency(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float]) -> float:
        if not self._is_argumentative_pair(anchor, candidate):
            return 0.0
        specific_title_bridge = self._specific_title_bridge_score(anchor, candidate)
        topic_family_match = metrics.get("topic_family_match", 0.0)
        normalized_title_overlap = min(metrics["title_overlap"] / 3.0, 1.0)
        normalized_content_overlap = min(metrics["content_overlap_score"] / 0.3, 1.0)
        normalized_mention = min(metrics["mention_score"] / 0.24, 1.0)
        family_weight = 0.34 if self._topic_family(anchor) and self._topic_family(candidate) else 0.0
        base_consistency = (
            0.34 * metrics["topic_cluster_match"]
            + 0.26 * specific_title_bridge
            + 0.16 * normalized_title_overlap
            + 0.14 * normalized_content_overlap
            + 0.10 * normalized_mention
        )
        return max(
            0.0,
            min(
                1.0,
                family_weight * topic_family_match + (1.0 - family_weight) * base_consistency,
            ),
        )

    def _contrastive_argument_reuse(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float]) -> float:
        if not self._is_argumentative_pair(anchor, candidate):
            return 0.0
        argument_bridge_strength = self._argument_bridge_strength(anchor, candidate, metrics)
        topic_consistency = self._argument_topic_consistency(anchor, candidate, metrics)
        if metrics["stance_contrast"] < 1.0:
            return 0.0
        return max(
            0.0,
            min(
                1.0,
                0.46 * argument_bridge_strength
                + 0.24 * topic_consistency
                + 0.18 * metrics["content_overlap_score"]
                + 0.08 * metrics.get("topic_family_match", 0.0)
                + 0.12 * min(metrics["title_overlap"] / 3.0, 1.0),
            ),
        )

    def _evidence_bridge_strength(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float]) -> float:
        if not self._supports_same_concept_pair(anchor, candidate):
            return 0.0
        specific_title_bridge = self._specific_title_bridge_score(anchor, candidate)
        return max(
            0.0,
            min(
                1.0,
                0.24 * metrics["dense_score"]
                + 0.16 * metrics["family_bridge_score"]
                + 0.12 * metrics["shared_dominant_family"]
                + 0.14 * metrics["topic_alignment"]
                + 0.1 * metrics["topic_cluster_match"]
                + 0.12 * metrics["novelty_bridge_score"]
                + 0.12 * specific_title_bridge,
            ),
        )

    def _specific_title_bridge_potential(self, anchor: DocBrief, corpus: list[DocBrief]) -> float:
        anchor_terms = self._specific_title_terms(anchor)
        if not anchor_terms:
            return 0.0
        scores: list[float] = []
        for candidate in corpus:
            if candidate.doc_id == anchor.doc_id:
                continue
            shared = anchor_terms & self._specific_title_terms(candidate)
            if not shared:
                continue
            scores.append(min(len(shared) / 2.0, 1.0))
        if not scores:
            return 0.0
        scores.sort(reverse=True)
        head = scores[:2]
        return min(sum(head) / len(head), 1.0)

    def _detail_density(self, brief: DocBrief) -> float:
        content_terms = self._content_terms(brief)
        cue_hits = len(content_terms & DETAIL_CUES)
        return min(cue_hits / 6.0, 1.0)

    def _implementation_direction_score(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float]) -> float:
        candidate_spec = self._title_specificity_score(candidate)
        anchor_spec = self._title_specificity_score(anchor)
        detail_gap = self._detail_density(candidate) - self._detail_density(anchor)
        reference_gap = metrics["forward_reference_score"] - 0.85 * metrics["reverse_reference_score"]
        score = (
            0.55 * (candidate_spec - anchor_spec)
            + 0.2 * detail_gap
            + 0.2 * reference_gap
            + 0.15 * (metrics["family_bridge_score"] - 0.5)
        )
        return max(-1.0, min(score, 1.0))

    def _has_listing_context(self, brief: DocBrief) -> float:
        return 1.0 if any(word in self._brief_text(brief) for word in LISTING_WORDS) else 0.0

    def _doc_stage(self, brief: DocBrief) -> str:
        title_tokens = set(tokenize(brief.title))
        content_terms = self._content_terms(brief)
        topic = str(brief.metadata.get("topic", "")).lower()
        source_dataset = self._source_dataset(brief)

        argument_hits = len(content_terms & ARGUMENT_STAGE_TOKENS) + (2 if title_tokens & {"argument", "debate", "claim", "claims"} else 0)
        scientific_hits = len(content_terms & SCIENTIFIC_BRIDGE_TERMS) + (1 if title_tokens & {"evidence", "study", "studies"} else 0)
        clinical_hits = len(content_terms & CLINICAL_BRIDGE_TERMS) + (1 if title_tokens & {"clinical", "treatment", "therapy", "patient", "patients"} else 0)

        if topic == "argument" or argument_hits >= 2 or (self._topic_cluster(brief) and self._stance(brief)):
            return "argument_claim"
        if topic == "scientific_claims" or (scientific_hits >= 2 and len([claim for claim in brief.claims if claim.strip()]) >= 1):
            return "scientific_evidence"
        if topic == "clinical_retrieval" or (clinical_hits >= 2 and len([claim for claim in brief.claims if claim.strip()]) >= 1):
            return "clinical_passage"
        if source_dataset == "arguana":
            return "argument_claim"
        if source_dataset == "scifact":
            return "scientific_evidence"
        if source_dataset == "nfcorpus":
            return "clinical_passage"
        if topic == "evaluation":
            metric_hits = len(content_terms & EVAL_METRIC_STAGE_TOKENS) + (2 if title_tokens & {"ann", "metric", "metrics"} else 0)
            report_hits = len(content_terms & EVAL_REPORT_STAGE_TOKENS) + (2 if title_tokens & {"report", "reporting"} else 0)
            if metric_hits >= max(2, report_hits):
                return "eval_metrics"
            if report_hits >= 2:
                return "eval_report"

        if title_tokens & {"overlay", "graph"}:
            return "logic_graph"
        if title_tokens & {"policy", "jump"}:
            return "logic_policy"
        if title_tokens & {"fusion", "ranker"}:
            return "logic_fusion"
        if title_tokens & {"hybrid", "retrieval"}:
            return "retrieval_overview"
        if title_tokens & {"deepagents", "overview"}:
            return "agent_overview"
        if title_tokens & ROLE_STAGE_TOKENS:
            return "agent_roles"
        if title_tokens & {"memory", "backend"} and (content_terms & MEMORY_STAGE_TOKENS or topic in {"deepagents", "agents"}):
            return "agent_memory"
        if title_tokens & {"jobs", "workers"}:
            return "ops_overview"
        if title_tokens & {"fastapi", "service"}:
            return "ops_service"
        if title_tokens & {"registry", "sqlite"}:
            return "ops_registry"
        if title_tokens & {"revalidation", "revalidate"}:
            return "ops_revalidation"

        if topic == "retrieval" and (title_tokens & {"hybrid", "retrieval"} or content_terms & {"hybrid", "ranker"}):
            return "retrieval_overview"
        if topic == "logic":
            if title_tokens & {"overlay", "graph"} or len(content_terms & GRAPH_STAGE_TOKENS) >= 2:
                return "logic_graph"
            if title_tokens & {"policy", "jump"} or len(content_terms & POLICY_STAGE_TOKENS) >= 2:
                return "logic_policy"
            if title_tokens & {"fusion", "ranker"} or len(content_terms & FUSION_STAGE_TOKENS) >= 2:
                return "logic_fusion"
        if topic in {"deepagents", "agents"}:
            if title_tokens & {"deepagents", "overview"}:
                return "agent_overview"
            if title_tokens & ROLE_STAGE_TOKENS or len(content_terms & ROLE_STAGE_TOKENS) >= 2:
                return "agent_roles"
            if title_tokens & {"memory", "backend"} or len(content_terms & MEMORY_STAGE_TOKENS) >= 2:
                return "agent_memory"
        if topic == "ops":
            if title_tokens & {"jobs", "workers"}:
                return "ops_overview"
            if title_tokens & {"fastapi", "service"} or len(content_terms & SERVICE_STAGE_TOKENS) >= 2:
                return "ops_service"
            if title_tokens & {"registry", "sqlite"} or len(content_terms & REGISTRY_STAGE_TOKENS) >= 2:
                return "ops_registry"
            if title_tokens & {"revalidation", "revalidate"} or len(content_terms & REVALIDATION_STAGE_TOKENS) >= 2:
                return "ops_revalidation"
            if len(content_terms & OVERVIEW_STAGE_TOKENS) >= 2:
                return "ops_overview"
        if topic == "evaluation":
            if len(content_terms & EVAL_METRIC_STAGE_TOKENS) >= max(2, len(content_terms & EVAL_REPORT_STAGE_TOKENS)):
                return "eval_metrics"
            if len(content_terms & EVAL_REPORT_STAGE_TOKENS) >= 2:
                return "eval_report"
        return ""

    def _relation_stage_bonus(self, anchor: DocBrief, candidate: DocBrief, relation_type: str, metrics: dict[str, float]) -> float:
        anchor_stage = self._doc_stage(anchor)
        candidate_stage = self._doc_stage(candidate)
        pair = (anchor_stage, candidate_stage)
        bonus = 0.0
        if relation_type == "implementation_detail":
            positive = {
                ("retrieval_overview", "logic_fusion"): 0.3,
                ("logic_graph", "logic_policy"): 0.34,
                ("agent_overview", "agent_roles"): 0.28,
                ("agent_overview", "agent_memory"): 0.28,
                ("ops_overview", "ops_registry"): 0.32,
                ("ops_service", "ops_registry"): 0.28,
                ("eval_metrics", "eval_report"): 0.26,
            }
            negative = {
                ("logic_fusion", "retrieval_overview"): -0.3,
                ("logic_policy", "logic_graph"): -0.34,
                ("agent_roles", "agent_overview"): -0.28,
                ("agent_memory", "agent_overview"): -0.28,
                ("ops_registry", "ops_overview"): -0.34,
                ("ops_registry", "ops_service"): -0.28,
                ("ops_service", "ops_overview"): -0.32,
                ("eval_report", "eval_metrics"): -0.26,
            }
            bonus = positive.get(pair, negative.get(pair, 0.0))
            if bonus == 0.0:
                if candidate_stage.endswith("_overview") and anchor_stage and candidate_stage != anchor_stage:
                    bonus -= 0.28
                if metrics["specific_role_score"] >= 0.5 and anchor_stage not in {"agent_overview", "agent_roles"}:
                    bonus -= 0.22
        elif relation_type == "supporting_evidence":
            positive = {
                ("logic_graph", "retrieval_overview"): 0.28,
                ("logic_policy", "logic_fusion"): 0.3,
                ("scientific_evidence", "scientific_evidence"): 0.18,
                ("clinical_passage", "clinical_passage"): 0.16,
            }
            negative = {
                ("retrieval_overview", "logic_graph"): -0.28,
                ("logic_fusion", "logic_policy"): -0.32,
                ("ops_revalidation", "agent_roles"): -0.28,
            }
            bonus = positive.get(pair, negative.get(pair, 0.0))
            if "judge" in set(tokenize(anchor.title)) and candidate_stage == "ops_revalidation":
                bonus = max(bonus, 0.28)
        elif relation_type == "prerequisite":
            bonus = {
                ("agent_roles", "agent_roles"): 0.28,
            }.get(pair, 0.0)
            if anchor_stage == "agent_roles" and metrics["specific_role_score"] >= 0.5 and self._has_listing_context(anchor) >= 1.0:
                bonus = max(bonus, 0.3)
            if "scout" in set(tokenize(anchor.title)) and "judge" in set(tokenize(candidate.title)):
                bonus = max(bonus, 0.24)
        elif relation_type == "comparison":
            if pair == ("argument_claim", "argument_claim"):
                bonus = 0.24 if metrics.get("stance_contrast", 0.0) >= 1.0 else 0.12
            elif pair == ("scientific_evidence", "scientific_evidence"):
                bonus = 0.08
        elif relation_type == "same_concept":
            if pair == ("scientific_evidence", "scientific_evidence"):
                bonus = 0.14
            elif pair == ("clinical_passage", "clinical_passage"):
                bonus = 0.12
        return bonus

    def _relation_fit_scores(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float]) -> dict[str, float]:
        direction_score = self._implementation_direction_score(anchor, candidate, metrics)
        listing_context = self._has_listing_context(anchor)
        reference_balance = max(metrics["forward_reference_score"] - 0.75 * metrics["reverse_reference_score"], 0.0)
        cue_terms = self._content_terms(anchor) | self._content_terms(candidate)
        detail_cue_score = min(len(cue_terms & DETAIL_CUES) / 5.0, 1.0)
        support_cue_score = min(len(cue_terms & SUPPORT_CUES) / 4.0, 1.0)
        comparison_cue_score = min(len(cue_terms & ARGUMENT_COMPARISON_CUES) / 5.0, 1.0)
        methodology_penalty = self._same_concept_methodology_penalty(anchor, candidate)
        argument_bridge_strength = self._argument_bridge_strength(anchor, candidate, metrics)
        argument_topic_consistency = self._argument_topic_consistency(anchor, candidate, metrics)
        evidence_bridge_strength = self._evidence_bridge_strength(anchor, candidate, metrics)

        implementation_detail = (
            0.32 * metrics["dense_score"]
            + 0.16 * metrics["family_bridge_score"]
            + 0.12 * metrics["shared_dominant_family"]
            + 0.16 * max(direction_score, 0.0)
            + 0.1 * metrics["content_overlap_score"]
            + 0.06 * metrics["mention_score"]
            + 0.06 * metrics["topic_alignment"]
            + 0.08 * reference_balance
            + 0.06 * detail_cue_score
            + self._relation_stage_bonus(anchor, candidate, "implementation_detail", metrics)
            - 0.18 * metrics["service_surface_score"]
            - 0.16 * max(-direction_score, 0.0)
            - 0.06 * (listing_context * metrics["specific_role_score"])
        )
        supporting_evidence = (
            0.18 * metrics["dense_score"]
            + 0.22 * metrics["mention_score"]
            + 0.16 * metrics["content_overlap_score"]
            + 0.12 * metrics["family_bridge_score"]
            + 0.08 * reference_balance
            + 0.08 * metrics["topic_alignment"]
            + 0.1 * support_cue_score
            + self._relation_stage_bonus(anchor, candidate, "supporting_evidence", metrics)
            - 0.24 * metrics["service_surface_score"]
        )
        if self._supports_same_concept_pair(anchor, candidate):
            supporting_evidence += 0.1 * evidence_bridge_strength
        prerequisite = (
            0.34 * metrics["specific_role_score"]
            + 0.24 * metrics["role_listing_score"]
            + 0.12 * metrics["mention_score"]
            + 0.08 * metrics["content_overlap_score"]
            + 0.1 * listing_context
            + 0.08 * metrics["topic_alignment"]
            + self._relation_stage_bonus(anchor, candidate, "prerequisite", metrics)
            - 0.08 * metrics["service_surface_score"]
        )
        comparison = (
            0.2 * metrics["dense_score"]
            + 0.18 * metrics["overlap_score"]
            + 0.14 * metrics["content_overlap_score"]
            + 0.16 * metrics["topic_alignment"]
            + 0.14 * metrics["topic_family_match"]
            + 0.14 * metrics["topic_cluster_match"]
            + 0.18 * metrics["stance_contrast"]
            + 0.08 * metrics["novelty_bridge_score"]
            + 0.08 * comparison_cue_score
            + self._relation_stage_bonus(anchor, candidate, "comparison", metrics)
            - 0.12 * metrics["service_surface_score"]
        )
        comparison += 0.12 * argument_bridge_strength + 0.12 * argument_topic_consistency
        if self._is_argumentative_pair(anchor, candidate) and argument_topic_consistency < 0.56:
            comparison -= 0.2 * (0.56 - argument_topic_consistency) / 0.56
        same_concept = (
            0.26 * metrics["dense_score"]
            + 0.22 * metrics["overlap_score"]
            + 0.18 * metrics["content_overlap_score"]
            + 0.16 * metrics["mention_score"]
            + 0.16 * metrics["topic_alignment"]
            + 0.12 * metrics["topic_cluster_match"]
            + 0.08 * metrics["shared_dominant_family"]
            + 0.14 * metrics["novelty_bridge_score"]
            + max(self._relation_stage_bonus(anchor, candidate, "same_concept", metrics), 0.0)
            - 0.16 * metrics["stance_contrast"]
            - 0.22 * methodology_penalty
            - 0.1 * max(0.0, 0.45 - metrics["family_bridge_score"])
            - 0.12 * max(0.0, 0.14 - metrics["novel_term_ratio"])
            - 0.1 * max(0.0, metrics["novel_term_ratio"] - 0.82)
        )
        if self._supports_same_concept_pair(anchor, candidate):
            same_concept += 0.08 * evidence_bridge_strength
        return {
            "implementation_detail": max(0.0, min(implementation_detail, 1.4)),
            "supporting_evidence": max(0.0, min(supporting_evidence, 1.2)),
            "prerequisite": max(0.0, min(prerequisite, 1.2)),
            "comparison": max(0.0, min(comparison, 1.2)),
            "same_concept": max(0.0, min(same_concept, 1.1)),
        }

    def _utility_score(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float], fit_scores: dict[str, float], relation_type: str) -> float:
        bridge_gain = self._bridge_information_gain(anchor, candidate)
        duplicate_penalty = self._near_duplicate_penalty(anchor, candidate, metrics)
        specific_title_bridge = self._specific_title_bridge_score(anchor, candidate)
        contrastive_bridge_score = self._contrastive_argument_reuse(anchor, candidate, metrics)
        query_surface_match = float(
            metrics.get(
                "query_surface_match",
                0.5 * metrics.get("topic_alignment", 0.0)
                + 0.5 * max(metrics.get("overlap_score", 0.0), metrics.get("content_overlap_score", 0.0)),
            )
            or 0.0
        )
        score = (
            0.26 * metrics["dense_score"]
            + 0.2 * metrics["local_support"]
            + 0.1 * metrics["mention_score"]
            + 0.14 * fit_scores.get(relation_type, 0.0)
            + 0.08 * metrics["forward_reference_score"]
            + 0.06 * metrics["content_overlap_score"]
            + 0.08 * metrics["topic_alignment"]
            + 0.12 * bridge_gain
            + 0.08 * specific_title_bridge
            + 0.08 * metrics["novelty_bridge_score"]
            + 0.08 * contrastive_bridge_score
            + 0.06 * query_surface_match
        )
        score += 0.06 * metrics["topic_cluster_match"]
        score += 0.06 * metrics["topic_family_match"]
        score += 0.04 * metrics["stance_contrast"]
        score -= 0.18 * metrics["service_surface_score"]
        score -= 0.18 * min(metrics["topic_drift"], 1.0)
        score -= 0.24 * duplicate_penalty
        return max(0.0, min(score, 1.0))

    def _signal_bundle(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float], fit_scores: dict[str, float], relation_type: str) -> JudgeSignals:
        stage_pair = f"{self._doc_stage(anchor)}->{self._doc_stage(candidate)}".strip("->")
        duplicate_penalty = self._near_duplicate_penalty(anchor, candidate, metrics)
        contrastive_bridge_score = self._contrastive_argument_reuse(anchor, candidate, metrics)
        signal_report = self._signal_runtime.build_signal_report(
            anchor,
            candidate,
            local_signals={
                **metrics,
                "utility_score": self._utility_score(anchor, candidate, metrics, fit_scores, relation_type),
                "bridge_gain": self._bridge_information_gain(anchor, candidate),
                "duplicate_penalty": duplicate_penalty,
                "contrastive_bridge_score": contrastive_bridge_score,
            },
        )
        risk_flags: list[str] = []
        if metrics["topic_drift"] >= 1.0 or float(signal_report.get("drift_risk", 0.0) or 0.0) >= 0.75:
            risk_flags.append("topic_drift")
        if metrics["service_surface_score"] >= 0.5:
            risk_flags.append("service_surface")
        if float(signal_report.get("bridge_information_gain", 0.0) or 0.0) < 0.34:
            risk_flags.append("low_bridge_gain")
        if float(signal_report.get("topic_consistency", 0.0) or 0.0) < 0.24 and float(signal_report.get("drift_risk", 0.0) or 0.0) >= 0.55:
            risk_flags.append("weak_topic_consistency")
        if duplicate_penalty >= 0.34 and float(signal_report.get("contrast_evidence", contrastive_bridge_score) or 0.0) < 0.56:
            risk_flags.append("near_duplicate")
        if metrics["local_support"] < 0.22 and float(signal_report.get("query_surface_match", 0.0) or 0.0) < 0.12:
            risk_flags.append("low_query_alignment")
        return JudgeSignals(
            dense_score=metrics["dense_score"],
            sparse_score=max(metrics["overlap_score"], metrics["content_overlap_score"]),
            overlap_score=metrics["overlap_score"],
            content_overlap_score=metrics["content_overlap_score"],
            mention_score=metrics["mention_score"],
            role_listing_score=metrics["role_listing_score"],
            forward_reference_score=metrics["forward_reference_score"],
            reverse_reference_score=metrics["reverse_reference_score"],
            direction_score=self._implementation_direction_score(anchor, candidate, metrics),
            local_support=metrics["local_support"],
            utility_score=self._utility_score(anchor, candidate, metrics, fit_scores, relation_type),
            best_relation=relation_type,
            stage_pair=stage_pair,
            risk_flags=risk_flags,
            relation_fit_scores={key: round(value, 4) for key, value in fit_scores.items()},
            topic_family_match=metrics["topic_family_match"],
            topic_cluster_match=metrics["topic_cluster_match"],
            stance_contrast=metrics["stance_contrast"],
            bridge_gain=float(signal_report.get("bridge_information_gain", self._bridge_information_gain(anchor, candidate)) or 0.0),
            duplicate_penalty=duplicate_penalty,
            contrastive_bridge_score=max(float(signal_report.get("contrast_evidence", 0.0) or 0.0), contrastive_bridge_score),
            topic_consistency=float(signal_report.get("topic_consistency", 0.0) or 0.0),
            duplicate_risk=float(signal_report.get("duplicate_risk", duplicate_penalty) or 0.0),
            query_surface_match=float(signal_report.get("query_surface_match", 0.0) or 0.0),
            uncertainty_hint=float(signal_report.get("uncertainty_hint", 0.0) or 0.0),
            drift_risk=float(signal_report.get("drift_risk", 0.0) or 0.0),
        )

    def _pair_rerank(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float]) -> tuple[float, str, dict[str, float]]:
        fit_scores = self._relation_fit_scores(anchor, candidate, metrics)
        relation_type, score = max(fit_scores.items(), key=lambda item: (item[1], item[0]))
        return score, relation_type, fit_scores

    def _candidate_signal_report(
        self,
        anchor: DocBrief,
        candidate: DocBrief,
        metrics: dict[str, float],
        fit_scores: dict[str, float],
        relation_type: str,
    ) -> dict[str, float]:
        return self._signal_runtime.build_signal_report(
            anchor,
            candidate,
            local_signals={
                **metrics,
                "best_relation": relation_type,
                "relation_fit": fit_scores.get(relation_type, 0.0),
                "same_concept_fit": fit_scores.get("same_concept", 0.0),
                "comparison_fit": fit_scores.get("comparison", 0.0),
                "supporting_evidence_fit": fit_scores.get("supporting_evidence", 0.0),
            },
        )

    @staticmethod
    def _graph_hygiene_failure(signal_report: dict[str, float]) -> bool:
        return (
            float(signal_report.get("drift_risk", 0.0) or 0.0) >= 0.82
            and float(signal_report.get("topic_consistency", 0.0) or 0.0) < 0.2
            and float(signal_report.get("bridge_information_gain", 0.0) or 0.0) < 0.28
        )

    def _candidate_priority_score(
        self,
        *,
        base_score: float,
        metrics: dict[str, float],
        fit_scores: dict[str, float],
        signal_report: dict[str, float],
    ) -> float:
        report = self._signal_runtime.compute_candidate_priority(
            base_score=base_score,
            metrics=metrics,
            fit_scores=fit_scores,
            signal_report=signal_report,
        )
        return float(report.get("priority_score", base_score) or base_score)

    def _select_diverse_proposals(
        self,
        anchor: DocBrief,
        scored: list[tuple[float, str, CandidateProposal]],
        limit: int,
    ) -> list[CandidateProposal]:
        selected: list[CandidateProposal] = []
        selected_ids: set[str] = set()
        for score, relation_type, proposal in sorted(scored, key=lambda item: (-item[0], item[1], item[2].doc_id)):
            if len(selected) >= limit:
                break
            if proposal.doc_id in selected_ids:
                continue
            selected.append(proposal)
            selected_ids.add(proposal.doc_id)
        return selected[:limit]

    def _select_diverse_assessments(
        self,
        anchor: DocBrief,
        accepted: list[CandidateAssessment],
        candidate_map: dict[str, DocBrief],
        limit: int,
    ) -> list[CandidateAssessment]:
        if len(accepted) <= limit:
            return accepted
        remaining = list(accepted)
        selected: list[CandidateAssessment] = []
        selected_terms: list[set[str]] = []
        selected_specific_terms: list[set[str]] = []
        while remaining and len(selected) < limit:
            best_index = 0
            best_score = float("-inf")
            for index, item in enumerate(remaining):
                candidate = candidate_map.get(item.candidate_doc_id)
                candidate_terms = self._normalized_tokens(self._content_terms(candidate)) if candidate is not None else set()
                candidate_specific_terms = self._specific_title_terms(candidate) if candidate is not None else set()
                novelty = 1.0
                if selected_terms and candidate_terms:
                    similarities = [
                        len(candidate_terms & chosen_terms) / max(1, len(candidate_terms | chosen_terms))
                        for chosen_terms in selected_terms
                    ]
                    novelty = 1.0 - max(similarities, default=0.0)
                specific_novelty = 1.0
                if selected_specific_terms and candidate_specific_terms:
                    specific_similarities = [
                        len(candidate_specific_terms & chosen_terms) / max(1, len(candidate_specific_terms | chosen_terms))
                        for chosen_terms in selected_specific_terms
                    ]
                    specific_novelty = 1.0 - max(specific_similarities, default=0.0)
                activation_profile = getattr(item.edge, "activation_profile", {}) if item.edge is not None else {}
                score = float(
                    self._signal_runtime.compute_edge_budget_score(
                        score=item.score,
                        utility_score=max(0.0, min(getattr(item.edge, "utility_score", 0.0), 1.0)),
                        activation_prior=max(0.0, min(activation_profile.get("activation_prior", 0.0), 1.0)),
                        novelty=novelty,
                        specific_novelty=specific_novelty,
                        drift_risk=max(0.0, min(activation_profile.get("drift_risk", 0.0), 1.0)),
                        duplicate_risk=max(0.0, min(activation_profile.get("duplicate_risk", 0.0), 1.0)),
                    ).get("selection_score", item.score)
                    or item.score
                )
                if score > best_score:
                    best_score = score
                    best_index = index
            chosen = remaining.pop(best_index)
            selected.append(chosen)
            chosen_candidate = candidate_map.get(chosen.candidate_doc_id)
            selected_terms.append(self._normalized_tokens(self._content_terms(chosen_candidate)) if chosen_candidate is not None else set())
            selected_specific_terms.append(self._specific_title_terms(chosen_candidate) if chosen_candidate is not None else set())
        return selected

    def _dense_neighbor_candidate_proposals(self, anchor: DocBrief, corpus: list[DocBrief], *, expanded: bool = False) -> list[tuple[float, CandidateProposal]]:
        anchor_vec = self._embed_brief(anchor)
        if anchor_vec is None:
            return []
        proposals: list[tuple[float, CandidateProposal]] = []
        dense_floor = 0.52 if expanded else 0.56
        cap = 14 if expanded else 10
        for candidate in corpus:
            if candidate.doc_id == anchor.doc_id:
                continue
            candidate_vec = self._embed_brief(candidate)
            if candidate_vec is None:
                continue
            dense_score = max(cosine(anchor_vec, candidate_vec), 0.0)
            if dense_score < dense_floor:
                continue
            metrics = self._candidate_metrics(anchor, candidate)
            rerank_score, relation_type, fit_scores = self._pair_rerank(anchor, candidate, metrics)
            signal_report = self._candidate_signal_report(anchor, candidate, metrics, fit_scores, relation_type)
            if self._is_live_provider() and self._graph_hygiene_failure(signal_report):
                continue
            score = self._candidate_priority_score(
                base_score=0.62 * dense_score + 0.18 * rerank_score,
                metrics=metrics,
                fit_scores=fit_scores,
                signal_report=signal_report,
            )
            proposals.append(
                (
                    score,
                    CandidateProposal(
                        doc_id=candidate.doc_id,
                        reason=f"dense neighbor aligns with {relation_type}",
                        query=" ".join((candidate.relation_hints + candidate.keywords + [candidate.title])[:4]),
                        score_hint=min(score, 0.99),
                    ),
                )
            )
        proposals.sort(key=lambda item: (-item[0], item[1].doc_id))
        return proposals[:cap]

    def _mention_score(self, anchor: DocBrief, candidate: DocBrief) -> float:
        anchor_text = self._brief_text(anchor)
        candidate_text = self._brief_text(candidate)
        candidate_title = candidate.title.lower()
        anchor_title = anchor.title.lower()
        candidate_title_tokens = {token for token in tokenize(candidate.title) if len(token) > 3}
        anchor_title_tokens = {token for token in tokenize(anchor.title) if len(token) > 3}
        anchor_text_tokens = set(tokenize(anchor.summary + " " + " ".join(anchor.claims)))
        candidate_text_tokens = set(tokenize(candidate.summary + " " + " ".join(candidate.claims)))
        score = 0.0
        if candidate_title and candidate_title in anchor_text:
            score += 0.55
        if anchor_title and anchor_title in candidate_text:
            score += 0.35
        score += 0.08 * len(candidate_title_tokens & anchor_text_tokens)
        score += 0.06 * len(anchor_title_tokens & candidate_text_tokens)
        return min(score, 1.0)

    def _role_listing_score(self, anchor: DocBrief, candidate: DocBrief) -> float:
        anchor_text = self._brief_text(anchor)
        candidate_text = self._brief_text(candidate)
        candidate_title_tokens = {token for token in tokenize(candidate.title) if len(token) > 3}
        score = 0.0
        if any(word in anchor_text for word in LISTING_WORDS):
            score += 0.3
        if any(word in candidate_text for word in ROLE_WORDS):
            score += 0.2
        if candidate.title.lower() in anchor_text:
            score += 0.35
        if candidate_title_tokens & set(tokenize(anchor_text)):
            score += 0.15
        return min(score, 1.0)

    def _candidate_metrics(self, anchor: DocBrief, candidate: DocBrief) -> dict[str, float]:
        anchor_terms = self._brief_terms(anchor)
        candidate_terms = self._brief_terms(candidate)
        anchor_content_terms = self._normalized_tokens(self._content_terms(anchor))
        candidate_content_terms = self._normalized_tokens(self._content_terms(candidate))
        keyword_overlap = len(set(anchor.keywords) & set(candidate.keywords))
        entity_overlap = len(set(anchor.entities) & set(candidate.entities))
        hint_overlap = len(set(anchor.relation_hints) & set(candidate.relation_hints))
        title_overlap = len(
            self._normalized_tokens(set(tokenize(anchor.title)))
            & self._normalized_tokens(set(tokenize(candidate.title)))
        )
        overlap_score = min((keyword_overlap + entity_overlap + hint_overlap + title_overlap) / 5.0, 1.0)
        content_overlap = len(anchor_content_terms & candidate_content_terms)
        content_overlap_score = min(content_overlap / 6.0, 1.0)
        candidate_only_terms = candidate_content_terms - anchor_content_terms
        anchor_only_terms = anchor_content_terms - candidate_content_terms
        novel_term_ratio = min(len(candidate_only_terms) / max(1, min(len(candidate_content_terms), 8)), 1.0)
        mutual_novelty = min((len(candidate_only_terms) + len(anchor_only_terms)) / max(1, min(len(anchor_content_terms | candidate_content_terms), 10)), 1.0)
        if 0.18 <= novel_term_ratio <= 0.72:
            novelty_bridge_score = 1.0 - abs(novel_term_ratio - 0.42) / 0.42
        else:
            novelty_bridge_score = max(0.0, 0.35 - min(abs(novel_term_ratio - 0.42), 0.42))
        mention_score = self._mention_score(anchor, candidate)
        role_listing_score = self._role_listing_score(anchor, candidate)
        forward_reference_score = self._reference_score(anchor, candidate)
        reverse_reference_score = self._reference_score(candidate, anchor)
        anchor_vec = self._embed_brief(anchor)
        candidate_vec = self._embed_brief(candidate)
        dense_score = max(cosine(anchor_vec, candidate_vec), 0.0) if anchor_vec is not None and candidate_vec is not None else 0.0
        length_ratio = min(len(candidate.summary), len(anchor.summary)) / max(len(candidate.summary), len(anchor.summary), 1)
        family_bridge_score = self._family_bridge_score(anchor_content_terms, candidate_content_terms)
        anchor_family = self._dominant_family(anchor_content_terms)
        candidate_family = self._dominant_family(candidate_content_terms)
        topic_family_match = 1.0 if self._topic_family(anchor) and self._topic_family(anchor) == self._topic_family(candidate) else 0.0
        topic_cluster_match = 1.0 if self._topic_cluster(anchor) and self._topic_cluster(anchor) == self._topic_cluster(candidate) else 0.0
        stance_contrast = 1.0 if self._stance(anchor) and self._stance(candidate) and self._stance(anchor) != self._stance(candidate) else 0.0
        topic_alignment = 1.0 if (
            (anchor.metadata.get("topic") and anchor.metadata.get("topic") == candidate.metadata.get("topic"))
            or topic_family_match >= 1.0
            or topic_cluster_match >= 1.0
        ) else 0.0
        service_surface_score = min(len(candidate_content_terms & SERVICE_SURFACE_TERMS) / 3.0, 1.0)
        specific_role_score = 1.0 if set(tokenize(candidate.title)) & SPECIFIC_ROLE_TERMS else 0.0
        topic_drift = 1.0 if mention_score < 0.2 and title_overlap == 0 and keyword_overlap + entity_overlap + hint_overlap == 0 and dense_score < 0.45 and topic_cluster_match < 1.0 else 0.0
        local_support = (
            0.42 * dense_score
            + 0.2 * overlap_score
            + 0.22 * mention_score
            + 0.08 * role_listing_score
            + 0.08 * min(length_ratio * 1.5, 1.0)
            + 0.06 * topic_cluster_match
            + 0.04 * stance_contrast
        )
        return {
            "dense_score": dense_score,
            "overlap_score": overlap_score,
            "content_overlap_score": content_overlap_score,
            "novel_term_ratio": novel_term_ratio,
            "mutual_novelty": mutual_novelty,
            "novelty_bridge_score": novelty_bridge_score,
            "keyword_overlap": float(keyword_overlap),
            "entity_overlap": float(entity_overlap),
            "hint_overlap": float(hint_overlap),
            "title_overlap": float(title_overlap),
            "length_ratio": float(length_ratio),
            "mention_score": mention_score,
            "role_listing_score": role_listing_score,
            "forward_reference_score": forward_reference_score,
            "reverse_reference_score": reverse_reference_score,
            "family_bridge_score": family_bridge_score,
            "shared_dominant_family": 1.0 if anchor_family and anchor_family == candidate_family else 0.0,
            "topic_family_match": topic_family_match,
            "topic_alignment": topic_alignment,
            "topic_cluster_match": topic_cluster_match,
            "stance_contrast": stance_contrast,
            "service_surface_score": service_surface_score,
            "specific_role_score": specific_role_score,
            "topic_drift": topic_drift,
            "local_support": local_support,
            "anchor_terms": float(len(anchor_terms)),
            "candidate_terms": float(len(candidate_terms)),
        }

    def _evidence_quality(self, anchor: DocBrief, candidate: DocBrief, result) -> float:
        evidence_chunks = [span.strip() for span in result.evidence_spans if span.strip()]
        if not evidence_chunks:
            return 0.0
        evidence_text = " ".join(evidence_chunks + [result.rationale, getattr(result, "decision_reason", "")]).strip()
        evidence_terms = set(tokenize(evidence_text))
        if not evidence_terms:
            return 0.0
        anchor_terms = self._brief_terms(anchor)
        candidate_terms = self._brief_terms(candidate)
        anchor_cov = len(evidence_terms & anchor_terms) / max(1, min(len(anchor_terms), 6))
        candidate_cov = len(evidence_terms & candidate_terms) / max(1, min(len(candidate_terms), 6))
        mention_bonus = 0.12 if candidate.title.lower() in evidence_text.lower() or anchor.title.lower() in evidence_text.lower() else 0.0
        span_score = min(len(evidence_chunks) / 2.0, 1.0)
        contradiction_penalty = 0.15 if getattr(result, "contradiction_flags", None) else 0.0
        return max(0.0, 0.36 * min(anchor_cov + candidate_cov, 1.0) + 0.38 * span_score + 0.14 + mention_bonus - contradiction_penalty)

    def _relation_threshold(self, relation_type: str) -> RelationQualityConfig:
        return self._edge_quality().relation_thresholds.get(relation_type, RelationQualityConfig())

    def _is_live_provider(self) -> bool:
        provider = self._provider()
        return provider is not None and provider.__class__.__name__ == "OpenAICompatibleProvider"

    def _relation_prior(self, relation_type: str) -> float:
        provider = self._provider()
        if provider is None:
            return 1.0
        return float(getattr(provider, "relation_priors", {}).get(relation_type, 1.0))

    def _merge_spans(self, *groups: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for span in group:
                value = str(span).strip()
                if not value:
                    continue
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(value)
                if len(merged) >= 4:
                    return merged
        return merged

    def _review_consensus_result(
        self,
        anchor: DocBrief,
        candidate: DocBrief,
        result,
        review_result,
        metrics: dict[str, float],
        fit_scores: dict[str, float],
        signal_bundle: JudgeSignals,
    ):
        chosen = review_result or result
        resolved_relation = (
            chosen.relation_type
            if getattr(chosen, "relation_type", "") in RELATION_TYPES
            else (
                result.relation_type
                if getattr(result, "relation_type", "") in RELATION_TYPES
                else signal_bundle.best_relation
            )
        )
        utility_score = max(signal_bundle.utility_score, float(getattr(chosen, "utility_score", 0.0)))
        support_score = max(signal_bundle.local_support, float(getattr(chosen, "support_score", 0.0)))
        confidence = max(0.0, min(0.99, float(getattr(chosen, "confidence", 0.0))))
        uncertainty = max(
            float(getattr(chosen, "uncertainty", 0.0)),
            float(signal_bundle.uncertainty_hint),
        )
        accepted = bool(getattr(chosen, "accepted", False))
        decision_reason = str(
            getattr(chosen, "decision_reason", "")
            or ("review_rejected" if review_result is not None and not accepted else "agent_consensus")
        )[:200]
        return SimpleNamespace(
            accepted=accepted,
            relation_type=resolved_relation,
            confidence=confidence,
            evidence_spans=self._merge_spans(
                getattr(result, "evidence_spans", []),
                getattr(review_result, "evidence_spans", []),
            ),
            rationale=(getattr(chosen, "rationale", "") or getattr(result, "rationale", ""))[:200],
            support_score=max(0.0, min(1.0, support_score)),
            contradiction_flags=self._merge_spans(
                [str(flag) for flag in getattr(result, "contradiction_flags", []) or []],
                [str(flag) for flag in getattr(review_result, "contradiction_flags", []) or []],
            ),
            decision_reason=decision_reason,
            semantic_relation_label=str(getattr(chosen, "semantic_relation_label", resolved_relation) or resolved_relation)[:80],
            canonical_relation=resolved_relation if accepted else "none",
            utility_score=max(0.0, min(1.0, utility_score)),
            uncertainty=uncertainty,
            reject_reason=decision_reason,
        )

    def should_attempt_discovery(self, anchor: DocBrief) -> bool:
        anchor_terms = self._content_terms(anchor)
        claim_score = min(len([claim for claim in anchor.claims if claim.strip()]) / 2.0, 1.0)
        richness = (
            0.26 * claim_score
            + 0.18 * min(len(anchor.keywords) / 6.0, 1.0)
            + 0.16 * min(len(anchor.entities) / 4.0, 1.0)
            + 0.14 * min(len(anchor.relation_hints) / 5.0, 1.0)
            + 0.14 * min(len(anchor_terms) / 12.0, 1.0)
            + 0.12 * self._dataset_edge_signal(anchor)
        )
        if self._topic_cluster(anchor):
            richness += 0.06
        if self._stance(anchor):
            richness += 0.04
        text_tokens = set(tokenize(self._brief_text(anchor)))
        return richness >= 0.36 or bool(text_tokens & DISCOVERY_TERMS) or len(anchor.relation_hints) >= 4

    def discovery_anchor_priority(self, anchor: DocBrief) -> float:
        content_terms = self._content_terms(anchor)
        report = self._signal_runtime.compute_anchor_priority(
            anchor,
            features={
                "claim_score": min(len([claim for claim in anchor.claims if claim.strip()]) / 2.0, 1.0),
                "keyword_score": min(len(anchor.keywords) / 8.0, 1.0),
                "entity_score": min(len(anchor.entities) / 4.0, 1.0),
                "cue_score": min(len(set(anchor.relation_hints)) / 5.0, 1.0),
                "content_score": min(len(content_terms) / 14.0, 1.0),
                "dataset_edge_signal": self._dataset_edge_signal(anchor),
                "title_specificity": max(self._title_specificity_score(anchor), 0.0),
            },
            corpus_profile={},
        )
        return float(report.get("priority_score", 0.0) or 0.0)

    def _corpus_graph_profile(self, briefs: list[DocBrief]) -> dict[str, float]:
        if not briefs:
            return {"graph_potential": 0.0, "argument_ratio": 0.0, "evidence_ratio": 0.0}
        argument_ratio = sum(1.0 for brief in briefs if self._is_argumentative_doc(brief)) / len(briefs)
        evidence_ratio = sum(1.0 for brief in briefs if self._is_evidence_like_doc(brief)) / len(briefs)
        cluster_ratio = sum(1.0 for brief in briefs if self._topic_cluster(brief)) / len(briefs)
        cue_ratio = sum(min(len(brief.relation_hints) / 4.0, 1.0) for brief in briefs) / len(briefs)
        bridge_ratio = sum(self._dataset_edge_signal(brief) for brief in briefs) / len(briefs)
        specificity_ratio = sum(max(self._title_specificity_score(brief), 0.0) for brief in briefs) / len(briefs)
        graph_potential = min(
            1.0,
            0.28 * cue_ratio
            + 0.24 * bridge_ratio
            + 0.16 * cluster_ratio
            + 0.16 * specificity_ratio
            + 0.16 * max(argument_ratio, evidence_ratio),
        )
        return {
            "graph_potential": graph_potential,
            "argument_ratio": argument_ratio,
            "evidence_ratio": evidence_ratio,
        }

    def _dense_anchor_neighborhoods(self, briefs: list[DocBrief], *, top_k: int = 6, min_similarity: float = 0.52) -> tuple[dict[str, float], dict[str, list[tuple[str, float]]]]:
        vectors = {brief.doc_id: self._embed_brief(brief) for brief in briefs}
        scores: dict[str, float] = {}
        neighborhoods: dict[str, list[tuple[str, float]]] = {}
        for brief in briefs:
            vector = vectors.get(brief.doc_id)
            if vector is None:
                scores[brief.doc_id] = 0.0
                neighborhoods[brief.doc_id] = [(brief.doc_id, 1.0)]
                continue
            sims: list[tuple[float, str]] = []
            for other in briefs:
                if other.doc_id == brief.doc_id:
                    continue
                other_vec = vectors.get(other.doc_id)
                if other_vec is None:
                    continue
                sims.append((max(cosine(vector, other_vec), 0.0), other.doc_id))
            sims.sort(key=lambda item: (-item[0], item[1]))
            if not sims:
                scores[brief.doc_id] = 0.0
                neighborhoods[brief.doc_id] = [(brief.doc_id, 1.0)]
                continue
            top = sims[: min(5, len(sims))]
            scores[brief.doc_id] = sum(score for score, _ in top) / len(top)
            neighbors = [(brief.doc_id, 1.0)]
            for score, doc_id in sims[:top_k]:
                if score < min_similarity:
                    continue
                neighbors.append((doc_id, score))
            neighborhoods[brief.doc_id] = neighbors
        return scores, neighborhoods

    def rank_discovery_anchors(self, briefs: list[DocBrief]) -> list[str]:
        if not briefs:
            return []
        grouped: dict[str, list[DocBrief]] = {}
        for brief in briefs:
            grouped.setdefault(self._source_dataset(brief), []).append(brief)
        ordered: list[str] = []
        for _, group in grouped.items():
            profile = self._corpus_graph_profile(group)
            if profile["graph_potential"] < 0.16:
                ordered.extend(brief.doc_id for brief in group[: min(len(group), 6)])
                continue
            centrality, neighborhoods = self._dense_anchor_neighborhoods(group)
            specific_bridge_potential = {
                brief.doc_id: self._specific_title_bridge_potential(brief, group)
                for brief in group
            }
            cap_base = max(6, min(len(group), len(group) // 8 + 4))
            floor = max(0.34, min(0.54, 0.54 - 0.16 * profile["graph_potential"] + 0.04 * profile["argument_ratio"]))
            priority_map = {}
            bridge_pressure = sum(1 for brief in group if specific_bridge_potential.get(brief.doc_id, 0.0) >= 0.75) / max(1, len(group))
            for brief in group:
                priority_map[brief.doc_id] = float(
                    self._signal_runtime.compute_anchor_priority(
                        brief,
                        features={
                            "claim_score": min(len([claim for claim in brief.claims if claim.strip()]) / 2.0, 1.0),
                            "keyword_score": min(len(brief.keywords) / 8.0, 1.0),
                            "entity_score": min(len(brief.entities) / 4.0, 1.0),
                            "cue_score": min(len(set(brief.relation_hints)) / 5.0, 1.0),
                            "content_score": min(len(self._content_terms(brief)) / 14.0, 1.0),
                            "dataset_edge_signal": self._dataset_edge_signal(brief),
                            "centrality": centrality.get(brief.doc_id, 0.0),
                            "bridge_potential": specific_bridge_potential.get(brief.doc_id, 0.0),
                            "title_specificity": max(self._title_specificity_score(brief), 0.0),
                        },
                        corpus_profile={
                            **profile,
                            "bridge_pressure": bridge_pressure,
                        },
                    ).get("priority_score", 0.0)
                    or 0.0
                )
            eligible = [brief for brief in group if priority_map.get(brief.doc_id, 0.0) >= floor]
            if not eligible:
                eligible = list(group)
            eligibility_pressure = len(eligible) / max(1, len(group))
            pressure = max(eligibility_pressure, 0.6 * eligibility_pressure + 0.4 * bridge_pressure)
            cap_multiplier = 0.9 + 0.42 * profile["graph_potential"] + 0.8 * pressure
            cap = max(6, min(len(group), int(round(cap_base * cap_multiplier))))
            coverage: dict[str, float] = {}
            kept: list[str] = []
            selected_clusters: set[str] = set()
            selected_families: set[str] = set()
            while eligible and len(kept) < cap:
                best_id = ""
                best_score = float("-inf")
                best_index = 0
                for index, brief in enumerate(eligible):
                    coverage_gain = 0.0
                    for neighbor_id, weight in neighborhoods.get(brief.doc_id, [(brief.doc_id, 1.0)]):
                        coverage_gain += max(0.0, weight - coverage.get(neighbor_id, 0.0))
                    score = float(
                        self._signal_runtime.compute_anchor_priority(
                            brief,
                            features={
                                "claim_score": min(len([claim for claim in brief.claims if claim.strip()]) / 2.0, 1.0),
                                "keyword_score": min(len(brief.keywords) / 8.0, 1.0),
                                "entity_score": min(len(brief.entities) / 4.0, 1.0),
                                "cue_score": min(len(set(brief.relation_hints)) / 5.0, 1.0),
                                "content_score": min(len(self._content_terms(brief)) / 14.0, 1.0),
                                "dataset_edge_signal": self._dataset_edge_signal(brief),
                                "centrality": centrality.get(brief.doc_id, 0.0),
                                "bridge_potential": specific_bridge_potential.get(brief.doc_id, 0.0),
                                "title_specificity": max(self._title_specificity_score(brief), 0.0),
                                "coverage_gain": coverage_gain,
                                "cluster_novelty": 1.0 if self._topic_cluster(brief) and self._topic_cluster(brief) not in selected_clusters else 0.0,
                                "family_novelty": 1.0 if self._topic_family(brief) and self._topic_family(brief) not in selected_families else 0.0,
                            },
                            corpus_profile={
                                **profile,
                                "bridge_pressure": bridge_pressure,
                            },
                        ).get("priority_score", 0.0)
                        or 0.0
                    )
                    if score > best_score:
                        best_score = score
                        best_id = brief.doc_id
                        best_index = index
                if not best_id:
                    break
                chosen = eligible.pop(best_index)
                kept.append(best_id)
                for neighbor_id, weight in neighborhoods.get(best_id, [(best_id, 1.0)]):
                    coverage[neighbor_id] = max(coverage.get(neighbor_id, 0.0), weight)
                cluster = self._topic_cluster(chosen)
                if cluster:
                    selected_clusters.add(cluster)
                family = self._topic_family(chosen)
                if family:
                    selected_families.add(family)
            reserve_cap = max(0, min(len(group) - len(kept), max(4, cap // 2)))
            if reserve_cap > 0:
                reserve_ranked: list[tuple[float, DocBrief]] = []
                for brief in group:
                    if brief.doc_id in kept:
                        continue
                    bridge_potential = specific_bridge_potential.get(brief.doc_id, 0.0)
                    if bridge_potential < 0.5:
                        continue
                    reserve_score = float(
                        self._signal_runtime.compute_anchor_priority(
                            brief,
                            features={
                                "claim_score": min(len([claim for claim in brief.claims if claim.strip()]) / 2.0, 1.0),
                                "keyword_score": min(len(brief.keywords) / 8.0, 1.0),
                                "entity_score": min(len(brief.entities) / 4.0, 1.0),
                                "cue_score": min(len(set(brief.relation_hints)) / 5.0, 1.0),
                                "content_score": min(len(self._content_terms(brief)) / 14.0, 1.0),
                                "dataset_edge_signal": self._dataset_edge_signal(brief),
                                "centrality": centrality.get(brief.doc_id, 0.0),
                                "bridge_potential": bridge_potential,
                                "title_specificity": max(self._title_specificity_score(brief), 0.0),
                                "cluster_novelty": 1.0 if self._topic_cluster(brief) and self._topic_cluster(brief) not in selected_clusters else 0.0,
                                "family_novelty": 1.0 if self._topic_family(brief) and self._topic_family(brief) not in selected_families else 0.0,
                            },
                            corpus_profile={
                                **profile,
                                "bridge_pressure": bridge_pressure,
                            },
                        ).get("priority_score", 0.0)
                        or 0.0
                    )
                    reserve_ranked.append((reserve_score, brief))
                reserve_ranked.sort(key=lambda item: (-item[0], item[1].doc_id))
                reserve_kept = 0
                reserve_clusters: set[str] = set()
                reserve_families: set[str] = set()
                for _, brief in reserve_ranked:
                    cluster = self._topic_cluster(brief)
                    if cluster and cluster in reserve_clusters:
                        continue
                    family = self._topic_family(brief)
                    if family and family in reserve_families:
                        continue
                    kept.append(brief.doc_id)
                    reserve_kept += 1
                    if cluster:
                        reserve_clusters.add(cluster)
                        selected_clusters.add(cluster)
                    if family:
                        reserve_families.add(family)
                        selected_families.add(family)
                    if reserve_kept >= reserve_cap:
                        break
            if not kept:
                ranked = sorted(
                    ((priority_map.get(brief.doc_id, 0.0), brief) for brief in group),
                    key=lambda item: (-item[0], item[1].doc_id),
                )
                kept = [brief.doc_id for _, brief in ranked[: min(cap, max(6, len(group) // 6 or 1))]]
            ordered.extend(kept)
        deduped: list[str] = []
        seen: set[str] = set()
        for doc_id in ordered:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            deduped.append(doc_id)
        return deduped

    def select_discovery_anchors(self, briefs: list[DocBrief]) -> set[str]:
        return set(self.rank_discovery_anchors(briefs))

    def _assessment_for(self, anchor: DocBrief, candidate: DocBrief, result, review_result=None) -> CandidateAssessment:
        metrics = self._candidate_metrics(anchor, candidate)
        rerank_score, rerank_relation, fit_scores = self._pair_rerank(anchor, candidate, metrics)
        signal_bundle = self._signal_bundle(anchor, candidate, metrics, fit_scores, rerank_relation)
        final_result = self._review_consensus_result(anchor, candidate, result, review_result, metrics, fit_scores, signal_bundle)

        if not final_result.accepted:
            return CandidateAssessment(
                candidate.doc_id,
                False,
                getattr(final_result, "reject_reason", "agent_rejected"),
                0.0,
                0.0,
                0.0,
                final_result.relation_type,
                final_result.confidence,
            )
        if final_result.relation_type not in RELATION_TYPES:
            return CandidateAssessment(candidate.doc_id, False, "unsupported_relation", 0.0, 0.0, 0.0, final_result.relation_type, final_result.confidence)

        model_support = max(0.0, min(float(getattr(final_result, "support_score", 0.0)), 1.0))
        reviewer_support = max(0.0, min(float(getattr(review_result, "support_score", model_support)) if review_result is not None else model_support, 1.0))
        blended_support = 0.55 * metrics["local_support"] + 0.45 * max(model_support, reviewer_support)
        model_utility = max(0.0, min(float(getattr(final_result, "utility_score", 0.0)), 1.0))
        reviewer_utility = max(0.0, min(float(getattr(review_result, "utility_score", model_utility)) if review_result is not None else model_utility, 1.0))
        blended_utility = max(signal_bundle.utility_score, 0.45 * signal_bundle.utility_score + 0.55 * max(model_utility, reviewer_utility))
        evidence_quality = self._evidence_quality(anchor, candidate, final_result)
        threshold = self._relation_threshold(final_result.relation_type)
        hard_signal_failure = self._graph_hygiene_failure(
            {
                "drift_risk": signal_bundle.drift_risk,
                "topic_consistency": signal_bundle.topic_consistency,
                "bridge_information_gain": signal_bundle.bridge_gain,
            }
        ) or (
            signal_bundle.drift_risk >= 0.62
            and signal_bundle.topic_consistency < 0.32
            and signal_bundle.bridge_gain < 0.38
        )
        if hard_signal_failure:
            return CandidateAssessment(candidate.doc_id, False, "graph_hygiene_failure", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if signal_bundle.duplicate_risk >= 0.9 and signal_bundle.bridge_gain < 0.25:
            return CandidateAssessment(candidate.doc_id, False, "duplicate_risk", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if final_result.confidence < threshold.min_confidence:
            return CandidateAssessment(candidate.doc_id, False, "low_confidence", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if blended_support < threshold.min_support:
            return CandidateAssessment(candidate.doc_id, False, "low_support", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if blended_utility < 0.24:
            return CandidateAssessment(candidate.doc_id, False, "low_utility", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if evidence_quality < threshold.min_evidence_quality:
            return CandidateAssessment(candidate.doc_id, False, "weak_evidence", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if getattr(final_result, "contradiction_flags", None) and not bool(review_result and review_result.accepted):
            return CandidateAssessment(candidate.doc_id, False, "contradiction", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)

        bridge_gain = signal_bundle.bridge_gain
        score = final_result.confidence * max(blended_support, 0.01) * max(evidence_quality, 0.01) * self._relation_prior(final_result.relation_type)
        score *= 0.84 + 0.32 * fit_scores.get(final_result.relation_type, rerank_score)
        score *= 0.82 + 0.36 * blended_utility
        score *= 0.86 + 0.28 * bridge_gain
        score *= 0.88 + 0.2 * signal_bundle.topic_consistency
        score *= 0.9 + 0.1 * max(0.0, 1.0 - signal_bundle.duplicate_risk)
        edge_confidence = final_result.confidence
        activation_profile = self._signal_runtime.compute_query_activation_profile(
            anchor,
            candidate,
            final_result.relation_type,
            local_signals={
                **metrics,
                "topic_consistency": signal_bundle.topic_consistency,
                "duplicate_risk": signal_bundle.duplicate_risk,
                "bridge_information_gain": bridge_gain,
                "query_surface_match": signal_bundle.query_surface_match,
                "utility_score": blended_utility,
                "drift_risk": signal_bundle.drift_risk,
            },
            verdict={
                "utility_score": blended_utility,
                "decision_reason": getattr(final_result, "decision_reason", ""),
                "rationale": getattr(final_result, "rationale", ""),
                "contradiction_flags": getattr(final_result, "contradiction_flags", []) or [],
            },
        )
        edge = LogicEdge(
            src_doc_id=anchor.doc_id,
            dst_doc_id=candidate.doc_id,
            relation_type=final_result.relation_type,
            confidence=edge_confidence,
            evidence_spans=final_result.evidence_spans,
            discovery_path=["scout", "judge", "review", "gate"] if self.edge_reviewer is not None else ["scout", "judge", "gate"],
            edge_card_text=f"[REL={final_result.relation_type}] {anchor.title} -> {candidate.title}: {final_result.rationale}",
            created_at=DEFAULT_TIMESTAMP,
            last_validated_at=DEFAULT_TIMESTAMP,
            utility_score=max(0.0, min(1.0, blended_utility)),
            activation_profile=activation_profile,
        )
        return CandidateAssessment(
            candidate_doc_id=candidate.doc_id,
            accepted=True,
            reject_reason="",
            score=score,
            local_support=blended_support,
            evidence_quality=evidence_quality,
            relation_type=final_result.relation_type,
            confidence=final_result.confidence,
            edge=edge,
        )

    def profile(self, doc: DocRecord) -> DocBrief:
        return self.doc_profiler.run(doc)

    def profile_many(self, docs: list[DocRecord]) -> list[DocBrief]:
        if hasattr(self.doc_profiler, "run_many"):
            return self.doc_profiler.run_many(docs)
        return [self.profile(doc) for doc in docs]

    def _live_candidate_limit(self, anchor: DocBrief, *, expanded: bool = False) -> int:
        base_limit = max(self._edge_quality().max_judge_candidates_live, 6)
        richness = (
            min(len(anchor.claims) / 3.0, 1.0)
            + min(len(anchor.relation_hints) / 4.0, 1.0)
            + self._dataset_edge_signal(anchor)
            + max(self._title_specificity_score(anchor), 0.0)
        ) / 4.0
        if richness >= 0.6:
            base_limit += 2
        if expanded:
            base_limit += 2
        return base_limit

    def scout(self, anchor: DocBrief, corpus: list[DocBrief], *, expanded: bool = False):
        brief_map = {brief.doc_id: brief for brief in corpus}
        merged: dict[str, tuple[float, CandidateProposal]] = {}
        for proposal in self.corpus_scout.run(anchor, corpus):
            candidate = brief_map.get(proposal.doc_id)
            if candidate is None:
                continue
            metrics = self._candidate_metrics(anchor, candidate)
            rerank_score, relation_type, fit_scores = self._pair_rerank(anchor, candidate, metrics)
            signal_report = self._candidate_signal_report(anchor, candidate, metrics, fit_scores, relation_type)
            if self._is_live_provider() and self._graph_hygiene_failure(signal_report):
                continue
            score = self._candidate_priority_score(
                base_score=0.58 * proposal.score_hint + 0.24 * rerank_score,
                metrics=metrics,
                fit_scores=fit_scores,
                signal_report=signal_report,
            )
            previous = merged.get(proposal.doc_id)
            if previous is None or score > previous[0]:
                merged[proposal.doc_id] = (score, CandidateProposal(doc_id=proposal.doc_id, reason=proposal.reason, query=proposal.query, score_hint=min(score, 0.99)))

        for score, proposal in self._dense_neighbor_candidate_proposals(anchor, corpus, expanded=expanded):
            previous = merged.get(proposal.doc_id)
            if previous is None or score > previous[0]:
                merged[proposal.doc_id] = (score, proposal)

        limit = self._live_candidate_limit(anchor, expanded=expanded) if self._is_live_provider() else (8 if expanded else 6)
        scored: list[tuple[float, str, CandidateProposal]] = []
        for score, proposal in merged.values():
            candidate = brief_map.get(proposal.doc_id)
            if candidate is None:
                continue
            metrics = self._candidate_metrics(anchor, candidate)
            rerank_score, relation_type, fit_scores = self._pair_rerank(anchor, candidate, metrics)
            scored.append((score + 0.18 * rerank_score, relation_type, proposal))
        scored.sort(key=lambda item: (-item[0], item[1], item[2].doc_id))
        return self._select_diverse_proposals(anchor, scored, limit)

    def judge(self, anchor: DocBrief, candidate: DocBrief) -> LogicEdge | None:
        result = self.relation_judge.run(anchor, candidate)
        review = None
        if self.edge_reviewer is not None and hasattr(self.edge_reviewer, "run_with_signals"):
            metrics = self._candidate_metrics(anchor, candidate)
            _, relation_type, fit_scores = self._pair_rerank(anchor, candidate, metrics)
            review = self.edge_reviewer.run_with_signals(anchor, candidate, self._signal_bundle(anchor, candidate, metrics, fit_scores, relation_type), result)
        assessment = self._assessment_for(anchor, candidate, result, review)
        return assessment.edge if assessment.accepted else None

    def _apply_assessment_cap(
        self,
        anchor: DocBrief,
        assessments: list[CandidateAssessment],
        candidate_map: dict[str, DocBrief],
    ) -> list[CandidateAssessment]:
        accepted = [item for item in assessments if item.accepted and item.edge is not None]
        accepted.sort(key=lambda item: (-item.score, item.candidate_doc_id))

        cap = self._edge_quality().max_edges_per_anchor_live if self._is_live_provider() else 4
        if len(accepted) >= 2:
            second = accepted[1]
            if (
                accepted[0].score - second.score <= self._edge_quality().second_edge_margin
                and second.edge is not None
                and second.edge.utility_score >= 0.5
            ):
                cap = max(cap, min(cap + 1, len(accepted)))

        kept_ids = {item.candidate_doc_id for item in self._select_diverse_assessments(anchor, accepted, candidate_map, cap)}
        final: list[CandidateAssessment] = []
        for item in assessments:
            if item.accepted and item.candidate_doc_id not in kept_ids:
                final.append(
                    CandidateAssessment(
                        candidate_doc_id=item.candidate_doc_id,
                        accepted=False,
                        reject_reason="ranked_out",
                        score=item.score,
                        local_support=item.local_support,
                        evidence_quality=item.evidence_quality,
                        relation_type=item.relation_type,
                        confidence=item.confidence,
                    )
                )
            else:
                final.append(item)
        final.sort(key=lambda item: (-item.score, item.candidate_doc_id))
        return final

    def assess_candidates_from_verdicts(
        self,
        anchor: DocBrief,
        candidates: list[DocBrief],
        verdicts: dict[str, Any],
        reviews: dict[str, Any] | None = None,
    ) -> list[CandidateAssessment]:
        candidate_map = {candidate.doc_id: candidate for candidate in candidates}
        review_map = reviews or {}
        assessments = [
            self._assessment_for(anchor, candidate, verdicts.get(candidate.doc_id), review_map.get(candidate.doc_id))
            for candidate in candidates
            if verdicts.get(candidate.doc_id) is not None
        ]
        return self._apply_assessment_cap(anchor, assessments, candidate_map)

    def judge_many_with_diagnostics(self, anchor: DocBrief, candidates: list[DocBrief]) -> list[CandidateAssessment]:
        signal_map: dict[str, JudgeSignals] = {}
        if hasattr(self.relation_judge, "run_many_with_signals"):
            candidate_pairs = []
            for candidate in candidates:
                metrics = self._candidate_metrics(anchor, candidate)
                _, relation_type, fit_scores = self._pair_rerank(anchor, candidate, metrics)
                signal_map[candidate.doc_id] = self._signal_bundle(anchor, candidate, metrics, fit_scores, relation_type)
                candidate_pairs.append((candidate, signal_map[candidate.doc_id]))
            verdicts = self.relation_judge.run_many_with_signals(anchor, candidate_pairs)
        elif hasattr(self.relation_judge, "run_many"):
            verdicts = self.relation_judge.run_many(anchor, candidates)
        else:
            verdicts = {candidate.doc_id: self.relation_judge.run(anchor, candidate) for candidate in candidates}
        if not signal_map:
            for candidate in candidates:
                metrics = self._candidate_metrics(anchor, candidate)
                _, relation_type, fit_scores = self._pair_rerank(anchor, candidate, metrics)
                signal_map[candidate.doc_id] = self._signal_bundle(anchor, candidate, metrics, fit_scores, relation_type)
        reviews: dict[str, Any] = {}
        if self.edge_reviewer is not None and hasattr(self.edge_reviewer, "run_many_with_signals"):
            review_pairs = [
                (candidate, signal_map[candidate.doc_id], verdicts[candidate.doc_id])
                for candidate in candidates
                if candidate.doc_id in verdicts and signal_map.get(candidate.doc_id) is not None
            ]
            reviews = self.edge_reviewer.run_many_with_signals(anchor, review_pairs)
        return self.assess_candidates_from_verdicts(anchor, candidates, verdicts, reviews)

    def judge_many(self, anchor: DocBrief, candidates: list[DocBrief]) -> list[LogicEdge]:
        return [item.edge for item in self.judge_many_with_diagnostics(anchor, candidates) if item.accepted and item.edge is not None]

    def curate(self, anchor: DocBrief, accepted: list[LogicEdge], rejected: list[str]) -> dict:
        return self.memory_curator.run(anchor, accepted, rejected)
