from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

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
    deepagent: object | None = None
    retrieval_config: RetrievalConfig | None = None
    _embedding_cache: dict[str, Any] | None = None

    def __post_init__(self):
        if self._embedding_cache is None:
            self._embedding_cache = {}
        if self.retrieval_config is None:
            self.retrieval_config = RetrievalConfig()

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

    def _source_dataset(self, brief: DocBrief) -> str:
        return str(brief.metadata.get("source_dataset", "")).lower()

    def _topic_cluster(self, brief: DocBrief) -> str:
        return str(brief.metadata.get("topic_cluster", "")).lower()

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

        if source_dataset == "arguana" or topic == "argument":
            return "argument_claim"
        if source_dataset == "scifact" or topic == "scientific_claims":
            return "scientific_evidence"
        if source_dataset == "nfcorpus" or topic == "clinical_retrieval":
            return "clinical_passage"

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
            }
            negative = {
                ("logic_fusion", "retrieval_overview"): -0.3,
                ("logic_policy", "logic_graph"): -0.34,
                ("agent_roles", "agent_overview"): -0.28,
                ("agent_memory", "agent_overview"): -0.28,
                ("ops_registry", "ops_overview"): -0.34,
                ("ops_registry", "ops_service"): -0.28,
                ("ops_service", "ops_overview"): -0.32,
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
        return bonus

    def _relation_fit_scores(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float]) -> dict[str, float]:
        direction_score = self._implementation_direction_score(anchor, candidate, metrics)
        listing_context = self._has_listing_context(anchor)
        reference_balance = max(metrics["forward_reference_score"] - 0.75 * metrics["reverse_reference_score"], 0.0)
        cue_terms = self._content_terms(anchor) | self._content_terms(candidate)
        detail_cue_score = min(len(cue_terms & DETAIL_CUES) / 5.0, 1.0)
        support_cue_score = min(len(cue_terms & SUPPORT_CUES) / 4.0, 1.0)
        comparison_cue_score = min(len(cue_terms & ARGUMENT_COMPARISON_CUES) / 5.0, 1.0)

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
            + 0.18 * metrics["topic_cluster_match"]
            + 0.18 * metrics["stance_contrast"]
            + 0.08 * comparison_cue_score
            + self._relation_stage_bonus(anchor, candidate, "comparison", metrics)
            - 0.12 * metrics["service_surface_score"]
        )
        same_concept = (
            0.26 * metrics["dense_score"]
            + 0.22 * metrics["overlap_score"]
            + 0.18 * metrics["content_overlap_score"]
            + 0.16 * metrics["mention_score"]
            + 0.16 * metrics["topic_alignment"]
            + 0.12 * metrics["topic_cluster_match"]
            - 0.16 * metrics["stance_contrast"]
        )
        return {
            "implementation_detail": max(0.0, min(implementation_detail, 1.4)),
            "supporting_evidence": max(0.0, min(supporting_evidence, 1.2)),
            "prerequisite": max(0.0, min(prerequisite, 1.2)),
            "comparison": max(0.0, min(comparison, 1.2)),
            "same_concept": max(0.0, min(same_concept, 1.1)),
        }

    def _utility_score(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float], fit_scores: dict[str, float], relation_type: str) -> float:
        score = (
            0.28 * metrics["dense_score"]
            + 0.22 * metrics["local_support"]
            + 0.14 * metrics["mention_score"]
            + 0.14 * fit_scores.get(relation_type, 0.0)
            + 0.08 * metrics["forward_reference_score"]
            + 0.06 * metrics["content_overlap_score"]
            + 0.08 * metrics["topic_alignment"]
            + max(self._relation_stage_bonus(anchor, candidate, relation_type, metrics), 0.0) * 0.1
        )
        score -= 0.16 * metrics["service_surface_score"]
        if metrics["topic_drift"] >= 1.0:
            score -= 0.24
        if relation_type == "supporting_evidence" and self._is_foundational_candidate(candidate):
            score -= 0.22
        if relation_type == "comparison":
            score += 0.08 * metrics["topic_cluster_match"] + 0.08 * metrics["stance_contrast"]
        return max(0.0, min(score, 1.0))

    def _signal_bundle(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float], fit_scores: dict[str, float], relation_type: str) -> JudgeSignals:
        stage_pair = f"{self._doc_stage(anchor)}->{self._doc_stage(candidate)}".strip("->")
        risk_flags: list[str] = []
        if metrics["topic_drift"] >= 1.0:
            risk_flags.append("topic_drift")
        if metrics["service_surface_score"] >= 0.5:
            risk_flags.append("service_surface")
        if relation_type == "supporting_evidence" and self._is_foundational_candidate(candidate):
            risk_flags.append("foundational_support")
        if relation_type == "implementation_detail" and self._implementation_direction_score(anchor, candidate, metrics) < 0.08:
            risk_flags.append("weak_direction")
        if relation_type == "comparison" and metrics["topic_cluster_match"] < 1.0:
            risk_flags.append("weak_topic_match")
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
        )

    def _pair_rerank(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float]) -> tuple[float, str, dict[str, float]]:
        fit_scores = self._relation_fit_scores(anchor, candidate, metrics)
        relation_type, score = max(fit_scores.items(), key=lambda item: (item[1], item[0]))
        return score, relation_type, fit_scores

    def _is_foundational_candidate(self, candidate: DocBrief) -> bool:
        title_tokens = set(tokenize(candidate.title))
        content_terms = self._content_terms(candidate)
        topic = str(candidate.metadata.get("topic", "")).lower()
        return (
            topic == "retrieval"
            and ("similarity" in title_tokens or len(content_terms & FOUNDATIONAL_TERMS) >= 2)
            and not (content_terms & {"hybrid", "overlay", "logic", "jump", "fusion"})
        )

    def _workflow_prerequisite_signal(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float]) -> bool:
        combined = f"{anchor.title} {anchor.summary} {' '.join(anchor.claims)} {candidate.title} {candidate.summary} {' '.join(candidate.claims)}".lower()
        anchor_tokens = set(tokenize(anchor.title))
        candidate_tokens = set(tokenize(candidate.title))
        if "scout" in anchor_tokens and "judge" in candidate_tokens:
            return True
        if any(cue in combined for cue in ORDER_CUES):
            return metrics["specific_role_score"] >= 0.5 or self._doc_stage(candidate) == "agent_roles"
        return False

    def _topic_drift_exception(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float], fit_scores: dict[str, float]) -> bool:
        if metrics["topic_drift"] < 1.0:
            return True
        if fit_scores["prerequisite"] >= 0.72 and (
            metrics["specific_role_score"] >= 0.5
            or self._relation_stage_bonus(anchor, candidate, "prerequisite", metrics) >= 0.22
            or self._workflow_prerequisite_signal(anchor, candidate, metrics)
        ):
            return True
        if (
            fit_scores["supporting_evidence"] >= 0.28
            and self._relation_stage_bonus(anchor, candidate, "supporting_evidence", metrics) >= 0.22
            and not self._is_foundational_candidate(candidate)
        ):
            return True
        if (
            fit_scores["implementation_detail"] >= 0.72
            and self._relation_stage_bonus(anchor, candidate, "implementation_detail", metrics) >= 0.22
        ):
            return True
        if (
            self._is_argumentative_pair(anchor, candidate)
            and fit_scores["comparison"] >= 0.52
            and metrics["topic_cluster_match"] >= 1.0
            and (metrics["stance_contrast"] >= 1.0 or metrics["content_overlap_score"] >= 0.18)
        ):
            return True
        return False

    def _targeted_candidate_proposals(self, anchor: DocBrief, corpus: list[DocBrief]) -> list[tuple[float, CandidateProposal]]:
        anchor_stage = self._doc_stage(anchor)
        proposals: list[tuple[float, CandidateProposal]] = []
        for candidate in corpus:
            if candidate.doc_id == anchor.doc_id:
                continue
            metrics = self._candidate_metrics(anchor, candidate)
            _, relation_type, fit_scores = self._pair_rerank(anchor, candidate, metrics)
            if not self._topic_drift_exception(anchor, candidate, metrics, fit_scores):
                continue
            stage_bonus = max(
                self._relation_stage_bonus(anchor, candidate, "implementation_detail", metrics),
                self._relation_stage_bonus(anchor, candidate, "supporting_evidence", metrics),
                self._relation_stage_bonus(anchor, candidate, "prerequisite", metrics),
                self._relation_stage_bonus(anchor, candidate, "comparison", metrics),
            )
            score = max(fit_scores.values()) + 0.22 * stage_bonus
            if anchor_stage == "agent_roles" and fit_scores["prerequisite"] >= 0.72 and (
                self._has_listing_context(anchor) >= 1.0 or self._workflow_prerequisite_signal(anchor, candidate, metrics)
            ):
                proposals.append(
                    (
                        score + 0.16,
                        CandidateProposal(
                            doc_id=candidate.doc_id,
                            reason="targeted role/workflow prerequisite candidate",
                            query=" ".join((candidate.keywords + candidate.relation_hints + [candidate.title])[:4]),
                            score_hint=min(score + 0.16, 0.99),
                        ),
                    )
                )
            if fit_scores["supporting_evidence"] >= 0.28 and self._relation_stage_bonus(anchor, candidate, "supporting_evidence", metrics) >= 0.22:
                proposals.append(
                    (
                        score + 0.1,
                        CandidateProposal(
                            doc_id=candidate.doc_id,
                            reason="targeted stage-supporting evidence candidate",
                            query=" ".join((candidate.relation_hints + candidate.keywords + [candidate.title])[:4]),
                            score_hint=min(score + 0.1, 0.99),
                        ),
                    )
                )
            if anchor_stage == "agent_overview" and fit_scores["implementation_detail"] >= 0.78 and self._relation_stage_bonus(anchor, candidate, "implementation_detail", metrics) >= 0.22:
                proposals.append(
                    (
                        score + 0.08,
                        CandidateProposal(
                            doc_id=candidate.doc_id,
                            reason="targeted overview implementation-detail candidate",
                            query=" ".join((candidate.keywords + [candidate.title])[:4]),
                            score_hint=min(score + 0.08, 0.99),
                        ),
                    )
                )
            if (
                self._is_argumentative_pair(anchor, candidate)
                and metrics["topic_cluster_match"] >= 1.0
                and fit_scores["comparison"] >= 0.58
                and (metrics["stance_contrast"] >= 1.0 or metrics["content_overlap_score"] >= 0.18)
            ):
                proposals.append(
                    (
                        score + 0.18 + 0.08 * metrics["stance_contrast"],
                        CandidateProposal(
                            doc_id=candidate.doc_id,
                            reason="targeted argumentative comparison candidate",
                            query=" ".join((candidate.keywords + candidate.relation_hints + [candidate.title])[:4]),
                            score_hint=min(score + 0.18, 0.99),
                        ),
                    )
                )
        proposals.sort(key=lambda item: (-item[0], item[1].doc_id))
        dedup: dict[str, tuple[float, CandidateProposal]] = {}
        for score, proposal in proposals:
            previous = dedup.get(proposal.doc_id)
            if previous is None or score > previous[0]:
                dedup[proposal.doc_id] = (score, proposal)
        ranked = sorted(dedup.values(), key=lambda item: (-item[0], item[1].doc_id))
        return ranked[:8]

    def _proposal_bucket_quotas(self, anchor: DocBrief, limit: int) -> dict[str, int]:
        stage = self._doc_stage(anchor)
        if stage == "argument_claim":
            return {"comparison": min(limit, 4), "same_concept": min(limit, 2), "supporting_evidence": min(limit, 1)}
        if stage == "agent_roles":
            return {"prerequisite": min(limit, 4), "implementation_detail": min(limit, 2), "supporting_evidence": min(limit, 1)}
        if stage in {"logic_graph", "retrieval_overview", "ops_overview", "ops_service", "agent_overview"}:
            return {"implementation_detail": min(limit, 3), "supporting_evidence": min(limit, 2), "prerequisite": min(limit, 1)}
        return {"implementation_detail": min(limit, 2), "supporting_evidence": min(limit, 2), "prerequisite": min(limit, 1)}

    def _select_diverse_proposals(
        self,
        anchor: DocBrief,
        scored: list[tuple[float, str, CandidateProposal]],
        limit: int,
    ) -> list[CandidateProposal]:
        quotas = self._proposal_bucket_quotas(anchor, limit)
        buckets: dict[str, list[tuple[float, CandidateProposal]]] = {key: [] for key in quotas}
        for score, relation_type, proposal in scored:
            buckets.setdefault(relation_type, []).append((score, proposal))

        selected: list[CandidateProposal] = []
        selected_ids: set[str] = set()
        for relation_type, quota in quotas.items():
            for score, proposal in buckets.get(relation_type, []):
                if len(selected) >= limit:
                    break
                if quota <= 0 or proposal.doc_id in selected_ids:
                    continue
                selected.append(proposal)
                selected_ids.add(proposal.doc_id)
                quota -= 1
            if len(selected) >= limit:
                break

        for score, relation_type, proposal in sorted(scored, key=lambda item: (-item[0], item[1], item[2].doc_id)):
            if len(selected) >= limit:
                break
            if proposal.doc_id in selected_ids:
                continue
            selected.append(proposal)
            selected_ids.add(proposal.doc_id)
        return selected[:limit]

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
        anchor_content_terms = self._content_terms(anchor)
        candidate_content_terms = self._content_terms(candidate)
        keyword_overlap = len(set(anchor.keywords) & set(candidate.keywords))
        entity_overlap = len(set(anchor.entities) & set(candidate.entities))
        hint_overlap = len(set(anchor.relation_hints) & set(candidate.relation_hints))
        title_overlap = len(set(tokenize(anchor.title)) & set(tokenize(candidate.title)))
        overlap_score = min((keyword_overlap + entity_overlap + hint_overlap + title_overlap) / 5.0, 1.0)
        content_overlap = len(anchor_content_terms & candidate_content_terms)
        content_overlap_score = min(content_overlap / 6.0, 1.0)
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
        topic_cluster_match = 1.0 if self._topic_cluster(anchor) and self._topic_cluster(anchor) == self._topic_cluster(candidate) else 0.0
        stance_contrast = 1.0 if self._stance(anchor) and self._stance(candidate) and self._stance(anchor) != self._stance(candidate) else 0.0
        topic_alignment = 1.0 if (
            (anchor.metadata.get("topic") and anchor.metadata.get("topic") == candidate.metadata.get("topic"))
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

    def _relation_cues(self, anchor: DocBrief, candidate: DocBrief, result) -> bool:
        relation_type = result.relation_type
        text = " ".join(
            [
                anchor.title,
                anchor.summary,
                candidate.title,
                candidate.summary,
                result.rationale,
                getattr(result, "decision_reason", ""),
                *result.evidence_spans,
            ]
        ).lower()
        if relation_type == "supporting_evidence":
            return any(cue in text for cue in SUPPORT_CUES)
        if relation_type == "implementation_detail":
            return any(cue in text for cue in DETAIL_CUES) or candidate.title.lower() in self._brief_text(anchor)
        if relation_type == "prerequisite":
            return any(cue in text for cue in ORDER_CUES | LISTING_WORDS)
        if relation_type == "same_concept":
            return any(cue in text for cue in {"same", "alias", "equivalent", "become", "becomes"})
        if relation_type == "comparison":
            return self._is_argumentative_pair(anchor, candidate) or any(cue in text for cue in {"compare", "comparison", "contrast", "versus", "vs", "counterargument", "opposing"})
        return False

    def _passes_structural_gate(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float], result) -> bool:
        relation_type = result.relation_type
        if relation_type == "implementation_detail":
            direction_score = self._implementation_direction_score(anchor, candidate, metrics)
            semantic_detail_bridge = (
                metrics["dense_score"] >= 0.68
                and metrics["service_surface_score"] < 0.55
                and metrics["family_bridge_score"] >= 0.75
                and metrics["shared_dominant_family"] >= 1.0
                and direction_score >= 0.08
                and (
                    metrics["content_overlap_score"] >= 0.18
                    or metrics["topic_alignment"] >= 1.0
                    or metrics["mention_score"] >= 0.18
                )
            )
            return (
                metrics["mention_score"] >= 0.3
                or metrics["role_listing_score"] >= 0.55
                or (metrics["overlap_score"] >= 0.6 and metrics["dense_score"] >= 0.55)
                or semantic_detail_bridge
            )
        if relation_type == "supporting_evidence":
            return (
                metrics["service_surface_score"] < 0.35
                and metrics["dense_score"] >= 0.58
                and (
                    metrics["mention_score"] >= 0.18
                    or metrics["content_overlap_score"] >= 0.18
                    or metrics["family_bridge_score"] >= 0.42
                )
            )
        if relation_type == "prerequisite":
            return metrics["specific_role_score"] >= 0.5 and (
                metrics["role_listing_score"] >= 0.5 or (metrics["mention_score"] >= 0.28 and metrics["dense_score"] >= 0.52)
            )
        if relation_type == "comparison":
            return (
                metrics["topic_cluster_match"] >= 1.0
                and (
                    (
                        metrics["stance_contrast"] >= 1.0
                        and (
                            metrics["dense_score"] >= 0.18
                            or metrics["content_overlap_score"] >= 0.12
                            or metrics["overlap_score"] >= 0.18
                        )
                    )
                    or metrics["content_overlap_score"] >= 0.22
                    or metrics["overlap_score"] >= 0.28
                )
            )
        if relation_type == "same_concept":
            return metrics["dense_score"] >= 0.55 and (
                metrics["overlap_score"] >= 0.3
                or metrics["topic_cluster_match"] >= 1.0
            )
        return True

    def _relation_threshold(self, relation_type: str) -> RelationQualityConfig:
        return self._edge_quality().relation_thresholds.get(relation_type, RelationQualityConfig())

    def _effective_threshold(self, anchor: DocBrief, candidate: DocBrief, relation_type: str) -> RelationQualityConfig:
        threshold = self._relation_threshold(relation_type)
        if relation_type == "comparison" and self._is_argumentative_pair(anchor, candidate):
            return RelationQualityConfig(enabled=True, min_confidence=0.8, min_support=0.26, min_evidence_quality=0.28)
        if relation_type == "same_concept" and self._is_argumentative_pair(anchor, candidate):
            return RelationQualityConfig(enabled=True, min_confidence=0.82, min_support=0.3, min_evidence_quality=0.28)
        return threshold

    def _is_live_provider(self) -> bool:
        provider = self._provider()
        return provider is not None and provider.__class__.__name__ == "OpenAICompatibleProvider"

    def _relation_prior(self, relation_type: str) -> float:
        provider = self._provider()
        if provider is None:
            return 1.0
        return float(getattr(provider, "relation_priors", {}).get(relation_type, 1.0))

    def _make_fallback_result(self, anchor: DocBrief, candidate: DocBrief, relation_type: str, confidence: float, reason: str, support_score: float):
        return SimpleNamespace(
            accepted=True,
            relation_type=relation_type,
            confidence=confidence,
            evidence_spans=[anchor.summary[:160], candidate.summary[:160]],
            rationale=reason,
            support_score=support_score,
            contradiction_flags=[],
            decision_reason="Accepted by local fallback after strong lexical and structural match.",
        )

    def should_attempt_discovery(self, anchor: DocBrief) -> bool:
        topic = str(anchor.metadata.get("topic", "")).lower()
        source_dataset = str(anchor.metadata.get("source_dataset", "")).lower()
        anchor_terms = self._content_terms(anchor)
        title_tokens = set(tokenize(anchor.title))
        if source_dataset == "arguana":
            return len(anchor.claims) >= 1 and len(anchor_terms) >= 8 and bool(self._topic_cluster(anchor))
        if source_dataset in {"scifact", "nfcorpus"}:
            return len(anchor.claims) >= 1 and (len(anchor.keywords) >= 3 or len(anchor.entities) >= 2 or len(anchor_terms) >= 8)
        if topic in {"hnsw", "evaluation"}:
            return False
        if "similarity" in title_tokens:
            return False
        if topic == "retrieval" and len(anchor_terms & FOUNDATIONAL_TERMS) >= 2 and not (anchor_terms & {"hybrid", "overlay", "logic"}):
            return False
        if topic in {"retrieval", "logic", "deepagents", "ops"}:
            return True
        text_tokens = set(tokenize(self._brief_text(anchor)))
        return bool(text_tokens & DISCOVERY_TERMS) or len(anchor.relation_hints) >= 4

    def discovery_anchor_priority(self, anchor: DocBrief) -> float:
        source_dataset = self._source_dataset(anchor)
        content_terms = self._content_terms(anchor)
        keyword_score = min(len(anchor.keywords) / 8.0, 1.0)
        entity_score = min(len(anchor.entities) / 4.0, 1.0)
        claim_score = min(len([claim for claim in anchor.claims if claim.strip()]) / 2.0, 1.0)
        summary_score = min(len(anchor.summary) / 180.0, 1.0)
        hint_terms = set(anchor.relation_hints)
        if source_dataset == "arguana":
            return (
                0.3 * (1.0 if self._topic_cluster(anchor) else 0.0)
                + 0.24 * (1.0 if self._stance(anchor) else 0.0)
                + 0.18 * claim_score
                + 0.16 * keyword_score
                + 0.12 * summary_score
            )
        if source_dataset == "scifact":
            evidence_hint = 1.0 if hint_terms & {"claim", "evidence", "study"} else 0.0
            return (
                0.28 * entity_score
                + 0.24 * claim_score
                + 0.18 * evidence_hint
                + 0.16 * keyword_score
                + 0.14 * min(len(content_terms) / 14.0, 1.0)
            )
        if source_dataset == "nfcorpus":
            clinical_hint = 1.0 if hint_terms & {"clinical", "condition", "treatment"} else 0.0
            return (
                0.24 * claim_score
                + 0.22 * keyword_score
                + 0.18 * entity_score
                + 0.18 * clinical_hint
                + 0.18 * summary_score
            )
        return 1.0

    def select_discovery_anchors(self, briefs: list[DocBrief]) -> set[str]:
        if not briefs:
            return set()
        grouped: dict[str, list[DocBrief]] = {}
        for brief in briefs:
            grouped.setdefault(self._source_dataset(brief), []).append(brief)
        selected: set[str] = set()
        for source_dataset, group in grouped.items():
            if source_dataset not in {"arguana", "scifact", "nfcorpus"}:
                selected.update(brief.doc_id for brief in group)
                continue
            if source_dataset == "arguana":
                cap = max(12, min(len(group), len(group) // 5 + 8))
                floor = 0.5
            elif source_dataset == "scifact":
                cap = max(16, min(len(group), len(group) // 4))
                floor = 0.48
            else:
                cap = max(14, min(len(group), len(group) // 4))
                floor = 0.46
            ranked = sorted(
                ((self.discovery_anchor_priority(brief), brief.doc_id) for brief in group),
                key=lambda item: (-item[0], item[1]),
            )
            kept = [doc_id for score, doc_id in ranked if score >= floor][:cap]
            if not kept:
                kept = [doc_id for _, doc_id in ranked[: min(cap, max(6, len(group) // 6 or 1))]]
            selected.update(kept)
        return selected

    def _local_relation_override(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float], result):
        anchor_text = self._brief_text(anchor)
        candidate_text = self._brief_text(candidate)
        direct_title_link = candidate.title.lower() in anchor_text or anchor.title.lower() in candidate_text
        _, _, fit_scores = self._pair_rerank(anchor, candidate, metrics)
        if (
            self._doc_stage(anchor) == "agent_roles"
            and self._has_listing_context(anchor) >= 1.0
            and metrics["specific_role_score"] >= 0.5
            and fit_scores["prerequisite"] >= 0.72
            and metrics["role_listing_score"] >= 0.5
        ):
            return self._make_fallback_result(
                anchor,
                candidate,
                "prerequisite",
                confidence=max(0.9, float(getattr(result, "confidence", 0.0))),
                reason="Anchor enumerates specialized roles and candidate is one listed role.",
                support_score=min(0.95, metrics["local_support"] + 0.18),
            )
        if (
            self._doc_stage(anchor) == "agent_roles"
            and self._workflow_prerequisite_signal(anchor, candidate, metrics)
            and fit_scores["prerequisite"] >= 0.74
        ):
            return self._make_fallback_result(
                anchor,
                candidate,
                "prerequisite",
                confidence=max(0.9, float(getattr(result, "confidence", 0.0))),
                reason="Candidate is the next explicit workflow role implied by the anchor's ordering cues.",
                support_score=min(0.9, metrics["local_support"] + 0.16),
            )
        if (
            metrics["service_surface_score"] < 0.55
            and fit_scores["implementation_detail"] >= 0.6
            and (
                (direct_title_link and metrics["mention_score"] >= 0.28 and any(cue in candidate_text or cue in anchor_text for cue in DETAIL_CUES))
                or (
                    metrics["dense_score"] >= 0.68
                    and metrics["family_bridge_score"] >= 0.75
                    and metrics["shared_dominant_family"] >= 1.0
                    and self._implementation_direction_score(anchor, candidate, metrics) >= 0.08
                    and (
                        metrics["content_overlap_score"] >= 0.18
                        or metrics["topic_alignment"] >= 1.0
                        or metrics["mention_score"] >= 0.18
                    )
                )
            )
        ):
            return self._make_fallback_result(
                anchor,
                candidate,
                "implementation_detail",
                confidence=max(0.86, float(getattr(result, "confidence", 0.0))),
                reason="Anchor and candidate share a high-confidence mechanism or component-level semantic match.",
                support_score=min(0.92, metrics["local_support"] + 0.12),
            )
        if (
            metrics["service_surface_score"] < 0.35
            and fit_scores["supporting_evidence"] >= 0.45
            and direct_title_link
            and metrics["mention_score"] >= 0.28
            and metrics["content_overlap_score"] >= 0.14
            and any(cue in candidate_text or cue in anchor_text for cue in SUPPORT_CUES)
        ):
            return self._make_fallback_result(
                anchor,
                candidate,
                "supporting_evidence",
                confidence=max(0.84, float(getattr(result, "confidence", 0.0))),
                reason="Candidate explains or constrains a behavior that the anchor depends on.",
                support_score=min(0.9, metrics["local_support"] + 0.1),
            )
        if (
            self._is_argumentative_pair(anchor, candidate)
            and fit_scores["comparison"] >= 0.58
            and metrics["topic_cluster_match"] >= 1.0
            and (metrics["stance_contrast"] >= 1.0 or metrics["content_overlap_score"] >= 0.18)
        ):
            return self._make_fallback_result(
                anchor,
                candidate,
                "comparison",
                confidence=max(0.84, float(getattr(result, "confidence", 0.0))),
                reason="The documents argue about the same topic from contrasting or alternative positions.",
                support_score=min(0.9, metrics["local_support"] + 0.12),
            )
        return None

    def _assessment_for(self, anchor: DocBrief, candidate: DocBrief, result) -> CandidateAssessment:
        metrics = self._candidate_metrics(anchor, candidate)
        rerank_score, rerank_relation, fit_scores = self._pair_rerank(anchor, candidate, metrics)
        signal_bundle = self._signal_bundle(anchor, candidate, metrics, fit_scores, rerank_relation)
        final_result = result
        fallback = self._local_relation_override(anchor, candidate, metrics, result)
        if fallback is not None:
            force_override = (
                fallback.relation_type == "prerequisite"
                and (
                    (metrics["specific_role_score"] >= 0.5 and fit_scores["prerequisite"] >= 0.72)
                    or self._workflow_prerequisite_signal(anchor, candidate, metrics)
                )
            )
            detail_override = (
                fallback.relation_type == "implementation_detail"
                and metrics["dense_score"] >= 0.68
                and metrics["service_surface_score"] < 0.55
                and (
                    not result.accepted
                    or result.relation_type in {"same_concept", "comparison", "prerequisite", "supporting_evidence"}
                    or float(result.confidence) < 0.88
                )
            )
            if (
                force_override
                or detail_override
                or not result.accepted
                or result.relation_type in {"same_concept", "comparison"}
                or (
                    fallback.relation_type == "prerequisite"
                    and float(result.confidence) < 0.9
                )
                or float(result.confidence) < 0.84
            ):
                final_result = fallback

        if not final_result.accepted:
            return CandidateAssessment(candidate.doc_id, False, "model_rejected", 0.0, 0.0, 0.0, final_result.relation_type, final_result.confidence)
        if final_result.relation_type not in RELATION_TYPES:
            return CandidateAssessment(candidate.doc_id, False, "unsupported_relation", 0.0, 0.0, 0.0, final_result.relation_type, final_result.confidence)
        if (
            self._is_live_provider()
            and final_result.relation_type in {"same_concept", "comparison"}
            and not (final_result.relation_type == "comparison" and self._is_argumentative_pair(anchor, candidate))
        ):
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, 0.0, 0.0, final_result.relation_type, final_result.confidence)

        model_support = max(0.0, min(float(getattr(final_result, "support_score", 0.0)), 1.0))
        blended_support = 0.7 * metrics["local_support"] + 0.3 * model_support
        model_utility = max(0.0, min(float(getattr(final_result, "utility_score", 0.0)), 1.0))
        blended_utility = max(signal_bundle.utility_score, 0.6 * signal_bundle.utility_score + 0.4 * model_utility)
        evidence_quality = self._evidence_quality(anchor, candidate, final_result)
        threshold = self._effective_threshold(anchor, candidate, final_result.relation_type)

        if metrics["topic_drift"] >= 1.0 and not self._topic_drift_exception(anchor, candidate, metrics, fit_scores):
            return CandidateAssessment(candidate.doc_id, False, "topic_drift", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if not threshold.enabled and not self._relation_cues(anchor, candidate, final_result):
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if not self._relation_cues(anchor, candidate, final_result):
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if final_result.relation_type == "implementation_detail" and fit_scores["implementation_detail"] + 0.04 < max(fit_scores["supporting_evidence"], fit_scores["prerequisite"]):
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if final_result.relation_type == "supporting_evidence" and fit_scores["supporting_evidence"] < 0.25:
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if final_result.relation_type == "prerequisite" and fit_scores["prerequisite"] < 0.38:
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if final_result.relation_type == "comparison" and fit_scores["comparison"] < 0.4:
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if final_result.relation_type == "supporting_evidence" and self._is_foundational_candidate(candidate):
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if not self._passes_structural_gate(anchor, candidate, metrics, final_result):
            return CandidateAssessment(candidate.doc_id, False, "weak_link", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if final_result.relation_type == "implementation_detail":
            direction_score = self._implementation_direction_score(anchor, candidate, metrics)
            if direction_score < 0.08:
                return CandidateAssessment(candidate.doc_id, False, "wrong_direction", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
            if self._relation_stage_bonus(anchor, candidate, "implementation_detail", metrics) < -0.08:
                return CandidateAssessment(candidate.doc_id, False, "wrong_direction", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if final_result.relation_type == "supporting_evidence":
            if self._relation_stage_bonus(anchor, candidate, "supporting_evidence", metrics) < -0.08:
                return CandidateAssessment(candidate.doc_id, False, "wrong_direction", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if final_result.relation_type == "comparison" and self._is_argumentative_pair(anchor, candidate) and metrics["topic_cluster_match"] < 1.0:
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if final_result.confidence < threshold.min_confidence:
            return CandidateAssessment(candidate.doc_id, False, "low_confidence", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if blended_support < threshold.min_support:
            return CandidateAssessment(candidate.doc_id, False, "low_support", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if self._is_live_provider() and blended_utility < 0.24:
            return CandidateAssessment(candidate.doc_id, False, "low_utility", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if evidence_quality < threshold.min_evidence_quality:
            return CandidateAssessment(candidate.doc_id, False, "weak_evidence", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if getattr(final_result, "contradiction_flags", None):
            return CandidateAssessment(candidate.doc_id, False, "contradiction", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)

        score = final_result.confidence * max(blended_support, 0.01) * max(evidence_quality, 0.01) * self._relation_prior(final_result.relation_type)
        score *= 1.0 + 0.16 * fit_scores.get(final_result.relation_type, rerank_score)
        score *= 0.85 + 0.4 * blended_utility
        if final_result.relation_type == "implementation_detail":
            score *= 1.0 + max(self._implementation_direction_score(anchor, candidate, metrics), 0.0)
        edge = LogicEdge(
            src_doc_id=anchor.doc_id,
            dst_doc_id=candidate.doc_id,
            relation_type=final_result.relation_type,
            confidence=final_result.confidence,
            evidence_spans=final_result.evidence_spans,
            discovery_path=["scout", "judge", "gate"],
            edge_card_text=f"[REL={final_result.relation_type}] {anchor.title} -> {candidate.title}: {final_result.rationale}",
            created_at=DEFAULT_TIMESTAMP,
            last_validated_at=DEFAULT_TIMESTAMP,
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

    def _local_candidate_proposals(self, anchor: DocBrief, corpus: list[DocBrief]) -> list[tuple[float, CandidateProposal]]:
        proposals: list[tuple[float, CandidateProposal]] = []
        for candidate in corpus:
            if candidate.doc_id == anchor.doc_id:
                continue
            metrics = self._candidate_metrics(anchor, candidate)
            rerank_score, relation_type, fit_scores = self._pair_rerank(anchor, candidate, metrics)
            if self._is_live_provider() and not self._topic_drift_exception(anchor, candidate, metrics, fit_scores):
                continue
            score = (
                metrics["local_support"]
                + 0.42 * rerank_score
                + 0.08 * fit_scores["implementation_detail"]
                + 0.06 * fit_scores["supporting_evidence"]
                + 0.08 * fit_scores.get("comparison", 0.0)
            )
            proposals.append(
                (
                    score,
                    CandidateProposal(
                        doc_id=candidate.doc_id,
                        reason=f"local rerank favors {relation_type}",
                        query=" ".join((candidate.relation_hints + candidate.keywords + [candidate.title])[:4]),
                        score_hint=min(score, 0.99),
                    ),
                )
            )
        proposals.sort(key=lambda item: (-item[0], item[1].doc_id))
        return proposals[:8]

    def _live_candidate_limit(self, anchor: DocBrief) -> int:
        anchor_text = self._brief_text(anchor)
        if self._is_argumentative_doc(anchor):
            return max(self._edge_quality().max_judge_candidates_live, 8)
        if any(word in anchor_text for word in LISTING_WORDS):
            return max(self._edge_quality().max_judge_candidates_live, 8)
        return max(self._edge_quality().max_judge_candidates_live, 6)

    def scout(self, anchor: DocBrief, corpus: list[DocBrief]):
        brief_map = {brief.doc_id: brief for brief in corpus}
        merged: dict[str, tuple[float, CandidateProposal]] = {}
        for proposal in self.corpus_scout.run(anchor, corpus):
            candidate = brief_map.get(proposal.doc_id)
            if candidate is None:
                continue
            metrics = self._candidate_metrics(anchor, candidate)
            rerank_score, relation_type, fit_scores = self._pair_rerank(anchor, candidate, metrics)
            if self._is_live_provider() and not self._topic_drift_exception(anchor, candidate, metrics, fit_scores):
                continue
            score = (
                0.55 * proposal.score_hint
                + metrics["local_support"]
                + 0.34 * rerank_score
                + 0.06 * fit_scores["implementation_detail"]
                + 0.04 * fit_scores["supporting_evidence"]
                + 0.08 * fit_scores.get("comparison", 0.0)
            )
            previous = merged.get(proposal.doc_id)
            if previous is None or score > previous[0]:
                merged[proposal.doc_id] = (score, CandidateProposal(doc_id=proposal.doc_id, reason=proposal.reason, query=proposal.query, score_hint=min(score, 0.99)))

        for score, proposal in self._local_candidate_proposals(anchor, corpus):
            previous = merged.get(proposal.doc_id)
            if previous is None or score > previous[0]:
                merged[proposal.doc_id] = (score, proposal)

        for score, proposal in self._targeted_candidate_proposals(anchor, corpus):
            previous = merged.get(proposal.doc_id)
            if previous is None or score > previous[0]:
                merged[proposal.doc_id] = (score, proposal)

        limit = self._live_candidate_limit(anchor) if self._is_live_provider() else 6
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
        assessment = self._assessment_for(anchor, candidate, result)
        return assessment.edge if assessment.accepted else None

    def judge_many_with_diagnostics(self, anchor: DocBrief, candidates: list[DocBrief]) -> list[CandidateAssessment]:
        if hasattr(self.relation_judge, "run_many_with_signals"):
            candidate_pairs = []
            for candidate in candidates:
                metrics = self._candidate_metrics(anchor, candidate)
                _, relation_type, fit_scores = self._pair_rerank(anchor, candidate, metrics)
                candidate_pairs.append((candidate, self._signal_bundle(anchor, candidate, metrics, fit_scores, relation_type)))
            verdicts = self.relation_judge.run_many_with_signals(anchor, candidate_pairs)
        elif hasattr(self.relation_judge, "run_many"):
            verdicts = self.relation_judge.run_many(anchor, candidates)
        else:
            verdicts = {candidate.doc_id: self.relation_judge.run(anchor, candidate) for candidate in candidates}

        assessments = [self._assessment_for(anchor, candidate, verdicts.get(candidate.doc_id)) for candidate in candidates if verdicts.get(candidate.doc_id) is not None]
        accepted = [item for item in assessments if item.accepted and item.edge is not None]
        accepted.sort(key=lambda item: (-item.score, item.candidate_doc_id))

        cap = self._edge_quality().max_edges_per_anchor_live if self._is_live_provider() else 4
        if self._is_live_provider() and any(word in self._brief_text(anchor) for word in LISTING_WORDS):
            prerequisite_group = [item for item in accepted if item.relation_type == "prerequisite" and item.local_support >= 0.42]
            if len(prerequisite_group) >= 2:
                cap = max(cap, min(4, len(prerequisite_group)))
        if self._is_live_provider() and self._doc_stage(anchor) == "agent_overview" and len(accepted) >= 2:
            if accepted[0].score - accepted[1].score <= self._edge_quality().second_edge_margin:
                cap = max(cap, 2)
        if self._is_live_provider() and self._doc_stage(anchor) == "argument_claim":
            comparison_group = [item for item in accepted if item.relation_type == "comparison" and item.local_support >= 0.34]
            if len(comparison_group) >= 2 and comparison_group[0].score - comparison_group[1].score <= self._edge_quality().second_edge_margin + 0.04:
                cap = max(cap, 2)

        kept_ids = {item.candidate_doc_id for item in accepted[:cap]}
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

    def judge_many(self, anchor: DocBrief, candidates: list[DocBrief]) -> list[LogicEdge]:
        return [item.edge for item in self.judge_many_with_diagnostics(anchor, candidates) if item.accepted and item.edge is not None]

    def curate(self, anchor: DocBrief, accepted: list[LogicEdge], rejected: list[str]) -> dict:
        return self.memory_curator.run(anchor, accepted, rejected)
