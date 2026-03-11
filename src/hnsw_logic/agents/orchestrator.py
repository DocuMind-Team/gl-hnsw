from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from hnsw_logic.config.schema import RelationQualityConfig, RetrievalConfig
from hnsw_logic.core.constants import DEFAULT_TIMESTAMP, RELATION_TYPES
from hnsw_logic.core.models import DocBrief, DocRecord, LogicEdge
from hnsw_logic.core.utils import cosine, tokenize
from hnsw_logic.embedding.provider import CandidateProposal


ROLE_WORDS = {"role", "roles", "subagent", "subagents", "profiler", "scout", "judge", "curator"}
LISTING_WORDS = {"include", "includes", "including", "role", "roles", "subagent", "subagents"}
DETAIL_CUES = {"detail", "formula", "score", "policy", "fusion", "registry", "storage", "backend", "worker", "report", "metrics", "tracked", "path"}
SUPPORT_CUES = {"support", "supports", "allow", "allows", "govern", "governs", "control", "controls", "later", "after", "used", "enters"}
ORDER_CUES = {"before", "after", "first", "then", "prior", "precedes", "depends"}
DISCOVERY_TERMS = {
    "logic", "hybrid", "jump", "candidate", "fusion", "subagent", "subagents", "memory",
    "profiler", "scout", "judge", "curator", "worker", "workers", "job", "jobs", "registry",
    "report", "benchmark", "metrics", "revalidate", "revalidation",
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
        keyword_overlap = len(set(anchor.keywords) & set(candidate.keywords))
        entity_overlap = len(set(anchor.entities) & set(candidate.entities))
        hint_overlap = len(set(anchor.relation_hints) & set(candidate.relation_hints))
        title_overlap = len(set(tokenize(anchor.title)) & set(tokenize(candidate.title)))
        overlap_score = min((keyword_overlap + entity_overlap + hint_overlap + title_overlap) / 5.0, 1.0)
        mention_score = self._mention_score(anchor, candidate)
        role_listing_score = self._role_listing_score(anchor, candidate)
        anchor_vec = self._embed_brief(anchor)
        candidate_vec = self._embed_brief(candidate)
        dense_score = max(cosine(anchor_vec, candidate_vec), 0.0) if anchor_vec is not None and candidate_vec is not None else 0.0
        length_ratio = min(len(candidate.summary), len(anchor.summary)) / max(len(candidate.summary), len(anchor.summary), 1)
        topic_drift = 1.0 if mention_score < 0.2 and title_overlap == 0 and keyword_overlap + entity_overlap + hint_overlap == 0 and dense_score < 0.45 else 0.0
        local_support = (
            0.42 * dense_score
            + 0.2 * overlap_score
            + 0.22 * mention_score
            + 0.08 * role_listing_score
            + 0.08 * min(length_ratio * 1.5, 1.0)
        )
        return {
            "dense_score": dense_score,
            "overlap_score": overlap_score,
            "keyword_overlap": float(keyword_overlap),
            "entity_overlap": float(entity_overlap),
            "hint_overlap": float(hint_overlap),
            "title_overlap": float(title_overlap),
            "length_ratio": float(length_ratio),
            "mention_score": mention_score,
            "role_listing_score": role_listing_score,
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
            return any(cue in text for cue in {"compare", "comparison", "contrast", "versus", "vs"})
        return False

    def _passes_structural_gate(self, metrics: dict[str, float], result) -> bool:
        relation_type = result.relation_type
        if relation_type == "implementation_detail":
            return (
                metrics["mention_score"] >= 0.3
                or metrics["role_listing_score"] >= 0.55
                or (metrics["overlap_score"] >= 0.6 and metrics["dense_score"] >= 0.55)
            )
        if relation_type == "supporting_evidence":
            return metrics["mention_score"] >= 0.28 or (metrics["dense_score"] >= 0.68 and metrics["overlap_score"] >= 0.4)
        if relation_type == "prerequisite":
            return metrics["role_listing_score"] >= 0.5 or (metrics["mention_score"] >= 0.28 and metrics["dense_score"] >= 0.52)
        if relation_type in {"same_concept", "comparison"}:
            return False
        return True

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
        text_tokens = set(tokenize(self._brief_text(anchor)))
        return bool(text_tokens & DISCOVERY_TERMS) or len(anchor.relation_hints) >= 4

    def _local_relation_override(self, anchor: DocBrief, candidate: DocBrief, metrics: dict[str, float], result):
        anchor_text = self._brief_text(anchor)
        candidate_text = self._brief_text(candidate)
        direct_title_link = candidate.title.lower() in anchor_text or anchor.title.lower() in candidate_text
        if metrics["role_listing_score"] >= 0.62 and metrics["mention_score"] >= 0.32:
            return self._make_fallback_result(
                anchor,
                candidate,
                "prerequisite",
                confidence=max(0.87, float(getattr(result, "confidence", 0.0))),
                reason="Anchor enumerates specialized roles and candidate is one listed role.",
                support_score=min(0.95, metrics["local_support"] + 0.18),
            )
        if direct_title_link and metrics["mention_score"] >= 0.4 and any(cue in candidate_text or cue in anchor_text for cue in DETAIL_CUES):
            return self._make_fallback_result(
                anchor,
                candidate,
                "implementation_detail",
                confidence=max(0.86, float(getattr(result, "confidence", 0.0))),
                reason="Candidate title or function is explicitly referenced and then elaborated with concrete details.",
                support_score=min(0.92, metrics["local_support"] + 0.12),
            )
        if direct_title_link and metrics["mention_score"] >= 0.38 and any(cue in candidate_text or cue in anchor_text for cue in SUPPORT_CUES):
            return self._make_fallback_result(
                anchor,
                candidate,
                "supporting_evidence",
                confidence=max(0.84, float(getattr(result, "confidence", 0.0))),
                reason="Candidate explains or constrains a behavior that the anchor depends on.",
                support_score=min(0.9, metrics["local_support"] + 0.1),
            )
        return None

    def _assessment_for(self, anchor: DocBrief, candidate: DocBrief, result) -> CandidateAssessment:
        metrics = self._candidate_metrics(anchor, candidate)
        final_result = result
        fallback = self._local_relation_override(anchor, candidate, metrics, result)
        if fallback is not None:
            force_override = fallback.relation_type == "prerequisite" and metrics["role_listing_score"] >= 0.62
            if force_override or not result.accepted or result.relation_type in {"same_concept", "comparison"} or float(result.confidence) < 0.84:
                final_result = fallback

        if not final_result.accepted:
            return CandidateAssessment(candidate.doc_id, False, "model_rejected", 0.0, 0.0, 0.0, final_result.relation_type, final_result.confidence)
        if final_result.relation_type not in RELATION_TYPES:
            return CandidateAssessment(candidate.doc_id, False, "unsupported_relation", 0.0, 0.0, 0.0, final_result.relation_type, final_result.confidence)
        if self._is_live_provider() and final_result.relation_type in {"same_concept", "comparison"}:
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, 0.0, 0.0, final_result.relation_type, final_result.confidence)

        model_support = max(0.0, min(float(getattr(final_result, "support_score", 0.0)), 1.0))
        blended_support = 0.7 * metrics["local_support"] + 0.3 * model_support
        evidence_quality = self._evidence_quality(anchor, candidate, final_result)
        threshold = self._relation_threshold(final_result.relation_type)

        if metrics["topic_drift"] >= 1.0:
            return CandidateAssessment(candidate.doc_id, False, "topic_drift", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if not threshold.enabled and not self._relation_cues(anchor, candidate, final_result):
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if not self._relation_cues(anchor, candidate, final_result):
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if not self._passes_structural_gate(metrics, final_result):
            return CandidateAssessment(candidate.doc_id, False, "weak_link", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if final_result.confidence < threshold.min_confidence:
            return CandidateAssessment(candidate.doc_id, False, "low_confidence", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if blended_support < threshold.min_support:
            return CandidateAssessment(candidate.doc_id, False, "low_support", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if evidence_quality < threshold.min_evidence_quality:
            return CandidateAssessment(candidate.doc_id, False, "weak_evidence", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)
        if getattr(final_result, "contradiction_flags", None):
            return CandidateAssessment(candidate.doc_id, False, "contradiction", 0.0, blended_support, evidence_quality, final_result.relation_type, final_result.confidence)

        score = final_result.confidence * max(blended_support, 0.01) * max(evidence_quality, 0.01) * self._relation_prior(final_result.relation_type)
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
            if self._is_live_provider() and metrics["topic_drift"] >= 1.0:
                continue
            score = metrics["local_support"] + 0.35 * metrics["mention_score"] + 0.15 * metrics["role_listing_score"]
            proposals.append(
                (
                    score,
                    CandidateProposal(
                        doc_id=candidate.doc_id,
                        reason="local dense+lexical match",
                        query=" ".join((candidate.relation_hints + candidate.keywords + [candidate.title])[:4]),
                        score_hint=min(score, 0.99),
                    ),
                )
            )
        proposals.sort(key=lambda item: (-item[0], item[1].doc_id))
        return proposals[:8]

    def _live_candidate_limit(self, anchor: DocBrief) -> int:
        anchor_text = self._brief_text(anchor)
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
            if self._is_live_provider() and metrics["topic_drift"] >= 1.0:
                continue
            score = proposal.score_hint + metrics["local_support"] + 0.2 * metrics["mention_score"]
            previous = merged.get(proposal.doc_id)
            if previous is None or score > previous[0]:
                merged[proposal.doc_id] = (score, proposal)

        for score, proposal in self._local_candidate_proposals(anchor, corpus):
            previous = merged.get(proposal.doc_id)
            if previous is None or score > previous[0]:
                merged[proposal.doc_id] = (score, proposal)

        ranked = sorted(merged.values(), key=lambda item: (-item[0], item[1].doc_id))
        limit = self._live_candidate_limit(anchor) if self._is_live_provider() else 6
        return [proposal for _, proposal in ranked[:limit]]

    def judge(self, anchor: DocBrief, candidate: DocBrief) -> LogicEdge | None:
        result = self.relation_judge.run(anchor, candidate)
        assessment = self._assessment_for(anchor, candidate, result)
        return assessment.edge if assessment.accepted else None

    def judge_many_with_diagnostics(self, anchor: DocBrief, candidates: list[DocBrief]) -> list[CandidateAssessment]:
        if hasattr(self.relation_judge, "run_many"):
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
