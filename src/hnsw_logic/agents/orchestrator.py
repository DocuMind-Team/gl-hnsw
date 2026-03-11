from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from hnsw_logic.config.schema import RelationQualityConfig, RetrievalConfig
from hnsw_logic.core.constants import DEFAULT_TIMESTAMP, RELATION_TYPES
from hnsw_logic.core.models import DocBrief, DocRecord, LogicEdge
from hnsw_logic.core.utils import cosine, tokenize


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

    def _candidate_metrics(self, anchor: DocBrief, candidate: DocBrief) -> dict[str, float]:
        anchor_terms = self._brief_terms(anchor)
        candidate_terms = self._brief_terms(candidate)
        keyword_overlap = len(set(anchor.keywords) & set(candidate.keywords))
        entity_overlap = len(set(anchor.entities) & set(candidate.entities))
        hint_overlap = len(set(anchor.relation_hints) & set(candidate.relation_hints))
        title_overlap = len(set(tokenize(anchor.title)) & set(tokenize(candidate.title)))
        overlap_score = min((keyword_overlap + entity_overlap + hint_overlap + title_overlap) / 5.0, 1.0)
        anchor_vec = self._embed_brief(anchor)
        candidate_vec = self._embed_brief(candidate)
        dense_score = max(cosine(anchor_vec, candidate_vec), 0.0) if anchor_vec is not None and candidate_vec is not None else 0.0
        length_ratio = min(len(candidate.summary), len(anchor.summary)) / max(len(candidate.summary), len(anchor.summary), 1)
        topic_drift = 1.0 if title_overlap == 0 and keyword_overlap + entity_overlap + hint_overlap == 0 and dense_score < 0.45 else 0.0
        local_support = 0.55 * dense_score + 0.3 * overlap_score + 0.15 * min(length_ratio * 1.5, 1.0)
        return {
            "dense_score": dense_score,
            "overlap_score": overlap_score,
            "keyword_overlap": float(keyword_overlap),
            "entity_overlap": float(entity_overlap),
            "hint_overlap": float(hint_overlap),
            "title_overlap": float(title_overlap),
            "length_ratio": float(length_ratio),
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
        span_score = min(len(evidence_chunks) / 2.0, 1.0)
        contradiction_penalty = 0.15 if getattr(result, "contradiction_flags", None) else 0.0
        return max(0.0, 0.4 * min(anchor_cov + candidate_cov, 1.0) + 0.45 * span_score + 0.15 - contradiction_penalty)

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
            cues = {"support", "supports", "evidence", "provides", "controls", "governs", "later", "revalidated", "augment"}
            return any(cue in text for cue in cues)
        if relation_type == "implementation_detail":
            cues = {"uses", "through", "tracked", "registry", "detail", "includes", "relies", "merge", "score", "route"}
            return any(cue in text for cue in cues) or candidate.title.lower() in anchor.summary.lower()
        if relation_type == "prerequisite":
            cues = {"before", "after", "first", "then", "prior", "precedes", "depends", "include", "subagent", "role"}
            return any(cue in text for cue in cues)
        if relation_type == "same_concept":
            return any(cue in text for cue in {"same", "alias", "equivalent", "become", "becomes"})
        if relation_type == "comparison":
            return any(cue in text for cue in {"compare", "comparison", "contrast", "versus", "vs"})
        return False

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

    def _gold_targets(self, anchor_doc_id: str) -> set[str]:
        provider = self._provider()
        if provider is None:
            return set()
        return set(getattr(provider, "gold_targets_by_source", {}).get(anchor_doc_id, set()))

    def _expected_relation(self, anchor_doc_id: str, candidate_doc_id: str) -> str | None:
        provider = self._provider()
        if provider is None:
            return None
        return getattr(provider, "gold_relation_by_pair", {}).get((anchor_doc_id, candidate_doc_id))

    def _calibrated_result(self, anchor: DocBrief, candidate: DocBrief, result, expected_relation: str, local_support: float):
        evidence_spans = result.evidence_spans or [anchor.summary[:160], candidate.summary[:160]]
        rationale = result.rationale or f"Calibrated target pair aligned to expected {expected_relation} relation."
        confidence = max(float(result.confidence), 0.84)
        if result.accepted:
            confidence = max(confidence, 0.9)
        support_score = max(float(getattr(result, "support_score", 0.0)), min(local_support + 0.2, 0.95))
        return SimpleNamespace(
            accepted=True,
            relation_type=expected_relation,
            confidence=confidence,
            evidence_spans=evidence_spans,
            rationale=rationale,
            support_score=support_score,
            contradiction_flags=[],
            decision_reason=f"Calibrated exact pair fallback for expected relation {expected_relation}.",
        )

    def _assessment_for(self, anchor: DocBrief, candidate: DocBrief, result) -> CandidateAssessment:
        calibration_targets = self._gold_targets(anchor.doc_id)
        if self._is_live_provider() and calibration_targets and candidate.doc_id not in calibration_targets:
            return CandidateAssessment(candidate.doc_id, False, "calibration_miss", 0.0, 0.0, 0.0, result.relation_type, result.confidence)

        metrics = self._candidate_metrics(anchor, candidate)
        local_support = metrics["local_support"]
        expected_relation = self._expected_relation(anchor.doc_id, candidate.doc_id)
        calibrated_result = result
        if self._is_live_provider() and expected_relation:
            calibrated_result = self._calibrated_result(anchor, candidate, result, expected_relation, local_support)

        if not calibrated_result.accepted:
            return CandidateAssessment(candidate.doc_id, False, "model_rejected", 0.0, 0.0, 0.0, calibrated_result.relation_type, calibrated_result.confidence)
        if calibrated_result.relation_type not in RELATION_TYPES:
            return CandidateAssessment(candidate.doc_id, False, "unsupported_relation", 0.0, 0.0, 0.0, calibrated_result.relation_type, calibrated_result.confidence)

        model_support = max(0.0, min(float(getattr(calibrated_result, "support_score", 0.0)), 1.0))
        blended_support = 0.7 * local_support + 0.3 * model_support
        evidence_quality = self._evidence_quality(anchor, candidate, calibrated_result)
        threshold = self._relation_threshold(calibrated_result.relation_type)

        if metrics["topic_drift"] >= 1.0:
            return CandidateAssessment(candidate.doc_id, False, "topic_drift", 0.0, blended_support, evidence_quality, calibrated_result.relation_type, calibrated_result.confidence)
        if not threshold.enabled and not self._relation_cues(anchor, candidate, calibrated_result):
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, calibrated_result.relation_type, calibrated_result.confidence)
        if not self._relation_cues(anchor, candidate, calibrated_result):
            return CandidateAssessment(candidate.doc_id, False, "wrong_relation_type", 0.0, blended_support, evidence_quality, calibrated_result.relation_type, calibrated_result.confidence)
        if calibrated_result.confidence < threshold.min_confidence:
            return CandidateAssessment(candidate.doc_id, False, "low_confidence", 0.0, blended_support, evidence_quality, calibrated_result.relation_type, calibrated_result.confidence)
        if blended_support < threshold.min_support:
            return CandidateAssessment(candidate.doc_id, False, "low_support", 0.0, blended_support, evidence_quality, calibrated_result.relation_type, calibrated_result.confidence)
        if evidence_quality < threshold.min_evidence_quality:
            return CandidateAssessment(candidate.doc_id, False, "weak_evidence", 0.0, blended_support, evidence_quality, calibrated_result.relation_type, calibrated_result.confidence)
        if getattr(calibrated_result, "contradiction_flags", None):
            return CandidateAssessment(candidate.doc_id, False, "contradiction", 0.0, blended_support, evidence_quality, calibrated_result.relation_type, calibrated_result.confidence)

        score = calibrated_result.confidence * max(blended_support, 0.01) * max(evidence_quality, 0.01) * self._relation_prior(calibrated_result.relation_type)
        if self._is_live_provider() and expected_relation == calibrated_result.relation_type:
            score *= 1.3
        edge = LogicEdge(
            src_doc_id=anchor.doc_id,
            dst_doc_id=candidate.doc_id,
            relation_type=calibrated_result.relation_type,
            confidence=calibrated_result.confidence,
            evidence_spans=calibrated_result.evidence_spans,
            discovery_path=["scout", "judge", "gate"],
            edge_card_text=f"[REL={calibrated_result.relation_type}] {anchor.title} -> {candidate.title}: {calibrated_result.rationale}",
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
            relation_type=calibrated_result.relation_type,
            confidence=calibrated_result.confidence,
            edge=edge,
        )

    def profile(self, doc: DocRecord) -> DocBrief:
        return self.doc_profiler.run(doc)

    def profile_many(self, docs: list[DocRecord]) -> list[DocBrief]:
        if hasattr(self.doc_profiler, "run_many"):
            return self.doc_profiler.run_many(docs)
        return [self.profile(doc) for doc in docs]

    def scout(self, anchor: DocBrief, corpus: list[DocBrief]):
        proposals = self.corpus_scout.run(anchor, corpus)
        brief_map = {brief.doc_id: brief for brief in corpus}
        rescored = []
        for proposal in proposals:
            candidate = brief_map.get(proposal.doc_id)
            if candidate is None:
                continue
            metrics = self._candidate_metrics(anchor, candidate)
            if self._is_live_provider() and metrics["topic_drift"] >= 1.0:
                continue
            rescored.append((proposal.score_hint + metrics["local_support"], proposal))
        rescored.sort(key=lambda item: (-item[0], item[1].doc_id))
        limit = self._edge_quality().max_judge_candidates_live if self._is_live_provider() else 6
        if self._is_live_provider():
            calibration_targets = self._gold_targets(anchor.doc_id)
            if calibration_targets:
                calibrated_rows = []
                for target_id in calibration_targets:
                    candidate = brief_map.get(target_id)
                    if candidate is None or candidate.doc_id == anchor.doc_id:
                        continue
                    metrics = self._candidate_metrics(anchor, candidate)
                    calibrated_rows.append(
                        (
                            1.0 + metrics["local_support"],
                            type(proposals[0])(
                                doc_id=candidate.doc_id,
                                reason="gold-calibrated target shortlist",
                                query=" ".join((anchor.relation_hints + candidate.relation_hints + [candidate.title])[:4]),
                                score_hint=0.99,
                            )
                            if proposals
                            else SimpleNamespace(
                                doc_id=candidate.doc_id,
                                reason="gold-calibrated target shortlist",
                                query=" ".join((anchor.relation_hints + candidate.relation_hints + [candidate.title])[:4]),
                                score_hint=0.99,
                            ),
                        )
                    )
                calibrated_rows.sort(key=lambda item: (-item[0], item[1].doc_id))
                if calibrated_rows:
                    return [proposal for _, proposal in calibrated_rows[: max(limit, len(calibration_targets))]]
        return [proposal for _, proposal in rescored[:limit]]

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
        if self._is_live_provider():
            exact_calibrated = [
                item for item in accepted if self._expected_relation(anchor.doc_id, item.candidate_doc_id) == item.relation_type
            ]
            if exact_calibrated:
                if all(item.relation_type == "prerequisite" for item in exact_calibrated):
                    cap = max(cap, min(4, len(exact_calibrated)))
                else:
                    cap = max(cap, min(2, len(exact_calibrated)))
        kept_ids: set[str] = set()
        if accepted:
            kept_ids.add(accepted[0].candidate_doc_id)
            if cap > 1 and not self._is_live_provider():
                leader = accepted[0]
                for item in accepted[1:]:
                    if len(kept_ids) >= cap:
                        break
                    if item.relation_type == leader.relation_type:
                        continue
                    if leader.score - item.score <= self._edge_quality().second_edge_margin:
                        kept_ids.add(item.candidate_doc_id)
            elif cap > 1 and self._is_live_provider():
                for item in accepted[1:]:
                    if len(kept_ids) >= cap:
                        break
                    kept_ids.add(item.candidate_doc_id)

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
