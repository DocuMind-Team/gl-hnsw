from __future__ import annotations

from typing import Iterable

import numpy as np

from hnsw_logic.core.constants import RELATION_TYPES
from hnsw_logic.core.facets import enrich_brief
from hnsw_logic.core.models import DocBrief, DocRecord, LogicEdge
from hnsw_logic.core.utils import deterministic_vector, tokenize, top_terms
from hnsw_logic.embedding.provider_base import ProviderBase
from hnsw_logic.embedding.provider_types import CandidateProposal, JudgeResult, JudgeSignals


class StubProvider(ProviderBase):
    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([deterministic_vector(text, self.embedding_dim) for text in texts]).astype(np.float32)

    def profile_doc(self, doc: DocRecord) -> DocBrief:
        tokens = tokenize(doc.text)
        entities = sorted(
            {
                token
                for token in tokens
                if token in {"hnsw", "deepagents", "fastapi", "sqlite", "memory", "subagents", "skills"}
            }
        )
        claims = [sentence.strip() for sentence in doc.text.split(". ")[:2] if sentence.strip()]
        relation_hints = top_terms(doc.text, limit=3)
        return enrich_brief(
            DocBrief(
                doc_id=doc.doc_id,
                title=doc.title,
                summary=" ".join(claims)[:320],
                entities=entities,
                keywords=top_terms(doc.text, limit=6),
                claims=claims[:3],
                relation_hints=relation_hints,
                metadata=doc.metadata,
            )
        )

    def propose_candidates(self, anchor: DocBrief, corpus: list[DocBrief]) -> list[CandidateProposal]:
        anchor_terms = set(anchor.keywords + anchor.entities + anchor.relation_hints)
        proposals: list[CandidateProposal] = []
        for candidate in corpus:
            if candidate.doc_id == anchor.doc_id:
                continue
            candidate_terms = set(candidate.keywords + candidate.entities)
            overlap = sorted(anchor_terms & candidate_terms)
            if not overlap:
                continue
            score_hint = min(0.99, 0.2 + 0.15 * len(overlap))
            proposals.append(
                CandidateProposal(
                    doc_id=candidate.doc_id,
                    reason=f"shared terms: {', '.join(overlap[:4])}",
                    query=" ".join(overlap[:4]),
                    score_hint=score_hint,
                )
            )
        proposals.sort(key=lambda item: (-item.score_hint, item.doc_id))
        return proposals[:6]

    def judge_relation(self, anchor: DocBrief, candidate: DocBrief) -> JudgeResult:
        shared = sorted(set(anchor.keywords + anchor.entities) & set(candidate.keywords + candidate.entities))
        if not shared:
            return JudgeResult(
                accepted=False,
                relation_type="comparison",
                confidence=0.0,
                evidence_spans=[],
                rationale="no meaningful overlap",
                support_score=0.0,
                contradiction_flags=["no_overlap"],
                decision_reason="local overlap check failed",
            )
        relation = "same_concept"
        if "memory" in shared:
            relation = "implementation_detail"
        elif "subagents" in shared:
            relation = "prerequisite"
        elif "retrieval" in shared or "logic" in shared:
            relation = "supporting_evidence"
        confidence = min(0.95, 0.45 + 0.12 * len(shared))
        return JudgeResult(
            accepted=confidence >= 0.6,
            relation_type=relation,
            confidence=confidence,
            evidence_spans=[anchor.summary[:160], candidate.summary[:160]],
            rationale=f"shared terms: {', '.join(shared[:5])}",
            support_score=min(1.0, 0.25 * len(shared)),
            contradiction_flags=[],
            decision_reason="accepted by deterministic overlap heuristic" if confidence >= 0.6 else "confidence below heuristic threshold",
            semantic_relation_label=relation,
            canonical_relation=relation,
            utility_score=min(1.0, confidence * 0.9),
            uncertainty=max(0.0, 1.0 - confidence),
        )

    def review_relation_with_signals(
        self,
        anchor: DocBrief,
        candidate: DocBrief,
        signals: JudgeSignals,
        verdict: JudgeResult,
    ) -> JudgeResult:
        fit_scores = dict(signals.relation_fit_scores or {})
        current_relation = verdict.relation_type if verdict.relation_type in RELATION_TYPES else signals.best_relation
        if current_relation not in RELATION_TYPES:
            current_relation = "implementation_detail"
        best_relation = signals.best_relation if signals.best_relation in RELATION_TYPES else current_relation
        current_fit = float(fit_scores.get(current_relation, 0.0))
        best_fit = float(fit_scores.get(best_relation, current_fit))
        risk_flags = set(signals.risk_flags or [])
        local_alignment = (
            0.34 * signals.local_support
            + 0.24 * signals.utility_score
            + 0.16 * max(signals.overlap_score, signals.content_overlap_score)
            + 0.14 * signals.mention_score
            + 0.12 * max(current_fit, best_fit)
        )
        risk_penalty = 0.12 * len(risk_flags)
        if "service_surface" in risk_flags:
            risk_penalty += 0.16
        if "foundational_support" in risk_flags:
            risk_penalty += 0.18
        if "weak_direction" in risk_flags:
            risk_penalty += 0.12
        if "methodology_gap" in risk_flags:
            risk_penalty += 0.12

        reviewed_relation = current_relation
        if best_relation in RELATION_TYPES and best_fit >= current_fit + 0.08:
            reviewed_relation = best_relation
        if (
            reviewed_relation == "comparison"
            and signals.stance_contrast >= 1.0
            and signals.contrastive_bridge_score >= 0.56
            and signals.bridge_gain >= 0.38
        ):
            risk_penalty = max(0.0, risk_penalty - 0.3)
            risk_flags = {flag for flag in risk_flags if flag not in {"near_duplicate", "near_duplicate_bridge"}}
            risk_flags = {
                flag
                for flag in risk_flags
                if not (
                    flag.startswith("contradict")
                    or flag.startswith("counterargument")
                    or flag.startswith("oppos")
                    or flag.startswith("contrasting")
                    or "contrast" in flag
                    or flag.startswith("alternative_position")
                )
            }
        if (
            reviewed_relation == "same_concept"
            and "methodology_gap" in risk_flags
            and fit_scores.get("supporting_evidence", 0.0) >= current_fit - 0.05
        ):
            reviewed_relation = "supporting_evidence"

        reviewer_accepts = verdict.accepted
        reject_reason = ""
        if local_alignment - risk_penalty < 0.24:
            reviewer_accepts = False
            reject_reason = "low_alignment"
        if reviewed_relation == "supporting_evidence" and {"service_surface", "foundational_support"} & risk_flags:
            reviewer_accepts = False
            reject_reason = "generic_support"
        if reviewed_relation == "implementation_detail" and "weak_direction" in risk_flags and best_fit < 0.72:
            reviewer_accepts = False
            reject_reason = "weak_direction"
        if reviewed_relation == "same_concept" and {"low_novelty", "excess_novelty"} & risk_flags:
            reviewer_accepts = False
            reject_reason = "poor_novelty_bridge"
        if reviewed_relation == "same_concept" and "weak_family_bridge" in risk_flags and best_fit < 0.76:
            reviewer_accepts = False
            reject_reason = "weak_family_bridge"
        if not verdict.accepted and reviewer_accepts:
            reviewer_accepts = (
                best_fit >= 0.56
                and signals.utility_score >= 0.38
                and not {"service_surface", "foundational_support"} & risk_flags
            )
            if not reviewer_accepts:
                reject_reason = reject_reason or "insufficient_recovery"

        confidence = max(
            0.0,
            min(0.98, 0.52 * verdict.confidence + 0.28 * max(current_fit, best_fit) + 0.2 * signals.local_support - 0.12 * len(risk_flags)),
        )
        utility = max(signals.utility_score, 0.5 * verdict.utility_score + 0.5 * local_alignment - 0.1 * len(risk_flags))
        uncertainty = max(verdict.uncertainty, 0.18 + 0.16 * len(risk_flags))
        if not reviewer_accepts:
            return JudgeResult(
                accepted=False,
                relation_type=reviewed_relation,
                confidence=max(0.0, min(confidence, 0.72)),
                evidence_spans=verdict.evidence_spans[:4],
                rationale=verdict.rationale[:200],
                support_score=max(0.0, min(1.0, 0.5 * verdict.support_score + 0.5 * signals.local_support)),
                contradiction_flags=sorted(risk_flags)[:4],
                decision_reason=f"Rejected by generic reviewer: {reject_reason or 'low utility'}."[:200],
                semantic_relation_label=verdict.semantic_relation_label or reviewed_relation,
                canonical_relation="none",
                utility_score=max(0.0, min(utility, 0.45)),
                uncertainty=min(1.0, uncertainty + 0.12),
            )

        return JudgeResult(
            accepted=True,
            relation_type=reviewed_relation,
            confidence=confidence,
            evidence_spans=verdict.evidence_spans[:4],
            rationale=(verdict.rationale or "Approved by generic reviewer after cross-checking local signals.")[:200],
            support_score=max(0.0, min(1.0, 0.45 * verdict.support_score + 0.55 * signals.local_support)),
            contradiction_flags=sorted(risk_flags)[:4],
            decision_reason=(
                f"Reviewed for edge utility with relation {reviewed_relation}; "
                f"local alignment={local_alignment:.2f}, risk_penalty={risk_penalty:.2f}."
            )[:200],
            semantic_relation_label=verdict.semantic_relation_label or reviewed_relation,
            canonical_relation=reviewed_relation,
            utility_score=max(0.0, min(1.0, utility)),
            uncertainty=min(1.0, uncertainty),
        )

    def curate_memory(self, anchor: DocBrief, accepted: list[LogicEdge], rejected: list[str]) -> dict:
        aliases: dict[str, list[str]] = {}
        for entity in anchor.entities:
            aliases[entity] = sorted({entity, entity.replace("_", " ")})
        return {
            "active_hypotheses": anchor.relation_hints[:3],
            "successful_queries": [edge.edge_card_text for edge in accepted[:3]],
            "failed_queries": rejected[:3],
            "aliases": aliases,
            "relation_patterns": {
                edge.relation_type: sorted({edge.src_doc_id, edge.dst_doc_id}) for edge in accepted
            },
        }
