from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from hnsw_logic.config.schema import ProviderConfig
from hnsw_logic.core.models import DocBrief, DocRecord, LogicEdge
from hnsw_logic.embedding.provider_types import CandidateProposal, JudgeResult, JudgeSignals


class ProviderBase:
    def __init__(self, config: ProviderConfig, root_dir: Path | None = None):
        self.config = config
        self.root_dir = Path(root_dir) if root_dir else None
        self.live_reasoning = {
            "scout": True,
            "judge": True,
            "reviewer": True,
            "curator": True,
        }
        self.relation_priors = {
            "supporting_evidence": 1.02,
            "implementation_detail": 1.0,
            "same_concept": 0.82,
            "comparison": 0.9,
            "prerequisite": 0.98,
        }
        self.judge_few_shot_text = self._build_generic_judge_examples()
        self.review_few_shot_text = self._build_generic_review_examples()

    @staticmethod
    def _normalize_risk_flag(flag: str) -> str:
        return str(flag).strip().lower().replace("-", "_").replace(" ", "_")

    def _has_verdict_contrast_signal(self, verdict: JudgeResult) -> bool:
        text = " ".join(
            [
                *(verdict.contradiction_flags or []),
                verdict.decision_reason,
                verdict.rationale,
            ]
        ).lower()
        return any(
            cue in text
            for cue in (
                "contrast",
                "contrasting",
                "opposing",
                "opposed",
                "counterargument",
                "counter-argument",
                "alternative position",
                "direct contrast",
            )
        )

    def _is_contrastive_comparison_bridge(self, signals: JudgeSignals, verdict: JudgeResult) -> bool:
        if verdict.relation_type != "comparison":
            return False
        topic_consistent = (
            signals.topic_family_match >= 1.0
            or signals.topic_cluster_match >= 1.0
            or max(signals.overlap_score, signals.content_overlap_score) >= 0.18
            or signals.mention_score >= 0.18
        )
        contrast_signal = (
            signals.stance_contrast >= 1.0
            or signals.contrastive_bridge_score >= 0.56
            or self._has_verdict_contrast_signal(verdict)
        )
        bridge_signal = signals.bridge_gain >= 0.38 or verdict.utility_score >= 0.72
        return topic_consistent and contrast_signal and bridge_signal

    def _normalize_counterevidence_result(
        self,
        anchor: DocBrief,
        candidate: DocBrief,
        signals: JudgeSignals,
        verdict: JudgeResult,
        result: dict,
    ) -> dict:
        risk_flags = [str(flag) for flag in result.get("risk_flags", []) if str(flag)]
        normalized_flags = {self._normalize_risk_flag(flag) for flag in risk_flags}
        keep = bool(result.get("keep", True))
        risk_penalty = float(result.get("risk_penalty", 0.0) or 0.0)
        decision_reason = str(result.get("decision_reason", ""))[:220]

        contradiction_like = {
            flag
            for flag in normalized_flags
            if flag.startswith("contradict")
            or flag.startswith("counterargument")
            or flag.startswith("oppos")
            or flag.startswith("contrasting")
            or "contrast" in flag
            or flag.startswith("alternative_position")
        }
        duplicate_only_flags = {"near_duplicate", "near_duplicate_bridge"}
        hard_blockers = {"same_stance", "topic_drift", "weak_topic_match", "low_retrieval_utility", "weak_direction"}

        if self._is_contrastive_comparison_bridge(signals, verdict) and "same_stance" not in normalized_flags:
            normalized_flags -= duplicate_only_flags
            remaining_blockers = {flag for flag in normalized_flags if flag in hard_blockers}
            if not remaining_blockers:
                keep = True
                risk_penalty = min(risk_penalty, 0.22)
                if not decision_reason:
                    decision_reason = "Kept as a same-topic contrast bridge despite duplicate risk."
                normalized_flags |= contradiction_like

        return {
            "keep": keep,
            "risk_flags": sorted(normalized_flags),
            "counterevidence": [str(flag) for flag in result.get("counterevidence", [])][:8],
            "decision_reason": decision_reason,
            "risk_penalty": round(risk_penalty, 6),
        }

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def configure_live_reasoning(self, live_reasoning_config) -> None:
        self.live_reasoning = {
            "scout": live_reasoning_config.enable_scout_thinking,
            "judge": live_reasoning_config.enable_judge_thinking,
            "reviewer": live_reasoning_config.enable_reviewer_thinking,
            "curator": live_reasoning_config.enable_curator_thinking,
        }

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError

    def profile_doc(self, doc: DocRecord) -> DocBrief:
        raise NotImplementedError

    def profile_docs(self, docs: list[DocRecord], on_brief: Callable[[DocBrief], None] | None = None) -> list[DocBrief]:
        return [self.profile_doc(doc) for doc in docs]

    def propose_candidates(self, anchor: DocBrief, corpus: list[DocBrief]) -> list[CandidateProposal]:
        raise NotImplementedError

    def judge_relation(self, anchor: DocBrief, candidate: DocBrief) -> JudgeResult:
        raise NotImplementedError

    def judge_relations(self, anchor: DocBrief, candidates: list[DocBrief]) -> dict[str, JudgeResult]:
        return {candidate.doc_id: self.judge_relation(anchor, candidate) for candidate in candidates}

    def judge_relation_with_signals(self, anchor: DocBrief, candidate: DocBrief, signals: JudgeSignals) -> JudgeResult:
        return self.judge_relation(anchor, candidate)

    def judge_relations_with_signals(
        self,
        anchor: DocBrief,
        candidates: list[tuple[DocBrief, JudgeSignals]],
    ) -> dict[str, JudgeResult]:
        return {candidate.doc_id: self.judge_relation_with_signals(anchor, candidate, signals) for candidate, signals in candidates}

    def review_relation_with_signals(
        self,
        anchor: DocBrief,
        candidate: DocBrief,
        signals: JudgeSignals,
        verdict: JudgeResult,
    ) -> JudgeResult:
        return verdict

    def review_relations_with_signals(
        self,
        anchor: DocBrief,
        candidates: list[tuple[DocBrief, JudgeSignals, JudgeResult]],
    ) -> dict[str, JudgeResult]:
        return {
            candidate.doc_id: self.review_relation_with_signals(anchor, candidate, signals, verdict)
            for candidate, signals, verdict in candidates
        }

    def plan_indexing_batch(self, payload: dict) -> dict:
        return payload

    def judge_with_signals(self, anchor: DocBrief, candidate: DocBrief, signals: JudgeSignals) -> JudgeResult:
        return self.judge_relation_with_signals(anchor, candidate, signals)

    def check_counterevidence(self, anchor: DocBrief, candidate: DocBrief, signals: JudgeSignals, verdict: JudgeResult) -> dict:
        risk_flags = list(signals.risk_flags or [])
        penalty = 0.0
        if "weak_direction" in risk_flags:
            penalty += 0.1
        if "service_surface" in risk_flags:
            penalty += 0.14
        if "foundational_support" in risk_flags:
            penalty += 0.18
        if "methodology_gap" in risk_flags:
            penalty += 0.12
        if (
            verdict.relation_type == "comparison"
            and signals.stance_contrast >= 1.0
            and signals.contrastive_bridge_score >= 0.56
            and signals.bridge_gain >= 0.38
        ):
            penalty = max(0.0, penalty - 0.24)
            risk_flags = [flag for flag in risk_flags if flag not in {"near_duplicate", "near_duplicate_bridge"}]
        result = {
            "keep": penalty < 0.42,
            "risk_flags": sorted(set(risk_flags)),
            "counterevidence": [],
            "decision_reason": "base checker local decision",
            "risk_penalty": round(penalty, 6),
        }
        return self._normalize_counterevidence_result(anchor, candidate, signals, verdict, result)

    def check_counterevidence_many(
        self,
        anchor: DocBrief,
        candidates: list[tuple[DocBrief, JudgeSignals, JudgeResult]],
    ) -> dict[str, dict]:
        return {
            candidate.doc_id: self.check_counterevidence(anchor, candidate, signals, verdict)
            for candidate, signals, verdict in candidates
        }

    def review_with_utility(self, anchor: DocBrief, candidate: DocBrief, signals: JudgeSignals, verdict: JudgeResult) -> JudgeResult:
        return self.review_relation_with_signals(anchor, candidate, signals, verdict)

    def summarize_memory_learnings(self, payload: dict) -> dict:
        accepted = payload.get("accepted", [])
        rejected = payload.get("rejected", [])
        return {
            "learned_patterns": [str(item) for item in accepted[:8]],
            "failure_patterns": [str(item) for item in rejected[:12]],
            "reference_updates": {},
        }

    def curate_memory(self, anchor: DocBrief, accepted: list[LogicEdge], rejected: list[str]) -> dict:
        raise NotImplementedError

    def _build_generic_judge_examples(self) -> str:
        examples = [
            {
                "label": "positive",
                "anchor_title": "Hybrid Retrieval",
                "anchor_text": "The retriever combines geometric recall with logical expansions and then ranks the merged candidates.",
                "candidate_title": "Candidate Fusion",
                "candidate_text": "The fusion stage computes a weighted score over the geometric score and the logic score.",
                "expected_relation_type": "implementation_detail",
                "why": "The candidate defines the scoring mechanism used by the anchor.",
            },
            {
                "label": "positive",
                "anchor_title": "Subagents",
                "anchor_text": "The system includes a profiler role, a scout role, a judge role, and a curator role.",
                "candidate_title": "Relation Judge",
                "candidate_text": "The judge role verifies whether an anchor and a candidate should form a durable edge.",
                "expected_relation_type": "prerequisite",
                "why": "The candidate is one of the explicitly listed roles in the anchor.",
            },
            {
                "label": "positive",
                "anchor_title": "Jump Policy",
                "anchor_text": "The policy decides whether a logical candidate may enter the final ranker.",
                "candidate_title": "Candidate Fusion",
                "candidate_text": "Only approved logical candidates contribute logic score to the final ranking stage.",
                "expected_relation_type": "supporting_evidence",
                "why": "The candidate explains the downstream effect of the policy's approval.",
            },
            {
                "label": "positive",
                "anchor_title": "Public policy should prioritize public transit",
                "anchor_text": "The argument claims cities should invest in transit instead of adding highway capacity.",
                "candidate_title": "Road expansion remains the best congestion solution",
                "candidate_text": "The counterargument claims adding lanes improves mobility more reliably than transit spending.",
                "expected_relation_type": "comparison",
                "why": "Both documents discuss the same policy topic from contrasting positions.",
            },
            {
                "label": "negative",
                "anchor_title": "Artists should be allowed to offend social taboos",
                "anchor_text": "The argument focuses on provocative art and social disgust in the arts.",
                "candidate_title": "Gangsta rap should be censored because it causes harm",
                "candidate_text": "The counterargument focuses on censorship in a broader free-speech debate rather than the same arts dispute.",
                "expected_relation_type": "none",
                "why": "Opposing stance in a broad policy area is not enough when the pair does not share the same topic family or bridge surface.",
            },
            {
                "label": "positive",
                "anchor_title": "Obesity increases the risk of chronic kidney disease",
                "anchor_text": "The passage links obesity and metabolic risk factors to later chronic kidney disease outcomes.",
                "candidate_title": "Metabolic syndrome and chronic kidney disease progression",
                "candidate_text": "The candidate passage explains how metabolic syndrome worsens chronic kidney disease progression and patient risk.",
                "expected_relation_type": "supporting_evidence",
                "why": "The candidate adds clinically aligned risk and outcome evidence that can strengthen retrieval for the anchor claim family.",
            },
            {
                "label": "positive",
                "anchor_title": "Vitamin D deficiency and bone fracture risk",
                "anchor_text": "The evidence passage focuses on vitamin D deficiency as a driver of fracture risk.",
                "candidate_title": "Fracture outcomes in patients with low vitamin D",
                "candidate_text": "The candidate describes the same deficiency-risk family with overlapping outcome terminology and clinically specific evidence.",
                "expected_relation_type": "same_concept",
                "why": "The pair describes the same clinical finding family with aligned specific bridge terms and retrieval utility.",
            },
            {
                "label": "negative",
                "anchor_title": "Logic Overlay Graph",
                "anchor_text": "The overlay stores durable document-to-document relations used after initial recall.",
                "candidate_title": "ANN Metrics",
                "candidate_text": "Metrics such as recall and MRR summarize retrieval quality.",
                "expected_relation_type": "none",
                "why": "The topics belong to the same project but do not form a durable document edge.",
            },
            {
                "label": "negative",
                "anchor_title": "Document Profiler",
                "anchor_text": "The profiler produces a structured brief for an input document.",
                "candidate_title": "SQLite Job Registry",
                "candidate_text": "A registry stores job ids, states, and timestamps for background workers.",
                "expected_relation_type": "none",
                "why": "There is no direct semantic dependency between profiling and job persistence.",
            },
            {
                "label": "negative",
                "anchor_title": "Background Jobs",
                "anchor_text": "Workers run expensive offline tasks through a lightweight registry.",
                "candidate_title": "Public API Service",
                "candidate_text": "The service exposes endpoints that can submit jobs to the registry.",
                "expected_relation_type": "none",
                "why": "A service using the same registry is not by itself durable supporting evidence for the worker design.",
            },
            {
                "label": "negative",
                "anchor_title": "Public transit should replace highway expansion",
                "anchor_text": "The argument says public transit investment is better than highway expansion for city mobility.",
                "candidate_title": "Transit funding improves city mobility outcomes",
                "candidate_text": "The supporting document agrees that transit funding improves urban mobility and should be expanded.",
                "expected_relation_type": "none",
                "why": "Shared stance and topic overlap alone are not enough for a durable comparison edge.",
            },
        ]
        return "\n".join(json.dumps(example, ensure_ascii=False) for example in examples)

    def _build_generic_review_examples(self) -> str:
        examples = [
            {
                "label": "approve",
                "anchor_title": "Hybrid Retrieval",
                "candidate_title": "Candidate Fusion",
                "judge_verdict": {
                    "accepted": True,
                    "canonical_relation": "implementation_detail",
                    "confidence": 0.88,
                    "utility_score": 0.79,
                },
                "signals": {
                    "best_relation": "implementation_detail",
                    "utility_score": 0.74,
                    "risk_flags": [],
                },
                "expected": {
                    "accepted": True,
                    "canonical_relation": "implementation_detail",
                    "why": "Mechanism-level detail with aligned utility and no major risk.",
                },
            },
            {
                "label": "reject",
                "anchor_title": "Background Jobs",
                "candidate_title": "Public API Service",
                "judge_verdict": {
                    "accepted": True,
                    "canonical_relation": "supporting_evidence",
                    "confidence": 0.82,
                    "utility_score": 0.31,
                },
                "signals": {
                    "best_relation": "supporting_evidence",
                    "utility_score": 0.22,
                    "risk_flags": ["service_surface"],
                },
                "expected": {
                    "accepted": False,
                    "canonical_relation": "none",
                    "why": "Shared service surface is too generic to be a durable retrieval edge.",
                },
            },
            {
                "label": "downgrade",
                "anchor_title": "Scientific claim about disease risk",
                "candidate_title": "Measurement protocol for the same condition",
                "judge_verdict": {
                    "accepted": True,
                    "canonical_relation": "same_concept",
                    "confidence": 0.84,
                    "utility_score": 0.43,
                },
                "signals": {
                    "best_relation": "supporting_evidence",
                    "utility_score": 0.47,
                    "risk_flags": ["methodology_gap"],
                },
                "expected": {
                    "accepted": True,
                    "canonical_relation": "supporting_evidence",
                    "why": "The pair is related, but the candidate looks more like supporting context than the same concept.",
                },
            },
            {
                "label": "approve-comparison",
                "anchor_title": "Transit investment should replace highway expansion",
                "candidate_title": "Highway expansion remains the stronger congestion policy",
                "judge_verdict": {
                    "accepted": True,
                    "canonical_relation": "comparison",
                    "confidence": 0.84,
                    "utility_score": 0.63,
                },
                "signals": {
                    "best_relation": "comparison",
                    "utility_score": 0.66,
                    "risk_flags": [],
                },
                "expected": {
                    "accepted": True,
                    "canonical_relation": "comparison",
                    "why": "Opposing positions on the same policy topic can create a durable contrast edge for retrieval.",
                },
            },
            {
                "label": "approve-contrast-despite-overlap",
                "anchor_title": "Universities should restrict hate speech on campus",
                "candidate_title": "Universities should not restrict hate speech on campus",
                "judge_verdict": {
                    "accepted": True,
                    "canonical_relation": "comparison",
                    "confidence": 0.86,
                    "utility_score": 0.67,
                },
                "signals": {
                    "best_relation": "comparison",
                    "utility_score": 0.7,
                    "risk_flags": ["near_duplicate"],
                },
                "expected": {
                    "accepted": True,
                    "canonical_relation": "comparison",
                    "why": "High topic overlap is acceptable when the pair supplies a reusable same-topic contrast with opposing stances.",
                },
            },
            {
                "label": "reject-cross-family-analogy",
                "anchor_title": "Artists should be allowed to offend social taboos",
                "candidate_title": "Gangsta rap should be censored because it causes harm",
                "judge_verdict": {
                    "accepted": True,
                    "canonical_relation": "comparison",
                    "confidence": 0.81,
                    "utility_score": 0.63,
                },
                "signals": {
                    "best_relation": "comparison",
                    "utility_score": 0.58,
                    "risk_flags": ["topic_mismatch"],
                },
                "expected": {
                    "accepted": False,
                    "canonical_relation": "none",
                    "why": "Broad free-speech analogy is weaker than a same-topic arts contrast bridge and should not survive as a durable comparison edge.",
                },
            },
            {
                "label": "approve-clinical-support",
                "anchor_title": "Obesity increases chronic kidney disease risk",
                "candidate_title": "Metabolic syndrome worsens chronic kidney disease progression",
                "judge_verdict": {
                    "accepted": True,
                    "canonical_relation": "supporting_evidence",
                    "confidence": 0.8,
                    "utility_score": 0.58,
                },
                "signals": {
                    "best_relation": "supporting_evidence",
                    "utility_score": 0.64,
                    "risk_flags": [],
                },
                "expected": {
                    "accepted": True,
                    "canonical_relation": "supporting_evidence",
                    "why": "Clinically specific risk and outcome overlap can justify a durable evidence edge.",
                },
            },
        ]
        return "\n".join(json.dumps(example, ensure_ascii=False) for example in examples)
