from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hnsw_logic.agents.orchestrator import CandidateAssessment
from hnsw_logic.agents.runtime_models import MemoryLearningBundle
from hnsw_logic.core.models import DocBrief
from hnsw_logic.core.utils import read_json
from hnsw_logic.embedding.provider import JudgeResult


class OfflineIndexingSupervisor:
    def __init__(
        self,
        *,
        orchestrator,
        discovery_service,
        deepagent,
        runtime_toolsets: dict[str, dict[str, Any]],
        workspace_root: Path,
        agents_config,
        self_update_manager,
        agents_memory_path: Path,
    ):
        self.orchestrator = orchestrator
        self.discovery_service = discovery_service
        self.deepagent = deepagent
        self.runtime_toolsets = runtime_toolsets
        self.workspace_root = workspace_root
        self.agents_config = agents_config
        self.self_update_manager = self_update_manager
        self.agents_memory_path = agents_memory_path
        self.supervisor_task_delegation_enabled = os.getenv("GL_HNSW_ENABLE_DEEPAGENT_SUPERVISOR", "0") == "1"
        self.anchor_task_delegation_enabled = (
            self.supervisor_task_delegation_enabled
            and os.getenv("GL_HNSW_ENABLE_ANCHOR_TASK_DELEGATION", "0") == "1"
        )

    @staticmethod
    def _normalize_risk_flag(flag: str) -> str:
        return str(flag).strip().lower().replace("-", "_").replace(" ", "_")

    def _stage_path(self, stage: str, doc_id: str, suffix: str = ".json") -> Path:
        return self.workspace_root / "indexing" / stage / f"{doc_id}{suffix}"

    def _plan_path(self) -> Path:
        return self.workspace_root / "indexing" / "plans" / "indexing_plan.json"

    def _invoke_main_agent(self, prompt: str) -> None:
        if self.deepagent is None:
            raise RuntimeError("deepagent runtime unavailable")
        self.deepagent.invoke({"messages": [{"role": "user", "content": prompt}]})

    def _run_stage_locally(self, agent_name: str, tool_name: str, **kwargs) -> dict:
        tool = self.runtime_toolsets.get(agent_name, {}).get(tool_name)
        if tool is None:
            return {}
        return tool(**kwargs)

    def _build_plan(self) -> dict:
        if self.deepagent is not None and self.agents_config.planner_enabled and self.supervisor_task_delegation_enabled:
            try:
                self._invoke_main_agent(
                    "Create the offline indexing plan. Use the task tool to delegate to the index_planner subagent. "
                    "The subagent must call execute_index_planning and produce /data/workspace/indexing/plans/indexing_plan.json. "
                    "Return a short confirmation once the plan file exists."
                )
            except Exception:
                self._run_stage_locally("index_planner", "execute_index_planning")
        else:
            self._run_stage_locally("index_planner", "execute_index_planning")
        return read_json(self._plan_path(), {}) or {}

    def _run_anchor_workflow_local(self, anchor_doc_id: str) -> None:
        self._run_stage_locally("doc_profiler", "execute_doc_profiling", anchor_doc_id=anchor_doc_id)
        self._run_stage_locally("corpus_scout", "execute_candidate_expansion", anchor_doc_id=anchor_doc_id)
        self._run_stage_locally("relation_judge", "execute_relation_judging", anchor_doc_id=anchor_doc_id)
        if self.agents_config.counterevidence_enabled:
            self._run_stage_locally("counterevidence_checker", "execute_counterevidence_check", anchor_doc_id=anchor_doc_id)
        self._run_stage_locally("edge_reviewer", "execute_edge_review", anchor_doc_id=anchor_doc_id)
        self._run_stage_locally("memory_curator", "execute_memory_summarization", anchor_doc_id=anchor_doc_id)

    def _run_anchor_workflow_with_deepagents(self, anchor_doc_id: str) -> None:
        try:
            self._invoke_main_agent(
                "\n".join(
                    [
                        f"Run the offline indexing workflow for anchor `{anchor_doc_id}`.",
                        "Use the task tool and delegate in this order:",
                        "1. doc_profiler -> ensure the anchor dossier exists by calling execute_doc_profiling.",
                        "2. corpus_scout -> build the candidate bundle with execute_candidate_expansion.",
                        "3. relation_judge -> build the judgment bundle with execute_relation_judging.",
                        "4. counterevidence_checker -> build the counterevidence bundle with execute_counterevidence_check.",
                        "5. edge_reviewer -> build the review bundle with execute_edge_review.",
                        "6. memory_curator -> build the memory learning bundle with execute_memory_summarization.",
                        "Do not edit code or configs. Use workspace files as the handoff boundary and return only a short completion summary.",
                    ]
                )
            )
        except Exception:
            self._run_anchor_workflow_local(anchor_doc_id)

    def _load_candidate_assets(self, anchor_doc_id: str, briefs: list[DocBrief]) -> tuple[list[DocBrief], dict, dict, dict]:
        brief_map = {brief.doc_id: brief for brief in briefs}
        candidate_payload = read_json(self._stage_path("candidates", anchor_doc_id), {}) or {}
        judgment_payload = read_json(self._stage_path("judgments", anchor_doc_id), {}) or {}
        check_payload = read_json(self._stage_path("checks", anchor_doc_id), {}) or {}
        review_payload = read_json(self._stage_path("reviews", anchor_doc_id), {}) or {}
        candidates = [
            brief_map[item["candidate_doc_id"]]
            for item in candidate_payload.get("candidates", [])
            if item.get("candidate_doc_id") in brief_map
        ]
        verdicts = {
            item["candidate_doc_id"]: JudgeResult(**item.get("verdict", {}))
            for item in judgment_payload.get("judgments", [])
            if item.get("candidate_doc_id")
        }
        reviews = {}
        for item in review_payload.get("reviews", []):
            candidate_doc_id = item.get("candidate_doc_id")
            final_verdict = item.get("final_verdict")
            if candidate_doc_id and isinstance(final_verdict, dict):
                reviews[candidate_doc_id] = JudgeResult(**final_verdict)
        checks = {item["candidate_doc_id"]: item for item in check_payload.get("checks", []) if item.get("candidate_doc_id")}
        review_rows = {item["candidate_doc_id"]: item for item in review_payload.get("reviews", []) if item.get("candidate_doc_id")}
        return candidates, verdicts, reviews, {"checks": checks, "reviews": review_rows}

    def _apply_review_consensus(
        self,
        anchor: DocBrief,
        candidates: list[DocBrief],
        verdicts: dict[str, JudgeResult],
        reviews: dict[str, JudgeResult],
        bundle_lookup: dict[str, dict],
    ) -> list[CandidateAssessment]:
        assessments = [
            self.orchestrator._assessment_for(anchor, candidate, verdicts.get(candidate.doc_id), reviews.get(candidate.doc_id))
            for candidate in candidates
            if candidate.doc_id in verdicts
        ]
        review_rows = bundle_lookup.get("reviews", {})
        check_rows = bundle_lookup.get("checks", {})
        adjusted: list[CandidateAssessment] = []
        for assessment in assessments:
            review_row = review_rows.get(assessment.candidate_doc_id, {})
            check_row = check_rows.get(assessment.candidate_doc_id, {})
            review_keep = bool(review_row.get("keep", True))
            check_keep = bool(check_row.get("keep", True))
            reviewed_utility = float(review_row.get("reviewed_utility_score", 0.0) or 0.0)
            risk_penalty = float(check_row.get("risk_penalty", 0.0) or 0.0)
            risk_flags = {
                self._normalize_risk_flag(str(flag))
                for flag in [*review_row.get("risk_flags", []), *check_row.get("risk_flags", [])]
                if str(flag)
            }
            soft_bridge_risks = {"excess_novelty", "weak_family_bridge", "topic_only_overlap", "low_retrieval_utility", "weak_direction"}
            same_concept_soft_keep = (
                assessment.edge is not None
                and assessment.edge.relation_type == "same_concept"
                and assessment.edge.utility_score >= 0.78
                and assessment.local_support >= 0.68
                and assessment.evidence_quality >= 0.84
                and risk_penalty <= 0.22
                and risk_flags.issubset(soft_bridge_risks)
            )
            rescue_keep = (
                assessment.accepted
                and assessment.edge is not None
                and not ({"service_surface", "foundational_support"} & risk_flags)
                and (
                    (
                        reviewed_utility >= 0.74
                        and risk_penalty <= 0.18
                    )
                    or (
                        same_concept_soft_keep
                    )
                )
            )
            keep = (review_keep and check_keep) or rescue_keep
            if not keep:
                adjusted.append(
                    CandidateAssessment(
                        candidate_doc_id=assessment.candidate_doc_id,
                        accepted=False,
                        reject_reason=str(review_row.get("decision_reason") or check_row.get("decision_reason") or "review_rejected")[:160],
                        score=assessment.score,
                        local_support=assessment.local_support,
                        evidence_quality=assessment.evidence_quality,
                        relation_type=assessment.relation_type,
                        confidence=assessment.confidence,
                    )
                )
                continue
            next_assessment = assessment
            reviewed_utility = review_row.get("reviewed_utility_score")
            reviewed_confidence = review_row.get("reviewed_confidence")
            if assessment.edge is not None:
                if reviewed_utility is not None:
                    reviewed_utility_value = max(0.0, min(1.0, float(reviewed_utility)))
                    if rescue_keep and not (review_keep and check_keep) and reviewed_utility_value < max(0.35, next_assessment.edge.utility_score * 0.6):
                        reviewed_utility_value = next_assessment.edge.utility_score
                    next_assessment.edge.utility_score = reviewed_utility_value
                    next_assessment.score *= 0.85 + 0.35 * next_assessment.edge.utility_score
                if reviewed_confidence is not None:
                    next_assessment.edge.confidence = max(next_assessment.edge.confidence, float(reviewed_confidence))
                if rescue_keep and not (review_keep and check_keep):
                    next_assessment.score *= 0.94
                    next_assessment.edge.utility_score = max(0.0, min(1.0, next_assessment.edge.utility_score * 0.97))
            adjusted.append(next_assessment)
        return self.orchestrator._apply_assessment_cap(anchor, adjusted, {candidate.doc_id: candidate for candidate in candidates})

    def _apply_memory_updates(self, anchor_doc_id: str) -> None:
        payload = read_json(self._stage_path("memory", anchor_doc_id), {}) or {}
        if not payload:
            return
        bundle = MemoryLearningBundle(
            anchor_doc_id=str(payload.get("anchor_doc_id", anchor_doc_id)),
            generated_at=str(payload.get("generated_at", "")),
            learned_patterns=[str(item) for item in payload.get("learned_patterns", [])],
            failure_patterns=[str(item) for item in payload.get("failure_patterns", [])],
            reference_updates={
                str(key): [str(item) for item in value]
                for key, value in dict(payload.get("reference_updates", {})).items()
            },
        )
        self.self_update_manager.apply(bundle, self.agents_memory_path)

    def discover_for_anchor(self, anchor_doc_id: str, briefs: list[DocBrief]) -> list:
        brief_map = {brief.doc_id: brief for brief in briefs}
        anchor = brief_map[anchor_doc_id]
        if self.deepagent is not None and self.anchor_task_delegation_enabled:
            self._run_anchor_workflow_with_deepagents(anchor_doc_id)
        else:
            self._run_anchor_workflow_local(anchor_doc_id)
        candidates, verdicts, reviews, bundle_lookup = self._load_candidate_assets(anchor_doc_id, briefs)
        if not candidates or not verdicts:
            return self.discovery_service.discover_for_anchor(anchor_doc_id, briefs)
        assessments = self._apply_review_consensus(anchor, candidates, verdicts, reviews, bundle_lookup)
        accepted = self.discovery_service.commit_assessments(anchor, assessments)
        self._apply_memory_updates(anchor_doc_id)
        return accepted

    def discover_edges(self, docs, briefs: list[DocBrief]) -> list:
        plan = self._build_plan()
        ordered = [item.get("doc_id") for item in plan.get("anchors", []) if item.get("doc_id")]
        if not ordered:
            ordered = self.orchestrator.rank_discovery_anchors(briefs)
        brief_map = {brief.doc_id: brief for brief in briefs}
        accepted = []
        for doc_id in ordered:
            anchor = brief_map.get(doc_id)
            if anchor is None or not self.orchestrator.should_attempt_discovery(anchor):
                continue
            if doc_id not in {doc.doc_id for doc in docs}:
                continue
            accepted.extend(self.discover_for_anchor(doc_id, briefs))
        return accepted
