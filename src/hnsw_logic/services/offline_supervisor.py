from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hnsw_logic.agents.orchestrator import CandidateAssessment
from hnsw_logic.agents.runtime_models import ExecutionAudit, MemoryLearningBundle
from hnsw_logic.agents.tools.deepagents_runtime import (
    audit_execution_state,
    record_manifest_stage_event,
    required_indexing_stages,
)
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
        self.supervisor_task_delegation_enabled = self._flag_from_env(
            "GL_HNSW_ENABLE_DEEPAGENT_SUPERVISOR",
            self.agents_config.enable_supervisor_delegation,
        )
        self.anchor_task_delegation_enabled = (
            self.supervisor_task_delegation_enabled
            and self._flag_from_env(
                "GL_HNSW_ENABLE_ANCHOR_TASK_DELEGATION",
                self.agents_config.enable_anchor_task_delegation,
            )
        )
        self._stage_runners = {
            "dossiers": ("doc_profiler", "execute_doc_profiling"),
            "candidates": ("corpus_scout", "execute_candidate_expansion"),
            "judgments": ("relation_judge", "execute_relation_judging"),
            "checks": ("counterevidence_checker", "execute_counterevidence_check"),
            "reviews": ("edge_reviewer", "execute_edge_review"),
            "memory": ("memory_curator", "execute_memory_summarization"),
        }

    @staticmethod
    def _flag_from_env(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _normalize_risk_flag(flag: str) -> str:
        return str(flag).strip().lower().replace("-", "_").replace(" ", "_")

    def _stage_path(self, stage: str, doc_id: str, suffix: str = ".json") -> Path:
        return self.workspace_root / "indexing" / stage / f"{doc_id}{suffix}"

    def _plan_path(self) -> Path:
        return self.workspace_root / "indexing" / "plans" / "indexing_plan.json"

    def _required_stages(self) -> list[str]:
        return required_indexing_stages(self.agents_config.counterevidence_enabled)

    def _audit_anchor_execution(self, anchor_doc_id: str) -> ExecutionAudit:
        return audit_execution_state(
            self.workspace_root,
            anchor_doc_id,
            counterevidence_enabled=self.agents_config.counterevidence_enabled,
            task_iteration_cap=self.agents_config.task_iteration_cap,
        )

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
                    "\n".join(
                        [
                            "Create the offline indexing plan.",
                            "Use the delegation-policy skill and keep the plan auditable.",
                            f"Respect a maximum of {self.agents_config.max_parallel_tasks} active task slots.",
                            "Use the task tool to delegate to the index_planner subagent.",
                            "The subagent must call execute_index_planning and materialize the plan in the current workspace indexing/plans directory.",
                            "Audit the plan by checking the file exists before returning.",
                        ]
                    )
                )
            except Exception:
                self._run_stage_locally("index_planner", "execute_index_planning")
        else:
            self._run_stage_locally("index_planner", "execute_index_planning")
        return read_json(self._plan_path(), {}) or {}

    @staticmethod
    def _ordered_anchor_ids_from_plan(plan: dict[str, Any]) -> list[str]:
        ordered: list[str] = []
        if isinstance(plan.get("anchors"), list):
            ordered.extend(str(item.get("doc_id")) for item in plan["anchors"] if isinstance(item, dict) and item.get("doc_id"))
        anchor_selection = plan.get("anchor_selection")
        if isinstance(anchor_selection, dict):
            ordered.extend(str(doc_id) for doc_id in anchor_selection.get("priority_order", []) if doc_id)
        if isinstance(plan.get("batches"), list):
            for batch in plan["batches"]:
                if not isinstance(batch, dict):
                    continue
                ordered.extend(str(doc_id) for doc_id in batch.get("anchors", []) if doc_id)
        deduped: list[str] = []
        seen: set[str] = set()
        for doc_id in ordered:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            deduped.append(doc_id)
        return deduped

    def _run_anchor_workflow_local(self, anchor_doc_id: str, stages: list[str] | None = None) -> None:
        target_stages = stages or self._required_stages()
        for stage in target_stages:
            runner = self._stage_runners.get(stage)
            if runner is None:
                continue
            agent_name, tool_name = runner
            self._run_stage_locally(agent_name, tool_name, anchor_doc_id=anchor_doc_id)

    def _deepagent_stage_prompt(self, anchor_doc_id: str, stage: str, audit: ExecutionAudit) -> str:
        runner = self._stage_runners[stage]
        agent_name, tool_name = runner
        return "\n".join(
            [
                f"Resume the offline indexing workflow for anchor `{anchor_doc_id}`.",
                "Use the execution-audit, resume-recovery, and delegation-policy skills.",
                f"First call audit_anchor_execution for `{anchor_doc_id}` and confirm the next missing stage is `{stage}`.",
                f"Then use the task tool to delegate only the `{stage}` stage to `{agent_name}`.",
                f"The subagent must call `{tool_name}` and rely on workspace files as the handoff boundary.",
                "After the delegated task returns, call audit_anchor_execution again and verify progress.",
                "Do not edit code, configs, benchmark labels, or core SKILL.md files.",
                f"Respect the iteration cap of {self.agents_config.task_iteration_cap} and the task-slot budget of {self.agents_config.max_parallel_tasks}.",
                f"Already completed stages: {', '.join(audit.completed_stages) or 'none'}.",
            ]
        )

    def _run_anchor_workflow_with_deepagents(self, anchor_doc_id: str) -> None:
        rounds = 0
        max_rounds = max(1, self.agents_config.task_iteration_cap) * max(1, len(self._required_stages()))
        while rounds < max_rounds:
            audit = self._audit_anchor_execution(anchor_doc_id)
            if audit.workflow_complete:
                return
            if not audit.next_stage:
                break
            if audit.should_fallback:
                break
            stage = audit.next_stage
            record_manifest_stage_event(
                self.workspace_root,
                anchor_doc_id,
                stage=stage,
                status="started",
                note=f"delegating `{stage}` via deepagent",
                increment_round=True,
            )
            try:
                self._invoke_main_agent(self._deepagent_stage_prompt(anchor_doc_id, stage, audit))
            except Exception as exc:
                record_manifest_stage_event(
                    self.workspace_root,
                    anchor_doc_id,
                    stage=stage,
                    status="failed",
                    error=str(exc),
                    note="deepagent stage invocation failed",
                )
            post_audit = self._audit_anchor_execution(anchor_doc_id)
            if stage in post_audit.completed_stages:
                rounds += 1
                continue
            record_manifest_stage_event(
                self.workspace_root,
                anchor_doc_id,
                stage=stage,
                status="failed",
                error=f"delegated stage `{stage}` did not materialize the expected artifact",
            )
            rounds += 1
        final_audit = self._audit_anchor_execution(anchor_doc_id)
        if not final_audit.workflow_complete:
            record_manifest_stage_event(
                self.workspace_root,
                anchor_doc_id,
                stage=final_audit.next_stage or final_audit.current_stage or "fallback",
                status="started",
                note="switching to local fallback after bounded delegation loop",
                force_fallback=True,
            )
            self._run_anchor_workflow_local(anchor_doc_id, stages=final_audit.missing_stages)

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
            contradiction_like_risks = {
                flag
                for flag in risk_flags
                if flag.startswith("contradict")
                or flag.startswith("counterargument")
                or flag.startswith("oppos")
                or flag.startswith("alternative_position")
            }
            soft_bridge_risks = {"excess_novelty", "weak_family_bridge", "topic_only_overlap", "low_retrieval_utility", "weak_direction"}
            duplicate_only_comparison_risks = {"near_duplicate", "near_duplicate_bridge"}
            same_concept_soft_keep = (
                assessment.edge is not None
                and assessment.edge.relation_type == "same_concept"
                and assessment.edge.utility_score >= 0.78
                and assessment.local_support >= 0.68
                and assessment.evidence_quality >= 0.84
                and risk_penalty <= 0.22
                and risk_flags.issubset(soft_bridge_risks)
            )
            comparison_bridge_keep = (
                assessment.edge is not None
                and assessment.edge.relation_type == "comparison"
                and assessment.edge.utility_score >= 0.86
                and assessment.local_support >= 0.52
                and assessment.evidence_quality >= 0.76
                and risk_penalty <= 0.28
                and risk_flags
                and (risk_flags - contradiction_like_risks).issubset(duplicate_only_comparison_risks)
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
                    or (
                        comparison_bridge_keep
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
        used_delegation = self.deepagent is not None and self.anchor_task_delegation_enabled
        if used_delegation:
            self._run_anchor_workflow_with_deepagents(anchor_doc_id)
        else:
            self._run_anchor_workflow_local(anchor_doc_id, stages=self._audit_anchor_execution(anchor_doc_id).missing_stages)
        candidates, verdicts, reviews, bundle_lookup = self._load_candidate_assets(anchor_doc_id, briefs)
        if not candidates:
            self._apply_memory_updates(anchor_doc_id)
            return []
        if not verdicts:
            self._apply_memory_updates(anchor_doc_id)
            if used_delegation:
                audit = self._audit_anchor_execution(anchor_doc_id)
                raise RuntimeError(
                    f"deepagents workflow for `{anchor_doc_id}` did not materialize judgments; "
                    f"missing stages={audit.missing_stages}, completed={audit.completed_stages}"
                )
            return []
        assessments = self._apply_review_consensus(anchor, candidates, verdicts, reviews, bundle_lookup)
        accepted = self.discovery_service.commit_assessments(anchor, assessments)
        self._apply_memory_updates(anchor_doc_id)
        return accepted

    def discover_edges(self, docs, briefs: list[DocBrief]) -> list:
        plan = self._build_plan()
        ordered = self._ordered_anchor_ids_from_plan(plan)
        if not ordered:
            ordered = self.orchestrator.rank_discovery_anchors(briefs)
        brief_map = {brief.doc_id: brief for brief in briefs}
        accepted = []
        eligible = []
        available_doc_ids = {doc.doc_id for doc in docs}
        for doc_id in ordered:
            anchor = brief_map.get(doc_id)
            if anchor is None or not self.orchestrator.should_attempt_discovery(anchor):
                continue
            if doc_id not in available_doc_ids:
                continue
            eligible.append(doc_id)
        batch_size = max(1, self.agents_config.max_parallel_tasks)
        for offset in range(0, len(eligible), batch_size):
            for doc_id in eligible[offset : offset + batch_size]:
                accepted.extend(self.discover_for_anchor(doc_id, briefs))
        return accepted
