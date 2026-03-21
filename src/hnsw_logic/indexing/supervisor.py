from __future__ import annotations

from pathlib import Path
from typing import Any

from hnsw_logic.agents.orchestration.orchestrator import CandidateAssessment
from hnsw_logic.agents.runtime.execution import (
    audit_execution_state,
    normalize_indexing_workspace,
    record_manifest_stage_event,
    required_indexing_stages,
)
from hnsw_logic.agents.runtime.models import ExecutionAudit, MemoryLearningBundle
from hnsw_logic.domain.models import DocBrief
from hnsw_logic.domain.serialization import read_json
from hnsw_logic.embedding.providers.types import JudgeResult


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
        self._stage_runners = {
            "dossiers": ("doc_profiler", "execute_doc_profiling"),
            "candidates": ("corpus_scout", "execute_candidate_expansion"),
            "judgments": ("relation_judge", "execute_relation_judging"),
            "checks": ("counterevidence_checker", "execute_counterevidence_check"),
            "reviews": ("edge_reviewer", "execute_edge_review"),
            "memory": ("memory_curator", "execute_memory_summarization"),
        }

    def _stage_path(self, stage: str, doc_id: str, suffix: str = ".json") -> Path:
        return self.workspace_root / "indexing" / stage / f"{doc_id}{suffix}"

    def _plan_path(self) -> Path:
        return self.workspace_root / "indexing" / "plans" / "indexing_plan.json"

    def _required_stages(self) -> list[str]:
        return required_indexing_stages(self.agents_config.counterevidence_enabled)

    def _relative_stage_artifact_path(self, stage: str, anchor_doc_id: str) -> str:
        """Return the workspace-relative artifact path for a delegated stage."""
        return self._stage_path(stage, anchor_doc_id).relative_to(self.workspace_root).as_posix()

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

    def _run_stage_test_harness(self, agent_name: str, tool_name: str, **kwargs) -> dict:
        tool = self.runtime_toolsets.get(agent_name, {}).get(tool_name)
        if tool is None:
            return {}
        return tool(**kwargs)

    def _build_plan(self) -> dict:
        normalize_indexing_workspace(self.workspace_root)
        if self.deepagent is None:
            raise RuntimeError("deepagent runtime unavailable for offline indexing planning")
        self._invoke_main_agent(
            "\n".join(
                [
                    "Create the offline indexing plan.",
                    "Use the delegation-policy skill and keep the plan auditable.",
                    f"Respect a maximum of {self.agents_config.max_parallel_tasks} active task slots.",
                    "Use the task tool to delegate to the index_planner subagent.",
                    "The subagent must call execute_index_planning and materialize the plan at the exact workspace path `indexing/plans/indexing_plan.json`.",
                    "Never write freeform notes directly to the stage directory path `indexing/plans`.",
                    "Audit the plan by checking that `indexing/plans/indexing_plan.json` exists before returning.",
                ]
            )
        )
        normalize_indexing_workspace(self.workspace_root)
        plan = read_json(self._plan_path(), {}) or {}
        if not plan:
            raise RuntimeError(
                "deepagents planning did not materialize `indexing/plans/indexing_plan.json`"
            )
        return plan

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

    def run_anchor_workflow_test_harness(self, anchor_doc_id: str, stages: list[str] | None = None) -> None:
        target_stages = stages or self._required_stages()
        for stage in target_stages:
            runner = self._stage_runners.get(stage)
            if runner is None:
                continue
            agent_name, tool_name = runner
            self._run_stage_test_harness(agent_name, tool_name, anchor_doc_id=anchor_doc_id)

    def run_index_planning_test_harness(self) -> dict:
        normalize_indexing_workspace(self.workspace_root)
        self._run_stage_test_harness("index_planner", "execute_index_planning")
        return read_json(self._plan_path(), {}) or {}

    def _deepagent_stage_prompt(self, anchor_doc_id: str, stage: str, audit: ExecutionAudit) -> str:
        runner = self._stage_runners[stage]
        agent_name, tool_name = runner
        artifact_path = self._relative_stage_artifact_path(stage, anchor_doc_id)
        stage_instruction = (
            f"The delegated subagent must call `{tool_name}(anchor_doc_id=\"{anchor_doc_id}\", "
            f"output_path=\"{artifact_path}\")` unless `{artifact_path}` already exists and can be reused as-is."
        )
        stage_completion_rule = f"The stage is complete only when `{artifact_path}` exists."
        stage_context = ""
        if stage == "memory":
            stage_context = (
                "Read the review bundle, summarize durable learned and failure patterns, "
                "and materialize the memory bundle. Do not stop at prose notes or proposed updates."
            )
        return "\n".join(
            [
                f"Resume the offline indexing workflow for anchor `{anchor_doc_id}`.",
                "Use the execution-audit, resume-recovery, and delegation-policy skills.",
                f"First call audit_anchor_execution for `{anchor_doc_id}` and confirm the next missing stage is `{stage}`.",
                f"Then use the task tool to delegate only the `{stage}` stage to `{agent_name}`.",
                stage_instruction,
                stage_completion_rule,
                stage_context,
                "Rely on workspace files as the handoff boundary.",
                "After the delegated task returns, call audit_anchor_execution again and verify progress.",
                "Do not edit code, configs, benchmark labels, or core SKILL.md files.",
                f"Respect the iteration cap of {self.agents_config.task_iteration_cap} and the task-slot budget of {self.agents_config.max_parallel_tasks}.",
                f"Already completed stages: {', '.join(audit.completed_stages) or 'none'}.",
            ]
        )

    def _run_anchor_workflow_with_deepagents(self, anchor_doc_id: str) -> None:
        if self.deepagent is None:
            raise RuntimeError("deepagent runtime unavailable for anchor delegation")
        rounds = 0
        max_rounds = max(1, self.agents_config.task_iteration_cap) * max(1, len(self._required_stages()))
        while rounds < max_rounds:
            audit = self._audit_anchor_execution(anchor_doc_id)
            if audit.workflow_complete:
                return
            if not audit.next_stage:
                break
            if audit.retry_exhausted:
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
                stage=final_audit.next_stage or final_audit.current_stage or "delegation",
                status="failed",
                error="deepagent delegation did not materialize the full anchor workflow",
                force_halt=True,
            )
            raise RuntimeError(
                f"deepagents workflow for `{anchor_doc_id}` did not complete; "
                f"missing stages={final_audit.missing_stages}, completed={final_audit.completed_stages}"
            )

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

    def _validate_candidate_assets(
        self,
        anchor_doc_id: str,
        candidates: list[DocBrief],
        verdicts: dict[str, JudgeResult],
        reviews: dict[str, JudgeResult],
        bundle_lookup: dict[str, dict],
    ) -> None:
        candidate_ids = {candidate.doc_id for candidate in candidates}
        if not candidate_ids:
            return
        missing_judgments = sorted(candidate_ids - set(verdicts))
        if missing_judgments:
            raise RuntimeError(
                f"deepagents workflow for `{anchor_doc_id}` materialized incomplete judgments: "
                f"missing={missing_judgments}"
            )
        missing_reviews = sorted(candidate_ids - set(bundle_lookup.get('reviews', {})))
        if missing_reviews:
            raise RuntimeError(
                f"deepagents workflow for `{anchor_doc_id}` materialized incomplete reviews: "
                f"missing={missing_reviews}"
            )
        missing_review_verdicts = sorted(candidate_ids - set(reviews))
        if missing_review_verdicts:
            raise RuntimeError(
                f"deepagents workflow for `{anchor_doc_id}` materialized review rows without final verdicts: "
                f"missing={missing_review_verdicts}"
            )
        if self.agents_config.counterevidence_enabled:
            missing_checks = sorted(candidate_ids - set(bundle_lookup.get("checks", {})))
            if missing_checks:
                raise RuntimeError(
                    f"deepagents workflow for `{anchor_doc_id}` materialized incomplete checks: "
                    f"missing={missing_checks}"
                )

    def _apply_review_consensus(
        self,
        anchor: DocBrief,
        candidates: list[DocBrief],
        verdicts: dict[str, JudgeResult],
        reviews: dict[str, JudgeResult],
        bundle_lookup: dict[str, dict],
    ) -> list[CandidateAssessment]:
        review_rows = bundle_lookup.get("reviews", {})
        check_rows = bundle_lookup.get("checks", {})
        adjusted: list[CandidateAssessment] = []
        for candidate in candidates:
            if candidate.doc_id not in verdicts:
                continue
            review_verdict = reviews.get(candidate.doc_id)
            assessment = self.orchestrator._assessment_for(
                anchor,
                candidate,
                verdicts.get(candidate.doc_id),
                review_verdict,
            )
            review_row = review_rows.get(assessment.candidate_doc_id, {})
            check_row = check_rows.get(assessment.candidate_doc_id, {}) if self.agents_config.counterevidence_enabled else {"keep": True}
            reject_reason = ""
            if not review_row:
                reject_reason = "missing_review_artifact"
            elif self.agents_config.counterevidence_enabled and not check_row:
                reject_reason = "missing_check_artifact"
            elif review_verdict is None:
                reject_reason = "missing_review_verdict"
            review_keep = bool(review_row.get("keep", False)) if review_row else False
            check_keep = bool(check_row.get("keep", False)) if check_row else False
            reject_reason = str(
                reject_reason
                or review_row.get("decision_reason")
                or check_row.get("decision_reason")
                or assessment.reject_reason
                or "review_rejected"
            )[:160]
            keep = review_keep and check_keep
            if not keep:
                adjusted.append(
                    CandidateAssessment(
                        candidate_doc_id=assessment.candidate_doc_id,
                        accepted=False,
                        reject_reason=reject_reason,
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
                    next_assessment.edge.utility_score = max(0.0, min(1.0, float(reviewed_utility)))
                    next_assessment.score *= 0.85 + 0.35 * next_assessment.edge.utility_score
                if reviewed_confidence is not None:
                    next_assessment.edge.confidence = max(next_assessment.edge.confidence, float(reviewed_confidence))
                activation_profile = review_row.get("activation_profile")
                if isinstance(activation_profile, dict) and activation_profile:
                    next_assessment.edge.activation_profile = dict(activation_profile)
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
        normalize_indexing_workspace(self.workspace_root, anchor_doc_id)
        brief_map = {brief.doc_id: brief for brief in briefs}
        anchor = brief_map[anchor_doc_id]
        self._run_anchor_workflow_with_deepagents(anchor_doc_id)
        candidates, verdicts, reviews, bundle_lookup = self._load_candidate_assets(anchor_doc_id, briefs)
        if not candidates:
            self._apply_memory_updates(anchor_doc_id)
            return []
        try:
            self._validate_candidate_assets(anchor_doc_id, candidates, verdicts, reviews, bundle_lookup)
        except RuntimeError:
            self._apply_memory_updates(anchor_doc_id)
            raise
        assessments = self._apply_review_consensus(anchor, candidates, verdicts, reviews, bundle_lookup)
        accepted = self.discovery_service.commit_assessments(anchor, assessments)
        self._apply_memory_updates(anchor_doc_id)
        return accepted

    def discover_edges(self, docs, briefs: list[DocBrief]) -> list:
        plan = self._build_plan()
        ordered = self._ordered_anchor_ids_from_plan(plan)
        if not ordered:
            raise RuntimeError("deepagents planning returned no ordered anchors")
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
