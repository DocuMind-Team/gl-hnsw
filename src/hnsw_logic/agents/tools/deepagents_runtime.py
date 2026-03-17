from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from hnsw_logic.agents.runtime_models import (
    AnchorDossier,
    AnchorPlan,
    CandidateBundle,
    CandidateBundleItem,
    CounterevidenceBundle,
    CounterevidenceBundleItem,
    ExecutionAudit,
    ExecutionManifest,
    IndexingPlan,
    JudgmentBundle,
    JudgmentBundleItem,
    MemoryLearningBundle,
    ReviewBundle,
    ReviewBundleItem,
)
from hnsw_logic.core.models import DocBrief, LogicEdge
from hnsw_logic.core.utils import read_json, to_jsonable, utc_now, write_json


STAGE_SEQUENCE = ("dossiers", "candidates", "judgments", "checks", "reviews", "memory")
COMMIT_REQUIRED_STAGES = ("dossiers", "candidates", "judgments", "checks", "reviews")


def required_indexing_stages(counterevidence_enabled: bool = True) -> list[str]:
    return [stage for stage in STAGE_SEQUENCE if counterevidence_enabled or stage != "checks"]


def manifest_path(workspace_root: Path, anchor_doc_id: str) -> Path:
    return workspace_root / "indexing" / "manifests" / f"{anchor_doc_id}.json"


def stage_artifact_path(workspace_root: Path, stage: str, anchor_doc_id: str, suffix: str = ".json") -> Path:
    return workspace_root / "indexing" / stage / f"{anchor_doc_id}{suffix}"


def load_execution_manifest(workspace_root: Path, anchor_doc_id: str) -> ExecutionManifest:
    payload = read_json(manifest_path(workspace_root, anchor_doc_id))
    if isinstance(payload, dict):
        return ExecutionManifest(**payload)
    now = utc_now()
    manifest = ExecutionManifest(anchor_doc_id=anchor_doc_id, generated_at=now, updated_at=now)
    save_execution_manifest(workspace_root, manifest)
    return manifest


def save_execution_manifest(workspace_root: Path, manifest: ExecutionManifest) -> None:
    manifest.updated_at = utc_now()
    write_json(manifest_path(workspace_root, manifest.anchor_doc_id), manifest)


def record_manifest_stage_event(
    workspace_root: Path,
    anchor_doc_id: str,
    *,
    stage: str,
    status: str,
    note: str = "",
    error: str = "",
    increment_round: bool = False,
    force_fallback: bool = False,
) -> ExecutionManifest:
    manifest = load_execution_manifest(workspace_root, anchor_doc_id)
    manifest.current_stage = stage
    if increment_round:
        manifest.delegation_round += 1
    if status == "completed":
        if stage not in manifest.completed_stages:
            manifest.completed_stages.append(stage)
        manifest.failed_stages.pop(stage, None)
        manifest.last_error = ""
    elif status == "failed":
        manifest.retry_counts[stage] = int(manifest.retry_counts.get(stage, 0)) + 1
        manifest.failed_stages[stage] = (error or note or "stage failed")[:240]
        manifest.last_error = manifest.failed_stages[stage]
    elif status == "started":
        manifest.failed_stages.pop(stage, None)
    if note:
        manifest.notes = [*manifest.notes[-11:], note[:240]]
    if force_fallback:
        manifest.needs_fallback = True
    save_execution_manifest(workspace_root, manifest)
    return manifest


def audit_execution_state(
    workspace_root: Path,
    anchor_doc_id: str,
    *,
    counterevidence_enabled: bool = True,
    task_iteration_cap: int = 2,
) -> ExecutionAudit:
    manifest = load_execution_manifest(workspace_root, anchor_doc_id)
    required = required_indexing_stages(counterevidence_enabled)
    artifact_paths = {stage: str(stage_artifact_path(workspace_root, stage, anchor_doc_id)) for stage in required}
    completed = []
    for stage in required:
        if stage_artifact_path(workspace_root, stage, anchor_doc_id).exists():
            completed.append(stage)
        elif stage in manifest.completed_stages:
            completed.append(stage)
    completed = [stage for stage in required if stage in completed]
    if completed != manifest.completed_stages:
        manifest.completed_stages = completed
        save_execution_manifest(workspace_root, manifest)
    missing = [stage for stage in required if stage not in completed]
    required_for_commit = [stage for stage in COMMIT_REQUIRED_STAGES if counterevidence_enabled or stage != "checks"]
    ready_for_commit = all(stage in completed for stage in required_for_commit)
    workflow_complete = all(stage in completed for stage in required)
    should_fallback = bool(manifest.needs_fallback)
    if not should_fallback and missing:
        next_stage = missing[0]
        should_fallback = int(manifest.retry_counts.get(next_stage, 0)) >= max(task_iteration_cap, 1)
    return ExecutionAudit(
        anchor_doc_id=anchor_doc_id,
        generated_at=utc_now(),
        current_stage=manifest.current_stage,
        completed_stages=completed,
        missing_stages=missing,
        retry_counts=dict(manifest.retry_counts),
        artifact_paths=artifact_paths,
        next_stage=missing[0] if missing else "",
        ready_for_commit=ready_for_commit,
        workflow_complete=workflow_complete,
        should_fallback=should_fallback,
        notes=list(manifest.notes),
    )


def build_deepagent_toolsets(
    *,
    provider,
    corpus_store,
    brief_store,
    graph_store,
    anchor_memory_store,
    semantic_memory_store,
    graph_memory_store,
    hnsw_searcher,
    orchestrator,
    workspace_root: Path,
    config_subagents: dict[str, Any],
    counterevidence_enabled: bool,
    task_iteration_cap: int,
) -> dict[str, dict[str, Callable]]:
    processed_docs_cache: list | None = None

    def processed_docs():
        nonlocal processed_docs_cache
        if processed_docs_cache is None:
            processed_docs_cache = corpus_store.read_processed()
        return processed_docs_cache

    def brief_map() -> dict[str, Any]:
        return {brief.doc_id: brief for brief in brief_store.all()}

    def doc_map() -> dict[str, Any]:
        return {doc.doc_id: doc for doc in processed_docs()}

    def stage_dir(name: str) -> Path:
        path = workspace_root / "indexing" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_path(stage: str, doc_id: str, suffix: str = ".json") -> Path:
        return stage_artifact_path(workspace_root, stage, doc_id, suffix)

    def mark_stage_started(anchor_doc_id: str, stage: str, note: str = "") -> None:
        record_manifest_stage_event(workspace_root, anchor_doc_id, stage=stage, status="started", note=note)

    def mark_stage_completed(anchor_doc_id: str, stage: str, note: str = "") -> None:
        record_manifest_stage_event(workspace_root, anchor_doc_id, stage=stage, status="completed", note=note)

    def mark_stage_failed(anchor_doc_id: str, stage: str, error: str) -> None:
        record_manifest_stage_event(workspace_root, anchor_doc_id, stage=stage, status="failed", error=error)

    def read_graph_stats() -> dict:
        """Read persisted graph statistics for offline indexing planning."""
        return graph_memory_store.read()

    def read_semantic_memory() -> dict:
        """Read summarized semantic memory state used by offline agents."""
        return to_jsonable(semantic_memory_store.read())

    def read_failure_patterns() -> dict:
        """Read accumulated rejection and failure patterns from semantic memory."""
        memory = semantic_memory_store.read()
        return {"rejection_patterns": memory.rejection_patterns}

    def read_anchor_dossier(doc_id: str) -> dict | None:
        """Read a previously materialized anchor dossier from the workspace."""
        return read_json(stage_path("dossiers", doc_id))

    def read_candidate_bundle(doc_id: str) -> dict | None:
        """Read a candidate bundle for a specific anchor from the workspace."""
        return read_json(stage_path("candidates", doc_id))

    def read_judgment_bundle(doc_id: str) -> dict | None:
        """Read a judgment bundle for a specific anchor from the workspace."""
        return read_json(stage_path("judgments", doc_id))

    def read_counterevidence_bundle(doc_id: str) -> dict | None:
        """Read a counterevidence bundle for a specific anchor from the workspace."""
        return read_json(stage_path("checks", doc_id))

    def read_review_bundle(doc_id: str) -> dict | None:
        """Read a review bundle for a specific anchor from the workspace."""
        return read_json(stage_path("reviews", doc_id))

    def read_execution_manifest(doc_id: str) -> dict:
        """Read the execution manifest that tracks stage completion for an anchor."""
        return to_jsonable(load_execution_manifest(workspace_root, doc_id))

    def audit_anchor_execution(doc_id: str) -> dict:
        """Audit workspace artifacts for an anchor and report the next required stage."""
        return to_jsonable(
            audit_execution_state(
                workspace_root,
                doc_id,
                counterevidence_enabled=counterevidence_enabled,
                task_iteration_cap=task_iteration_cap,
            )
        )

    def evaluate_anchor_utility(doc_id: str) -> dict:
        """Summarize review-stage utility signals for an anchor to support prioritization and auditing."""
        review_payload = read_review_bundle(doc_id) or {}
        reviews = list(review_payload.get("reviews", []))
        kept = [item for item in reviews if item.get("keep")]
        utilities = [float(item.get("reviewed_utility_score", 0.0) or 0.0) for item in reviews]
        return {
            "anchor_doc_id": doc_id,
            "review_count": len(reviews),
            "kept_count": len(kept),
            "top_reviewed_utility": round(max(utilities), 6) if utilities else 0.0,
            "mean_reviewed_utility": round(sum(utilities) / len(utilities), 6) if utilities else 0.0,
            "risk_flags": sorted(
                {
                    str(flag)
                    for item in reviews
                    for flag in item.get("risk_flags", [])
                    if str(flag)
                }
            )[:16],
        }

    def execute_index_planning(output_path: str = "/data/workspace/indexing/plans/indexing_plan.json") -> dict:
        """Generate the offline indexing plan and persist it to the workspace."""
        destination = workspace_root.parent.parent / output_path.lstrip("/") if output_path.startswith("/") else workspace_root / output_path
        existing_plan = read_json(destination)
        if existing_plan:
            return {"plan_path": str(destination), "anchors": len(existing_plan.get("anchors", [])), "cached": True}
        briefs = brief_store.all()
        ordered = orchestrator.rank_discovery_anchors(briefs)
        anchor_limit = min(
            len(ordered),
            max(
                4,
                min(
                    getattr(orchestrator.retrieval_config, "adaptive_graph_seed_cap", 15),
                    int(max(len(briefs), 1) * 0.3),
                ),
            ),
        )
        ordered = ordered[:anchor_limit]
        graph_profile = orchestrator._corpus_graph_profile(briefs) if briefs else {"graph_potential": 0.0}
        batches: list[AnchorPlan] = []
        batch_size = max(1, min(getattr(orchestrator.retrieval_config, "adaptive_graph_seed_cap", 15), 12))
        brief_lookup = {brief.doc_id: brief for brief in briefs}
        for index, doc_id in enumerate(ordered):
            brief = brief_lookup.get(doc_id)
            if brief is None:
                continue
            batches.append(
                AnchorPlan(
                    doc_id=doc_id,
                    priority=round(orchestrator.discovery_anchor_priority(brief), 6),
                    batch_id=f"batch-{index // batch_size + 1:03d}",
                    bridge_potential=round(orchestrator._specific_title_bridge_potential(brief, briefs), 6),
                    coverage_pressure=round(orchestrator._dataset_edge_signal(brief), 6),
                    reason="priority-ranked by offline utility and coverage pressure",
                )
            )
        dataset_hint = str(briefs[0].metadata.get("dataset", briefs[0].metadata.get("topic", "unknown"))) if briefs else "unknown"
        plan = IndexingPlan(
            generated_at=utc_now(),
            dataset_hint=dataset_hint,
            graph_potential=round(float(graph_profile.get("graph_potential", 0.0)), 6),
            anchors=batches,
            notes=[
                "offline deepagents supervisor plan",
                "anchors sorted by utility, centrality, coverage pressure, and bridge reserve",
            ],
        )
        planning_payload = provider.plan_indexing_batch(to_jsonable(plan))
        if isinstance(planning_payload, dict):
            if planning_payload.get("notes"):
                plan.notes = [str(item) for item in planning_payload.get("notes", [])][:8]
            if planning_payload.get("graph_potential") is not None:
                plan.graph_potential = round(float(planning_payload["graph_potential"]), 6)
        write_json(destination, plan)
        return {"plan_path": str(destination), "anchors": len(plan.anchors)}

    def execute_doc_profiling(anchor_doc_id: str, output_path: str | None = None) -> dict:
        """Build or refresh the dossier for a single anchor document."""
        destination = Path(output_path) if output_path else stage_path("dossiers", anchor_doc_id)
        existing_dossier = read_json(destination)
        if existing_dossier:
            existing_brief = existing_dossier.get("brief")
            if brief_store.read(anchor_doc_id) is None and isinstance(existing_brief, dict):
                brief_store.write(DocBrief(**existing_brief))
            mark_stage_completed(anchor_doc_id, "dossiers", "reused cached dossier")
            return {"dossier_path": str(destination), "anchor_doc_id": anchor_doc_id, "cached": True}
        mark_stage_started(anchor_doc_id, "dossiers", "building anchor dossier")
        brief = brief_store.read(anchor_doc_id)
        if brief is None:
            doc = doc_map()[anchor_doc_id]
            brief = orchestrator.profile(doc)
            brief_store.write(brief)
        payload = AnchorDossier(
            anchor_doc_id=anchor_doc_id,
            dataset_hint=str(brief.metadata.get("dataset", brief.metadata.get("topic", "unknown"))),
            brief=to_jsonable(brief),
            full_doc=to_jsonable(doc_map().get(anchor_doc_id)),
            anchor_memory=to_jsonable(anchor_memory_store.read(anchor_doc_id)),
            semantic_memory=to_jsonable(semantic_memory_store.read()),
            graph_stats=graph_memory_store.read(),
            surrogate_query_terms=orchestrator._surrogate_query_terms(brief),
            active_hypotheses=anchor_memory_store.read(anchor_doc_id).active_hypotheses,
        )
        write_json(destination, payload)
        mark_stage_completed(anchor_doc_id, "dossiers", "anchor dossier materialized")
        return {"dossier_path": str(destination), "anchor_doc_id": anchor_doc_id}

    def execute_candidate_expansion(anchor_doc_id: str, output_path: str | None = None, expanded: bool = False) -> dict:
        """Create a candidate bundle for an anchor using scout and local signals."""
        destination = Path(output_path) if output_path else stage_path("candidates", anchor_doc_id)
        existing_bundle = read_json(destination)
        if existing_bundle:
            mark_stage_completed(anchor_doc_id, "candidates", "reused cached candidate bundle")
            return {"candidate_bundle_path": str(destination), "candidate_count": len(existing_bundle.get("candidates", [])), "cached": True}
        mark_stage_started(anchor_doc_id, "candidates", "expanding candidate bundle")
        briefs = brief_store.all()
        lookup = {brief.doc_id: brief for brief in briefs}
        anchor = lookup[anchor_doc_id]
        proposals = orchestrator.scout(anchor, briefs, expanded=expanded)
        items: list[CandidateBundleItem] = []
        for proposal in proposals:
            candidate = lookup.get(proposal.doc_id)
            if candidate is None:
                continue
            metrics = orchestrator._candidate_metrics(anchor, candidate)
            _, relation_type, fit_scores = orchestrator._pair_rerank(anchor, candidate, metrics)
            signals = orchestrator._signal_bundle(anchor, candidate, metrics, fit_scores, relation_type)
            items.append(
                CandidateBundleItem(
                    candidate_doc_id=candidate.doc_id,
                    proposal_reason=proposal.reason,
                    query=proposal.query,
                    score_hint=proposal.score_hint,
                    signals=to_jsonable(signals),
                    candidate_brief=to_jsonable(candidate),
                )
            )
        bundle = CandidateBundle(anchor_doc_id=anchor_doc_id, generated_at=utc_now(), candidates=items)
        write_json(destination, bundle)
        mark_stage_completed(anchor_doc_id, "candidates", f"candidate bundle with {len(items)} items")
        return {"candidate_bundle_path": str(destination), "candidate_count": len(items)}

    def execute_relation_judging(anchor_doc_id: str, output_path: str | None = None) -> dict:
        """Create a judgment bundle for all candidates attached to an anchor."""
        destination = Path(output_path) if output_path else stage_path("judgments", anchor_doc_id)
        existing_bundle = read_json(destination)
        if existing_bundle:
            mark_stage_completed(anchor_doc_id, "judgments", "reused cached judgment bundle")
            return {"judgment_bundle_path": str(destination), "judgment_count": len(existing_bundle.get("judgments", [])), "cached": True}
        mark_stage_started(anchor_doc_id, "judgments", "judging candidate relations")
        anchor = brief_map()[anchor_doc_id]
        bundle_payload = read_candidate_bundle(anchor_doc_id) or {}
        items = bundle_payload.get("candidates", [])
        from hnsw_logic.embedding.provider import JudgeSignals

        normalized_pairs = []
        for item in items:
            candidate = brief_map().get(item["candidate_doc_id"])
            if candidate is None:
                continue
            normalized_pairs.append((candidate, JudgeSignals(**item.get("signals", {}))))
        verdicts = orchestrator.relation_judge.run_many_with_signals(anchor, normalized_pairs) if normalized_pairs else {}
        bundle = JudgmentBundle(
            anchor_doc_id=anchor_doc_id,
            generated_at=utc_now(),
            judgments=[
                JudgmentBundleItem(candidate_doc_id=doc_id, verdict=to_jsonable(verdict), signals=next(item.get("signals", {}) for item in items if item["candidate_doc_id"] == doc_id))
                for doc_id, verdict in verdicts.items()
            ],
        )
        write_json(destination, bundle)
        mark_stage_completed(anchor_doc_id, "judgments", f"judgment bundle with {len(bundle.judgments)} verdicts")
        return {"judgment_bundle_path": str(destination), "judgment_count": len(bundle.judgments)}

    def execute_counterevidence_check(anchor_doc_id: str, output_path: str | None = None) -> dict:
        """Create a counterevidence bundle that checks tentative edges for risk and duplication."""
        destination = Path(output_path) if output_path else stage_path("checks", anchor_doc_id)
        existing_bundle = read_json(destination)
        if existing_bundle:
            mark_stage_completed(anchor_doc_id, "checks", "reused cached counterevidence bundle")
            return {
                "counterevidence_bundle_path": str(destination),
                "check_count": len(existing_bundle.get("checks", [])),
                "cached": True,
            }
        mark_stage_started(anchor_doc_id, "checks", "checking counterevidence and duplicate bridges")
        anchor = brief_map()[anchor_doc_id]
        candidate_payload = read_candidate_bundle(anchor_doc_id) or {}
        judgment_payload = read_judgment_bundle(anchor_doc_id) or {}
        from hnsw_logic.embedding.provider import JudgeResult, JudgeSignals

        prepared: list[tuple[Any, JudgeSignals, JudgeResult]] = []
        for item in judgment_payload.get("judgments", []):
            candidate = brief_map().get(item["candidate_doc_id"])
            if candidate is None:
                continue
            signals_obj = JudgeSignals(**item.get("signals", {}))
            verdict_obj = (
                orchestrator.relation_judge.provider._verdict_from_payload(item.get("verdict", {}))
                if hasattr(orchestrator.relation_judge.provider, "_verdict_from_payload")
                else JudgeResult(**item.get("verdict", {}))
            )
            prepared.append((candidate, signals_obj, verdict_obj))
        provider_payloads = provider.check_counterevidence_many(anchor, prepared)
        checks: list[CounterevidenceBundleItem] = []
        for item in judgment_payload.get("judgments", []):
            candidate = brief_map().get(item["candidate_doc_id"])
            if candidate is None:
                continue
            signals = item.get("signals", {})
            verdict = item.get("verdict", {})
            metrics = orchestrator._candidate_metrics(anchor, candidate)
            duplicate_penalty = orchestrator._near_duplicate_penalty(anchor, candidate, metrics)
            bridge_gain = orchestrator._bridge_information_gain(anchor, candidate)
            provider_payload = provider_payloads.get(candidate.doc_id, {})
            risk_flags = sorted(set(signals.get("risk_flags", [])) | set(verdict.get("contradiction_flags", [])) | set(provider_payload.get("risk_flags", [])))
            counterevidence: list[str] = [str(item) for item in provider_payload.get("counterevidence", [])]
            risk_penalty = float(provider_payload.get("risk_penalty", 0.0))
            if duplicate_penalty >= 0.28 and "near_duplicate_bridge" not in risk_flags:
                risk_flags.append("near_duplicate_bridge")
                counterevidence.append("duplicate penalty suggests low incremental bridge value")
                risk_penalty += min(duplicate_penalty, 0.45)
            if bridge_gain < 0.34 and "low_bridge_gain" not in risk_flags:
                risk_flags.append("low_bridge_gain")
                counterevidence.append("candidate adds little new retrieval surface")
                risk_penalty += 0.12
            keep = bool(provider_payload.get("keep", True)) and risk_penalty < 0.42
            checks.append(
                CounterevidenceBundleItem(
                    candidate_doc_id=item["candidate_doc_id"],
                    keep=keep,
                    risk_flags=sorted(set(risk_flags)),
                    counterevidence=counterevidence,
                    decision_reason=str(provider_payload.get("decision_reason", "keep after checker" if keep else "drop after checker due to duplicate or weak bridge value"))[:220],
                    risk_penalty=round(risk_penalty, 6),
                )
            )
        bundle = CounterevidenceBundle(anchor_doc_id=anchor_doc_id, generated_at=utc_now(), checks=checks)
        write_json(destination, bundle)
        mark_stage_completed(anchor_doc_id, "checks", f"counterevidence bundle with {len(checks)} checks")
        return {"counterevidence_bundle_path": str(destination), "check_count": len(checks)}

    def execute_edge_review(anchor_doc_id: str, output_path: str | None = None) -> dict:
        """Create a review bundle with reviewed utility scores and keep/drop decisions."""
        destination = Path(output_path) if output_path else stage_path("reviews", anchor_doc_id)
        existing_bundle = read_json(destination)
        if existing_bundle:
            mark_stage_completed(anchor_doc_id, "reviews", "reused cached review bundle")
            return {"review_bundle_path": str(destination), "review_count": len(existing_bundle.get("reviews", [])), "cached": True}
        mark_stage_started(anchor_doc_id, "reviews", "reviewing edge utility and risk")
        anchor = brief_map()[anchor_doc_id]
        candidate_payload = read_candidate_bundle(anchor_doc_id) or {}
        judgment_payload = read_judgment_bundle(anchor_doc_id) or {}
        check_payload = read_counterevidence_bundle(anchor_doc_id) or {}
        candidate_lookup = {item["candidate_doc_id"]: item for item in candidate_payload.get("candidates", [])}
        check_lookup = {item["candidate_doc_id"]: item for item in check_payload.get("checks", [])}
        review_pairs = []
        for item in judgment_payload.get("judgments", []):
            candidate = brief_map().get(item["candidate_doc_id"])
            if candidate is None:
                continue
            from hnsw_logic.embedding.provider import JudgeResult, JudgeSignals

            signals = JudgeSignals(**item.get("signals", {}))
            verdict = JudgeResult(**item.get("verdict", {}))
            review_pairs.append((candidate, signals, verdict))
        reviewed = orchestrator.edge_reviewer.run_many_with_signals(anchor, review_pairs) if review_pairs else {}
        review_rows: list[ReviewBundleItem] = []
        for candidate, signals, verdict in review_pairs:
            reviewed_verdict = reviewed.get(candidate.doc_id, verdict)
            check = check_lookup.get(candidate.doc_id, {})
            keep = bool(getattr(reviewed_verdict, "accepted", False)) and bool(check.get("keep", True))
            final_utility = max(float(getattr(reviewed_verdict, "utility_score", 0.0)) - float(check.get("risk_penalty", 0.0)), 0.0)
            risk_flags = sorted(set(getattr(reviewed_verdict, "contradiction_flags", []) or []) | set(check.get("risk_flags", [])))
            review_rows.append(
                ReviewBundleItem(
                    candidate_doc_id=candidate.doc_id,
                    keep=keep,
                    reviewed_utility_score=round(final_utility, 6),
                    reviewed_confidence=round(float(getattr(reviewed_verdict, "confidence", 0.0)), 6),
                    relation_type=str(getattr(reviewed_verdict, "relation_type", "comparison")),
                    decision_reason=(str(getattr(reviewed_verdict, "decision_reason", "")) or str(check.get("decision_reason", "")))[:220],
                    final_verdict=to_jsonable(reviewed_verdict),
                    risk_flags=risk_flags,
                )
            )
        review_rows.sort(key=lambda item: (-item.reviewed_utility_score, -item.reviewed_confidence, item.candidate_doc_id))
        bundle = ReviewBundle(anchor_doc_id=anchor_doc_id, generated_at=utc_now(), reviews=review_rows)
        write_json(destination, bundle)
        mark_stage_completed(anchor_doc_id, "reviews", f"review bundle with {len(review_rows)} items")
        return {"review_bundle_path": str(destination), "review_count": len(review_rows)}

    def execute_memory_summarization(anchor_doc_id: str, output_path: str | None = None) -> dict:
        """Summarize accepted and rejected outcomes into a memory learning bundle."""
        destination = Path(output_path) if output_path else stage_path("memory", anchor_doc_id)
        existing_bundle = read_json(destination)
        if existing_bundle:
            mark_stage_completed(anchor_doc_id, "memory", "reused cached memory bundle")
            return {
                "memory_bundle_path": str(destination),
                "learned": len(existing_bundle.get("learned_patterns", [])),
                "failed": len(existing_bundle.get("failure_patterns", [])),
                "cached": True,
            }
        mark_stage_started(anchor_doc_id, "memory", "summarizing memory learnings")
        review_payload = read_review_bundle(anchor_doc_id) or {}
        kept = [item for item in review_payload.get("reviews", []) if item.get("keep")]
        dropped = [item for item in review_payload.get("reviews", []) if not item.get("keep")]
        provider_payload = provider.summarize_memory_learnings(
            {
                "anchor_doc_id": anchor_doc_id,
                "accepted": [f"{item['relation_type']}:{item['candidate_doc_id']}" for item in kept[:8]],
                "rejected": [
                    f"{item['candidate_doc_id']}:{','.join(item.get('risk_flags', [])[:4]) or 'rejected'}"
                    for item in dropped[:12]
                ],
            }
        )
        learned_patterns = [str(item) for item in provider_payload.get("learned_patterns", [])][:8]
        failure_patterns = [str(item) for item in provider_payload.get("failure_patterns", [])][:12]
        bundle = MemoryLearningBundle(
            anchor_doc_id=anchor_doc_id,
            generated_at=utc_now(),
            learned_patterns=learned_patterns,
            failure_patterns=failure_patterns,
            reference_updates={
                ".deepagents/skills/graph-hygiene/references/hygiene-rules.md": failure_patterns[:6],
                **{
                    str(key): [str(item) for item in value][:12]
                    for key, value in dict(provider_payload.get("reference_updates", {})).items()
                },
            },
        )
        write_json(destination, bundle)
        mark_stage_completed(anchor_doc_id, "memory", f"memory bundle with {len(learned_patterns)} learned patterns")
        return {"memory_bundle_path": str(destination), "learned": len(learned_patterns), "failed": len(failure_patterns)}

    shared_tools: dict[str, Callable] = {
        "read_graph_stats": read_graph_stats,
        "read_semantic_memory": read_semantic_memory,
        "read_failure_patterns": read_failure_patterns,
        "read_anchor_dossier": read_anchor_dossier,
        "read_candidate_bundle": read_candidate_bundle,
        "read_judgment_bundle": read_judgment_bundle,
        "read_counterevidence_bundle": read_counterevidence_bundle,
        "read_review_bundle": read_review_bundle,
        "read_execution_manifest": read_execution_manifest,
        "audit_anchor_execution": audit_anchor_execution,
        "evaluate_anchor_utility": evaluate_anchor_utility,
    }

    project_tools = {
        "search_summaries": None,
        "lookup_entities": None,
        "get_hnsw_neighbors": None,
        "read_doc_brief": lambda doc_id: to_jsonable(brief_store.read(doc_id)),
        "read_doc_full": lambda doc_id: to_jsonable(doc_map().get(doc_id)),
        "load_anchor_memory": lambda doc_id: to_jsonable(anchor_memory_store.read(doc_id)),
    }

    from hnsw_logic.agents.tools.registry import build_agent_tools

    project_tool_impls = build_agent_tools(
        corpus_store,
        brief_store,
        graph_store,
        anchor_memory_store,
        semantic_memory_store,
        hnsw_searcher,
    )
    for key in list(project_tools):
        if key in project_tool_impls:
            project_tools[key] = project_tool_impls[key]
    project_tools = {key: value for key, value in project_tools.items() if value is not None}

    stage_tools = {
        "index_planner": {"execute_index_planning": execute_index_planning},
        "doc_profiler": {"execute_doc_profiling": execute_doc_profiling},
        "corpus_scout": {"execute_candidate_expansion": execute_candidate_expansion},
        "relation_judge": {"execute_relation_judging": execute_relation_judging},
        "counterevidence_checker": {"execute_counterevidence_check": execute_counterevidence_check},
        "edge_reviewer": {"execute_edge_review": execute_edge_review},
        "memory_curator": {"execute_memory_summarization": execute_memory_summarization},
    }

    scoped: dict[str, dict[str, Callable]] = {}
    for agent_name, config in config_subagents.items():
        allowed = dict(stage_tools.get(agent_name, {}))
        for scope_name in getattr(config, "tool_scopes", []):
            if scope_name in shared_tools:
                allowed[scope_name] = shared_tools[scope_name]
            elif scope_name in project_tools:
                allowed[scope_name] = project_tools[scope_name]
        scoped[agent_name] = allowed
    return scoped


def build_deepagent_supervisor_tools(
    *,
    workspace_root: Path,
    graph_memory_store,
    counterevidence_enabled: bool,
    task_iteration_cap: int,
) -> list[Callable]:
    def read_indexing_plan() -> dict:
        """Read the current offline indexing plan from the workspace."""
        return read_json(workspace_root / "indexing" / "plans" / "indexing_plan.json", {}) or {}

    def read_execution_manifest(doc_id: str) -> dict:
        """Read the execution manifest for an anchor and inspect stage progress."""
        return to_jsonable(load_execution_manifest(workspace_root, doc_id))

    def audit_anchor_execution(doc_id: str) -> dict:
        """Audit an anchor workflow and report missing stages, readiness, and fallback state."""
        return to_jsonable(
            audit_execution_state(
                workspace_root,
                doc_id,
                counterevidence_enabled=counterevidence_enabled,
                task_iteration_cap=task_iteration_cap,
            )
        )

    def evaluate_anchor_metrics(doc_id: str) -> dict:
        """Summarize graph coverage and review utility signals for an anchor."""
        review_payload = read_json(stage_artifact_path(workspace_root, "reviews", doc_id), {}) or {}
        reviews = list(review_payload.get("reviews", []))
        graph_stats = graph_memory_store.read()
        utilities = [float(item.get("reviewed_utility_score", 0.0) or 0.0) for item in reviews]
        return {
            "anchor_doc_id": doc_id,
            "review_count": len(reviews),
            "kept_count": sum(1 for item in reviews if item.get("keep")),
            "top_reviewed_utility": round(max(utilities), 6) if utilities else 0.0,
            "graph_stats": graph_stats,
        }

    return [
        read_indexing_plan,
        read_execution_manifest,
        audit_anchor_execution,
        evaluate_anchor_metrics,
    ]
