from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from hnsw_logic.agents.runtime.models import (
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
from hnsw_logic.agents.runtime.skill_runtime import SkillSignalRuntime
from hnsw_logic.domain.models import DocBrief, DocRecord
from hnsw_logic.domain.serialization import read_json, to_jsonable, utc_now, write_json

STAGE_SEQUENCE = ("dossiers", "candidates", "judgments", "checks", "reviews", "memory")
COMMIT_REQUIRED_STAGES = ("dossiers", "candidates", "judgments", "checks", "reviews")


def required_indexing_stages(counterevidence_enabled: bool = True) -> list[str]:
    return [stage for stage in STAGE_SEQUENCE if counterevidence_enabled or stage != "checks"]


def manifest_path(workspace_root: Path, anchor_doc_id: str) -> Path:
    return workspace_root / "indexing" / "manifests" / f"{anchor_doc_id}.json"


def stage_artifact_path(workspace_root: Path, stage: str, anchor_doc_id: str, suffix: str = ".json") -> Path:
    return workspace_root / "indexing" / stage / f"{anchor_doc_id}{suffix}"


def _stage_default_filename(stage: str, anchor_doc_id: str | None = None) -> str | None:
    if stage == "plans":
        return "indexing_plan.json"
    if anchor_doc_id:
        return f"{anchor_doc_id}.json"
    return None


def normalize_stage_container_path(workspace_root: Path, stage: str, anchor_doc_id: str | None = None) -> list[str]:
    container = workspace_root / "indexing" / stage
    normalized: list[str] = []
    if not container.exists() or container.is_dir():
        container.mkdir(parents=True, exist_ok=True)
        return normalized
    raw_bytes = container.read_bytes()
    target_name = _stage_default_filename(stage, anchor_doc_id)
    container.unlink()
    container.mkdir(parents=True, exist_ok=True)
    if target_name:
        # Preserve valid JSON stage payloads in the expected default file; keep non-JSON notes separately.
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except Exception:
            parsed = None
        if isinstance(parsed, (dict, list)):
            target = container / target_name
            target.write_bytes(raw_bytes)
            normalized.append(str(target))
        else:
            note = container / "_stage_note.md"
            note.write_bytes(raw_bytes)
            normalized.append(str(note))
    else:
        note = container / "_stage_note.md"
        note.write_bytes(raw_bytes)
        normalized.append(str(note))
    return normalized


def normalize_indexing_workspace(workspace_root: Path, anchor_doc_id: str | None = None) -> list[str]:
    normalized: list[str] = []
    normalized.extend(normalize_stage_container_path(workspace_root, "plans"))
    for stage in STAGE_SEQUENCE:
        normalized.extend(normalize_stage_container_path(workspace_root, stage, anchor_doc_id))
    normalize_stage_container_path(workspace_root, "manifests", anchor_doc_id)
    return normalized


def resolve_workspace_output_path(workspace_root: Path, output_path: str | None, default_path: Path) -> Path:
    default_stage = default_path.parent.name
    default_anchor = None if default_stage == "plans" else default_path.stem
    normalize_stage_container_path(workspace_root, default_stage, default_anchor)
    if not output_path:
        return default_path
    path = Path(output_path)
    if not path.is_absolute():
        resolved = workspace_root / output_path
    else:
        raw = path.as_posix()
        if raw.startswith("/data/workspace/"):
            resolved = workspace_root / raw.removeprefix("/data/workspace/")
        elif raw.startswith("/data/"):
            resolved = workspace_root.parent / raw.removeprefix("/data/")
        elif raw.startswith("/workspace/"):
            resolved = workspace_root.parent.parent.parent / raw.removeprefix("/workspace/")
        else:
            resolved = path
    if resolved == default_path.parent or (not resolved.suffix and resolved.name == default_path.parent.name):
        return resolved / default_path.name
    return resolved


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
    force_halt: bool = False,
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
    if force_halt:
        manifest.halt_requested = True
    save_execution_manifest(workspace_root, manifest)
    return manifest


def audit_execution_state(
    workspace_root: Path,
    anchor_doc_id: str,
    *,
    counterevidence_enabled: bool = True,
    task_iteration_cap: int = 2,
) -> ExecutionAudit:
    normalize_indexing_workspace(workspace_root, anchor_doc_id)
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
    retry_exhausted = bool(manifest.halt_requested)
    if not retry_exhausted and missing:
        next_stage = missing[0]
        retry_exhausted = int(manifest.retry_counts.get(next_stage, 0)) >= max(task_iteration_cap, 1)
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
        retry_exhausted=retry_exhausted,
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
    processed_docs_cache: list[DocRecord] | None = None
    signal_runtime = SkillSignalRuntime()

    def processed_docs() -> list[DocRecord]:
        nonlocal processed_docs_cache
        if processed_docs_cache is None:
            processed_docs_cache = cast(list[DocRecord], corpus_store.read_processed())
        return processed_docs_cache

    def brief_map() -> dict[str, DocBrief]:
        return {brief.doc_id: brief for brief in brief_store.all()}

    def doc_map() -> dict[str, DocRecord]:
        return {doc.doc_id: doc for doc in processed_docs()}

    def stage_dir(name: str) -> Path:
        normalize_stage_container_path(workspace_root, name)
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

    def compute_topic_consistency(anchor_doc_id: str, candidate_doc_id: str) -> dict:
        """Compute a topic-consistency signal report for an anchor/candidate pair via skill scripts."""
        anchor = brief_map().get(anchor_doc_id)
        candidate = brief_map().get(candidate_doc_id)
        if anchor is None or candidate is None:
            return {}
        return signal_runtime.compute_topic_consistency(anchor, candidate)

    def compute_duplicate_risk(anchor_doc_id: str, candidate_doc_id: str) -> dict:
        """Compute duplicate-risk signals for an anchor/candidate pair via skill scripts."""
        anchor = brief_map().get(anchor_doc_id)
        candidate = brief_map().get(candidate_doc_id)
        if anchor is None or candidate is None:
            return {}
        return signal_runtime.compute_duplicate_risk(anchor, candidate)

    def compute_bridge_gain(anchor_doc_id: str, candidate_doc_id: str) -> dict:
        """Compute bridge-gain signals for an anchor/candidate pair via skill scripts."""
        anchor = brief_map().get(anchor_doc_id)
        candidate = brief_map().get(candidate_doc_id)
        if anchor is None or candidate is None:
            return {}
        return signal_runtime.compute_bridge_gain(anchor, candidate)

    def compute_contrast_evidence(anchor_doc_id: str, candidate_doc_id: str, verdict: dict | None = None) -> dict:
        """Compute contrast-evidence signals for an anchor/candidate pair via skill scripts."""
        anchor = brief_map().get(anchor_doc_id)
        candidate = brief_map().get(candidate_doc_id)
        if anchor is None or candidate is None:
            return {}
        return signal_runtime.compute_contrast_evidence(anchor, candidate, verdict=verdict)

    def compute_query_activation_profile(anchor_doc_id: str, candidate_doc_id: str, relation_type: str, verdict: dict | None = None) -> dict:
        """Build an edge activation profile via skill scripts."""
        anchor = brief_map().get(anchor_doc_id)
        candidate = brief_map().get(candidate_doc_id)
        if anchor is None or candidate is None:
            return {}
        return signal_runtime.compute_query_activation_profile(anchor, candidate, relation_type, verdict=verdict)

    def compute_relation_fit(
        anchor_doc_id: str,
        candidate_doc_id: str,
        metrics: dict | None = None,
        signal_report: dict | None = None,
    ) -> dict:
        """Score canonical relation fits for an anchor/candidate pair via skill scripts."""
        anchor = brief_map().get(anchor_doc_id)
        candidate = brief_map().get(candidate_doc_id)
        if anchor is None or candidate is None:
            return {}
        return signal_runtime.score_relation_fit(anchor, candidate, metrics or {}, signal_report or {})

    def compute_candidate_utility(
        anchor_doc_id: str,
        candidate_doc_id: str,
        relation_type: str,
        metrics: dict | None = None,
        fit_scores: dict | None = None,
        signal_report: dict | None = None,
    ) -> dict:
        """Score reviewed candidate utility for retrieval via skill scripts."""
        anchor = brief_map().get(anchor_doc_id)
        candidate = brief_map().get(candidate_doc_id)
        if anchor is None or candidate is None:
            return {}
        return signal_runtime.score_candidate_utility(
            relation_type=relation_type,
            metrics=metrics or {},
            fit_scores=fit_scores or {},
            signal_report=signal_report or {},
        )

    def execute_index_planning(output_path: str = "indexing/plans/indexing_plan.json") -> dict:
        """Generate the offline indexing plan and persist it to the workspace."""
        destination = resolve_workspace_output_path(
            workspace_root,
            output_path,
            stage_artifact_path(workspace_root, "plans", "indexing_plan", suffix=".json"),
        )
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
        destination = resolve_workspace_output_path(workspace_root, output_path, stage_path("dossiers", anchor_doc_id))
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
        destination = resolve_workspace_output_path(workspace_root, output_path, stage_path("candidates", anchor_doc_id))
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
            signal_report = signal_runtime.build_signal_report(anchor, candidate, local_signals=to_jsonable(signals))
            payload_signals = {
                **to_jsonable(signals),
                "signal_report": signal_report,
                "topic_consistency": signal_report.get("topic_consistency", getattr(signals, "topic_consistency", 0.0)),
                "duplicate_risk": signal_report.get("duplicate_risk", getattr(signals, "duplicate_risk", getattr(signals, "duplicate_penalty", 0.0))),
                "bridge_gain": signal_report.get("bridge_information_gain", getattr(signals, "bridge_gain", 0.0)),
                "contrastive_bridge_score": max(
                    float(signal_report.get("contrast_evidence", 0.0) or 0.0),
                    float(getattr(signals, "contrastive_bridge_score", 0.0) or 0.0),
                ),
                "query_surface_match": signal_report.get("query_surface_match", 0.0),
                "uncertainty_hint": signal_report.get("uncertainty_hint", 0.0),
                "drift_risk": signal_report.get("drift_risk", 0.0),
            }
            items.append(
                CandidateBundleItem(
                    candidate_doc_id=candidate.doc_id,
                    proposal_reason=proposal.reason,
                    query=proposal.query,
                    score_hint=proposal.score_hint,
                    signals=payload_signals,
                    candidate_brief=to_jsonable(candidate),
                )
            )
        bundle = CandidateBundle(anchor_doc_id=anchor_doc_id, generated_at=utc_now(), candidates=items)
        write_json(destination, bundle)
        mark_stage_completed(anchor_doc_id, "candidates", f"candidate bundle with {len(items)} items")
        return {"candidate_bundle_path": str(destination), "candidate_count": len(items)}

    def execute_relation_judging(anchor_doc_id: str, output_path: str | None = None) -> dict:
        """Create a judgment bundle for all candidates attached to an anchor."""
        destination = resolve_workspace_output_path(workspace_root, output_path, stage_path("judgments", anchor_doc_id))
        existing_bundle = read_json(destination)
        if existing_bundle:
            mark_stage_completed(anchor_doc_id, "judgments", "reused cached judgment bundle")
            return {"judgment_bundle_path": str(destination), "judgment_count": len(existing_bundle.get("judgments", [])), "cached": True}
        mark_stage_started(anchor_doc_id, "judgments", "judging candidate relations")
        anchor = brief_map()[anchor_doc_id]
        bundle_payload = read_candidate_bundle(anchor_doc_id) or {}
        items = bundle_payload.get("candidates", [])
        from hnsw_logic.embedding.providers.types import JudgeSignals

        allowed_signal_keys = set(JudgeSignals.__dataclass_fields__.keys())

        normalized_pairs = []
        for item in items:
            candidate = brief_map().get(item["candidate_doc_id"])
            if candidate is None:
                continue
            signal_payload = {key: value for key, value in item.get("signals", {}).items() if key in allowed_signal_keys}
            normalized_pairs.append((candidate, JudgeSignals(**signal_payload)))
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
        destination = resolve_workspace_output_path(workspace_root, output_path, stage_path("checks", anchor_doc_id))
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
        judgment_payload = read_judgment_bundle(anchor_doc_id) or {}
        from hnsw_logic.embedding.providers.types import JudgeResult, JudgeSignals
        allowed_signal_keys = set(JudgeSignals.__dataclass_fields__.keys())

        prepared: list[tuple[Any, JudgeSignals, JudgeResult]] = []
        for item in judgment_payload.get("judgments", []):
            candidate = brief_map().get(item["candidate_doc_id"])
            if candidate is None:
                continue
            signal_payload = {key: value for key, value in item.get("signals", {}).items() if key in allowed_signal_keys}
            signals_obj = JudgeSignals(**signal_payload)
            verdict_obj = JudgeResult(**item.get("verdict", {}))
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
            signal_report = signal_runtime.build_signal_report(anchor, candidate, local_signals=signals, verdict=verdict)
            duplicate_penalty = float(signal_report.get("duplicate_risk", orchestrator._near_duplicate_penalty(anchor, candidate, metrics)) or 0.0)
            bridge_gain = float(signal_report.get("bridge_information_gain", orchestrator._bridge_information_gain(anchor, candidate)) or 0.0)
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
        destination = resolve_workspace_output_path(workspace_root, output_path, stage_path("reviews", anchor_doc_id))
        existing_bundle = read_json(destination)
        if existing_bundle:
            mark_stage_completed(anchor_doc_id, "reviews", "reused cached review bundle")
            return {"review_bundle_path": str(destination), "review_count": len(existing_bundle.get("reviews", [])), "cached": True}
        mark_stage_started(anchor_doc_id, "reviews", "reviewing edge utility and risk")
        anchor = brief_map()[anchor_doc_id]
        judgment_payload = read_judgment_bundle(anchor_doc_id) or {}
        check_payload = read_counterevidence_bundle(anchor_doc_id) or {}
        check_lookup = {item["candidate_doc_id"]: item for item in check_payload.get("checks", [])}
        review_pairs: list[tuple[DocBrief, JudgeSignals, JudgeResult, dict[str, Any]]] = []
        for item in judgment_payload.get("judgments", []):
            candidate = brief_map().get(item["candidate_doc_id"])
            if candidate is None:
                continue
            from hnsw_logic.embedding.providers.types import JudgeResult, JudgeSignals
            allowed_signal_keys = set(JudgeSignals.__dataclass_fields__.keys())
            signal_payload = {key: value for key, value in item.get("signals", {}).items() if key in allowed_signal_keys}
            signals = JudgeSignals(**signal_payload)
            verdict = JudgeResult(**item.get("verdict", {}))
            review_pairs.append((candidate, signals, verdict, signal_payload))
        reviewed_inputs = [(candidate, signals, verdict) for candidate, signals, verdict, _ in review_pairs]
        reviewed = orchestrator.edge_reviewer.run_many_with_signals(anchor, reviewed_inputs) if review_pairs else {}
        review_rows: list[ReviewBundleItem] = []
        for candidate, _signals, verdict, signal_payload in review_pairs:
            reviewed_verdict = reviewed.get(candidate.doc_id, verdict)
            check = check_lookup.get(candidate.doc_id, {})
            risk_flags = sorted(set(getattr(reviewed_verdict, "contradiction_flags", []) or []) | set(check.get("risk_flags", [])))
            normalized_risk_flags = {
                str(flag).strip().lower().replace("-", "_").replace(" ", "_")
                for flag in risk_flags
                if str(flag)
            }
            contradiction_like = {
                flag
                for flag in normalized_risk_flags
                if flag.startswith("contradict")
                or flag.startswith("counterargument")
                or flag.startswith("oppos")
                or flag.startswith("contrasting")
                or "contrast" in flag
                or flag.startswith("alternative_position")
            }
            contrast_report = signal_runtime.compute_contrast_evidence(
                anchor,
                candidate,
                local_signals=signal_payload,
                verdict=to_jsonable(reviewed_verdict),
            )
            contrastive_comparison_bridge = bool(
                getattr(reviewed_verdict, "relation_type", "") == "comparison"
                and float(contrast_report.get("contrast_evidence", 0.0) or 0.0) >= 0.56
                and max(
                    float(signal_payload.get("topic_family_match", 0.0) or 0.0),
                    float(signal_payload.get("topic_cluster_match", 0.0) or 0.0),
                    float(contrast_report.get("topic_consistency", 0.0) or 0.0),
                )
                >= 0.32
            )
            hard_blockers = {"same_stance", "topic_drift", "weak_topic_match", "low_retrieval_utility", "weak_direction"}
            effective_risk_flags = set(normalized_risk_flags)
            risk_penalty = float(check.get("risk_penalty", 0.0))
            if contrastive_comparison_bridge and "same_stance" not in effective_risk_flags:
                effective_risk_flags -= {"near_duplicate", "near_duplicate_bridge"}
                effective_risk_flags -= contradiction_like
                if not (effective_risk_flags & hard_blockers):
                    risk_penalty = min(risk_penalty, 0.22)
            keep = bool(getattr(reviewed_verdict, "accepted", False)) and (
                bool(check.get("keep", True)) or (
                    contrastive_comparison_bridge and not (effective_risk_flags & hard_blockers)
                )
            )
            activation_profile = signal_runtime.compute_query_activation_profile(
                anchor,
                candidate,
                str(getattr(reviewed_verdict, "relation_type", "comparison")),
                local_signals={
                    **signal_payload,
                    "bridge_information_gain": float(signal_payload.get("bridge_information_gain", 0.0) or 0.0),
                    "utility_score": float(getattr(reviewed_verdict, "utility_score", 0.0) or 0.0),
                },
                verdict=to_jsonable(reviewed_verdict),
            )
            final_utility = max(float(getattr(reviewed_verdict, "utility_score", 0.0)) - risk_penalty, 0.0)
            review_rows.append(
                ReviewBundleItem(
                    candidate_doc_id=candidate.doc_id,
                    keep=keep,
                    reviewed_utility_score=round(final_utility, 6),
                    reviewed_confidence=round(float(getattr(reviewed_verdict, "confidence", 0.0)), 6),
                    relation_type=str(getattr(reviewed_verdict, "relation_type", "comparison")),
                    decision_reason=(str(getattr(reviewed_verdict, "decision_reason", "")) or str(check.get("decision_reason", "")))[:220],
                    final_verdict=to_jsonable(reviewed_verdict),
                    risk_flags=sorted(effective_risk_flags),
                    activation_profile=activation_profile,
                )
            )
        review_rows.sort(key=lambda item: (-item.reviewed_utility_score, -item.reviewed_confidence, item.candidate_doc_id))
        bundle = ReviewBundle(anchor_doc_id=anchor_doc_id, generated_at=utc_now(), reviews=review_rows)
        write_json(destination, bundle)
        mark_stage_completed(anchor_doc_id, "reviews", f"review bundle with {len(review_rows)} items")
        return {"review_bundle_path": str(destination), "review_count": len(review_rows)}

    def execute_memory_summarization(anchor_doc_id: str, output_path: str | None = None) -> dict:
        """Summarize accepted and rejected outcomes into a memory learning bundle."""
        destination = resolve_workspace_output_path(workspace_root, output_path, stage_path("memory", anchor_doc_id))
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
        "compute_topic_consistency": compute_topic_consistency,
        "compute_duplicate_risk": compute_duplicate_risk,
        "compute_bridge_gain": compute_bridge_gain,
        "compute_contrast_evidence": compute_contrast_evidence,
        "compute_query_activation_profile": compute_query_activation_profile,
        "compute_relation_fit": compute_relation_fit,
        "compute_candidate_utility": compute_candidate_utility,
    }

    project_tools = {
        "search_summaries": None,
        "lookup_entities": None,
        "get_hnsw_neighbors": None,
        "read_doc_brief": lambda doc_id: to_jsonable(brief_store.read(doc_id)),
        "read_doc_full": lambda doc_id: to_jsonable(doc_map().get(doc_id)),
        "load_anchor_memory": lambda doc_id: to_jsonable(anchor_memory_store.read(doc_id)),
    }

    from hnsw_logic.agents.runtime.toolsets import build_agent_tools

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
        normalize_indexing_workspace(workspace_root)
        return read_json(workspace_root / "indexing" / "plans" / "indexing_plan.json", {}) or {}

    def read_execution_manifest(doc_id: str) -> dict:
        """Read the execution manifest for an anchor and inspect stage progress."""
        return to_jsonable(load_execution_manifest(workspace_root, doc_id))

    def audit_anchor_execution(doc_id: str) -> dict:
        """Audit an anchor workflow and report missing stages, readiness, and escalation state."""
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
