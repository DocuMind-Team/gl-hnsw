from __future__ import annotations

import subprocess
from pathlib import Path
from hnsw_logic.agents.orchestrator import CandidateAssessment
from hnsw_logic.agents.tools.deepagents_runtime import (
    audit_execution_state,
    load_execution_manifest,
    normalize_indexing_workspace,
    normalize_stage_container_path,
    record_manifest_stage_event,
    resolve_workspace_output_path,
    stage_artifact_path,
)
from hnsw_logic.agents.tools.skill_signals import SkillSignalRuntime
from hnsw_logic.memory.self_update import ControlledSelfUpdateManager
from hnsw_logic.core.models import LogicEdge
from hnsw_logic.core.utils import write_json
from hnsw_logic.embedding.provider import JudgeResult
from hnsw_logic.agents.tools.registry import build_agent_tools


def _skill_frontmatter(content: str) -> dict[str, str]:
    assert content.startswith("---\n")
    _, frontmatter, _ = content.split("---", 2)
    fields: dict[str, str] = {}
    for line in frontmatter.strip().splitlines():
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def test_settings_enable_deepagents_runtime(test_root: Path):
    from hnsw_logic.config.settings import load_settings

    load_settings.cache_clear()
    settings = load_settings(test_root)
    assert settings.agents.skills_root == Path(".deepagents/skills")
    assert settings.agents.memory_files == [Path(".deepagents/AGENTS.md")]
    assert settings.agents.counterevidence_enabled is True


def test_skill_packages_have_frontmatter_and_resources(test_root: Path):
    skills_root = test_root / ".deepagents" / "skills"
    skill_dirs = [path for path in skills_root.iterdir() if path.is_dir()]
    assert skill_dirs
    for skill_dir in skill_dirs:
        skill_md = skill_dir / "SKILL.md"
        assert skill_md.exists(), skill_dir
        content = skill_md.read_text(encoding="utf-8")
        frontmatter = _skill_frontmatter(content)
        assert set(frontmatter) == {"name", "description"}, skill_dir
        assert all(frontmatter.values()), skill_dir
        assert (skill_dir / "references").exists(), skill_dir
        assert (skill_dir / "scripts").exists(), skill_dir


def test_legacy_skill_tree_is_removed(test_root: Path):
    assert not (test_root / "src" / "hnsw_logic" / "agents" / "skills").exists()


def test_runtime_skill_tree_has_no_python_cache():
    repo_root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", ".deepagents/skills"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not any("__pycache__" in path for path in tracked)
    assert not any(path.endswith(".pyc") for path in tracked)


def test_supervisor_runtime_has_no_local_runtime_escape_source(test_root: Path):
    repo_root = Path(__file__).resolve().parents[1]
    content = (repo_root / "src" / "hnsw_logic" / "services" / "offline_supervisor.py").read_text(encoding="utf-8")
    assert "_run_anchor_workflow_local" not in content
    assert "switching to local fallback" not in content


def test_key_runtime_skills_document_recommended_tools(test_root: Path):
    skills_root = test_root / ".deepagents" / "skills"
    expected = {
        "candidate-expansion",
        "relation-judging",
        "counterevidence-check",
        "edge-utility-review",
        "metric-evaluation",
        "delegation-policy",
    }
    for skill_name in expected:
        content = (skills_root / skill_name / "SKILL.md").read_text(encoding="utf-8")
        assert "Recommended tools" in content, skill_name


def test_runtime_skill_references_are_english(test_root: Path):
    skills_root = test_root / ".deepagents" / "skills"
    for reference in skills_root.glob("*/references/*.md"):
        content = reference.read_text(encoding="utf-8")
        assert not any("\u4e00" <= char <= "\u9fff" for char in content), reference


def test_skill_signal_runtime_returns_structured_signal_report(test_root: Path):
    runtime = SkillSignalRuntime(repo_root=test_root)
    anchor = {
        "doc_id": "a",
        "title": "Public transit should replace highway expansion",
        "summary": "The argument says transit investment is better than highway expansion.",
        "claims": ["Transit investment is preferable to highway expansion."],
        "keywords": ["transit", "mobility"],
        "relation_hints": ["comparison"],
        "metadata": {"topic_family": "culture-policy", "topic_cluster": "transit-mobility", "stance": "pro"},
    }
    candidate = {
        "doc_id": "b",
        "title": "Highway expansion is better than transit investment",
        "summary": "The argument says highway expansion is preferable for mobility.",
        "claims": ["Highway expansion is preferable to transit investment."],
        "keywords": ["highway", "mobility"],
        "relation_hints": ["comparison"],
        "metadata": {"topic_family": "culture-policy", "topic_cluster": "transit-mobility", "stance": "con"},
    }
    report = runtime.build_signal_report(anchor, candidate, local_signals={"topic_alignment": 1.0, "dense_score": 0.71})
    profile = runtime.compute_query_activation_profile(anchor, candidate, "comparison", local_signals=report, verdict={"utility_score": 0.78})
    anchor_priority = runtime.compute_anchor_priority(anchor, features={"claim_score": 1.0, "bridge_potential": 0.7})
    candidate_priority = runtime.compute_candidate_priority(
        base_score=0.62,
        metrics={"local_support": 0.71},
        fit_scores={"comparison": 0.8},
        signal_report=report,
    )
    relation_fit = runtime.score_relation_fit(anchor, candidate, {"dense_score": 0.71, "topic_alignment": 1.0}, report)
    utility = runtime.score_candidate_utility(
        relation_type="comparison",
        metrics={"dense_score": 0.71, "local_support": 0.68, "mention_score": 0.22},
        fit_scores=relation_fit["fit_scores"],
        signal_report=report,
    )
    edge_budget = runtime.compute_edge_budget_score(
        score=0.7,
        utility_score=0.8,
        activation_prior=0.75,
        novelty=0.6,
        specific_novelty=0.4,
    )

    assert report["topic_consistency"] >= 0.5
    assert "topic_report" in report and "duplicate_report" in report
    assert profile["activation_prior"] > 0.0
    assert "query_surface_terms" in profile
    assert anchor_priority["priority_score"] > 0.0
    assert candidate_priority["priority_score"] > 0.0
    assert relation_fit["best_relation"] in relation_fit["fit_scores"]
    assert utility["utility_score"] > 0.0
    assert edge_budget["selection_score"] > 0.0


def test_controlled_self_update_only_touches_allowlisted_targets(tmp_path: Path):
    repo_root = tmp_path / "repo"
    agents_path = repo_root / ".deepagents" / "AGENTS.md"
    ref_path = repo_root / ".deepagents" / "skills" / "graph-hygiene" / "references" / "hygiene-rules.md"
    blocked_path = repo_root / "src" / "blocked.py"
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    blocked_path.parent.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(
        "# AGENTS\n\n## Known Failure Patterns\n\n- none\n\n## Learned Patterns\n\n- none\n",
        encoding="utf-8",
    )
    blocked_path.write_text("print('blocked')\n", encoding="utf-8")
    manager = ControlledSelfUpdateManager(
        repo_root,
        [".deepagents/AGENTS.md", ".deepagents/skills/*/references/*.md"],
    )
    from hnsw_logic.agents.runtime_models import MemoryLearningBundle

    bundle = MemoryLearningBundle(
        anchor_doc_id="doc-1",
        generated_at="2026-03-16T00:00:00Z",
        learned_patterns=["keep:implementation_detail:doc-2"],
        failure_patterns=["drop:doc-3:weak_direction"],
        reference_updates={
            ".deepagents/skills/graph-hygiene/references/hygiene-rules.md": ["avoid duplicate bridge edges"],
            "src/blocked.py": ["should not write"],
        },
    )
    manager.apply(bundle, agents_path)
    agents_content = agents_path.read_text(encoding="utf-8")
    assert "drop:doc-3:weak_direction" in agents_content
    assert "keep:implementation_detail:doc-2" in agents_content
    assert "avoid duplicate bridge edges" in ref_path.read_text(encoding="utf-8")
    assert blocked_path.read_text(encoding="utf-8") == "print('blocked')\n"


def test_controlled_self_update_preserves_reference_content_and_appends_updates(tmp_path: Path):
    repo_root = tmp_path / "repo"
    ref_path = repo_root / ".deepagents" / "skills" / "graph-hygiene" / "references" / "hygiene-rules.md"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text("# Hygiene Rules\n\nKeep graph edges precise.\n", encoding="utf-8")
    manager = ControlledSelfUpdateManager(
        repo_root,
        [".deepagents/skills/*/references/*.md"],
    )

    manager.update_references(
        {
            ".deepagents/skills/graph-hygiene/references/hygiene-rules.md": [
                "avoid duplicate bridge edges",
                "preserve reviewed contrast bridges",
            ]
        }
    )

    content = ref_path.read_text(encoding="utf-8")
    assert "# Hygiene Rules" in content
    assert "Keep graph edges precise." in content
    assert "## Learned Updates" in content
    assert "- avoid duplicate bridge edges" in content
    assert "- preserve reviewed contrast bridges" in content


def test_runtime_tool_scopes_exclude_edge_commit(app_container):
    scoped = app_container.agent_factory.runtime_toolsets
    assert scoped["index_planner"]
    for agent_name, tools in scoped.items():
        assert "commit_logic_edge" not in tools
        assert "update_global_memory" not in tools
        if agent_name == "memory_curator":
            assert "execute_memory_summarization" in tools
        if agent_name == "edge_reviewer":
            assert "evaluate_anchor_utility" in tools
    supervisor_tool_names = {tool.__name__ for tool in app_container.agent_factory.supervisor_tools}
    assert {"read_indexing_plan", "read_execution_manifest", "audit_anchor_execution"} <= supervisor_tool_names


def test_registry_tools_serialize_slotted_models(app_container):
    tools = build_agent_tools(
        app_container.corpus_store,
        app_container.brief_store,
        app_container.graph_store,
        app_container.anchor_memory_store,
        app_container.semantic_memory_store,
        app_container.searcher,
    )
    app_container.pipeline.build_embeddings()
    app_container.pipeline.build_hnsw()
    briefs = app_container.discovery_service.ensure_briefs(app_container.corpus_store.read_processed())
    doc_id = briefs[0].doc_id

    assert "commit_logic_edge" not in tools
    assert "update_global_memory" not in tools
    assert isinstance(tools["read_doc_brief"](doc_id), dict)
    assert isinstance(tools["read_doc_full"](doc_id), dict)
    assert isinstance(tools["load_anchor_memory"](doc_id), dict)


def test_offline_supervisor_test_harness_writes_bundles(app_container):
    app_container.pipeline.build_embeddings()
    app_container.pipeline.build_hnsw()
    briefs = app_container.discovery_service.ensure_briefs(app_container.corpus_store.read_processed())
    anchor_doc_id = briefs[0].doc_id
    app_container.offline_supervisor.run_anchor_workflow_test_harness(anchor_doc_id)
    workspace = app_container.settings.root_dir / app_container.settings.app.paths.workspace_dir / "indexing"
    assert (workspace / "dossiers" / f"{anchor_doc_id}.json").exists()
    assert (workspace / "candidates" / f"{anchor_doc_id}.json").exists()
    assert (workspace / "judgments" / f"{anchor_doc_id}.json").exists()
    assert (workspace / "reviews" / f"{anchor_doc_id}.json").exists()
    assert (workspace / "memory" / f"{anchor_doc_id}.json").exists()


def test_runtime_tools_reuse_existing_stage_outputs(app_container):
    app_container.pipeline.build_embeddings()
    app_container.pipeline.build_hnsw()
    briefs = app_container.discovery_service.ensure_briefs(app_container.corpus_store.read_processed())
    anchor_doc_id = briefs[0].doc_id
    tools = app_container.agent_factory.runtime_toolsets

    first = tools["doc_profiler"]["execute_doc_profiling"](anchor_doc_id=anchor_doc_id)
    second = tools["doc_profiler"]["execute_doc_profiling"](anchor_doc_id=anchor_doc_id)

    assert "cached" not in first
    assert second["cached"] is True


def test_execution_audit_reports_missing_stages(tmp_path: Path):
    audit = audit_execution_state(tmp_path, "doc-1", counterevidence_enabled=True, task_iteration_cap=2)
    assert audit.next_stage == "dossiers"
    assert audit.workflow_complete is False

    write_json(stage_artifact_path(tmp_path, "dossiers", "doc-1"), {"anchor_doc_id": "doc-1"})
    record_manifest_stage_event(tmp_path, "doc-1", stage="dossiers", status="completed", note="seeded dossier")

    next_audit = audit_execution_state(tmp_path, "doc-1", counterevidence_enabled=True, task_iteration_cap=2)
    assert "dossiers" in next_audit.completed_stages
    assert next_audit.next_stage == "candidates"


def test_workspace_output_path_resolution_maps_data_prefixes(tmp_path: Path):
    workspace_root = tmp_path / "root" / "data" / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    default = workspace_root / "indexing" / "reviews" / "doc-1.json"

    assert resolve_workspace_output_path(workspace_root, None, default) == default
    assert resolve_workspace_output_path(
        workspace_root,
        "indexing/reviews/doc-1.json",
        default,
    ) == workspace_root / "indexing/reviews/doc-1.json"
    assert resolve_workspace_output_path(
        workspace_root,
        "/data/workspace/indexing/reviews/doc-1.json",
        default,
    ) == workspace_root / "indexing/reviews/doc-1.json"
    assert resolve_workspace_output_path(
        workspace_root,
        "/data/indexing/reviews/doc-1.json",
        default,
    ) == workspace_root.parent / "indexing/reviews/doc-1.json"
    assert resolve_workspace_output_path(
        workspace_root,
        "indexing/reviews",
        default,
    ) == workspace_root / "indexing/reviews/doc-1.json"


def test_normalize_stage_container_path_moves_json_payload_into_default_file(tmp_path: Path):
    workspace_root = tmp_path / "data" / "workspace"
    stage_file = workspace_root / "indexing" / "plans"
    stage_file.parent.mkdir(parents=True, exist_ok=True)
    stage_file.write_text('{"anchors":[{"doc_id":"doc-1"}]}', encoding="utf-8")

    normalized = normalize_stage_container_path(workspace_root, "plans")

    target = workspace_root / "indexing" / "plans" / "indexing_plan.json"
    assert str(target) in normalized
    assert target.exists()
    assert target.read_text(encoding="utf-8").startswith('{"anchors"')


def test_normalize_indexing_workspace_preserves_non_json_stage_notes(tmp_path: Path):
    workspace_root = tmp_path / "data" / "workspace"
    stage_file = workspace_root / "indexing" / "judgments"
    stage_file.parent.mkdir(parents=True, exist_ok=True)
    stage_file.write_text("# Judgments Directory\n", encoding="utf-8")

    normalized = normalize_indexing_workspace(workspace_root, "doc-1")

    note = workspace_root / "indexing" / "judgments" / "_stage_note.md"
    assert str(note) in normalized
    assert note.exists()
    assert note.read_text(encoding="utf-8").startswith("# Judgments")


def test_delegation_loop_raises_when_workflow_incomplete(app_container):
    app_container.pipeline.build_embeddings()
    app_container.pipeline.build_hnsw()
    briefs = app_container.discovery_service.ensure_briefs(app_container.corpus_store.read_processed())
    anchor_doc_id = briefs[0].doc_id

    class DummyDeepAgent:
        def __init__(self):
            self.calls = []

        def invoke(self, payload):
            self.calls.append(payload)

    app_container.offline_supervisor.deepagent = DummyDeepAgent()
    app_container.offline_supervisor.agents_config.task_iteration_cap = 1

    try:
        app_container.offline_supervisor._run_anchor_workflow_with_deepagents(anchor_doc_id)
    except RuntimeError as exc:
        assert "did not complete" in str(exc)
    else:
        raise AssertionError("expected incomplete delegation workflow to raise")

    manifest = load_execution_manifest(app_container.offline_supervisor.workspace_root, anchor_doc_id)
    assert manifest.halt_requested is True
    assert manifest.failed_stages


def test_discover_for_anchor_does_not_call_direct_discovery_runtime_path(app_container, monkeypatch):
    app_container.pipeline.build_embeddings()
    app_container.pipeline.build_hnsw()
    briefs = app_container.discovery_service.ensure_briefs(app_container.corpus_store.read_processed())
    anchor_doc_id = briefs[0].doc_id

    monkeypatch.setattr(
        app_container.offline_supervisor,
        "_run_anchor_workflow_with_deepagents",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        app_container.discovery_service,
        "discover_for_anchor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("direct discovery runtime path should not run")),
    )

    workspace = app_container.offline_supervisor.workspace_root / "indexing"
    write_json(workspace / "candidates" / f"{anchor_doc_id}.json", {"anchor_doc_id": anchor_doc_id, "candidates": []})
    write_json(workspace / "judgments" / f"{anchor_doc_id}.json", {"anchor_doc_id": anchor_doc_id, "judgments": []})

    accepted = app_container.offline_supervisor.discover_for_anchor(anchor_doc_id, briefs)
    assert accepted == []


def test_build_plan_requires_materialized_plan_under_full_delegation(app_container):
    class DummyDeepAgent:
        def invoke(self, _payload):
            return {}

    app_container.offline_supervisor.deepagent = DummyDeepAgent()

    try:
        app_container.offline_supervisor._build_plan()
    except RuntimeError as exc:
        assert "indexing/plans/indexing_plan.json" in str(exc)
    else:
        raise AssertionError("expected full delegation planning to require a materialized plan file")


def test_offline_supervisor_respects_explicit_review_rejection(app_container, monkeypatch):
    app_container.pipeline.build_embeddings()
    app_container.pipeline.build_hnsw()
    briefs = app_container.discovery_service.ensure_briefs(app_container.corpus_store.read_processed())
    anchor = briefs[0]
    candidate = briefs[1]
    accepted = CandidateAssessment(
        candidate_doc_id=candidate.doc_id,
        accepted=True,
        reject_reason="",
        score=0.91,
        local_support=0.81,
        evidence_quality=0.9,
        relation_type="same_concept",
        confidence=0.88,
        edge=LogicEdge(
            src_doc_id=anchor.doc_id,
            dst_doc_id=candidate.doc_id,
            relation_type="same_concept",
            confidence=0.88,
            evidence_spans=[anchor.summary, candidate.summary],
            discovery_path=["judge", "review", "gate"],
            edge_card_text=f"{anchor.title} -> {candidate.title}",
            created_at="2026-03-16T00:00:00Z",
            last_validated_at="2026-03-16T00:00:00Z",
            utility_score=0.92,
        ),
    )

    monkeypatch.setattr(
        type(app_container.offline_supervisor.orchestrator),
        "_assessment_for",
        lambda self, *_args, **_kwargs: accepted,
    )

    verdicts = {
        candidate.doc_id: JudgeResult(
            accepted=False,
            relation_type="comparison",
            confidence=0.72,
            evidence_spans=[],
            rationale="review says comparison",
            support_score=0.0,
            contradiction_flags=[],
            decision_reason="soft risk rejection",
            utility_score=0.0,
            uncertainty=0.6,
            canonical_relation="none",
            semantic_relation_label="none",
        )
    }
    reviews = dict(verdicts)
    bundle_lookup = {
        "reviews": {
            candidate.doc_id: {
                "keep": False,
                "reviewed_utility_score": 0.0,
                "reviewed_confidence": 0.72,
                "risk_flags": [
                    "excess_novelty",
                    "weak_family_bridge",
                    "topic-only overlap",
                    "low retrieval utility",
                    "weak direction",
                ],
            }
        },
        "checks": {
            candidate.doc_id: {
                "keep": False,
                "risk_penalty": 0.2,
                "risk_flags": [
                    "excess_novelty",
                    "weak_family_bridge",
                    "topic-only overlap",
                    "low retrieval utility",
                    "weak direction",
                ],
            }
        },
    }

    assessments = app_container.offline_supervisor._apply_review_consensus(
        anchor,
        [candidate],
        verdicts,
        reviews,
        bundle_lookup,
    )

    assert assessments
    assert assessments[0].accepted is False
    assert assessments[0].reject_reason == "review_rejected"


def test_offline_supervisor_applies_reviewed_activation_profile(app_container, monkeypatch):
    app_container.pipeline.build_embeddings()
    app_container.pipeline.build_hnsw()
    briefs = app_container.discovery_service.ensure_briefs(app_container.corpus_store.read_processed())
    anchor = briefs[0]
    candidate = briefs[1]
    accepted = CandidateAssessment(
        candidate_doc_id=candidate.doc_id,
        accepted=True,
        reject_reason="",
        score=0.88,
        local_support=0.6,
        evidence_quality=0.82,
        relation_type="comparison",
        confidence=0.84,
        edge=LogicEdge(
            src_doc_id=anchor.doc_id,
            dst_doc_id=candidate.doc_id,
            relation_type="same_concept",
            confidence=0.84,
            evidence_spans=[anchor.summary, candidate.summary],
            discovery_path=["judge", "review", "gate"],
            edge_card_text=f"{anchor.title} <> {candidate.title}",
            created_at="2026-03-16T00:00:00Z",
            last_validated_at="2026-03-16T00:00:00Z",
            utility_score=0.93,
        ),
    )

    monkeypatch.setattr(
        type(app_container.offline_supervisor.orchestrator),
        "_assessment_for",
        lambda self, *_args, **_kwargs: accepted,
    )

    verdicts = {
        candidate.doc_id: JudgeResult(
            accepted=True,
            relation_type="same_concept",
            confidence=0.84,
            evidence_spans=[],
            rationale="durable concept bridge",
            support_score=0.0,
            contradiction_flags=[],
            decision_reason="review accepted",
            utility_score=0.93,
            uncertainty=0.22,
            canonical_relation="same_concept",
            semantic_relation_label="same_concept",
        )
    }
    reviews = dict(verdicts)
    bundle_lookup = {
        "reviews": {
            candidate.doc_id: {
                "keep": True,
                "reviewed_utility_score": 0.74,
                "reviewed_confidence": 0.9,
                "activation_profile": {
                    "topic_signature": ["memory", "agent"],
                    "query_surface_terms": ["memory", "agent", "persistent"],
                    "edge_use_cases": ["concept-bridge"],
                    "drift_risk": 0.12,
                    "activation_prior": 0.78,
                    "negative_patterns": [],
                },
                "risk_flags": [],
            }
        },
        "checks": {
            candidate.doc_id: {
                "keep": True,
                "risk_penalty": 0.0,
                "risk_flags": [],
            }
        },
    }

    assessments = app_container.offline_supervisor._apply_review_consensus(
        anchor,
        [candidate],
        verdicts,
        reviews,
        bundle_lookup,
    )

    assert assessments
    assert assessments[0].accepted is True
    assert assessments[0].edge is not None
    assert assessments[0].edge.utility_score == 0.74
    assert assessments[0].edge.activation_profile["activation_prior"] == 0.78


def test_offline_supervisor_rejects_missing_review_artifact(app_container, monkeypatch):
    app_container.pipeline.build_embeddings()
    app_container.pipeline.build_hnsw()
    briefs = app_container.discovery_service.ensure_briefs(app_container.corpus_store.read_processed())
    anchor = briefs[0]
    candidate = briefs[1]
    accepted = CandidateAssessment(
        candidate_doc_id=candidate.doc_id,
        accepted=True,
        reject_reason="",
        score=0.88,
        local_support=0.6,
        evidence_quality=0.82,
        relation_type="comparison",
        confidence=0.84,
        edge=LogicEdge(
            src_doc_id=anchor.doc_id,
            dst_doc_id=candidate.doc_id,
            relation_type="same_concept",
            confidence=0.84,
            evidence_spans=[anchor.summary, candidate.summary],
            discovery_path=["judge", "review", "gate"],
            edge_card_text=f"{anchor.title} <> {candidate.title}",
            created_at="2026-03-16T00:00:00Z",
            last_validated_at="2026-03-16T00:00:00Z",
            utility_score=0.93,
        ),
    )

    monkeypatch.setattr(
        type(app_container.offline_supervisor.orchestrator),
        "_assessment_for",
        lambda self, *_args, **_kwargs: accepted,
    )

    verdicts = {
        candidate.doc_id: JudgeResult(
            accepted=True,
            relation_type="same_concept",
            confidence=0.84,
            evidence_spans=[],
            rationale="durable concept bridge",
            support_score=0.0,
            contradiction_flags=[],
            decision_reason="review accepted",
            utility_score=0.93,
            uncertainty=0.22,
            canonical_relation="same_concept",
            semantic_relation_label="same_concept",
        )
    }
    bundle_lookup = {
        "reviews": {},
        "checks": {
            candidate.doc_id: {
                "keep": True,
                "risk_penalty": 0.0,
                "risk_flags": [],
            }
        },
    }

    assessments = app_container.offline_supervisor._apply_review_consensus(
        anchor,
        [candidate],
        verdicts,
        {},
        bundle_lookup,
    )

    assert assessments
    assert assessments[0].accepted is False
    assert assessments[0].reject_reason == "missing_review_artifact"
