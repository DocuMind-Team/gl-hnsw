from __future__ import annotations

from pathlib import Path
from hnsw_logic.agents.orchestrator import CandidateAssessment
from hnsw_logic.memory.self_update import ControlledSelfUpdateManager
from hnsw_logic.core.models import LogicEdge
from hnsw_logic.embedding.provider import JudgeResult


def test_settings_enable_deepagents_runtime(test_root: Path):
    from hnsw_logic.config.settings import load_settings

    load_settings.cache_clear()
    settings = load_settings(test_root)
    assert settings.agents.runtime_mode == "deepagents"
    assert settings.agents.skills_root == Path(".deepagents/skills")
    assert settings.agents.memory_files == [Path(".deepagents/AGENTS.md")]
    assert settings.agents.planner_enabled is True
    assert settings.agents.counterevidence_enabled is True


def test_skill_packages_have_frontmatter_and_resources(test_root: Path):
    skills_root = test_root / ".deepagents" / "skills"
    skill_dirs = [path for path in skills_root.iterdir() if path.is_dir()]
    assert skill_dirs
    for skill_dir in skill_dirs:
        skill_md = skill_dir / "SKILL.md"
        assert skill_md.exists(), skill_dir
        content = skill_md.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "name:" in content and "description:" in content
        assert (skill_dir / "references").exists(), skill_dir
        assert (skill_dir / "scripts").exists(), skill_dir


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


def test_runtime_tool_scopes_exclude_edge_commit(app_container):
    scoped = app_container.agent_factory.runtime_toolsets
    assert scoped["index_planner"]
    for agent_name, tools in scoped.items():
        assert "commit_logic_edge" not in tools
        if agent_name == "memory_curator":
            assert "execute_memory_summarization" in tools


def test_offline_supervisor_local_workflow_writes_bundles(app_container):
    app_container.pipeline.build_embeddings()
    app_container.pipeline.build_hnsw()
    briefs = app_container.discovery_service.ensure_briefs(app_container.corpus_store.read_processed())
    anchor_doc_id = briefs[0].doc_id
    app_container.offline_supervisor._run_anchor_workflow_local(anchor_doc_id)
    workspace = app_container.settings.root_dir / app_container.settings.app.paths.workspace_dir / "indexing"
    assert (workspace / "dossiers" / f"{anchor_doc_id}.json").exists()
    assert (workspace / "candidates" / f"{anchor_doc_id}.json").exists()
    assert (workspace / "judgments" / f"{anchor_doc_id}.json").exists()
    assert (workspace / "reviews" / f"{anchor_doc_id}.json").exists()
    assert (workspace / "memory" / f"{anchor_doc_id}.json").exists()


def test_offline_supervisor_normalizes_soft_risk_flags_for_rescue(app_container, monkeypatch):
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
    assert assessments[0].accepted is True
