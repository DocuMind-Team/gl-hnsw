from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from hnsw_logic.agents.subagents.corpus_scout import CorpusScoutAgent
from hnsw_logic.agents.subagents.counterevidence_checker import CounterevidenceCheckerAgent
from hnsw_logic.agents.subagents.doc_profiler import DocProfilerAgent
from hnsw_logic.agents.subagents.edge_reviewer import EdgeReviewerAgent
from hnsw_logic.agents.subagents.index_planner import IndexPlannerAgent
from hnsw_logic.agents.subagents.memory_curator import MemoryCuratorAgent
from hnsw_logic.agents.subagents.relation_judge import RelationJudgeAgent
from hnsw_logic.agents.tools.deepagents_runtime import (
    build_deepagent_supervisor_tools,
    build_deepagent_toolsets,
)
from hnsw_logic.config.schema import AgentsConfig, ProviderConfig, RetrievalConfig
from hnsw_logic.core.utils import ensure_dir
from hnsw_logic.embedding.provider import OpenAICompatibleProvider
from hnsw_logic.embedding.provider_base import ProviderBase


class AgentFactory:
    def __init__(
        self,
        agents_config: AgentsConfig,
        provider_config: ProviderConfig,
        retrieval_config: RetrievalConfig,
        provider: ProviderBase,
        tools: dict[str, Any],
        skills_root: Path,
        workspace_root: Path,
        memories_root: Path,
        repo_root: Path,
        corpus_store,
        brief_store,
        graph_store,
        anchor_memory_store,
        semantic_memory_store,
        graph_memory_store,
        searcher,
    ):
        self.agents_config = agents_config
        self.provider_config = provider_config
        self.retrieval_config = retrieval_config
        self.provider = provider
        self.tools = tools
        self.skills_root = skills_root if skills_root.is_absolute() else (repo_root / skills_root)
        self.workspace_root = workspace_root
        self.memories_root = memories_root
        self.repo_root = repo_root
        self.corpus_store = corpus_store
        self.brief_store = brief_store
        self.graph_store = graph_store
        self.anchor_memory_store = anchor_memory_store
        self.semantic_memory_store = semantic_memory_store
        self.graph_memory_store = graph_memory_store
        self.searcher = searcher
        self.runtime_toolsets: dict[str, dict[str, Any]] = {}
        self.runtime_skill_views: dict[str, Path] = {}
        self.supervisor_tools: list[Any] = []
        if hasattr(self.provider, "configure_live_reasoning"):
            self.provider.configure_live_reasoning(self.agents_config.live_reasoning)

    def _backend_path(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.repo_root.resolve())
        return "/" + relative.as_posix()

    def _runtime_views_root(self) -> Path:
        path = self.repo_root / ".deepagents" / "runtime_views"
        ensure_dir(path)
        return path

    def _ensure_memory_files(self) -> list[Path]:
        files: list[Path] = []
        for item in self.agents_config.memory_files:
            path = item if item.is_absolute() else (self.repo_root / item)
            ensure_dir(path.parent)
            if not path.exists():
                path.write_text("# AGENTS\n\n", encoding="utf-8")
            files.append(path)
        return files

    def _build_skill_view(self, agent_name: str, skill_names: list[str]) -> Path:
        view_root = self._runtime_views_root() / agent_name
        if view_root.exists():
            shutil.rmtree(view_root)
        ensure_dir(view_root)
        for skill_name in skill_names:
            source = self.skills_root / skill_name
            if not source.exists():
                continue
            shutil.copytree(source, view_root / skill_name)
        return view_root

    def _prepare_skill_views(self) -> dict[str, Path]:
        views: dict[str, Path] = {}
        for name, config in self.agents_config.subagents.items():
            if not config.enabled:
                continue
            skill_names = [skill for skill in config.skills if (self.skills_root / skill).exists()]
            if not skill_names:
                continue
            views[name] = self._build_skill_view(name, skill_names)
        self.runtime_skill_views = views
        return views

    def _build_runtime_toolsets(self, orchestrator) -> dict[str, dict[str, Any]]:
        self.runtime_toolsets = build_deepagent_toolsets(
            provider=self.provider,
            corpus_store=self.corpus_store,
            brief_store=self.brief_store,
            graph_store=self.graph_store,
            anchor_memory_store=self.anchor_memory_store,
            semantic_memory_store=self.semantic_memory_store,
            graph_memory_store=self.graph_memory_store,
            hnsw_searcher=self.searcher,
            orchestrator=orchestrator,
            workspace_root=self.workspace_root,
            config_subagents=self.agents_config.subagents,
            counterevidence_enabled=self.agents_config.counterevidence_enabled,
            task_iteration_cap=self.agents_config.task_iteration_cap,
        )
        self.supervisor_tools = build_deepagent_supervisor_tools(
            workspace_root=self.workspace_root,
            graph_memory_store=self.graph_memory_store,
            counterevidence_enabled=self.agents_config.counterevidence_enabled,
            task_iteration_cap=self.agents_config.task_iteration_cap,
        )
        return self.runtime_toolsets

    def create_orchestrator(self):
        from hnsw_logic.agents.orchestrator import LogicOrchestrator

        orchestrator = LogicOrchestrator(
            doc_profiler=self.create_doc_profiler(),
            corpus_scout=self.create_corpus_scout(),
            relation_judge=self.create_relation_judge(),
            edge_reviewer=self.create_edge_reviewer(),
            memory_curator=self.create_memory_curator(),
            deepagent=None,
            retrieval_config=self.retrieval_config,
        )
        self._prepare_skill_views()
        self._build_runtime_toolsets(orchestrator)
        orchestrator.deepagent = self.try_create_deep_agent()
        return orchestrator

    def create_index_planner(self):
        return IndexPlannerAgent(self.provider)

    def create_doc_profiler(self):
        return DocProfilerAgent(self.provider)

    def create_corpus_scout(self):
        return CorpusScoutAgent(self.provider)

    def create_relation_judge(self):
        return RelationJudgeAgent(self.provider)

    def create_counterevidence_checker(self):
        return CounterevidenceCheckerAgent(self.provider)

    def create_edge_reviewer(self):
        return EdgeReviewerAgent(self.provider)

    def create_memory_curator(self):
        return MemoryCuratorAgent(self.provider)

    def create_deepagent_specs(self) -> list[dict[str, Any]]:
        subagents: list[dict[str, Any]] = []
        for name, config in self.agents_config.subagents.items():
            if not config.enabled:
                continue
            tools = list(self.runtime_toolsets.get(name, {}).values())
            skill_view = self.runtime_skill_views.get(name)
            spec: dict[str, Any] = {
                "name": name,
                "description": config.description or f"{name} specialist for offline indexing",
                "system_prompt": config.system_prompt or f"You are the {name} specialist for gl-hnsw offline indexing.",
                "tools": tools,
            }
            if skill_view is not None:
                spec["skills"] = [self._backend_path(skill_view)]
            subagents.append(spec)
        return subagents

    def try_create_deep_agent(self):
        if not isinstance(self.provider, OpenAICompatibleProvider):
            return None
        try:
            from deepagents import create_deep_agent
            from deepagents.backends import FilesystemBackend
            from langchain_openai import ChatOpenAI
        except Exception:
            return None

        api_key = os.getenv(self.provider_config.api_key_env)
        if not api_key:
            return None
        model = ChatOpenAI(
            base_url=self.provider_config.base_url,
            api_key=api_key,
            model=self.provider_config.chat_model,
            temperature=0,
        )
        memory_files = [self._backend_path(path) for path in self._ensure_memory_files()]
        root_skill_paths = [self._backend_path(self.skills_root)] if self.skills_root.exists() else []
        return create_deep_agent(
            model=model,
            skills=root_skill_paths,
            memory=memory_files,
            tools=self.supervisor_tools,
            subagents=self.create_deepagent_specs(),
            backend=FilesystemBackend(root_dir=str(self.repo_root), virtual_mode=True),
            system_prompt=(
                "You are the gl-hnsw offline indexing supervisor. "
                "Use the task tool to delegate indexing work to subagents, rely on workspace files as the handoff boundary, "
                "and never edit Python code, YAML configs, benchmark gold data, or core SKILL.md files. "
                "Only update allowed memory/reference files when a memory_curator task explicitly produces learned patterns."
            ),
            name="gl_hnsw_offline_supervisor",
        )
