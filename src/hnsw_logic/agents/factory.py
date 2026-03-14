from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hnsw_logic.agents.subagents.corpus_scout import CorpusScoutAgent
from hnsw_logic.agents.subagents.doc_profiler import DocProfilerAgent
from hnsw_logic.agents.subagents.memory_curator import MemoryCuratorAgent
from hnsw_logic.agents.subagents.query_strategy import QueryStrategyAgent
from hnsw_logic.agents.subagents.relation_judge import RelationJudgeAgent
from hnsw_logic.agents.tools.registry import build_agent_tools
from hnsw_logic.config.schema import AgentsConfig, ProviderConfig, RetrievalConfig
from hnsw_logic.embedding.provider import OpenAICompatibleProvider, ProviderBase


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
    ):
        self.agents_config = agents_config
        self.provider_config = provider_config
        self.retrieval_config = retrieval_config
        self.provider = provider
        self.tools = tools
        self.skills_root = skills_root
        self.workspace_root = workspace_root
        self.memories_root = memories_root
        if hasattr(self.provider, "configure_live_reasoning"):
            self.provider.configure_live_reasoning(self.agents_config.live_reasoning)

    def create_orchestrator(self):
        from hnsw_logic.agents.orchestrator import LogicOrchestrator

        return LogicOrchestrator(
            doc_profiler=self.create_doc_profiler(),
            corpus_scout=self.create_corpus_scout(),
            relation_judge=self.create_relation_judge(),
            memory_curator=self.create_memory_curator(),
            deepagent=self.try_create_deep_agent(),
            retrieval_config=self.retrieval_config,
        )

    def create_doc_profiler(self):
        return DocProfilerAgent(self.provider)

    def create_corpus_scout(self):
        return CorpusScoutAgent(self.provider)

    def create_relation_judge(self):
        return RelationJudgeAgent(self.provider)

    def create_memory_curator(self):
        return MemoryCuratorAgent(self.provider)

    def create_query_strategy(self):
        return QueryStrategyAgent()

    def create_deepagent_specs(self) -> list[dict[str, Any]]:
        subagents: list[dict[str, Any]] = []
        for name, config in self.agents_config.subagents.items():
            if not config.enabled:
                continue
            subagents.append(
                {
                    "name": name,
                    "description": f"{name} subagent for gl-hnsw",
                    "system_prompt": f"You are the {name} specialist for gl-hnsw. Follow the loaded skills and return concise structured outputs.",
                    "tools": list(self.tools.values()),
                }
            )
        return subagents

    def try_create_deep_agent(self):
        if self.agents_config.runtime_mode != "deepagents":
            return None
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
        skills = [str(self.skills_root)]
        return create_deep_agent(
            model=model,
            skills=skills,
            tools=list(self.tools.values()),
            subagents=self.create_deepagent_specs(),
            backend=FilesystemBackend(root_dir=str(self.workspace_root), virtual_mode=True),
            system_prompt="You are the gl-hnsw orchestrator. Use skills and tools to produce structured outputs for document profiling, discovery, judging, and memory updates.",
        )
