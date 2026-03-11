from __future__ import annotations

from hnsw_logic.core.models import DocBrief, LogicEdge
from hnsw_logic.embedding.provider import ProviderBase


class MemoryCuratorAgent:
    def __init__(self, provider: ProviderBase):
        self.provider = provider

    def run(self, anchor: DocBrief, accepted: list[LogicEdge], rejected: list[str]) -> dict:
        return self.provider.curate_memory(anchor, accepted, rejected)
