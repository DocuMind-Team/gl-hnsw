from __future__ import annotations

from hnsw_logic.embedding.provider_base import ProviderBase


class IndexPlannerAgent:
    def __init__(self, provider: ProviderBase):
        self.provider = provider

    def run(self, payload: dict) -> dict:
        return self.provider.plan_indexing_batch(payload)
