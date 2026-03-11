from __future__ import annotations

from hnsw_logic.core.models import DocBrief
from hnsw_logic.embedding.provider import JudgeResult, ProviderBase


class RelationJudgeAgent:
    def __init__(self, provider: ProviderBase):
        self.provider = provider

    def run(self, anchor: DocBrief, candidate: DocBrief) -> JudgeResult:
        return self.provider.judge_relation(anchor, candidate)

    def run_many(self, anchor: DocBrief, candidates: list[DocBrief]) -> dict[str, JudgeResult]:
        return self.provider.judge_relations(anchor, candidates)
