from __future__ import annotations

from hnsw_logic.core.models import DocBrief
from hnsw_logic.embedding.provider import JudgeResult, JudgeSignals, ProviderBase


class RelationJudgeAgent:
    def __init__(self, provider: ProviderBase):
        self.provider = provider

    def run(self, anchor: DocBrief, candidate: DocBrief) -> JudgeResult:
        return self.provider.judge_relation(anchor, candidate)

    def run_many(self, anchor: DocBrief, candidates: list[DocBrief]) -> dict[str, JudgeResult]:
        return self.provider.judge_relations(anchor, candidates)

    def run_with_signals(self, anchor: DocBrief, candidate: DocBrief, signals: JudgeSignals) -> JudgeResult:
        return self.provider.judge_relation_with_signals(anchor, candidate, signals)

    def run_many_with_signals(self, anchor: DocBrief, candidates: list[tuple[DocBrief, JudgeSignals]]) -> dict[str, JudgeResult]:
        return self.provider.judge_relations_with_signals(anchor, candidates)
