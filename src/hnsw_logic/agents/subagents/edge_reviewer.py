from __future__ import annotations

from hnsw_logic.core.models import DocBrief
from hnsw_logic.embedding.provider import ProviderBase
from hnsw_logic.embedding.provider_types import JudgeResult, JudgeSignals


class EdgeReviewerAgent:
    def __init__(self, provider: ProviderBase):
        self.provider = provider

    def run_with_signals(self, anchor: DocBrief, candidate: DocBrief, signals: JudgeSignals, verdict: JudgeResult) -> JudgeResult:
        return self.provider.review_relation_with_signals(anchor, candidate, signals, verdict)

    def run_many_with_signals(self, anchor: DocBrief, candidates: list[tuple[DocBrief, JudgeSignals, JudgeResult]]) -> dict[str, JudgeResult]:
        return self.provider.review_relations_with_signals(anchor, candidates)
