from __future__ import annotations

from hnsw_logic.core.models import DocBrief
from hnsw_logic.embedding.provider import ProviderBase
from hnsw_logic.embedding.provider_types import JudgeResult, JudgeSignals


class CounterevidenceCheckerAgent:
    def __init__(self, provider: ProviderBase):
        self.provider = provider

    def run_with_signals(self, anchor: DocBrief, candidate: DocBrief, signals: JudgeSignals, verdict: JudgeResult) -> dict:
        return self.provider.check_counterevidence(anchor, candidate, signals, verdict)
