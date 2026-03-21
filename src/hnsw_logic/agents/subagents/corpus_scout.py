from __future__ import annotations

from hnsw_logic.domain.models import DocBrief
from hnsw_logic.embedding.providers.base import ProviderBase
from hnsw_logic.embedding.providers.types import CandidateProposal


class CorpusScoutAgent:
    def __init__(self, provider: ProviderBase):
        self.provider = provider

    def run(self, anchor: DocBrief, corpus: list[DocBrief]) -> list[CandidateProposal]:
        return self.provider.propose_candidates(anchor, corpus)
