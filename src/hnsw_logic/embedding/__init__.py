"""Embedding package."""

from hnsw_logic.embedding.factory import build_provider
from hnsw_logic.embedding.providers.base import ProviderBase
from hnsw_logic.embedding.providers.live import OpenAICompatibleProvider
from hnsw_logic.embedding.providers.stub import StubProvider
from hnsw_logic.embedding.providers.types import CandidateProposal, JudgeResult, JudgeSignals

__all__ = [
    "CandidateProposal",
    "JudgeResult",
    "JudgeSignals",
    "OpenAICompatibleProvider",
    "ProviderBase",
    "StubProvider",
    "build_provider",
]
