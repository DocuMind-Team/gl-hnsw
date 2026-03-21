"""Embedding provider implementations and shared transport helpers."""

from hnsw_logic.embedding.providers.base import ProviderBase
from hnsw_logic.embedding.providers.client import OpenAIProviderTransportMixin
from hnsw_logic.embedding.providers.live import OpenAICompatibleProvider, build_provider
from hnsw_logic.embedding.providers.stub import StubProvider
from hnsw_logic.embedding.providers.types import CandidateProposal, JudgeResult, JudgeSignals

__all__ = [
    "CandidateProposal",
    "JudgeResult",
    "JudgeSignals",
    "OpenAICompatibleProvider",
    "OpenAIProviderTransportMixin",
    "ProviderBase",
    "StubProvider",
    "build_provider",
]
