from __future__ import annotations

from hnsw_logic.core.models import DocBrief, DocRecord
from hnsw_logic.embedding.provider import ProviderBase


class DocProfilerAgent:
    def __init__(self, provider: ProviderBase):
        self.provider = provider

    def run(self, doc: DocRecord) -> DocBrief:
        return self.provider.profile_doc(doc)

    def run_many(self, docs: list[DocRecord]) -> list[DocBrief]:
        return self.provider.profile_docs(docs)
