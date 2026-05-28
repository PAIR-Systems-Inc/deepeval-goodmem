"""DeepEval-traced retriever for GoodMem.

`GoodMemRetriever` wraps `GoodMemClient.retrieve_memories` with the
`@observe(type="retriever")` decorator so every call becomes a
retriever span on the active trace. Use it when you want
`LLMTestCase.retrieval_context` to be sourced from GoodMem during an
evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from deepeval.tracing import observe, update_retriever_span

from deepeval.integrations.goodmem.client import GoodMemClient
from deepeval.integrations.goodmem.types import GoodMemChunk


@dataclass
class GoodMemConfig:
    """Connection and retrieval settings for `GoodMemRetriever`.

    Supply either `space_id` (single space) or `space_ids` (one or more
    spaces). `top_k`, `reranker`, `relevance_threshold`,
    `metadata_filter`, and `embedder` are forwarded to every retrieval
    call.
    """

    base_url: str
    api_key: str
    space_ids: List[str] = field(default_factory=list)
    top_k: int = 5
    reranker: Optional[str] = None
    relevance_threshold: Optional[float] = None
    metadata_filter: Optional[str] = None
    embedder: Optional[str] = None
    verify_ssl: bool = True
    wait_for_indexing: bool = False
    max_wait_seconds: float = 10.0
    poll_interval: float = 2.0

    space_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if self.space_id and not self.space_ids:
            self.space_ids = [self.space_id]
        if not self.space_ids:
            raise ValueError("Provide space_id or space_ids")


class GoodMemRetriever:
    """Retriever that pulls chunks from GoodMem into a DeepEval trace.

    Example:

        from deepeval.integrations.goodmem import (
            GoodMemRetriever, GoodMemConfig,
        )

        retriever = GoodMemRetriever(GoodMemConfig(
            base_url="https://localhost:8080",
            api_key="gm_...",
            space_id="my-space",
            verify_ssl=False,
        ))

        texts = retriever.retrieve("What is machine learning?")
        chunks = retriever.retrieve_chunks("What is machine learning?")
    """

    def __init__(
        self,
        config: GoodMemConfig,
        client: Optional[GoodMemClient] = None,
    ) -> None:
        self.config = config
        self._client = client or GoodMemClient(
            base_url=config.base_url,
            api_key=config.api_key,
            verify_ssl=config.verify_ssl,
        )

    def retrieve(self, query: str) -> List[str]:
        """Return chunk text in a list, ready for `retrieval_context`."""
        chunks = self.retrieve_chunks(query)
        return [c.content for c in chunks if c.content]

    @observe(type="retriever", name="GoodMem Retriever")
    def retrieve_chunks(self, query: str) -> List[GoodMemChunk]:
        """Return ranked chunks with scores, IDs, and space IDs."""
        update_retriever_span(
            embedder=self.config.embedder,
            top_k=self.config.top_k,
        )
        return self._client.retrieve_memories(
            query=query,
            space_ids=self.config.space_ids,
            max_results=self.config.top_k,
            reranker_id=self.config.reranker,
            relevance_threshold=self.config.relevance_threshold,
            metadata_filter=self.config.metadata_filter,
            wait_for_indexing=self.config.wait_for_indexing,
            max_wait_seconds=self.config.max_wait_seconds,
            poll_interval=self.config.poll_interval,
        )

    def retrieve_as_context(self, query: str) -> List[str]:
        """Alias for `retrieve` suited to `LLMTestCase.retrieval_context`."""
        return self.retrieve(query)
