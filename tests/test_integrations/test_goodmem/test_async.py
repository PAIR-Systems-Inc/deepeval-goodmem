"""Asynchronous GoodMem tracing tests."""

import json
from unittest.mock import patch, MagicMock

import pytest

from deepeval.tracing import trace, observe
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import RetrieverSpan, TraceSpanStatus

from deepeval.integrations.goodmem import (
    GoodMemChunk,
    GoodMemConfig,
    GoodMemRetriever,
)

MOCK_NDJSON = "\n".join(
    [
        json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "c1",
                            "memoryId": "m1",
                            "chunkText": "Async chunk one.",
                        },
                        "relevanceScore": -0.20,
                    }
                }
            }
        ),
        json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "c2",
                            "memoryId": "m2",
                            "chunkText": "Async chunk two.",
                        },
                        "relevanceScore": -0.35,
                    }
                }
            }
        ),
    ]
)


def _mock_post(*args, **kwargs):
    resp = MagicMock()
    resp.text = MOCK_NDJSON
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def retriever():
    return GoodMemRetriever(
        GoodMemConfig(
            base_url="https://api.goodmem.ai",
            api_key="test-key",
            space_id="space-async",
            top_k=5,
            embedder="text-embedding-ada-002",
        )
    )


class TestAsyncRetrieverSpan:
    """Retriever spans should be recorded when called from async code."""

    @pytest.mark.asyncio
    @patch("requests.Session.post", side_effect=_mock_post)
    async def test_async_context_creates_span(self, mock_post, retriever):
        @observe(type="agent", name="Async RAG Agent")
        async def async_rag(query):
            return retriever.retrieve(query)

        with trace(name="async-goodmem-test"):
            result = await async_rag("async test query")

        assert result == ["Async chunk one.", "Async chunk two."]

        t = trace_manager.get_all_traces()[0]
        agent_span = t.root_spans[0]
        assert agent_span.name == "Async RAG Agent"

        retriever_span = agent_span.children[0]
        assert isinstance(retriever_span, RetrieverSpan)
        assert retriever_span.name == "GoodMem Retriever"
        assert retriever_span.embedder == "text-embedding-ada-002"
        assert retriever_span.top_k == 5
        assert retriever_span.status == TraceSpanStatus.SUCCESS

    @pytest.mark.asyncio
    @patch("requests.Session.post", side_effect=_mock_post)
    async def test_sequential_async_retrieves(self, mock_post, retriever):
        @observe(type="agent", name="Multi-Retrieve Agent")
        async def multi_retrieve(queries):
            results = []
            for q in queries:
                results.append(retriever.retrieve(q))
            return results

        with trace(name="sequential-async-test"):
            results = await multi_retrieve(["query 1", "query 2"])

        assert len(results) == 2
        t = trace_manager.get_all_traces()[0]
        agent_span = t.root_spans[0]
        retriever_spans = [
            s for s in agent_span.children if isinstance(s, RetrieverSpan)
        ]
        assert len(retriever_spans) == 2


class TestAsyncRetrieveChunks:
    """`retrieve_chunks()` should work the same under async code."""

    @pytest.mark.asyncio
    @patch("requests.Session.post", side_effect=_mock_post)
    async def test_async_retrieve_chunks(self, mock_post, retriever):
        @observe(type="agent", name="Async Chunks Agent")
        async def async_chunks(query):
            return retriever.retrieve_chunks(query)

        with trace(name="async-chunks-test"):
            result = await async_chunks("test query")

        assert len(result) == 2
        assert isinstance(result[0], GoodMemChunk)
        assert result[0].content == "Async chunk one."
        assert result[0].score == -0.20

        t = trace_manager.get_all_traces()[0]
        agent_span = t.root_spans[0]
        retriever_span = agent_span.children[0]
        assert isinstance(retriever_span, RetrieverSpan)
        assert retriever_span.status == TraceSpanStatus.SUCCESS
