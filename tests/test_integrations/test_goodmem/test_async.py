"""Asynchronous GoodMem tracing tests.

Trace structure is captured with the shared snapshot helper from
``tests/test_integrations/utils.py`` and compared against the committed
schemas under ``schemas/``. Regenerate them with::

    GENERATE_SCHEMAS=true pytest tests/test_integrations/test_goodmem/test_async.py
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from deepeval.tracing import observe

from deepeval.integrations.goodmem import (
    GoodMemChunk,
    GoodMemConfig,
    GoodMemRetriever,
)
from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)

_SCHEMAS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "schemas"
)


def trace_test(schema_name: str):
    schema_path = os.path.join(_SCHEMAS_DIR, schema_name)
    if is_generate_mode():
        return generate_trace_json(schema_path)
    return assert_trace_json(schema_path)


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


def _make_retriever() -> GoodMemRetriever:
    return GoodMemRetriever(
        GoodMemConfig(
            base_url="https://api.goodmem.ai",
            api_key="test-key",
            space_id="space-async",
            top_k=5,
            embedder="text-embedding-ada-002",
        )
    )


@observe(type="agent", name="Async RAG Agent")
async def _async_rag_agent(retriever: GoodMemRetriever, query: str):
    return retriever.retrieve(query)


class TestAsyncRetrieverTrace:
    """The retriever span is recorded when called from async code."""

    @pytest.mark.asyncio
    @trace_test("async_agent_with_retriever.json")
    async def test_async_agent_with_retriever(self):
        with patch("requests.Session.post", side_effect=_mock_post):
            await _async_rag_agent(_make_retriever(), "async test query")


class TestAsyncRetrieverReturns:
    """`retrieve_chunks` returns chunk objects under async code."""

    @pytest.mark.asyncio
    @patch("requests.Session.post", side_effect=_mock_post)
    async def test_async_retrieve_chunks(self, mock_post):
        @observe(type="agent", name="Async Chunks Agent")
        async def async_chunks(query: str):
            return _make_retriever().retrieve_chunks(query)

        result = await async_chunks("test query")
        assert len(result) == 2
        assert isinstance(result[0], GoodMemChunk)
        assert result[0].content == "Async chunk one."
        assert result[0].score == -0.20
