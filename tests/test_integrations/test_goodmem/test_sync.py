"""Synchronous GoodMem tracing tests.

Trace structure is captured with the shared snapshot helper from
``tests/test_integrations/utils.py`` and compared against the committed
schemas under ``schemas/``. Regenerate them with::

    GENERATE_SCHEMAS=true pytest tests/test_integrations/test_goodmem/test_sync.py

Return-value and error-path behaviour are asserted directly, since those
do not depend on the captured trace.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from deepeval.tracing import observe, trace
from deepeval.tracing.types import RetrieverSpan, TraceSpanStatus

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
                            "chunkText": "Python is a programming language.",
                        },
                        "relevanceScore": -0.25,
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
                            "chunkText": "Python was created by Guido van Rossum.",
                        },
                        "relevanceScore": -0.40,
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
            space_id="space-123",
            top_k=3,
            embedder="text-embedding-3-small",
        )
    )


@observe(type="agent", name="RAG Agent")
def _rag_agent(retriever: GoodMemRetriever, query: str):
    return retriever.retrieve(query)


class TestRetrieverTrace:
    """Snapshot the trace structure a GoodMem retrieval produces."""

    @trace_test("retriever_span.json")
    def test_retriever_span(self):
        with patch("requests.Session.post", side_effect=_mock_post):
            _make_retriever().retrieve_chunks("What is Python?")

    @trace_test("agent_with_retriever.json")
    def test_agent_with_retriever(self):
        with patch("requests.Session.post", side_effect=_mock_post):
            _rag_agent(_make_retriever(), "What is Python?")


class TestRetrieverReturns:
    """The retrieve methods return the expected text and chunk objects."""

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_retrieve_returns_texts(self, mock_post):
        result = _make_retriever().retrieve("test query")
        assert result == [
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
        ]

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_retrieve_chunks_returns_structured(self, mock_post):
        chunks = _make_retriever().retrieve_chunks("test query")
        assert len(chunks) == 2
        assert isinstance(chunks[0], GoodMemChunk)
        assert chunks[0].content == "Python is a programming language."
        assert chunks[0].score == -0.25
        assert chunks[0].chunk_id == "c1"
        assert chunks[0].memory_id == "m1"


class TestRetrieverErrors:
    """A failed HTTP call surfaces as an errored retriever span."""

    def test_span_errored_on_http_failure(self):
        retriever = _make_retriever()
        with trace() as active_trace:
            with patch(
                "requests.Session.post",
                side_effect=Exception("Connection refused"),
            ):
                with pytest.raises(Exception, match="Connection refused"):
                    retriever.retrieve("test query")

            span = active_trace.root_spans[0]
            assert isinstance(span, RetrieverSpan)
            assert span.status == TraceSpanStatus.ERRORED
            assert span.error is not None
