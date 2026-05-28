"""Synchronous GoodMem tracing tests."""

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


@pytest.fixture
def retriever():
    return GoodMemRetriever(
        GoodMemConfig(
            base_url="https://api.goodmem.ai",
            api_key="test-key",
            space_id="space-123",
            top_k=3,
            embedder="text-embedding-3-small",
        )
    )


class TestRetrieverSpanCreation:
    """`retrieve_chunks()` should create a single retriever span."""

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_creates_retriever_span(self, mock_post, retriever):
        with trace(name="goodmem-test"):
            retriever.retrieve_chunks("What is Python?")

        traces = trace_manager.get_all_traces()
        assert len(traces) == 1

        root_spans = traces[0].root_spans
        assert len(root_spans) == 1
        assert isinstance(root_spans[0], RetrieverSpan)

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_span_has_correct_name(self, mock_post, retriever):
        with trace(name="goodmem-test"):
            retriever.retrieve_chunks("test query")

        span = trace_manager.get_all_traces()[0].root_spans[0]
        assert span.name == "GoodMem Retriever"

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_span_captures_input(self, mock_post, retriever):
        with trace(name="goodmem-test"):
            retriever.retrieve_chunks("What is Python?")

        span = trace_manager.get_all_traces()[0].root_spans[0]
        assert span.input is not None
        assert "What is Python?" in str(span.input)

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_span_captures_output(self, mock_post, retriever):
        with trace(name="goodmem-test"):
            result = retriever.retrieve_chunks("test")

        span = trace_manager.get_all_traces()[0].root_spans[0]
        assert span.output is not None
        assert len(result) == 2
        assert isinstance(result[0], GoodMemChunk)

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_span_has_success_status(self, mock_post, retriever):
        with trace(name="goodmem-test"):
            retriever.retrieve_chunks("test")

        span = trace_manager.get_all_traces()[0].root_spans[0]
        assert span.status == TraceSpanStatus.SUCCESS

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_span_has_retriever_metadata(self, mock_post, retriever):
        with trace(name="goodmem-test"):
            retriever.retrieve_chunks("test")

        span = trace_manager.get_all_traces()[0].root_spans[0]
        assert isinstance(span, RetrieverSpan)
        assert span.embedder == "text-embedding-3-small"
        assert span.top_k == 3

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_retrieve_returns_text_list(self, mock_post, retriever):
        with trace(name="text-test"):
            result = retriever.retrieve("test")

        assert result == [
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
        ]


class TestTraceMetadata:
    """Trace-level metadata should reach the recorded trace."""

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_trace_tags(self, mock_post, retriever):
        with trace(
            name="goodmem-tagged",
            tags=["goodmem", "retrieval"],
            thread_id="thread-abc",
            user_id="user-xyz",
        ):
            retriever.retrieve("test")

        t = trace_manager.get_all_traces()[0]
        assert t.name == "goodmem-tagged"
        assert "goodmem" in t.tags
        assert "retrieval" in t.tags
        assert t.thread_id == "thread-abc"
        assert t.user_id == "user-xyz"


class TestSpanNesting:
    """Retriever spans should nest correctly under parent spans."""

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_nested_inside_agent_span(self, mock_post, retriever):
        @observe(type="agent", name="RAG Agent")
        def rag_agent(query):
            return retriever.retrieve(query)

        with trace(name="nested-test"):
            rag_agent("test query")

        t = trace_manager.get_all_traces()[0]
        assert len(t.root_spans) == 1
        agent_span = t.root_spans[0]
        assert agent_span.name == "RAG Agent"

        assert len(agent_span.children) == 1
        retriever_span = agent_span.children[0]
        assert isinstance(retriever_span, RetrieverSpan)
        assert retriever_span.name == "GoodMem Retriever"

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_multiple_retrieves_create_separate_traces(
        self, mock_post, retriever
    ):
        retriever.retrieve("query 1")
        retriever.retrieve("query 2")

        traces = trace_manager.get_all_traces()
        assert len(traces) == 2
        assert all(isinstance(t.root_spans[0], RetrieverSpan) for t in traces)


class TestRetrieveChunksSpan:
    """`retrieve_chunks()` should also create traced spans."""

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_retrieve_chunks_creates_span(self, mock_post, retriever):
        with trace(name="chunks-test"):
            result = retriever.retrieve_chunks("What is Python?")

        assert len(result) == 2
        assert isinstance(result[0], GoodMemChunk)
        assert result[0].score == -0.25

        traces = trace_manager.get_all_traces()
        assert len(traces) == 1
        span = traces[0].root_spans[0]
        assert isinstance(span, RetrieverSpan)
        assert span.name == "GoodMem Retriever"
        assert span.status == TraceSpanStatus.SUCCESS

    @patch("requests.Session.post", side_effect=_mock_post)
    def test_retrieve_chunks_has_metadata(self, mock_post, retriever):
        with trace(name="chunks-meta-test"):
            retriever.retrieve_chunks("test")

        span = trace_manager.get_all_traces()[0].root_spans[0]
        assert isinstance(span, RetrieverSpan)
        assert span.embedder == "text-embedding-3-small"
        assert span.top_k == 3


class TestErrorHandling:
    """Errors raised during retrieval should land on the retriever span."""

    def test_span_captures_error(self, retriever):
        with patch(
            "requests.Session.post",
            side_effect=Exception("Connection refused"),
        ):
            with trace(name="error-test"):
                with pytest.raises(Exception, match="Connection refused"):
                    retriever.retrieve("test")

        t = trace_manager.get_all_traces()[0]
        span = t.root_spans[0]
        assert span.status == TraceSpanStatus.ERRORED
        assert span.error is not None
