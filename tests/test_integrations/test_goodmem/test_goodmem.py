"""Unit tests for the GoodMem client and retriever."""

import json
from unittest.mock import patch, MagicMock

import pytest

from deepeval.integrations.goodmem import (
    GoodMemChunk,
    GoodMemClient,
    GoodMemConfig,
    GoodMemRetriever,
)
from deepeval.integrations.goodmem.utils import parse_ndjson_response

SAMPLE_NDJSON_RESPONSE = "\n".join(
    [
        json.dumps(
            {
                "resultSetBoundary": {
                    "resultSetId": "abc-123",
                    "kind": "BEGIN",
                    "stageName": "retrieve",
                    "expectedItems": 2,
                }
            }
        ),
        json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "chunk-1",
                            "memoryId": "mem-1",
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
                            "chunkId": "chunk-2",
                            "memoryId": "mem-2",
                            "chunkText": "Python was created by Guido van Rossum.",
                        },
                        "relevanceScore": -0.40,
                    }
                }
            }
        ),
        json.dumps(
            {
                "resultSetBoundary": {
                    "resultSetId": "abc-123",
                    "kind": "END",
                    "stageName": "",
                }
            }
        ),
    ]
)


@pytest.fixture
def config():
    return GoodMemConfig(
        base_url="https://api.goodmem.ai",
        api_key="test-key",
        space_id="space-123",
        top_k=3,
        embedder="text-embedding-3-small",
    )


@pytest.fixture
def retriever(config):
    return GoodMemRetriever(config)


@pytest.fixture
def client():
    return GoodMemClient(
        base_url="https://api.goodmem.ai",
        api_key="test-key",
    )


def _ndjson_response(body: str = SAMPLE_NDJSON_RESPONSE) -> MagicMock:
    resp = MagicMock()
    resp.text = body
    resp.raise_for_status = MagicMock()
    return resp


def _json_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    resp.headers = {"Content-Type": "application/json"}
    return resp


class TestParseNdjsonResponse:
    def test_parses_chunks(self):
        result = parse_ndjson_response(SAMPLE_NDJSON_RESPONSE)
        assert len(result) == 2
        assert isinstance(result[0], GoodMemChunk)
        assert result[0].content == "Python is a programming language."
        assert result[1].content == "Python was created by Guido van Rossum."

    def test_extracts_chunk_ids(self):
        result = parse_ndjson_response(SAMPLE_NDJSON_RESPONSE)
        assert result[0].chunk_id == "chunk-1"
        assert result[1].chunk_id == "chunk-2"

    def test_extracts_memory_ids(self):
        result = parse_ndjson_response(SAMPLE_NDJSON_RESPONSE)
        assert result[0].memory_id == "mem-1"

    def test_extracts_relevance_scores(self):
        result = parse_ndjson_response(SAMPLE_NDJSON_RESPONSE)
        assert result[0].score == -0.25

    def test_empty_response(self):
        result = parse_ndjson_response("")
        assert result == []

    def test_ignores_malformed_lines(self):
        text = "not json\n" + json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "c1",
                            "memoryId": "m1",
                            "chunkText": "valid",
                        },
                        "relevanceScore": -0.1,
                    }
                }
            }
        )
        result = parse_ndjson_response(text)
        assert len(result) == 1
        assert result[0].content == "valid"

    def test_handles_sse_data_prefix(self):
        text = "data: " + json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "c1",
                            "memoryId": "m1",
                            "chunkText": "sse",
                        },
                        "relevanceScore": -0.1,
                    }
                }
            }
        )
        result = parse_ndjson_response(text)
        assert len(result) == 1
        assert result[0].content == "sse"


class TestGoodMemClientRetrieve:
    @patch("requests.Session.post")
    def test_sends_correct_request(self, mock_post, client):
        mock_post.return_value = _ndjson_response()

        result = client.retrieve_memories(
            query="What is Python?",
            space_ids=["space-123"],
            max_results=3,
        )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        body = call_kwargs[1]["json"]
        assert body["message"] == "What is Python?"
        assert body["spaceKeys"] == [{"spaceId": "space-123"}]
        assert body["requestedSize"] == 3
        assert call_kwargs[1]["headers"]["x-api-key"] == "test-key"
        assert len(result) == 2
        assert isinstance(result[0], GoodMemChunk)

    @patch("requests.Session.post")
    def test_includes_metadata_filter(self, mock_post, client):
        mock_post.return_value = _ndjson_response()

        client.retrieve_memories(
            query="test",
            space_ids=["space-123", "space-456"],
            metadata_filter="source = 'wiki'",
        )

        body = mock_post.call_args[1]["json"]
        assert body["spaceKeys"][0]["filter"] == "source = 'wiki'"
        assert body["spaceKeys"][1]["filter"] == "source = 'wiki'"

    @patch("requests.Session.post")
    def test_omits_filter_when_none(self, mock_post, client):
        mock_post.return_value = _ndjson_response()

        client.retrieve_memories(
            query="test",
            space_ids=["space-123"],
            metadata_filter=None,
        )

        body = mock_post.call_args[1]["json"]
        assert "filter" not in body["spaceKeys"][0]

    @patch("requests.Session.post")
    def test_treats_empty_filter_as_none(self, mock_post, client):
        mock_post.return_value = _ndjson_response()

        client.retrieve_memories(
            query="test",
            space_ids=["space-123"],
            metadata_filter="",
        )

        body = mock_post.call_args[1]["json"]
        assert "filter" not in body["spaceKeys"][0]

    @patch("requests.Session.post")
    def test_multi_space_request(self, mock_post, client):
        mock_post.return_value = _ndjson_response()

        client.retrieve_memories(
            query="test",
            space_ids=["space-a", "space-b"],
        )

        body = mock_post.call_args[1]["json"]
        assert len(body["spaceKeys"]) == 2
        assert body["spaceKeys"][0] == {"spaceId": "space-a"}
        assert body["spaceKeys"][1] == {"spaceId": "space-b"}

    @patch("requests.Session.post")
    def test_include_memory_definition_default_true(self, mock_post, client):
        mock_post.return_value = _ndjson_response()
        client.retrieve_memories(query="q", space_ids=["s"])
        body = mock_post.call_args[1]["json"]
        assert body["fetchMemory"] is True

    @patch("requests.Session.post")
    def test_include_memory_definition_false(self, mock_post, client):
        mock_post.return_value = _ndjson_response()
        client.retrieve_memories(
            query="q",
            space_ids=["s"],
            include_memory_definition=False,
        )
        body = mock_post.call_args[1]["json"]
        assert body["fetchMemory"] is False

    @patch("requests.Session.post")
    def test_post_processor_with_reranker(self, mock_post, client):
        mock_post.return_value = _ndjson_response()
        client.retrieve_memories(
            query="q",
            space_ids=["s"],
            reranker_id="rerank-001",
            relevance_threshold=0.3,
        )
        body = mock_post.call_args[1]["json"]
        assert body["postProcessor"]["config"]["reranker_id"] == "rerank-001"
        assert body["postProcessor"]["config"]["relevance_threshold"] == 0.3

    @patch("requests.Session.post")
    def test_post_processor_with_llm(self, mock_post, client):
        mock_post.return_value = _ndjson_response()
        client.retrieve_memories(
            query="q",
            space_ids=["s"],
            llm_id="gpt-4o-mini",
            llm_temperature=0.2,
            chronological_resort=True,
        )
        config_payload = mock_post.call_args[1]["json"]["postProcessor"][
            "config"
        ]
        assert config_payload["llm_id"] == "gpt-4o-mini"
        assert config_payload["llm_temp"] == 0.2
        assert config_payload["chronological_resort"] is True

    def test_requires_at_least_one_space(self, client):
        with pytest.raises(ValueError, match="space_id"):
            client.retrieve_memories(query="q", space_ids=[])


class TestGoodMemClientPolling:
    @patch("deepeval.integrations.goodmem.client.time.sleep")
    @patch("requests.Session.post")
    def test_polls_until_results_arrive(self, mock_post, mock_sleep, client):
        empty_resp = _ndjson_response("")
        full_resp = _ndjson_response()
        mock_post.side_effect = [empty_resp, empty_resp, full_resp]

        result = client.retrieve_memories(
            query="q",
            space_ids=["s"],
            wait_for_indexing=True,
            max_wait_seconds=10,
            poll_interval=1,
        )

        assert len(result) == 2
        assert mock_post.call_count == 3

    @patch("deepeval.integrations.goodmem.client.time.time")
    @patch("deepeval.integrations.goodmem.client.time.sleep")
    @patch("requests.Session.post")
    def test_polling_respects_max_wait(
        self, mock_post, mock_sleep, mock_time, client
    ):
        mock_post.return_value = _ndjson_response("")
        mock_time.side_effect = [0.0, 0.0, 11.0]

        result = client.retrieve_memories(
            query="q",
            space_ids=["s"],
            wait_for_indexing=True,
            max_wait_seconds=10,
            poll_interval=1,
        )

        assert result == []

    @patch("requests.Session.post")
    def test_no_polling_by_default(self, mock_post, client):
        mock_post.return_value = _ndjson_response("")

        result = client.retrieve_memories(query="q", space_ids=["s"])

        assert result == []
        assert mock_post.call_count == 1


class TestGoodMemClientSpaces:
    @patch("requests.Session.get")
    def test_list_embedders_normalises_payload(self, mock_get, client):
        mock_get.return_value = _json_response(
            {
                "embedders": [
                    {
                        "embedderId": "emb-1",
                        "displayName": "OpenAI",
                        "modelIdentifier": "text-embedding-3-small",
                    }
                ]
            }
        )
        result = client.list_embedders()
        assert result == [
            {
                "embedderId": "emb-1",
                "displayName": "OpenAI",
                "modelIdentifier": "text-embedding-3-small",
            }
        ]

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_create_space_reuses_existing(self, mock_get, mock_post, client):
        mock_get.return_value = _json_response(
            {
                "spaces": [
                    {
                        "spaceId": "existing-id",
                        "name": "demo",
                        "spaceEmbedders": [{"embedderId": "emb-1"}],
                    }
                ]
            }
        )

        result = client.create_space(name="demo", embedder_id="emb-1")

        assert result["reused"] is True
        assert result["spaceId"] == "existing-id"
        mock_post.assert_not_called()

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_create_space_posts_when_missing(self, mock_get, mock_post, client):
        mock_get.return_value = _json_response({"spaces": []})
        mock_post.return_value = _json_response(
            {"spaceId": "new-id", "name": "demo"}
        )

        result = client.create_space(name="demo", embedder_id="emb-1")

        assert result["reused"] is False
        assert result["spaceId"] == "new-id"
        mock_post.assert_called_once()

    @patch("requests.Session.delete")
    def test_delete_space(self, mock_delete, client):
        mock_delete.return_value = _json_response({})
        result = client.delete_space("space-123")
        assert result["success"] is True
        assert result["spaceId"] == "space-123"

    def test_update_space_rejects_both_label_modes(self, client):
        result = client.update_space(
            "space-123",
            replace_labels={"a": "b"},
            merge_labels={"c": "d"},
        )
        assert result["success"] is False


class TestGoodMemClientMemories:
    @patch("requests.Session.post")
    def test_create_memory_text(self, mock_post, client):
        mock_post.return_value = _json_response(
            {
                "memoryId": "mem-1",
                "spaceId": "space-1",
                "processingStatus": "PENDING",
            }
        )

        result = client.create_memory(
            space_id="space-1",
            text_content="hello",
            metadata={"category": "feat"},
        )

        assert result["memoryId"] == "mem-1"
        body = mock_post.call_args[1]["json"]
        assert body["contentType"] == "text/plain"
        assert body["originalContent"] == "hello"
        assert body["metadata"] == {"category": "feat"}

    def test_create_memory_requires_content(self, client):
        result = client.create_memory(space_id="s")
        assert result["success"] is False

    @patch("requests.Session.get")
    def test_list_memories(self, mock_get, client):
        mock_get.return_value = _json_response(
            {"memories": [{"memoryId": "m1"}]}
        )
        result = client.list_memories(
            space_id="s",
            status_filter="COMPLETED",
            include_content=True,
            sort_by="created_at",
            sort_order="DESCENDING",
        )
        params = mock_get.call_args[1]["params"]
        assert params["statusFilter"] == "COMPLETED"
        assert params["includeContent"] == "true"
        assert params["sortBy"] == "created_at"
        assert params["sortOrder"] == "DESCENDING"
        assert result == [{"memoryId": "m1"}]

    @patch("requests.Session.delete")
    def test_delete_memory(self, mock_delete, client):
        mock_delete.return_value = _json_response({})
        result = client.delete_memory("mem-1")
        assert result["memoryId"] == "mem-1"


class TestGoodMemRetriever:
    @patch("requests.Session.post")
    def test_retrieve_returns_texts(self, mock_post, retriever):
        mock_post.return_value = _ndjson_response()

        result = retriever.retrieve("test query")
        assert result == [
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
        ]

    @patch("requests.Session.post")
    def test_retrieve_chunks_returns_structured(self, mock_post, retriever):
        mock_post.return_value = _ndjson_response()

        result = retriever.retrieve_chunks("test query")
        assert len(result) == 2
        assert isinstance(result[0], GoodMemChunk)
        assert result[0].content == "Python is a programming language."
        assert result[0].score == -0.25
        assert result[0].chunk_id == "chunk-1"
        assert result[0].memory_id == "mem-1"

    @patch("requests.Session.post")
    def test_retrieve_as_context_delegates(self, mock_post, retriever):
        mock_post.return_value = _ndjson_response()

        result = retriever.retrieve_as_context("query")
        assert result == [
            "Python is a programming language.",
            "Python was created by Guido van Rossum.",
        ]

    @patch("requests.Session.post")
    def test_config_metadata_filter_forwarded(self, mock_post):
        retriever = GoodMemRetriever(
            GoodMemConfig(
                base_url="https://api.goodmem.ai",
                api_key="test-key",
                space_id="space-x",
                metadata_filter="CAST(val('$.category') AS TEXT) = 'feat'",
            )
        )
        mock_post.return_value = _ndjson_response()

        retriever.retrieve("q")
        body = mock_post.call_args[1]["json"]
        assert (
            body["spaceKeys"][0]["filter"]
            == "CAST(val('$.category') AS TEXT) = 'feat'"
        )

    def test_config_defaults(self):
        config = GoodMemConfig(
            base_url="https://api.goodmem.ai",
            api_key="key",
            space_id="space",
        )
        assert config.top_k == 5
        assert config.reranker is None
        assert config.relevance_threshold is None
        assert config.metadata_filter is None
        assert config.embedder is None
        assert config.verify_ssl is True

    def test_config_space_id_backward_compat(self):
        config = GoodMemConfig(
            base_url="https://api.goodmem.ai",
            api_key="key",
            space_id="single-space",
        )
        assert config.space_ids == ["single-space"]

    def test_config_multi_space(self):
        config = GoodMemConfig(
            base_url="https://api.goodmem.ai",
            api_key="key",
            space_ids=["space-a", "space-b"],
        )
        assert config.space_ids == ["space-a", "space-b"]

    def test_config_requires_space(self):
        with pytest.raises(ValueError, match="space_id or space_ids"):
            GoodMemConfig(
                base_url="https://api.goodmem.ai",
                api_key="key",
            )
