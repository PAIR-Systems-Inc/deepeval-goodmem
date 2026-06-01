"""End-to-end test: GoodMem retrieval, OpenAI generation, DeepEval metrics.

The test provisions its own space, stores a small known corpus, runs the
retrieve-generate-evaluate flow against it, and deletes the space on
teardown. It is skipped automatically unless the required environment
variables are set.

Run with:

    pytest tests/test_integrations/test_goodmem/test_integration.py -m integration -v

Required env vars:
    GOODMEM_BASE_URL   GoodMem server URL, e.g. https://localhost:8080.
    GOODMEM_API_KEY    GoodMem API key.
    OPENAI_API_KEY     OpenAI API key for generation and metrics.

Optional:
    GOODMEM_VERIFY_SSL Set to ``false`` for local dev with a self-signed
                       certificate.
"""

import os

import pytest

REQUIRED_VARS = [
    "GOODMEM_BASE_URL",
    "GOODMEM_API_KEY",
    "OPENAI_API_KEY",
]
_missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        len(_missing) > 0,
        reason=f"Missing env vars: {', '.join(_missing)}",
    ),
]

GENERATION_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "Answer the question accurately using only the provided context. "
    "If the context does not contain enough information, say so."
)

KNOWLEDGE_BASE = [
    "The Eiffel Tower stands in Paris and was completed in 1889.",
    "Mount Everest is the highest mountain on Earth at 8,849 meters.",
    "The Pacific Ocean is the largest and deepest of Earth's oceans.",
    "Python is a programming language created by Guido van Rossum.",
]


def _verify_ssl() -> bool:
    return os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() not in (
        "false",
        "0",
        "no",
    )


@pytest.fixture(scope="module")
def retriever():
    import urllib3

    from deepeval.integrations.goodmem import (
        GoodMemClient,
        GoodMemConfig,
        GoodMemRetriever,
    )

    verify_ssl = _verify_ssl()
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    client = GoodMemClient(verify_ssl=verify_ssl, timeout=60.0)
    embedder_id = client.list_embedders()[0]["embedderId"]
    space_id = client.create_space(
        name="deepeval-goodmem-integration-test",
        embedder_id=embedder_id,
    )["spaceId"]
    for fact in KNOWLEDGE_BASE:
        client.create_memory(space_id=space_id, text_content=fact)

    retriever_under_test = GoodMemRetriever(
        GoodMemConfig(
            base_url=client.base_url,
            api_key=client.api_key,
            space_id=space_id,
            top_k=3,
            verify_ssl=verify_ssl,
            wait_for_indexing=True,
            max_wait_seconds=30.0,
            poll_interval=3.0,
        ),
        client=client,
    )

    try:
        yield retriever_under_test
    finally:
        client.delete_space(space_id)
        client.close()


@pytest.fixture(scope="module")
def openai_client():
    from openai import OpenAI

    return OpenAI()


class TestRetrieve:
    """Live retrieval should return usable results."""

    def test_retrieve_returns_strings(self, retriever):
        results = retriever.retrieve("Where is the Eiffel Tower?")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, str) for r in results)

    def test_retrieve_chunks_returns_structured(self, retriever):
        from deepeval.integrations.goodmem import GoodMemChunk

        chunks = retriever.retrieve_chunks("Where is the Eiffel Tower?")
        assert len(chunks) > 0
        assert all(isinstance(c, GoodMemChunk) for c in chunks)

    def test_chunk_has_metadata(self, retriever):
        chunks = retriever.retrieve_chunks("What is the highest mountain?")
        chunk = chunks[0]
        assert chunk.content
        assert chunk.score is not None
        assert chunk.chunk_id
        assert chunk.memory_id

    def test_top_k_respected(self, retriever):
        chunks = retriever.retrieve_chunks("Tell me about oceans.")
        assert len(chunks) <= retriever.config.top_k


class TestRAGPipeline:
    """Retrieve, generate, and evaluate with DeepEval metrics."""

    @staticmethod
    def _generate(client, chunks, query):
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{chr(10).join(chunks)}\n\n"
                        f"Question: {query}"
                    ),
                },
            ],
        )
        return response.choices[0].message.content

    def test_rag_answer_relevancy(self, retriever, openai_client):
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        query = "Where is the Eiffel Tower and when was it completed?"
        chunks = retriever.retrieve(query)
        answer = self._generate(openai_client, chunks, query)

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=chunks,
        )
        metric = AnswerRelevancyMetric()
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.score >= 0.0

    def test_rag_contextual_relevancy(self, retriever, openai_client):
        from deepeval.metrics import ContextualRelevancyMetric
        from deepeval.test_case import LLMTestCase

        query = "Where is the Eiffel Tower and when was it completed?"
        chunks = retriever.retrieve(query)
        answer = self._generate(openai_client, chunks, query)

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=chunks,
        )
        metric = ContextualRelevancyMetric()
        metric.measure(test_case)

        assert metric.score is not None
        assert metric.score >= 0.0

    def test_batch_evaluate(self, retriever, openai_client):
        from deepeval import evaluate
        from deepeval.evaluate import AsyncConfig
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        queries = [
            "Where is the Eiffel Tower and when was it completed?",
            "What is the highest mountain on Earth?",
        ]

        test_cases = []
        for query in queries:
            chunks = retriever.retrieve(query)
            answer = self._generate(openai_client, chunks, query)
            test_cases.append(
                LLMTestCase(
                    input=query,
                    actual_output=answer,
                    retrieval_context=chunks,
                )
            )

        results = evaluate(
            test_cases,
            [AnswerRelevancyMetric()],
            async_config=AsyncConfig(max_concurrent=2, throttle_value=1),
        )
        assert results is not None
