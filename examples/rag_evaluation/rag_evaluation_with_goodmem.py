"""Evaluate a GoodMem-backed RAG pipeline with DeepEval.

GoodMem is the retrieval backend: documents are stored in a space, and
``GoodMemRetriever`` pulls back the most relevant chunks for each question
while recording the call as a retriever span. The retrieved chunks become
the ``retrieval_context`` of an ``LLMTestCase``, an LLM produces a grounded
answer, and DeepEval scores the result with its RAG metrics.

The script runs two evaluations:
    1. Retrieval across the whole space, scored on a set of questions.
    2. The first question re-scored with a server-side ``metadata_filter``
       that scopes retrieval to a single category, for a same-question
       comparison.

Required environment variables:
    GOODMEM_API_KEY        GoodMem API key.
    GOODMEM_BASE_URL       GoodMem server URL, e.g. https://localhost:8080.
    OPENAI_API_KEY         OpenAI API key for answer generation and the
                           metric judges.

Optional:
    GOODMEM_VERIFY_SSL     Set to ``false`` for local dev with a
                           self-signed certificate.

Run from the repo root with:

    python examples/rag_evaluation/rag_evaluation_with_goodmem.py
"""

from __future__ import annotations

import os
import urllib3
import urllib3.exceptions

from openai import OpenAI

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.integrations.goodmem import (
    GoodMemClient,
    GoodMemConfig,
    GoodMemRetriever,
)
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

GENERATION_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "Answer the user's question using only the provided context. If the "
    "context does not contain the answer, say so plainly."
)

# Release-log knowledge base. Each entry is stored as one memory tagged with a
# category, which the metadata_filter run later uses to scope retrieval.
KNOWLEDGE_BASE = [
    ("Added single sign-on (SSO) support via SAML to the dashboard.", "feat"),
    ("Added a CSV export option to the billing reports page.", "feat"),
    ("Fixed a crash when opening attachments larger than 25 MB.", "fix"),
    (
        "Fixed slow login on the mobile app by adding a missing database index.",
        "fix",
    ),
    ("Upgraded the API gateway to require TLS 1.3 for all traffic.", "infra"),
    ("Migrated the search backend to OpenSearch 2.13.", "infra"),
]

# Questions paired with the answer a grounded pipeline should produce.
QUESTIONS = [
    (
        "What new features were shipped?",
        "Single sign-on via SAML and a CSV export option for billing reports.",
    ),
    (
        "What change was made to improve security?",
        "The API gateway was upgraded to require TLS 1.3 for all traffic.",
    ),
]

FEATURE_FILTER = "CAST(val('$.category') AS TEXT) = 'feat'"


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("false", "0", "no")


def _build_retriever(
    client: GoodMemClient,
    space_id: str,
    embedder_id: str,
    *,
    top_k: int = 4,
    metadata_filter: str | None = None,
) -> GoodMemRetriever:
    config = GoodMemConfig(
        base_url=client.base_url,
        api_key=client.api_key,
        space_id=space_id,
        top_k=top_k,
        embedder=embedder_id,
        verify_ssl=client.verify_ssl,
        metadata_filter=metadata_filter,
        wait_for_indexing=True,
        max_wait_seconds=30.0,
        poll_interval=3.0,
    )
    return GoodMemRetriever(config, client=client)


def _rag_metrics() -> list:
    return [
        AnswerRelevancyMetric(),
        FaithfulnessMetric(),
        ContextualPrecisionMetric(),
        ContextualRecallMetric(),
        ContextualRelevancyMetric(),
    ]


def generate_answer(
    openai_client: OpenAI, question: str, context: list[str]
) -> str:
    """Answer a question grounded only in the retrieved context."""
    joined = "\n".join(f"- {c}" for c in context) or "(no context retrieved)"
    response = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{joined}\n\nQuestion: {question}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def build_test_case(
    retriever: GoodMemRetriever,
    openai_client: OpenAI,
    question: str,
    expected_output: str,
) -> LLMTestCase:
    """Retrieve from GoodMem, generate an answer, and wrap both in a test case."""
    context = retriever.retrieve(question)
    answer = generate_answer(openai_client, question, context)
    return LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected_output,
        retrieval_context=context,
    )


def print_summary(title: str, result) -> None:
    """Print the mean score for each metric across the evaluated test cases."""
    totals: dict[str, list[float]] = {}
    for test_result in result.test_results:
        for metric in test_result.metrics_data or []:
            if metric.score is not None:
                totals.setdefault(metric.name, []).append(metric.score)

    print(f"\n{title}")
    for name, scores in totals.items():
        mean = sum(scores) / len(scores)
        print(f"  {name}: {mean:.2f} (n={len(scores)})")


def main() -> None:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    verify_ssl = _bool_env("GOODMEM_VERIFY_SSL", True)
    client = GoodMemClient(verify_ssl=verify_ssl, timeout=60.0)
    openai_client = OpenAI()

    embedders = client.list_embedders()
    if not embedders:
        raise RuntimeError(
            "No embedders registered on the GoodMem server. "
            "Register one before running this example."
        )
    embedder_id = embedders[0]["embedderId"]

    space_id = client.create_space(
        name="deepeval-goodmem-rag-eval", embedder_id=embedder_id
    )["spaceId"]
    for content, category in KNOWLEDGE_BASE:
        client.create_memory(
            space_id=space_id,
            text_content=content,
            metadata={"category": category},
        )

    async_config = AsyncConfig(max_concurrent=2, throttle_value=1)
    display_config = DisplayConfig(inspect_after_run=False)

    try:
        retriever = _build_retriever(client, space_id, embedder_id)
        cases = [
            build_test_case(retriever, openai_client, question, expected)
            for question, expected in QUESTIONS
        ]
        unfiltered = evaluate(
            test_cases=cases,
            metrics=_rag_metrics(),
            async_config=async_config,
            display_config=display_config,
        )

        feat_retriever = _build_retriever(
            client,
            space_id,
            embedder_id,
            top_k=6,
            metadata_filter=FEATURE_FILTER,
        )
        # Re-run the first question with the server-side filter so the
        # scores compare directly against the same question in pass 1.
        question, expected = QUESTIONS[0]
        filtered_case = build_test_case(
            feat_retriever, openai_client, question, expected
        )
        filtered = evaluate(
            test_cases=[filtered_case],
            metrics=_rag_metrics(),
            async_config=async_config,
            display_config=display_config,
        )
    finally:
        client.delete_space(space_id)
        client.close()

    print_summary("Mean scores, retrieval across the whole space:", unfiltered)
    print_summary(
        "Mean scores, retrieval filtered to the feat category:", filtered
    )


if __name__ == "__main__":
    main()
