"""GoodMem and DeepEval: retrieval, tracing, and grounded answers.

Three scenarios drive the GoodMem integration through DeepEval's
tracing layer:
    1. Persistent project context across phases.
    2. A two-role pipeline (Scribe and Analyst) with span metadata.
    3. Metadata-driven retrieval with server-side filtering.

Each scenario records a retriever span via ``GoodMemRetriever`` and an
LLM span via an instrumented OpenAI call, so the full retrieve-and-answer
flow shows up in the DeepEval trace.

Required environment variables:
    GOODMEM_API_KEY        GoodMem API key.
    GOODMEM_BASE_URL       GoodMem server URL, e.g. https://localhost:8080.
    OPENAI_API_KEY         OpenAI API key used for grounded answers.

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

from deepeval.integrations.goodmem import (
    GoodMemClient,
    GoodMemConfig,
    GoodMemRetriever,
)
from deepeval.tracing import (
    observe,
    trace,
    update_current_span,
    update_current_trace,
)

SYSTEM_PROMPT = (
    "Answer the user's question accurately using only the provided "
    "context. If the context lacks the answer, say so plainly."
)
GENERATION_MODEL = "gpt-4o-mini"


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
    top_k: int = 5,
    metadata_filter: str | None = None,
    wait_for_indexing: bool = True,
) -> GoodMemRetriever:
    config = GoodMemConfig(
        base_url=client.base_url,
        api_key=client.api_key,
        space_id=space_id,
        top_k=top_k,
        embedder=embedder_id,
        verify_ssl=client.verify_ssl,
        metadata_filter=metadata_filter,
        wait_for_indexing=wait_for_indexing,
        max_wait_seconds=20.0,
        poll_interval=2.0,
    )
    return GoodMemRetriever(config, client=client)


@observe(type="llm", model=GENERATION_MODEL)
def grounded_answer(
    openai_client: OpenAI, query: str, context: list[str]
) -> str:
    response = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{chr(10).join(context)}\n\n"
                    f"Question: {query}"
                ),
            },
        ],
    )
    answer = response.choices[0].message.content or ""
    update_current_span(input=query, output=answer)
    return answer


@observe(type="agent", name="Scribe")
def scribe_store_notes(
    client: GoodMemClient, space_id: str, notes: list[str]
) -> None:
    update_current_span(metadata={"role": "scribe"})
    for note in notes:
        client.create_memory(space_id=space_id, text_content=note)


@observe(type="agent", name="Analyst")
def analyst_answer(
    retriever: GoodMemRetriever,
    openai_client: OpenAI,
    query: str,
) -> str:
    update_current_span(metadata={"role": "analyst"})
    chunks = retriever.retrieve(query)
    return grounded_answer(openai_client, query, chunks)


def scenario_persistent_context(
    client: GoodMemClient,
    openai_client: OpenAI,
    space_id: str,
    embedder_id: str,
) -> None:
    print("\n=== Scenario 1: Persistent project context across phases ===")
    facts = [
        "I'm building a customer support assistant for our SaaS product.",
        "The team uses Python 3.12 with FastAPI and Postgres.",
        "For tests we use pytest with at least 80% coverage required.",
    ]

    with trace(
        name="goodmem-scenario-persistent",
        tags=["goodmem", "scenario-1"],
    ):
        update_current_trace(metadata={"scenario": "persistent-context"})
        for fact in facts:
            client.create_memory(space_id=space_id, text_content=fact)
            print(f"  stored: {fact}")

        retriever = _build_retriever(client, space_id, embedder_id, top_k=3)
        question = "Remind me what our coverage requirement is."
        chunks = retriever.retrieve(question)
        answer = grounded_answer(openai_client, question, chunks)

    print(f"\nUser:  {question}")
    print(f"Agent: {answer}")
    for idx, chunk in enumerate(chunks, start=1):
        print(f"  chunk {idx}: {chunk}")


def scenario_two_role_pipeline(
    client: GoodMemClient,
    openai_client: OpenAI,
    space_id: str,
    embedder_id: str,
) -> None:
    print("\n=== Scenario 2: Two-role pipeline (Scribe + Analyst) ===")
    notes = [
        "Q2 goal: reduce customer support response time to under 2 hours.",
        (
            "Our main services are auth-service, billing-service, and "
            "notifications-service."
        ),
        (
            "Known issue: notifications-service drops messages during "
            "high load."
        ),
        (
            "Team retro: the CI pipeline is too slow; we should "
            "parallelize tests."
        ),
    ]

    with trace(
        name="goodmem-scenario-two-role",
        tags=["goodmem", "scenario-2"],
    ):
        update_current_trace(metadata={"scenario": "two-role-pipeline"})
        scribe_store_notes(client, space_id, notes)

        retriever = _build_retriever(client, space_id, embedder_id, top_k=4)
        question = "What do we know about our services and current priorities?"
        answer = analyst_answer(retriever, openai_client, question)

    print(f"\nUser:  {question}")
    print(f"Agent: {answer}")


def scenario_metadata_filter(
    client: GoodMemClient,
    openai_client: OpenAI,
    space_id: str,
    embedder_id: str,
) -> None:
    print("\n=== Scenario 3: Metadata-driven retrieval ===")
    entries = [
        ("Added user profile editing to the dashboard.", "feat"),
        ("Built the CSV export feature.", "feat"),
        ("Resolved slow login on the mobile app.", "fix"),
        ("Fixed crash when opening large attachments.", "fix"),
        ("Upgraded Python version across services.", "chore"),
        ("Updated the API reference for billing endpoints.", "docs"),
    ]

    with trace(
        name="goodmem-scenario-metadata",
        tags=["goodmem", "scenario-3"],
    ):
        update_current_trace(metadata={"scenario": "metadata-filter"})
        for content, category in entries:
            client.create_memory(
                space_id=space_id,
                text_content=content,
                metadata={"category": category},
            )
            print(f"  stored ({category}): {content}")

        feat_retriever = _build_retriever(
            client,
            space_id,
            embedder_id,
            top_k=6,
            metadata_filter="CAST(val('$.category') AS TEXT) = 'feat'",
        )
        question = "Show me the new features we've shipped."
        chunks = feat_retriever.retrieve(question)
        answer = grounded_answer(openai_client, question, chunks)

    print(f"\nUser:  {question}")
    print(f"Agent: {answer}")
    print("\n  filtered chunks (feat only):")
    for idx, chunk in enumerate(chunks, start=1):
        print(f"    {idx}. {chunk}")


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

    project_space = client.create_space(
        name="deepeval-goodmem-project", embedder_id=embedder_id
    )["spaceId"]
    team_space = client.create_space(
        name="deepeval-goodmem-team", embedder_id=embedder_id
    )["spaceId"]
    tagged_space = client.create_space(
        name="deepeval-goodmem-tagged", embedder_id=embedder_id
    )["spaceId"]

    try:
        scenario_persistent_context(
            client, openai_client, project_space, embedder_id
        )
        scenario_two_role_pipeline(
            client, openai_client, team_space, embedder_id
        )
        scenario_metadata_filter(
            client, openai_client, tagged_space, embedder_id
        )
    finally:
        for space in (project_space, team_space, tagged_space):
            try:
                client.delete_space(space)
            except Exception as cleanup_error:
                print(f"  cleanup warning for {space}: {cleanup_error}")
        client.close()

    print(
        "\nDone. View the recorded traces on the Confident AI dashboard "
        "(deepeval login) or in the local trace manager."
    )


if __name__ == "__main__":
    main()
