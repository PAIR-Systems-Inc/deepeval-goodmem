"""HTTP client for the GoodMem service.

`GoodMemClient` exposes the GoodMem REST API as a small, typed Python
surface: spaces and memories CRUD, plus the NDJSON-streaming
retrieve endpoint with optional polling for newly-indexed content.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional

import requests

from deepeval.integrations.goodmem.types import GoodMemChunk
from deepeval.integrations.goodmem.utils import (
    get_mime_type,
    parse_ndjson_response,
)


class GoodMemClient:
    """Direct client for the GoodMem REST API.

    The client wraps an authenticated `requests.Session` and offers
    one method per server operation. It is independent of the
    DeepEval tracing layer; use `GoodMemRetriever` when you want
    retrieval calls to appear as retriever spans.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        verify_ssl: bool = True,
        timeout: float = 30.0,
    ) -> None:
        """Build an authenticated client.

        `base_url` and `api_key` fall back to the `GOODMEM_BASE_URL`
        and `GOODMEM_API_KEY` environment variables when not supplied.
        Set `verify_ssl=False` to accept the self-signed certificate
        served by the local dev environment.
        """
        resolved_base = base_url or os.environ.get("GOODMEM_BASE_URL", "")
        resolved_key = api_key or os.environ.get("GOODMEM_API_KEY", "")
        if not resolved_base:
            raise ValueError("base_url is required (or set GOODMEM_BASE_URL).")
        if not resolved_key:
            raise ValueError("api_key is required (or set GOODMEM_API_KEY).")

        self.base_url = resolved_base.rstrip("/")
        self.api_key = resolved_key
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self._session = requests.Session()
        self._session.verify = verify_ssl

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "GoodMemClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _headers(self, include_content_type: bool = True) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
        }
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers

    def list_embedders(self) -> List[Dict[str, Any]]:
        """List the embedder models registered on the server.

        Each entry exposes `embedderId`, `displayName`, and
        `modelIdentifier`. Use the `embedderId` value when creating a
        space.
        """
        response = self._session.get(
            f"{self.base_url}/v1/embedders",
            headers=self._headers(include_content_type=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        embedders = (
            body if isinstance(body, list) else body.get("embedders", [])
        )
        return [
            {
                "embedderId": e.get("embedderId") or e.get("id"),
                "displayName": (
                    e.get("displayName")
                    or e.get("name")
                    or e.get("modelIdentifier")
                    or "Unnamed"
                ),
                "modelIdentifier": (
                    e.get("modelIdentifier") or e.get("model") or "unknown"
                ),
            }
            for e in embedders
        ]

    def list_spaces(self) -> List[Dict[str, Any]]:
        """List every space the API key can see."""
        response = self._session.get(
            f"{self.base_url}/v1/spaces",
            headers=self._headers(include_content_type=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        spaces = body if isinstance(body, list) else body.get("spaces", [])
        return [
            {
                "spaceId": s.get("spaceId") or s.get("id"),
                "name": s.get("name") or "Unnamed",
                "spaceEmbedders": s.get("spaceEmbedders", []),
            }
            for s in spaces
        ]

    def get_space(self, space_id: str) -> Dict[str, Any]:
        """Fetch a single space by ID, including embedders and chunking."""
        response = self._session.get(
            f"{self.base_url}/v1/spaces/{space_id}",
            headers=self._headers(include_content_type=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def create_space(
        self,
        name: str,
        embedder_id: str,
        chunk_size: int = 256,
        chunk_overlap: int = 25,
        keep_strategy: str = "KEEP_END",
        length_measurement: str = "CHARACTER_COUNT",
    ) -> Dict[str, Any]:
        """Create a new space, or return the existing one with this name.

        Returns a dict with `spaceId`, `name`, `embedderId`, `reused`,
        and `message`. When a space named `name` already exists, the
        existing space is returned with `reused=True`; otherwise a new
        space is created with the supplied embedder and chunking
        configuration.
        """
        try:
            for space in self.list_spaces():
                if space.get("name") == name:
                    actual_embedder_id = embedder_id
                    space_embedders = space.get("spaceEmbedders", [])
                    if space_embedders:
                        actual_embedder_id = space_embedders[0].get(
                            "embedderId", embedder_id
                        )
                    return {
                        "success": True,
                        "spaceId": space["spaceId"],
                        "name": space["name"],
                        "embedderId": actual_embedder_id,
                        "message": "Space already exists; reusing it.",
                        "reused": True,
                    }
        except requests.RequestException:
            pass

        request_body: Dict[str, Any] = {
            "name": name,
            "spaceEmbedders": [
                {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
            ],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunk_size,
                    "chunkOverlap": chunk_overlap,
                    "separators": ["\n\n", "\n", ". ", " ", ""],
                    "keepStrategy": keep_strategy,
                    "separatorIsRegex": False,
                    "lengthMeasurement": length_measurement,
                },
            },
        }
        response = self._session.post(
            f"{self.base_url}/v1/spaces",
            headers=self._headers(),
            json=request_body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return {
            "success": True,
            "spaceId": body.get("spaceId"),
            "name": body.get("name"),
            "embedderId": embedder_id,
            "chunkingConfig": request_body["defaultChunkingConfig"],
            "message": "Space created.",
            "reused": False,
        }

    def update_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        public_read: Optional[bool] = None,
        replace_labels: Optional[Dict[str, str]] = None,
        merge_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Update mutable fields on a space.

        `replace_labels` overwrites the label map; `merge_labels` merges
        into the existing one. The two options are mutually exclusive.
        """
        if replace_labels is not None and merge_labels is not None:
            return {
                "success": False,
                "error": (
                    "replace_labels and merge_labels cannot both be set."
                ),
            }
        request_body: Dict[str, Any] = {}
        if name is not None:
            request_body["name"] = name
        if public_read is not None:
            request_body["publicRead"] = public_read
        if replace_labels is not None:
            request_body["replaceLabels"] = replace_labels
        if merge_labels is not None:
            request_body["mergeLabels"] = merge_labels

        response = self._session.put(
            f"{self.base_url}/v1/spaces/{space_id}",
            headers=self._headers(),
            json=request_body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def delete_space(self, space_id: str) -> Dict[str, Any]:
        """Delete a space and every memory inside it."""
        response = self._session.delete(
            f"{self.base_url}/v1/spaces/{space_id}",
            headers=self._headers(include_content_type=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return {
            "success": True,
            "spaceId": space_id,
            "message": "Space deleted.",
        }

    def create_memory(
        self,
        space_id: str,
        text_content: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a document as a memory in `space_id`.

        Provide `text_content` for plain text or `file_path` to upload a
        file from disk. Binary file types are base64-encoded; text-like
        types are sent as the raw decoded string. The optional
        `metadata` dict is attached to the memory for later filtering.
        """
        request_body: Dict[str, Any] = {"spaceId": space_id}

        if file_path:
            ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
            mime_type = get_mime_type(ext) or "application/octet-stream"
            with open(file_path, "rb") as fh:
                file_bytes = fh.read()
            request_body["contentType"] = mime_type
            if mime_type.startswith("text/"):
                request_body["originalContent"] = file_bytes.decode("utf-8")
            else:
                request_body["originalContentB64"] = base64.b64encode(
                    file_bytes
                ).decode("ascii")
        elif text_content is not None:
            request_body["contentType"] = "text/plain"
            request_body["originalContent"] = text_content
        else:
            return {
                "success": False,
                "error": "Provide text_content or file_path.",
            }

        if metadata:
            request_body["metadata"] = metadata

        response = self._session.post(
            f"{self.base_url}/v1/memories",
            headers=self._headers(),
            json=request_body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return {
            "success": True,
            "memoryId": body.get("memoryId"),
            "spaceId": body.get("spaceId"),
            "status": body.get("processingStatus", "PENDING"),
            "contentType": request_body["contentType"],
            "message": "Memory created.",
        }

    def list_memories(
        self,
        space_id: str,
        status_filter: Optional[str] = None,
        include_content: bool = False,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List memories in a space, with optional status filter and sort."""
        params: Dict[str, str] = {}
        if include_content:
            params["includeContent"] = "true"
        if status_filter:
            params["statusFilter"] = status_filter
        if sort_by:
            params["sortBy"] = sort_by
        if sort_order:
            params["sortOrder"] = sort_order

        response = self._session.get(
            f"{self.base_url}/v1/spaces/{space_id}/memories",
            headers=self._headers(include_content_type=False),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, list) else body.get("memories", [])

    def get_memory(
        self,
        memory_id: str,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """Fetch one memory and, optionally, its original content.

        Metadata is fetched from `GET /v1/memories/{id}`; the document
        body is fetched separately from `/content` so that status polling
        does not pull large blobs.
        """
        response = self._session.get(
            f"{self.base_url}/v1/memories/{memory_id}",
            headers=self._headers(include_content_type=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        result: Dict[str, Any] = {
            "success": True,
            "memory": response.json(),
        }

        if include_content:
            try:
                content_response = self._session.get(
                    f"{self.base_url}/v1/memories/{memory_id}/content",
                    headers=self._headers(include_content_type=False),
                    timeout=self.timeout,
                )
                content_response.raise_for_status()
                content_type = content_response.headers.get("Content-Type", "")
                if "text" in content_type:
                    result["content"] = content_response.text
                else:
                    result["content"] = content_response.content
            except requests.RequestException as request_error:
                result["contentError"] = (
                    f"Failed to fetch content: {request_error}"
                )

        return result

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a single memory."""
        response = self._session.delete(
            f"{self.base_url}/v1/memories/{memory_id}",
            headers=self._headers(include_content_type=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return {
            "success": True,
            "memoryId": memory_id,
            "message": "Memory deleted.",
        }

    def retrieve_memories(
        self,
        query: str,
        space_ids: List[str],
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = False,
        max_wait_seconds: float = 10.0,
        poll_interval: float = 2.0,
        reranker_id: Optional[str] = None,
        llm_id: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
        llm_temperature: Optional[float] = None,
        chronological_resort: bool = False,
        metadata_filter: Optional[str] = None,
    ) -> List[GoodMemChunk]:
        """Run a semantic retrieval across one or more spaces.

        Returns chunks ranked by relevance. When `wait_for_indexing` is
        true, the call polls until results appear or `max_wait_seconds`
        elapses, so newly stored memories have a chance to finish
        indexing. `metadata_filter` is a SQL/JSONPath predicate applied
        per space; pass a non-empty string to push the filter
        server-side.
        """
        space_keys: List[Dict[str, Any]] = [
            {"spaceId": sid} for sid in space_ids if sid
        ]
        if not space_keys:
            raise ValueError("At least one space_id is required.")
        if metadata_filter:
            for space_key in space_keys:
                space_key["filter"] = metadata_filter

        request_body: Dict[str, Any] = {
            "message": query,
            "spaceKeys": space_keys,
            "requestedSize": max_results,
            "fetchMemory": include_memory_definition,
        }

        if reranker_id or llm_id:
            config: Dict[str, Any] = {}
            if reranker_id:
                config["reranker_id"] = reranker_id
            if llm_id:
                config["llm_id"] = llm_id
            if relevance_threshold is not None:
                config["relevance_threshold"] = relevance_threshold
            if llm_temperature is not None:
                config["llm_temp"] = llm_temperature
            if max_results:
                config["max_results"] = max_results
            if chronological_resort:
                config["chronological_resort"] = True
            request_body["postProcessor"] = {
                "name": (
                    "com.goodmem.retrieval.postprocess"
                    ".ChatPostProcessorFactory"
                ),
                "config": config,
            }

        headers = {
            **self._headers(),
            "Accept": "application/x-ndjson",
        }
        url = f"{self.base_url}/v1/memories:retrieve"
        deadline = time.time() + max_wait_seconds

        while True:
            response = self._session.post(
                url,
                headers=headers,
                json=request_body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            chunks = parse_ndjson_response(response.text)

            if chunks or not wait_for_indexing:
                return chunks
            if time.time() >= deadline:
                return chunks
            time.sleep(poll_interval)
