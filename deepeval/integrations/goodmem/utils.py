"""Parsing and content-type helpers for the GoodMem client."""

import json
from typing import Dict, List, Optional

from deepeval.integrations.goodmem.types import GoodMemChunk

_MIME_TYPES: Dict[str, str] = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "txt": "text/plain",
    "html": "text/html",
    "md": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "doc": "application/msword",
    "docx": (
        "application/vnd.openxmlformats-officedocument"
        ".wordprocessingml.document"
    ),
    "xls": "application/vnd.ms-excel",
    "xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ),
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": (
        "application/vnd.openxmlformats-officedocument"
        ".presentationml.presentation"
    ),
}


def get_mime_type(extension: str) -> Optional[str]:
    """Return the MIME type for a file extension, or `None` if unknown."""
    return _MIME_TYPES.get(extension.lower().lstrip("."))


def parse_ndjson_response(text: str) -> List[GoodMemChunk]:
    """Parse GoodMem's NDJSON retrieve stream into `GoodMemChunk` objects.

    Lines are tolerated when they are blank, prefixed with `data:` (SSE
    framing), or fail to parse as JSON; only `retrievedItem` events
    contribute chunks to the result.
    """
    chunks: List[GoodMemChunk] = []

    for line in text.strip().split("\n"):
        json_str = line.strip()
        if not json_str:
            continue
        if json_str.startswith("data:"):
            json_str = json_str[5:].strip()
        if not json_str or json_str.startswith("event:"):
            continue
        try:
            event = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        if "retrievedItem" not in event:
            continue

        item = event["retrievedItem"]
        chunk_data = item.get("chunk", {})
        inner_chunk = chunk_data.get("chunk", chunk_data)

        chunks.append(
            GoodMemChunk(
                content=inner_chunk.get("chunkText", ""),
                score=chunk_data.get(
                    "relevanceScore", item.get("relevanceScore")
                ),
                chunk_id=inner_chunk.get("chunkId", ""),
                memory_id=inner_chunk.get("memoryId", ""),
                space_id=inner_chunk.get("spaceId", ""),
            )
        )

    return chunks


_parse_ndjson_response = parse_ndjson_response
