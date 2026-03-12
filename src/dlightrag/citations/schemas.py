"""Citation data models — ported from sandbox_agent, framework-agnostic."""

from __future__ import annotations

from pydantic import BaseModel


class ChunkSnippet(BaseModel):
    """Individual chunk reference with optional semantic highlights."""

    chunk_id: str
    chunk_idx: int | None = None
    page_idx: int | None = None
    content: str
    image_data: str | None = None  # base64-encoded page image (unified mode)
    highlight_phrases: list[str] | None = None
    highlight_phrases_map: dict[str, list[str]] | None = None

    model_config = {"extra": "forbid"}


class SourceReference(BaseModel):
    """Document-level source reference with cited chunks."""

    id: str
    title: str | None = None
    path: str
    type: str | None = None
    url: str | None = None
    snippet: str | None = None
    chunk_ids: list[str] | None = None
    cited_chunk_ids: list[str] | None = None
    chunks: list[ChunkSnippet] | None = None

    model_config = {"extra": "forbid"}
