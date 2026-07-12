"""Citation data models — ported from sandbox_agent, framework-agnostic."""

from typing import Any, Protocol

from pydantic import BaseModel, Field


class ChunkSnippet(BaseModel):
    """Individual chunk reference with optional semantic highlights."""

    chunk_id: str
    chunk_idx: int | None = None
    page_idx: int | None = None
    bbox: dict[str, Any] | None = None
    content: str
    image_url: str | None = None
    thumbnail_url: str | None = None
    highlight_phrases: list[str] | None = None

    model_config = {"extra": "forbid"}


class SourceReference(BaseModel):
    """Internal document source with durable download-routing metadata."""

    id: str
    title: str | None = None
    type: str | None = None
    source_uri: str
    workspace: str = Field(exclude=True, repr=False)
    download_locator: str = Field(exclude=True, repr=False)
    cited_chunk_ids: list[str] | None = None
    chunks: list[ChunkSnippet] | None = None

    model_config = {"extra": "forbid"}


class SourceReferencePayload(BaseModel):
    """Public source payload with an adapter-projected download URL."""

    id: str
    title: str | None = None
    type: str | None = None
    source_uri: str
    download_url: str | None = None
    cited_chunk_ids: list[str] | None = None
    chunks: list[ChunkSnippet] | None = None

    model_config = {"extra": "forbid"}


class HighlightSource(Protocol):
    """Structural source contract used by semantic highlight enrichment."""

    id: str
    chunks: list[ChunkSnippet] | None
