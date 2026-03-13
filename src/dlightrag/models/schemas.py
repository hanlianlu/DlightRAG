# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for reranking and answer output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VisualRerankScore(BaseModel):
    """Single pointwise relevance score from the unified-mode VLM visual reranker."""

    score: float = Field(ge=0, le=1, description="Relevance score 0.0-1.0")

    model_config = {"extra": "forbid"}


class RankedChunk(BaseModel):
    """Single reranked chunk result from LLM reranker."""

    index: int = Field(description="0-based chunk index from original list")
    relevance_score: float = Field(ge=0, le=1, description="Relevance score 0-1")

    model_config = {"extra": "forbid"}


class RerankResult(BaseModel):
    """LLM reranker output - list of chunks sorted by relevance."""

    ranked_chunks: list[RankedChunk] = Field(description="Chunks sorted by relevance descending")

    model_config = {"extra": "forbid"}


class Reference(BaseModel):
    """A document-level reference cited in the answer."""

    id: int = Field(description="Reference number matching [n] in inline citations")
    title: str = Field(description="Document title/filename")


class StructuredAnswer(BaseModel):
    """Structured answer with separated references."""

    answer: str = Field(description="Markdown answer with inline [n-m] citations")
    references: list[Reference] = Field(
        default_factory=list,
        description="Document-level references cited in the answer",
    )


__all__ = ["RankedChunk", "Reference", "RerankResult", "StructuredAnswer", "VisualRerankScore"]
