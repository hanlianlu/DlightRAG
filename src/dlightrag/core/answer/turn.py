# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-local input boundary for the answer pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlightrag.core.request.planner import QueryPlan

HISTORICAL_DOCUMENT_LOAD_FAILED = "HISTORICAL_DOCUMENT_LOAD_FAILED"
HISTORICAL_DOCUMENT_UNAVAILABLE = "HISTORICAL_DOCUMENT_UNAVAILABLE"
HISTORICAL_DOCUMENT_PARSE_FAILED = "HISTORICAL_DOCUMENT_PARSE_FAILED"


@dataclass(frozen=True, slots=True)
class DocumentWarning:
    """Public request-local warning for a historical document."""

    code: str
    filename: str
    message: str


@dataclass(frozen=True, slots=True)
class PreparedAnswerTurn:
    """Server-prepared answer input without identity or persistence state."""

    current_query: str
    retrieval_query: str
    text_history: tuple[dict[str, Any], ...] = ()
    # Current-turn uploads drive retrieval and are strict answer inputs.
    current_query_images: tuple[dict[str, Any], ...] = ()
    # Planner-selected Web history images are answer-only, best-effort inputs.
    history_query_images: tuple[dict[str, Any], ...] = ()
    # VLM descriptions of current-turn images (ordinal -> text), computed once in
    # prepare_answer_turn so the planner is image-aware without re-describing them
    # downstream in retrieval.
    current_image_descriptions: dict[str, str] = field(default_factory=dict)
    plan: QueryPlan | None = None
    history_image_catalog_count: int = 0
    history_image_resolution_status: str = "ok"
    # Composer-only evidence selected from current and referenced historical
    # Web attachments. Workspace RAG retrieval never sees or reranks these rows;
    # core only places them ahead of the untouched RAG rows for answer assembly.
    composer_context_chunks: tuple[dict[str, Any], ...] = ()
    composer_evidence_trace: dict[str, Any] = field(default_factory=dict)
    web_composer_visuals: bool = False
    # Current attachment id -> deterministic planner digest. The Web adapter
    # persists these with the completed turn so future planner turns can resolve
    # references to prior documents without reparsing them.
    current_attachment_digests: dict[str, str] = field(default_factory=dict)
    history_attachment_catalog_count: int = 0
    history_attachments_selected: int = 0
    document_warnings: tuple[DocumentWarning, ...] = ()

    @classmethod
    def stateless(
        cls,
        query: str,
        query_images: list[dict[str, Any]] | None = None,
        *,
        history: list[dict[str, Any]] | None = None,
    ) -> PreparedAnswerTurn:
        """Create the turn used by public answer methods.

        ``history`` holds caller-supplied prior messages (``role``/``content``).
        It is stateless: the caller owns persistence and passes it per request.
        """
        images = tuple(query_images or ())
        return cls(
            current_query=query,
            retrieval_query=query,
            text_history=tuple(history or ()),
            current_query_images=images,
        )


__all__ = [
    "HISTORICAL_DOCUMENT_LOAD_FAILED",
    "HISTORICAL_DOCUMENT_PARSE_FAILED",
    "HISTORICAL_DOCUMENT_UNAVAILABLE",
    "DocumentWarning",
    "PreparedAnswerTurn",
]
