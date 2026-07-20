# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-local input boundary for the answer pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlightrag.core.request.planner import QueryPlan


@dataclass(frozen=True, slots=True)
class PreparedAnswerTurn:
    """Server-prepared answer input without identity or persistence state."""

    current_query: str
    retrieval_query: str
    text_history: tuple[dict[str, Any], ...] = ()
    # Images shown to the answer model: current-turn uploads plus any resolved
    # history images the planner referenced.
    materialized_query_images: tuple[dict[str, Any], ...] = ()
    # Current-turn uploads only -- the retrieval-enhancement subset. History
    # images' retrieval semantics are owned by the planner (woven into the
    # standalone query), so they are excluded here to avoid re-describing them.
    current_query_images: tuple[dict[str, Any], ...] = ()
    # VLM descriptions of current-turn images (ordinal -> text), computed once in
    # prepare_answer_turn so the planner is image-aware without re-describing them
    # downstream in retrieval.
    current_image_descriptions: dict[str, str] = field(default_factory=dict)
    plan: QueryPlan | None = None
    history_image_catalog_count: int = 0
    history_image_resolution_status: str = "ok"
    # Request-local Web query-document attachments. Plain-dict context rows the
    # Web layer resolved (parse -> chunk -> text-token budget); core merges them
    # into retrieval chunks before generation. Empty for REST/MCP/SDK turns so
    # the core answer boundary never depends on the web layer.
    attachment_context_chunks: tuple[dict[str, Any], ...] = ()
    # Current attachment id -> deterministic planner digest. The Web adapter
    # persists these with the completed turn so future planner turns can resolve
    # references to prior documents without reparsing them.
    current_attachment_digests: dict[str, str] = field(default_factory=dict)
    history_attachment_catalog_count: int = 0
    history_attachments_selected: int = 0
    attachment_resolution_status: str = "ok"

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
            materialized_query_images=images,
            current_query_images=images,
        )


__all__ = ["PreparedAnswerTurn"]
