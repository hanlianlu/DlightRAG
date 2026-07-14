# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-local input boundary for the answer pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlightrag.core.query_planner import QueryPlan


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
    plan: QueryPlan | None = None
    history_image_catalog_count: int = 0
    history_image_resolution_status: str = "ok"

    @classmethod
    def stateless(
        cls,
        query: str,
        query_images: list[dict[str, Any]] | None = None,
    ) -> PreparedAnswerTurn:
        """Create the empty-history turn used by public answer methods."""
        images = tuple(query_images or ())
        return cls(
            current_query=query,
            retrieval_query=query,
            materialized_query_images=images,
            current_query_images=images,
        )


__all__ = ["PreparedAnswerTurn"]
