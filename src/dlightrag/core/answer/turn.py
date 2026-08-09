# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-local input boundary for the answer pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedAnswerTurn:
    """Server-prepared answer input without identity or persistence state.

    Current-turn images and documents are supplied through the request's
    ``resources`` list, not on the turn: the manager extracts verified current
    images from those resources into internal image blocks and registers the
    rest as request-local resources.
    """

    current_query: str
    retrieval_query: str
    text_history: tuple[dict[str, Any], ...] = ()

    @classmethod
    def stateless(
        cls,
        query: str,
        *,
        history: list[dict[str, Any]] | None = None,
    ) -> PreparedAnswerTurn:
        """Create the turn used by public answer methods.

        ``history`` holds caller-supplied prior messages (``role``/``content``).
        It is stateless: the caller owns persistence and passes it per request.
        """
        return cls(
            current_query=query,
            retrieval_query=query,
            text_history=tuple(history or ()),
        )


__all__ = ["PreparedAnswerTurn"]
