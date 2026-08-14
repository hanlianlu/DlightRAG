# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One session's earlier turns, as a request may replay them."""

from collections.abc import Callable
from typing import Any

from dlightrag_ai.tokens import estimate_content_tokens


class PriorTurns:
    """Earlier turns a request replays and never appends to.

    Every caller renders these text turns differently -- the planner as a
    transcript and answer/control calls as role/content messages -- but they
    shed them the same way: drop the oldest turn until the request fits.
    """

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self._messages = list(messages or [])

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def recent(self, *, max_messages: int, max_tokens: int) -> PriorTurns:
        """The newest turns within a configured window, before any request is sized."""
        if max_messages <= 0 or max_tokens <= 0:
            return PriorTurns()
        messages = self._messages
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
        total = 0
        cutoff = 0
        for index in range(len(messages) - 1, -1, -1):
            total += estimate_content_tokens(messages[index].get("content", ""))
            if total > max_tokens:
                cutoff = index + 1
                break
        return PriorTurns(messages[cutoff:])

    def fit(
        self,
        budget: int,
        measure: Callable[[list[dict[str, Any]]], int],
    ) -> list[dict[str, Any]]:
        """Drop whole turns, oldest first, until ``measure`` fits ``budget``."""
        kept = list(self._messages)
        while kept and measure(kept) > budget:
            kept = kept[_oldest_turn_width(kept) :]
        return kept


def _oldest_turn_width(messages: list[dict[str, Any]]) -> int:
    """A user turn and the assistant reply it earned leave together."""
    if (
        len(messages) >= 2
        and messages[0].get("role") == "user"
        and messages[1].get("role") == "assistant"
    ):
        return 2
    return 1


__all__ = ["PriorTurns"]
