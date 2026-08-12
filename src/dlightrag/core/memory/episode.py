# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One research run's episodic memory: every exchange the loop produced."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from dlightrag.utils.tokens import estimate_messages_tokens

# The budget pi replays verbatim before it reduces older turns.
_KEEP_RECENT_TOKENS = 20_000


class RunEpisode:
    """Every assistant/tool exchange one research run produced, newest first to replay.

    Provider-native reasoning is what makes an exchange expensive and is valid
    only as an unmodified replay, so the newest ``_KEEP_RECENT_TOKENS`` worth
    carry it and older exchanges keep just the call and its result: a later turn
    still sees which angle was spent without paying for the thinking behind it.
    """

    def __init__(self) -> None:
        self._exchanges: list[list[dict[str, Any]]] = []

    def record(self, exchange: list[dict[str, Any]]) -> None:
        self._exchanges.append(exchange)

    def export_state(self) -> dict[str, Any]:
        """Return every exchange, provider-native state included, in order."""
        return {"exchanges": [[dict(message) for message in ex] for ex in self._exchanges]}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Replace the episode with a previously exported one."""
        exchanges = state.get("exchanges")
        if not isinstance(exchanges, Sequence):
            raise ValueError("episode state has no exchanges")
        self._exchanges = [
            [dict(cast(Mapping[str, Any], message)) for message in cast(Sequence[Any], exchange)]
            for exchange in exchanges
        ]

    @property
    def last_exchange(self) -> list[dict[str, Any]]:
        return list(self._exchanges[-1]) if self._exchanges else []

    def messages(self) -> list[dict[str, Any]]:
        if not self._exchanges:
            return []
        newest = len(self._exchanges) - 1
        replay_from = newest
        budget = _KEEP_RECENT_TOKENS - estimate_messages_tokens(self._exchanges[newest])
        for index in reversed(range(newest)):
            budget -= estimate_messages_tokens(self._exchanges[index])
            if budget < 0:
                break
            replay_from = index
        messages: list[dict[str, Any]] = []
        for index, exchange in enumerate(self._exchanges):
            if index >= replay_from:
                messages.extend(exchange)
            else:
                messages.extend(_without_reasoning(message) for message in exchange)
        return messages


def _without_reasoning(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("role") != "assistant":
        return message
    reduced = {key: value for key, value in message.items() if key != "provider_state"}
    calls = cast(list[dict[str, Any]], reduced.get("tool_calls") or [])
    if calls:
        reduced["tool_calls"] = [
            {key: value for key, value in call.items() if key != "thought_signature"}
            for call in calls
        ]
    return reduced


__all__ = ["RunEpisode"]
