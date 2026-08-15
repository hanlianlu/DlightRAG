# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One session's earlier turns, as a request may replay them."""

from typing import Any


class PriorTurns:
    """Earlier turns already projected once for every reachable model call."""

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self._messages = list(messages or [])

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)


__all__ = ["PriorTurns"]
