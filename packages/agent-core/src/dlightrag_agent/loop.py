# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unlimited research loop: stop on silence, cancel, error, or all-terminate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from dlightrag_agent.tools.contracts import ExecutedTurn

LoopStopReason = Literal["model_stop", "cancelled", "provider_error", "all_terminate"]


class LoopCancelled(Exception):
    """The host observed durable cancellation at a turn boundary."""


class LoopProviderError(Exception):
    """The model or transport failed; the attempt must end."""


@dataclass(frozen=True, slots=True)
class LoopOutcome:
    """Why the loop stopped and the last executed turn, if any."""

    reason: LoopStopReason
    last_turn: ExecutedTurn | None = None


class AgentLoopHost(Protocol):
    """One Research host: cancel checks and a single model-plus-tools turn."""

    async def check_cancelled(self) -> None:
        """Raise LoopCancelled when the run must stop."""
        ...

    async def run_turn(self) -> ExecutedTurn:
        """Call the model once and execute any valid tool batch."""
        ...


class AgentLoop:
    """Run turns until the model emits no tool call or a terminal host signal."""

    async def run(self, host: AgentLoopHost) -> LoopOutcome:
        last: ExecutedTurn | None = None
        try:
            while True:
                await host.check_cancelled()
                turn = await host.run_turn()
                last = turn
                if _all_terminate(turn):
                    return LoopOutcome(reason="all_terminate", last_turn=turn)
                if not turn.assistant.tool_calls:
                    return LoopOutcome(reason="model_stop", last_turn=turn)
        except LoopCancelled:
            return LoopOutcome(reason="cancelled", last_turn=last)
        except LoopProviderError:
            return LoopOutcome(reason="provider_error", last_turn=last)


def _all_terminate(turn: ExecutedTurn) -> bool:
    if not turn.results:
        return False
    return all(execution.result.terminate for execution in turn.results)


__all__ = [
    "AgentLoop",
    "AgentLoopHost",
    "LoopCancelled",
    "LoopOutcome",
    "LoopProviderError",
    "LoopStopReason",
]
