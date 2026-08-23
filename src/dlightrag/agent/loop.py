# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Small event-driven Agent loop independent of products and storage."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from dlightrag.agent.events import AgentEvent
from dlightrag.agent.tools.contracts import ExecutedTurn

type EventSink = Callable[[AgentEvent], Awaitable[None]]


class AgentLoopCancelled(Exception):
    """A host translated its product-specific cancellation into loop control."""


class AgentTurnDriver(Protocol):
    """Host operations required by the product-neutral loop."""

    async def check_cancelled(self) -> None: ...

    async def run_turn(self, turn_number: int) -> ExecutedTurn: ...


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """Terminal loop outcome and the last complete provider turn."""

    turn_count: int
    stop_reason: str
    last_turn: ExecutedTurn | None


class AgentLoop:
    """Run complete model/tool turns until model silence or host cancellation.

    The driver owns context projection, durable intent settlement, tools, and
    product errors. The kernel owns only lifecycle ordering and termination.
    There is deliberately no hard turn cap.
    """

    def __init__(self, *, on_event: EventSink | None = None) -> None:
        self._on_event = on_event

    async def run(
        self,
        driver: AgentTurnDriver,
        *,
        starting_turn: int = 0,
    ) -> AgentLoopResult:
        if starting_turn < 0:
            raise ValueError("starting_turn cannot be negative")
        await self._emit(AgentEvent("agent_start", data={"starting_turn": starting_turn}))
        turn_count = starting_turn
        last_turn: ExecutedTurn | None = None
        stop_reason = "model_stop"
        try:
            while True:
                try:
                    await driver.check_cancelled()
                except AgentLoopCancelled:
                    stop_reason = "cancelled"
                    break
                turn_number = turn_count + 1
                await self._emit(AgentEvent("turn_start", turn_number=turn_number))
                last_turn = await driver.run_turn(turn_number)
                turn_count = turn_number
                await self._emit(
                    AgentEvent(
                        "turn_end",
                        turn_number=turn_number,
                        data={
                            "stop_reason": last_turn.assistant.stop_reason,
                            "tool_calls": len(last_turn.assistant.tool_calls),
                        },
                    )
                )
                if not last_turn.assistant.tool_calls:
                    continue_after_stop = getattr(driver, "continue_after_stop", None)
                    if continue_after_stop is None or not await continue_after_stop():
                        break
        except BaseException:
            stop_reason = "error"
            raise
        finally:
            await self._emit(
                AgentEvent(
                    "agent_end",
                    turn_number=turn_count or None,
                    data={"stop_reason": stop_reason},
                )
            )
        return AgentLoopResult(
            turn_count=turn_count,
            stop_reason=stop_reason,
            last_turn=last_turn,
        )

    async def _emit(self, event: AgentEvent) -> None:
        if self._on_event is not None:
            await self._on_event(event)


__all__ = [
    "AgentLoop",
    "AgentLoopCancelled",
    "AgentLoopResult",
    "AgentTurnDriver",
    "EventSink",
]
