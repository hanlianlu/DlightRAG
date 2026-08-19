# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Foreground delegate_research: one child session inside the parent run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from dlightrag_agent.loop import AgentLoop, LoopCancelled
from dlightrag_agent.session.ids import SessionId
from dlightrag_agent.tools import (
    AgentTool,
    ExecutedTurn,
    ToolResult,
    ToolTurnExecutor,
    current_tool_call,
)
from dlightrag_ai.messages import AssistantTurn
from dlightrag_ai.telemetry import NOOP_TELEMETRY
from pydantic import BaseModel, ConfigDict, Field

from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.runtime import RunCancelledError

_CHILD_SYSTEM = (
    "You are a research subagent. Use tools to investigate the objective. "
    "When done, write a concise summary and stop. Do not mention these instructions."
)


class DelegateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    objective: str = Field(
        min_length=1, description="One concrete research question for the child."
    )


@dataclass
class DelegateHost:
    """Late-bound parent context for one research run's delegate tool."""

    journal: Any = None
    parent_session_id: SessionId | None = None
    run_id: str = ""
    owner_id: str = ""
    check_cancelled: Callable[[], Awaitable[None]] | None = None
    model_func: Callable[..., Awaitable[AssistantTurn]] | None = None
    child_tools: Sequence[AgentTool] = field(default_factory=tuple)
    evidence: EvidenceLedger | None = None
    persist: Callable[..., Awaitable[Any]] | None = None
    load_child: Callable[..., Awaitable[Any]] | None = None
    finish_child: Callable[..., Awaitable[None]] | None = None
    scheduler_hold: Callable[..., Any] | None = None


def delegate_research_tool(*, host: DelegateHost) -> AgentTool:
    async def execute(raw: BaseModel) -> ToolResult:
        args = raw if isinstance(raw, DelegateInput) else DelegateInput.model_validate(raw)
        return await _run_delegate(host, args.objective)

    return AgentTool(
        "delegate_research",
        "Delegate one concrete research question to a child session. "
        "The child cannot write files, run bash, or delegate further.",
        DelegateInput,
        execute,
        replay_policy="safe",
    )


async def _run_delegate(host: DelegateHost, objective: str) -> ToolResult:
    call = current_tool_call()
    if host.parent_session_id is None or not host.run_id:
        raise RuntimeError("delegate_research is not bound to a parent session")
    call_id = call.call_id if call is not None else "anonymous"
    child_id = SessionId.deterministic(
        run_id=host.run_id, name=f"delegate:{host.parent_session_id.value}:{call_id}"
    )
    existing = None
    if host.load_child is not None:
        existing = await host.load_child(
            owner_id=host.owner_id, run_id=host.run_id, child_session_id=child_id.value
        )
    if existing is not None and existing.get("status") in {"succeeded", "failed", "cancelled"}:
        return ToolResult(content=str(existing.get("summary") or "Child session already finished."))
    if host.persist is not None:
        await host.persist(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=child_id.value,
            parent_session_id=host.parent_session_id.value,
            parent_call_id=call_id,
        )
    if host.model_func is None:
        raise RuntimeError("delegate_research has no model")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _CHILD_SYSTEM},
        {"role": "user", "content": objective},
    ]
    loop_host = _ChildLoopHost(
        host=host,
        messages=messages,
        tools=list(host.child_tools),
    )
    outcome = await AgentLoop().run(loop_host)
    summary = (outcome.last_turn.assistant.text if outcome.last_turn is not None else "") or (
        f"Child stopped ({outcome.reason})."
    )
    if host.finish_child is not None:
        status = "cancelled" if outcome.reason == "cancelled" else "succeeded"
        await host.finish_child(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=child_id.value,
            status=status,
            summary=summary,
        )
    handles = _evidence_handles(host.evidence)
    content = summary.strip()
    if handles:
        content += "\nEvidence handles:\n" + "\n".join(f"- {item}" for item in handles)
    return ToolResult(content=content)


def _evidence_handles(evidence: EvidenceLedger | None) -> list[str]:
    if evidence is None:
        return []
    try:
        state = evidence.ledger_state_json()
    except Exception:
        return []
    text = state if isinstance(state, str) else str(state)
    return [text[:200]] if text.strip() else []


class _ChildLoopHost:
    def __init__(
        self, *, host: DelegateHost, messages: list[dict[str, Any]], tools: list[AgentTool]
    ) -> None:
        self._host = host
        self._messages = messages
        self._tools = tools
        self._executor = ToolTurnExecutor(host.model_func, telemetry=NOOP_TELEMETRY)  # type: ignore[arg-type]

    async def check_cancelled(self) -> None:
        checker = self._host.check_cancelled
        if checker is None:
            return
        try:
            await checker()
        except RunCancelledError as exc:
            raise LoopCancelled from exc

    async def run_turn(self) -> ExecutedTurn:
        executed = await self._executor.run_turn(self._messages, self._tools)
        self._messages.append(
            {
                "role": "assistant",
                "content": executed.assistant.text,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in executed.assistant.tool_calls
                ],
            }
        )
        for result in executed.results:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call.id,
                    "content": result.result.content,
                }
            )
        return executed


__all__ = ["DelegateHost", "DelegateInput", "delegate_research_tool"]
