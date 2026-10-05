# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""How many children one spawn call starts, and how many a Run runs at once."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dlightrag.application.config import DlightragConfig
from dlightrag.application.settings import answer_executor_settings
from dlightrag.engine.agent.session.ids import SessionId
from dlightrag.engine.answer.tools.subagents import (
    MAX_CHILDREN_PER_SPAWN,
    ChildContextSnapshot,
    ChildOutcome,
    ChildRequest,
    SpawnAgentInput,
    subagent_declarations,
    subagent_tools,
)
from tests.tool_helpers import tool_runtime
from tests.unit.test_child_model_roles import _prepared_executor
from tests.unit.test_subagents import _context_snapshot


def _children(count: int) -> list[dict[str, str]]:
    return [{"objective": f"objective {index}"} for index in range(count)]


def test_a_spawn_call_admits_sixteen_children_and_refuses_a_seventeenth() -> None:
    declared = subagent_declarations()[0].definition.parameters
    assert declared["properties"]["children"]["maxItems"] == MAX_CHILDREN_PER_SPAWN == 16

    admitted = SpawnAgentInput.model_validate({"children": _children(16)})
    assert len(admitted.children) == 16
    with pytest.raises(ValidationError, match="at most 16 items"):
        SpawnAgentInput.model_validate({"children": _children(17)})


@pytest.mark.parametrize(
    ("configured", "running_at_once"),
    [(2, 2), (None, MAX_CHILDREN_PER_SPAWN)],
    ids=["configured to two", "default"],
)
async def test_child_concurrency_limits_how_many_children_run_at_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured: int | None,
    running_at_once: int,
) -> None:
    agent: dict[str, Any] = {} if configured is None else {"child_concurrency": configured}
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        deployment={"working_dir": str(tmp_path)},
        answer={"agent": agent},
    )
    executor, orchestrator, _provider, _pins = await _prepared_executor(
        monkeypatch, settings=answer_executor_settings(config)
    )
    try:
        host = orchestrator._subagent_host  # pyright: ignore[reportPrivateUsage]
        assert host is not None
        running = 0
        peak = 0
        release = asyncio.Event()

        async def run_child(
            child_id: SessionId,
            request: ChildRequest,
            _call_id: str,
            _snapshot: ChildContextSnapshot,
        ) -> ChildOutcome:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            try:
                await release.wait()
            finally:
                running -= 1
            return ChildOutcome(
                status="succeeded", summary=request.objective, child_session_id=child_id.value
            )

        parent_id = SessionId.new()
        host.parent_session_id = parent_id
        host.run_id = SessionId.new().value
        host.run_child = run_child
        host.context_snapshot = _context_snapshot(parent_id)

        spawned = await subagent_tools(host=host)[0].execute(
            SpawnAgentInput.model_validate({"children": _children(MAX_CHILDREN_PER_SPAWN)}),
            tool_runtime(call_id="spawn", tool_name="spawn_agent"),
        )
        assert spawned.details is not None
        child_ids = [item["child_session_id"] for item in spawned.details["children"]]
        assert len(child_ids) == MAX_CHILDREN_PER_SPAWN

        # Every child has a handle now; only as many as the limit hold a slot.
        for _ in range(100):
            await asyncio.sleep(0)
        assert running == running_at_once

        release.set()
        outcomes = await asyncio.gather(*(host.tasks[child_id] for child_id in child_ids))
        assert {outcome.status for outcome in outcomes} == {"succeeded"}
        assert peak == running_at_once
        await host.stop(cancel=False)
    finally:
        await executor._models.aclose()  # pyright: ignore[reportPrivateUsage]
