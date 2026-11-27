# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Schedule tools carry a turn's identity and say what was saved; the rules live behind the port.

The port's owner-side rules are tested with the real service in test_schedule_agent.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ValidationError

from dlightrag.engine.agent.environment.toolchain import SearchToolchain
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.tools.contracts import AgentTool, ToolResult
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.execution.executor import AnswerExecutor
from dlightrag.engine.answer.tools.composition import (
    CHILD_FORBIDDEN_TOOLS,
    compose_research_tools,
    research_tool_declarations,
)
from dlightrag.engine.answer.tools.schedules import (
    SCHEDULE_INVALID,
    SCHEDULE_LIMIT,
    SCHEDULE_NOT_FOUND,
    SCHEDULE_TOOL_NAMES,
    SCHEDULE_TOOLS_PIN,
    SCHEDULE_UNAVAILABLE,
    CreateScheduleArgs,
    DeleteScheduleArgs,
    ListSchedulesArgs,
    ScheduleHost,
    ScheduleRecord,
    ScheduleRefusal,
    UpdateScheduleArgs,
    When,
    describe_cadence,
    schedule_tool_declarations,
    schedule_tools,
    schedule_tools_offered,
)
from tests.support.tool_helpers import tool_runtime
from tests.unit.test_answer_executor import _executor

_OWNER = "owner-1"
_RUN = "run-1"
_NEXT = datetime(2026, 11, 30, 8, 0, tzinfo=UTC)
_WEEKDAYS_AT_NINE = When(cron="0 9 * * 1-5", zone="Europe/Stockholm")


class FakePort:
    """The port with no rules of its own: it keeps what it is given, per owner, or refuses."""

    def __init__(self) -> None:
        self.records: dict[str, tuple[str, ScheduleRecord]] = {}
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.refusal: ScheduleRefusal | None = None

    def _check(self) -> None:
        if self.refusal is not None:
            raise self.refusal

    def seed(self, owner_id: str, schedule_id: str, **changes: Any) -> ScheduleRecord:
        record = ScheduleRecord(
            schedule_id=schedule_id,
            name="Weekly review",
            prompt="Summarise the week.",
            mode="research",
            target="new_conversation",
            workspaces=("alpha",),
            zone="Europe/Stockholm",
            cron="0 8 * * 1",
            at=None,
            state="active",
            pause_reason=None,
            next_fire_at=_NEXT,
        )
        record = replace(record, **changes)
        self.records[schedule_id] = (owner_id, record)
        return record

    async def list(self, *, owner_id: str, run_id: str) -> tuple[ScheduleRecord, ...]:
        self._check()
        return tuple(record for owner, record in self.records.values() if owner == owner_id)

    async def create(
        self,
        *,
        owner_id: str,
        run_id: str,
        name: str,
        prompt: str,
        workspaces: Sequence[str],
        zone: str,
        cron: str | None,
        at: datetime | None,
        mode: str,
        target: str,
    ) -> ScheduleRecord:
        self._check()
        self.created.append(
            {"workspaces": tuple(workspaces), "zone": zone, "cron": cron, "at": at, "mode": mode}
        )
        return self.seed(
            owner_id,
            f"id-{len(self.records) + 1}",
            name=name,
            prompt=prompt,
            workspaces=tuple(workspaces),
            zone=zone,
            cron=cron,
            at=at,
            mode=mode,
            target=target,
        )

    async def update(
        self,
        *,
        owner_id: str,
        run_id: str,
        schedule_id: str,
        name: str | None,
        prompt: str | None,
        workspaces: Sequence[str] | None,
        mode: str | None,
        zone: str | None,
        cron: str | None,
        at: datetime | None,
        paused: bool | None,
    ) -> ScheduleRecord:
        self._check()
        self.updated.append({"zone": zone, "cron": cron, "at": at, "paused": paused})
        owner, record = self.records[schedule_id]
        assert owner == owner_id
        if paused is not None:
            record = replace(
                record,
                state="paused" if paused else "active",
                pause_reason=None,
                next_fire_at=None if paused else _NEXT,
            )
        record = replace(record, name=name or record.name)
        self.records[schedule_id] = (owner, record)
        return record

    async def delete(self, *, owner_id: str, run_id: str, schedule_id: str) -> ScheduleRecord:
        self._check()
        owner, record = self.records.pop(schedule_id)
        assert owner == owner_id
        return record


def _tools(
    port: FakePort, *, workspaces: tuple[str, ...] = ("alpha", "beta")
) -> dict[str, AgentTool]:
    host = ScheduleHost(service=port, owner_id=_OWNER, run_id=_RUN, workspaces=workspaces)
    return {tool.name: tool for tool in schedule_tools(host=host)}


async def _call(tool: AgentTool, args: BaseModel) -> ToolResult:
    return await tool.execute(args, tool_runtime(tool_name=tool.name))


def test_the_schedule_tools_are_declared_with_their_replay_facts() -> None:
    tools = _tools(FakePort())

    assert tuple(tools) == SCHEDULE_TOOL_NAMES
    assert [tool.name for tool in schedule_tool_declarations()] == list(SCHEDULE_TOOL_NAMES)
    for name in ("create_schedule", "update_schedule", "delete_schedule"):
        assert tools[name].replay_policy == "never"
    assert tools["list_schedules"].replay_policy == "replayable"
    assert tools["list_schedules"].read_only
    for tool in tools.values():
        assert tool.description
    assert "no goal and no spend ceiling" in tools["create_schedule"].description
    assert "Scheduled row in the sidebar" in tools["create_schedule"].description
    assert "at most one" in tools["create_schedule"].description


def test_no_tool_takes_a_goal_a_spend_ceiling_or_an_owner() -> None:
    """The Agent cannot name what only the owner decides, or whose Schedule it is."""

    def names(schema: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(schema, dict):
            found |= set(schema.get("properties", {}))
            for value in schema.values():
                found |= names(value)
        elif isinstance(schema, list):
            for value in schema:
                found |= names(value)
        return found

    declared = set()
    for declaration in schedule_tool_declarations():
        declared |= names(declaration.definition.parameters)

    assert {"cron", "at", "zone", "paused", "schedule_id"} <= declared
    for word in ("goal", "token", "ceiling", "owner", "principal", "user"):
        assert not [name for name in declared if word in name], word


async def test_create_says_what_was_saved_in_plain_words() -> None:
    port = FakePort()

    result = await _call(
        _tools(port)["create_schedule"],
        CreateScheduleArgs(
            name="Weekday digest", prompt="Summarise the new filings.", when=_WEEKDAYS_AT_NINE
        ),
    )

    assert not result.is_error
    text = result.text_content
    assert "Weekday digest" in text
    assert "every weekday (Monday to Friday) at 09:00, Europe/Stockholm time" in text
    assert "Mon 2026-11-30 09:00 Europe/Stockholm (2026-11-30 08:00 UTC)" in text
    assert "new conversation" in text and "research mode" in text
    assert "Scheduled list" in text
    # The Run's own workspaces are the default; nothing the model passed named an owner.
    assert port.created[0]["workspaces"] == ("alpha", "beta")
    assert port.records["id-1"][0] == _OWNER


async def test_a_single_time_without_an_offset_is_read_on_the_zone_clock() -> None:
    port = FakePort()

    result = await _call(
        _tools(port)["create_schedule"],
        CreateScheduleArgs(
            name="Filing check",
            prompt="Check the filing.",
            when=When(at="2026-12-01T09:00:00", zone="Europe/Stockholm"),
        ),
    )

    assert not result.is_error
    assert port.created[0]["at"] == datetime(2026, 12, 1, 8, 0, tzinfo=UTC)
    assert port.created[0]["cron"] is None
    assert "once, on Tue 2026-12-01 09:00 Europe/Stockholm" in result.text_content


async def test_a_schedule_may_narrow_the_workspaces_but_not_widen_them() -> None:
    port = FakePort()
    create = _tools(port)["create_schedule"]

    widened = await _call(
        create,
        CreateScheduleArgs(name="n", prompt="p", when=_WEEKDAYS_AT_NINE, workspaces=["gamma"]),
    )
    narrowed = await _call(
        create,
        CreateScheduleArgs(name="n", prompt="p", when=_WEEKDAYS_AT_NINE, workspaces=["beta"]),
    )

    assert widened.is_error
    assert "gamma is not a workspace this conversation reads" in widened.text_content
    assert not narrowed.is_error
    assert [item["workspaces"] for item in port.created] == [("beta",)]


@pytest.mark.parametrize(
    ("when", "needle"),
    [
        (When(cron="0 9 * * *"), "`when.zone` is required"),
        (When(zone="Europe/Stockholm"), "`when.cron` or `when.at`"),
        (When(at="next tuesday", zone="Europe/Stockholm"), "not an ISO 8601"),
        (When(at="2026-12-01T09:00:00", zone="Not/AZone"), "is not an IANA time zone"),
    ],
)
async def test_a_cadence_the_model_left_out_or_got_wrong_is_a_tool_error(
    when: When, needle: str
) -> None:
    port = FakePort()

    result = await _call(
        _tools(port)["create_schedule"], CreateScheduleArgs(name="n", prompt="p", when=when)
    )

    assert result.is_error
    assert needle in result.text_content
    assert (result.details or {})["schedule_error"] == SCHEDULE_INVALID
    assert port.created == []


def test_a_cadence_names_a_recurrence_or_a_single_time_never_both() -> None:
    with pytest.raises(ValidationError, match="not both"):
        When(cron="0 9 * * *", at="2026-12-01T09:00:00", zone="Europe/Stockholm")


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (SCHEDULE_INVALID, "slots may be at least 60 minutes apart"),
        (SCHEDULE_LIMIT, "an owner may hold at most 20 active Schedules"),
    ],
)
async def test_what_the_service_refuses_comes_back_as_an_error_the_model_can_act_on(
    code: str, message: str
) -> None:
    port = FakePort()
    port.refusal = ScheduleRefusal(code, message)

    result = await _call(
        _tools(port)["create_schedule"],
        CreateScheduleArgs(name="n", prompt="p", when=_WEEKDAYS_AT_NINE),
    )

    assert result.is_error
    assert f"schedule refused ({code}): {message}" in result.text_content
    assert (result.details or {})["schedule_error"] == code


async def test_list_shows_the_callers_own_schedules_only() -> None:
    port = FakePort()
    port.seed(_OWNER, "mine", name="Mine")
    port.seed("owner-2", "theirs", name="Theirs")
    tools = _tools(port)

    listed = await _call(tools["list_schedules"], ListSchedulesArgs())

    assert "mine" in listed.text_content and "Mine" in listed.text_content
    assert "Theirs" not in listed.text_content
    assert "every Monday at 08:00" in listed.text_content
    port.records.pop("mine")
    assert (await _call(tools["list_schedules"], ListSchedulesArgs())).text_content == (
        "the user has no Schedules"
    )


async def test_update_pauses_and_resumes_and_says_when_it_next_runs() -> None:
    port = FakePort()
    port.seed(_OWNER, "mine")
    update = _tools(port)["update_schedule"]

    paused = await _call(update, UpdateScheduleArgs(schedule_id="mine", paused=True))
    resumed = await _call(update, UpdateScheduleArgs(schedule_id="mine", paused=False))

    assert "Next run: none while it is paused" in paused.text_content
    assert not resumed.is_error
    assert "Next run: Mon 2026-11-30 09:00 Europe/Stockholm" in resumed.text_content
    assert port.records["mine"][1].state == "active"


async def test_an_edit_that_names_a_bare_time_leaves_it_on_the_schedules_own_zone() -> None:
    port = FakePort()
    port.seed(_OWNER, "mine")
    update = _tools(port)["update_schedule"]

    bare = await _call(
        update, UpdateScheduleArgs(schedule_id="mine", when=When(at="2026-12-01T09:00:00"))
    )
    named = await _call(
        update,
        UpdateScheduleArgs(
            schedule_id="mine", when=When(at="2026-12-01T09:00:00", zone="Europe/Stockholm")
        ),
    )

    assert not bare.is_error and not named.is_error
    # With no zone named the time reaches the Application naive, and it reads it on its own zone.
    assert port.updated[0]["at"] == datetime(2026, 12, 1, 9, 0)
    assert port.updated[1]["at"] == datetime(2026, 12, 1, 8, 0, tzinfo=UTC)


async def test_an_update_that_names_no_change_is_refused() -> None:
    port = FakePort()
    port.seed(_OWNER, "mine")
    update = _tools(port)["update_schedule"]

    for args in (
        UpdateScheduleArgs(schedule_id="mine"),
        UpdateScheduleArgs(schedule_id="mine", when=When()),
    ):
        result = await _call(update, args)
        assert result.is_error
        assert "name at least one change" in result.text_content
    assert port.updated == []


async def test_a_turn_makes_one_schedule_and_a_failed_attempt_does_not_use_it_up() -> None:
    port = FakePort()
    create = _tools(port)["create_schedule"]
    good = CreateScheduleArgs(name="Digest", prompt="p", when=_WEEKDAYS_AT_NINE)

    port.refusal = ScheduleRefusal(SCHEDULE_INVALID, "slots may be at least 60 minutes apart")
    failed = await _call(create, good)
    port.refusal = None
    first = await _call(create, good)
    second = await _call(create, good)

    assert failed.is_error and not first.is_error
    assert second.is_error
    assert "already made a Schedule" in second.text_content
    assert "ask the user to confirm another" in second.text_content
    assert (second.details or {})["schedule_error"] == SCHEDULE_LIMIT
    assert len(port.created) == 1
    # Another turn has its own host, so it makes its own.
    assert not (await _call(_tools(port)["create_schedule"], good)).is_error


async def test_delete_names_what_it_removed_and_a_missing_id_is_an_error() -> None:
    port = FakePort()
    port.seed(_OWNER, "mine")
    delete = _tools(port)["delete_schedule"]

    deleted = await _call(delete, DeleteScheduleArgs(schedule_id="mine"))
    assert 'Deleted the Schedule "Weekly review"' in deleted.text_content
    assert "mine" not in port.records

    port.refusal = ScheduleRefusal(SCHEDULE_NOT_FOUND, "the user has no Schedule with this id")
    missing = await _call(delete, DeleteScheduleArgs(schedule_id="mine"))
    assert missing.is_error
    assert (missing.details or {})["schedule_error"] == SCHEDULE_NOT_FOUND


@pytest.mark.parametrize(
    ("cron", "words"),
    [
        ("0 9 * * *", "every day at 09:00"),
        ("30 7 * * 1-5", "every weekday (Monday to Friday) at 07:30"),
        ("0 8 * * 1,4", "every Monday and Thursday at 08:00"),
        ("0 8 * * 0,7", "every Sunday at 08:00"),
        ("15 6 1 * *", "on day 1 of every month at 06:15"),
        ("30 * * * *", "every hour at minute 30"),
        ("0 9 1 */3 *", 'cron "0 9 1 */3 *", read in Europe/Stockholm'),
    ],
)
def test_the_common_recurrences_have_words_and_the_rest_keep_their_cron(
    cron: str, words: str
) -> None:
    record = FakePort().seed(_OWNER, "x", cron=cron)

    assert words in describe_cadence(record)


def test_only_the_kinds_of_turn_a_person_starts_are_offered_the_tools() -> None:
    """An allowlist: a kind nobody named is not offered them, a future one included."""
    for kind in (None, "follow_up", "fork"):
        assert schedule_tools_offered(kind), kind
    for kind in ("scheduled", "goal_round", "retry", ""):
        assert not schedule_tools_offered(kind), kind


def _compose(*, child: bool, narrow: tuple[str, ...] | None = None) -> list[AgentTool]:
    host = ScheduleHost(service=FakePort(), owner_id=_OWNER, run_id=_RUN, workspaces=("alpha",))
    return compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=AsyncMock(),
        search_web=None,
        register_web_source=None,
        injected_tools=[],
        schedule_host=host,
        child=child,
        tool_names=narrow,
        search_toolchain=SearchToolchain(),
    )


def test_a_child_never_holds_the_schedule_tools_even_when_it_asks_for_them() -> None:
    assert set(SCHEDULE_TOOL_NAMES) <= CHILD_FORBIDDEN_TOOLS
    parent = {tool.name for tool in _compose(child=False)}
    child = {tool.name for tool in _compose(child=True, narrow=SCHEDULE_TOOL_NAMES)}

    assert set(SCHEDULE_TOOL_NAMES) <= parent
    assert child & set(SCHEDULE_TOOL_NAMES) == set()
    # Without a host no Run offers them, whoever it is.
    plain = {tool.name for tool in research_tool_declarations(schedules=False)}
    assert plain & set(SCHEDULE_TOOL_NAMES) == set()


def _declared(executor: AnswerExecutor, *, schedules: bool) -> set[str]:
    declared = executor.research_tool_declarations(
        web_search=False,
        memory=False,
        agent_may_register=False,
        model_guidance="",
        injected=(),
        schedules=schedules,
    )
    return {tool.name for tool in declared}


@pytest.mark.parametrize("schedules", [True, False])
@pytest.mark.parametrize("service", [FakePort(), None])
def test_acceptance_declares_the_tools_it_was_told_to_whatever_the_process_holds(
    service: FakePort | None, schedules: bool
) -> None:
    held = _declared(_executor(schedule_service=service), schedules=schedules) & set(
        SCHEDULE_TOOL_NAMES
    )

    assert held == (set(SCHEDULE_TOOL_NAMES) if schedules else set())


def _run_session(*, accepted_with_tools: bool) -> Any:
    prepared = {SCHEDULE_TOOLS_PIN: accepted_with_tools}
    return SimpleNamespace(prepared_input=prepared, owner_id=_OWNER, run_id=_RUN)


@pytest.mark.parametrize("accepted_with_tools", [True, False])
@pytest.mark.parametrize("executing_has_service", [True, False])
def test_a_run_is_composed_with_the_tools_it_was_accepted_with_in_any_process(
    accepted_with_tools: bool, executing_has_service: bool
) -> None:
    """The API accepts, and an MCP or reader process may claim: the pin decides, not its flags."""
    pinned = AgentRunPlan.from_tools(
        research_tool_declarations(schedules=accepted_with_tools),
        model_role="query",
        context_policy_revision="r",
    )
    executor = _executor(schedule_service=FakePort() if executing_has_service else None)
    host = executor._schedule_host(
        _run_session(accepted_with_tools=accepted_with_tools),
        cast(Any, SimpleNamespace(workspaces=("alpha",))),
    )
    tools = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=AsyncMock(),
        search_web=None,
        register_web_source=None,
        injected_tools=[],
        schedule_host=host,
        search_toolchain=SearchToolchain(),
    )

    AnswerExecutor.validate_pinned_agent_run_plan(
        cast(Any, SimpleNamespace(agent_run_plan=pinned, context_policy_revision="r")), tools
    )

    assert (host is not None) == accepted_with_tools


async def test_a_process_with_no_schedule_service_answers_the_pinned_tools_as_unavailable() -> None:
    host = _executor()._schedule_host(
        _run_session(accepted_with_tools=True), cast(Any, SimpleNamespace(workspaces=()))
    )
    assert host is not None

    result = await _call(
        {tool.name: tool for tool in schedule_tools(host=host)}["list_schedules"],
        ListSchedulesArgs(),
    )

    assert result.is_error
    assert (result.details or {})["schedule_error"] == SCHEDULE_UNAVAILABLE
