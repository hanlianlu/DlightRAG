# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Run timeline read's own rules: one merge, one page, one projection for marks and page.

These tests stay away from a database and a route: they fix how stored step rows and the roster
become the Run's ordered steps, which of them a page may open, and how the page's words are put
back beside the steps they belong to.
"""

import datetime

from dlightrag.application.answer_runs.timeline import (
    RunTimelinePosition,
    agent_identities,
    decision_marks,
    merge_steps,
    page_of,
    parse_instant,
    with_page_content,
)

_T0 = datetime.datetime(2026, 8, 12, 12, 0, 0, tzinfo=datetime.UTC)


def _at(seconds: float) -> str:
    return (_T0 + datetime.timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _signal(
    *,
    sequence: int,
    second: float,
    entry_type: str,
    session: str = "main-session",
    is_main: bool = True,
    tool_calls: int = 0,
    has_content: bool = False,
    has_reasoning: bool = False,
    call_id: str | None = None,
    tool_name: str | None = None,
    failed: bool = False,
    degraded: bool = False,
    tool_call_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "session_id": session,
        "is_main": is_main,
        "sequence": sequence,
        "at": _at(second),
        "entry_type": entry_type,
        "tool_calls": tool_calls,
        "has_content": has_content,
        "has_reasoning": has_reasoning,
        "call_id": call_id,
        "tool_name": tool_name,
        "failed": failed,
        "degraded": degraded,
        "tool_call_ids": tool_call_ids,
    }


_CHILDREN = (
    {"child_session_id": "child-a", "objective": "Look up filings"},
    {"child_session_id": "child-b", "objective": "Draft summary"},
)


def test_agents_are_the_run_itself_then_the_roster_order() -> None:
    agents = agent_identities(_CHILDREN)

    assert [(agent.key, agent.label, agent.order) for agent in agents] == [
        ("main", "main", 0),
        ("child-a", "Look up filings", 1),
        ("child-b", "Draft summary", 2),
    ]


def test_the_merge_orders_agents_steps_on_the_one_clock_and_times_a_tool_by_its_call() -> None:
    rows = [
        # The invocation record states no step; it is the clock the result is timed against.
        _signal(
            sequence=1,
            second=0,
            entry_type="assistant_message",
            tool_calls=1,
            tool_call_ids=("call-1",),
            is_main=False,
            session="child-a",
        ),
        _signal(
            sequence=2,
            second=1.5,
            entry_type="tool_result",
            session="child-a",
            is_main=False,
            call_id="call-1",
            tool_name="read",
        ),
        _signal(
            sequence=1,
            second=0.5,
            entry_type="assistant_message",
            tool_calls=1,
            has_content=True,
        ),
        _signal(sequence=2, second=2, entry_type="assistant_message"),
    ]

    steps = merge_steps(
        agents=agent_identities(_CHILDREN),
        rows=rows,
        subjects={"call-1": "notes/a.md"},
    )

    assert [(step.agent, step.sequence, step.kind) for step in steps] == [
        ("main", 1, "say"),
        ("child-a", 2, "tool"),
        ("main", 2, "answer"),
    ]
    tool = steps[1]
    assert tool.duration_ms == 1500
    assert tool.subject == "notes/a.md"
    assert tool.name == "read"
    assert tool.agent_label == "Look up filings"
    assert tool.agent_order == 1
    assert tool.content == "", "content is read for the page, not the Run"


def test_a_session_past_the_roster_is_still_in_the_merge_for_its_marks() -> None:
    rows = [
        _signal(
            sequence=1,
            second=0,
            entry_type="assistant_message",
            tool_calls=1,
            has_content=True,
        ),
        _signal(
            sequence=1,
            second=1,
            entry_type="tool_result",
            session="child-z",
            is_main=False,
            call_id="call-z",
            tool_name="bash",
        ),
    ]

    steps = merge_steps(agents=agent_identities(()), rows=rows, subjects={})

    assert [(step.agent, step.kind) for step in steps] == [("main", "say"), ("child-z", "tool")]
    assert steps[1].agent_order == 1


def test_the_page_holds_the_newest_named_steps_older_than_the_cursor() -> None:
    rows = [
        _signal(
            sequence=sequence,
            second=sequence,
            entry_type="tool_result",
            call_id=f"call-{sequence}",
            tool_name="read",
        )
        for sequence in range(1, 6)
    ]
    agents = agent_identities(())
    steps = merge_steps(agents=agents, rows=rows, subjects={})
    boundary = RunTimelinePosition(
        instant=steps[3].instant,
        agent=steps[3].agent,
        sequence=steps[3].sequence,
        kind=steps[3].kind,
    )

    page, has_older = page_of(steps, limit=2, before=boundary, agents=agents)

    assert [step.sequence for step in page] == [2, 3]
    assert has_older, "the Run holds steps older than the page"


def test_a_page_never_opens_a_session_past_the_roster() -> None:
    rows = [
        _signal(
            sequence=1,
            second=0,
            entry_type="assistant_message",
            tool_calls=1,
            has_content=True,
        ),
        _signal(
            sequence=1,
            second=1,
            entry_type="tool_result",
            session="child-z",
            is_main=False,
            call_id="call-z",
            tool_name="bash",
        ),
    ]
    agents = agent_identities(())
    steps = merge_steps(agents=agents, rows=rows, subjects={})

    page, has_older = page_of(steps, limit=10, before=None, agents=agents)

    assert [step.agent for step in page] == ["main"]
    assert has_older is False


def test_page_content_is_put_back_only_where_the_page_named_it() -> None:
    rows = [
        _signal(
            sequence=sequence,
            second=sequence,
            entry_type="tool_result",
            call_id=f"call-{sequence}",
            tool_name="read",
        )
        for sequence in range(1, 3)
    ]
    steps = merge_steps(
        agents=agent_identities(()),
        rows=rows,
        subjects={},
    )

    page = with_page_content(
        steps,
        {("main-session", 2): {"content": "two files\nsecond line"}},
    )

    assert page[0].content == ""
    assert page[1].content == "two files\nsecond line"


def test_an_instant_is_read_from_the_wire_in_utc_or_none() -> None:
    assert parse_instant("2026-08-12T12:00:00Z") == _T0
    assert parse_instant("2026-08-12T14:00:00+02:00") == _T0
    assert parse_instant("2026-08-12T12:00:00") == _T0
    assert parse_instant("") is None
    assert parse_instant("not a time") is None
    assert parse_instant(None) is None


def test_equal_instants_are_ordered_by_agent_then_sequence_and_pages_keep_them() -> None:
    rows = [
        _signal(
            sequence=2,
            second=0,
            entry_type="tool_result",
            session="child-a",
            is_main=False,
            call_id="call-child",
            tool_name="read",
        ),
        _signal(sequence=4, second=0, entry_type="assistant_message", has_content=True),
        _signal(
            sequence=1,
            second=0,
            entry_type="tool_result",
            session="child-b",
            is_main=False,
            call_id="call-b",
            tool_name="read",
        ),
    ]
    agents = agent_identities(_CHILDREN)
    steps = merge_steps(agents=agents, rows=rows, subjects={})

    # Same instant: the Run's own agent first, then the roster order, then sequence.
    assert [(step.agent, step.sequence) for step in steps] == [
        ("main", 4),
        ("child-a", 2),
        ("child-b", 1),
    ]
    # The cursor names the child-a step: the page before it in the current order, not in the
    # order the steps happened to arrive in.
    boundary = RunTimelinePosition(
        instant=steps[1].instant,
        agent=steps[1].agent,
        sequence=steps[1].sequence,
        kind=steps[1].kind,
    )
    page, has_older = page_of(steps, limit=10, before=boundary, agents=agents)

    assert [(step.agent, step.sequence) for step in page] == [("main", 4)]
    assert has_older is False


def test_an_entry_type_the_transcript_scope_does_not_admit_is_not_a_step() -> None:
    rows = [
        _signal(sequence=1, second=0, entry_type="compaction"),
        _signal(sequence=2, second=1, entry_type="assistant_message", has_content=True),
    ]

    steps = merge_steps(agents=agent_identities(()), rows=rows, subjects={})

    assert [step.sequence for step in steps] == [2]


def test_a_retried_call_id_is_timed_from_the_latest_call_before_the_result() -> None:
    rows = [
        _signal(
            sequence=1,
            second=1,
            entry_type="assistant_message",
            tool_calls=1,
            tool_call_ids=("call-1",),
        ),
        _signal(sequence=2, second=2, entry_type="tool_result", call_id="call-1", tool_name="read"),
        _signal(
            sequence=5,
            second=5,
            entry_type="assistant_message",
            tool_calls=1,
            tool_call_ids=("call-1",),
        ),
        _signal(sequence=6, second=6, entry_type="tool_result", call_id="call-1", tool_name="read"),
    ]

    steps = merge_steps(agents=agent_identities(()), rows=rows, subjects={})

    assert [step.duration_ms for step in steps] == [1000, 1000], (
        "each result is timed from the attempt that called it, not the first row seen"
    )


def test_user_and_control_entries_state_instructions() -> None:
    rows = [
        _signal(sequence=1, second=0, entry_type="user_message"),
        _signal(sequence=2, second=1, entry_type="control_message"),
        _signal(sequence=3, second=2, entry_type="assistant_message"),
    ]
    steps = merge_steps(agents=agent_identities(()), rows=rows, subjects={})

    assert [(step.kind, step.sequence) for step in steps] == [
        ("instruction", 1),
        ("instruction", 2),
        ("answer", 3),
    ]


def test_a_row_without_a_readable_instant_states_no_step() -> None:
    rows = [
        _signal(sequence=1, second=0, entry_type="assistant_message", has_content=True),
        {**_signal(sequence=2, second=1, entry_type="assistant_message"), "at": None},
        {**_signal(sequence=3, second=2, entry_type="assistant_message"), "at": "not a time"},
    ]

    steps = merge_steps(agents=agent_identities(()), rows=rows, subjects={})

    assert [step.sequence for step in steps] == [1]


def test_a_tool_call_id_list_must_be_a_list_not_a_string_of_characters() -> None:
    rows = [
        {
            **_signal(sequence=1, second=0, entry_type="assistant_message", tool_calls=1),
            "tool_call_ids": "call-1",
        },
        _signal(sequence=2, second=1, entry_type="tool_result", call_id="call-1", tool_name="read"),
    ]

    steps = merge_steps(agents=agent_identities(()), rows=rows, subjects={})

    assert len(steps) == 1
    assert steps[0].duration_ms is None


def test_a_page_filling_the_limit_exactly_has_no_older_page() -> None:
    rows = [
        _signal(
            sequence=sequence,
            second=sequence,
            entry_type="tool_result",
            call_id=f"call-{sequence}",
            tool_name="read",
        )
        for sequence in range(1, 4)
    ]
    agents = agent_identities(())
    steps = merge_steps(agents=agents, rows=rows, subjects={})

    page, has_older = page_of(steps, limit=3, before=None, agents=agents)

    assert len(page) == 3
    assert has_older is False


def test_a_delivery_warning_states_a_delivery_step() -> None:
    rows = [
        _signal(sequence=1, second=0, entry_type="delivery_warning"),
        _signal(sequence=2, second=1, entry_type="assistant_message"),
    ]
    steps = merge_steps(agents=agent_identities(()), rows=rows, subjects={})

    assert [(step.kind, step.sequence) for step in steps] == [("delivery", 1), ("answer", 2)]


def test_an_assistant_thought_states_a_thinking_step_before_its_turn() -> None:
    """The thought that produced a turn is its own step, carrying the turn's identity."""
    rows = [
        _signal(
            sequence=1,
            second=0,
            entry_type="assistant_message",
            has_reasoning=True,
            has_content=True,
            tool_calls=1,
            tool_call_ids=("call-1",),
        ),
        _signal(sequence=2, second=1, entry_type="tool_result", call_id="call-1", tool_name="read"),
    ]
    steps = merge_steps(agents=agent_identities(()), rows=rows, subjects={})

    assert [(step.kind, step.sequence) for step in steps] == [
        ("thinking", 1),
        ("say", 1),
        ("tool", 2),
    ]
    thought = steps[0]
    assert (thought.agent, thought.agent_label, thought.agent_order) == ("main", "main", 0)
    assert thought.session_id == "main-session"
    assert thought.instant == steps[1].instant, "the thought and its turn share the entry's clock"
    assert thought.name is None
    assert thought.subject is None
    assert thought.outcome is None
    assert thought.duration_ms is None
    assert thought.content == "", "the page fills the wording in"


def test_a_tool_selection_turn_with_reasoning_and_no_words_is_only_its_thought() -> None:
    """An invocation record is not a step; the thought that chose the Tool still is."""
    rows = [
        _signal(
            sequence=1,
            second=0,
            entry_type="assistant_message",
            has_reasoning=True,
            tool_calls=1,
            tool_call_ids=("call-1",),
        ),
        _signal(sequence=2, second=1, entry_type="tool_result", call_id="call-1", tool_name="read"),
    ]
    steps = merge_steps(agents=agent_identities(()), rows=rows, subjects={})

    assert [(step.kind, step.sequence) for step in steps] == [("thinking", 1), ("tool", 2)]


def test_an_assistant_turn_without_reasoning_states_no_thinking_step() -> None:
    rows = [_signal(sequence=1, second=0, entry_type="assistant_message", has_content=True)]

    steps = merge_steps(agents=agent_identities(()), rows=rows, subjects={})

    assert [(step.kind, step.sequence) for step in steps] == [("answer", 1)]


def test_a_child_agents_thought_is_a_step_and_the_totals_count_it() -> None:
    rows = [
        _signal(
            sequence=1,
            second=0,
            entry_type="assistant_message",
            session="child-a",
            is_main=False,
            has_reasoning=True,
            has_content=True,
            tool_calls=1,
        ),
    ]
    agents = agent_identities(_CHILDREN)

    steps = merge_steps(agents=agents, rows=rows, subjects={})

    assert [(step.agent, step.kind, step.sequence) for step in steps] == [
        ("child-a", "thinking", 1),
        ("child-a", "say", 1),
    ]


def test_a_cursor_that_names_a_kind_pages_from_that_step_not_its_twin() -> None:
    """A thought and its turn share an agent and sequence, so the cursor must name the kind."""
    rows = [
        _signal(
            sequence=1,
            second=0,
            entry_type="assistant_message",
            has_reasoning=True,
            has_content=True,
            tool_calls=1,
        ),
    ]
    agents = agent_identities(())
    steps = merge_steps(agents=agents, rows=rows, subjects={})
    assert [step.kind for step in steps] == ["thinking", "say"]

    thought = RunTimelinePosition(
        instant=steps[0].instant, agent="main", sequence=1, kind="thinking"
    )
    page, has_older = page_of(steps, limit=10, before=thought, agents=agents)
    assert page == () and has_older is False, "nothing is older than the thought"

    said = RunTimelinePosition(instant=steps[1].instant, agent="main", sequence=1, kind="say")
    page, has_older = page_of(steps, limit=10, before=said, agents=agents)
    assert [step.kind for step in page] == ["thinking"]
    assert has_older is False


def test_a_thought_is_a_step_but_never_a_decision_point() -> None:
    """Thinking is a spine row, not a decision: the step count counts it, the marks never do."""
    rows = [
        _signal(
            sequence=1,
            second=0,
            entry_type="assistant_message",
            has_reasoning=True,
            has_content=True,
        ),
    ]
    steps = merge_steps(agents=agent_identities(()), rows=rows, subjects={})

    assert [step.kind for step in steps] == ["thinking", "answer"]
    assert decision_marks(steps) == {}, "a thought states no mechanical decision"
