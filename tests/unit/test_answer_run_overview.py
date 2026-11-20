# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Run overview's own rules: whole-Run totals, rollups, and the findings a panel lists.

These tests stay away from a database and a route: they fix how the merged index's steps and marks
become the numbers and rows the Web overview states — what a rollup covers, which steps become
findings, how the findings list is capped while its total is not, and what one finding's excerpt is
allowed to be.
"""

import datetime
from typing import Any

from dlightrag.application.answer_runs.overview import (
    FINDING_EXCERPT_MAX_CHARS,
    FINDINGS_MAX,
    MARK_KINDS,
    agent_rollups,
    excerpt_of,
    findings_of,
    mark_counts,
    overview_totals,
    run_usage,
)
from dlightrag.application.answer_runs.timeline import AgentIdentity, RunTimelineStep

_T0 = datetime.datetime(2026, 8, 12, 12, 0, 0, tzinfo=datetime.UTC)
_MAIN_SESSION = "00000000-0000-7000-8000-000000000001"


def _at(seconds: float) -> str:
    return (_T0 + datetime.timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _step(
    *,
    second: float,
    kind: str = "tool",
    agent: str = "main",
    agent_label: str = "main",
    agent_order: int = 0,
    session: str = _MAIN_SESSION,
    sequence: int = 1,
    name: str | None = "read",
    subject: str | None = None,
    duration_ms: int | None = None,
    failed: bool = False,
) -> RunTimelineStep:
    """One step of the Run, as the merged index and the read state it."""
    tool = kind == "tool"
    return RunTimelineStep(
        agent=agent,
        agent_label=agent_label,
        agent_order=agent_order,
        session_id=session,
        sequence=sequence,
        instant=_T0 + datetime.timedelta(seconds=second),
        at=_at(second),
        kind=kind,
        name=name if tool else None,
        subject=subject if tool else None,
        outcome=("failed" if failed else "succeeded") if tool else None,
        duration_ms=duration_ms if tool else None,
        failed=failed if tool else False,
    )


def _agents() -> tuple[AgentIdentity, ...]:
    return (
        AgentIdentity(key="main", label="main", session_id=None, order=0),
        AgentIdentity(key="child-a", label="Look up filings", session_id="child-a", order=1),
        AgentIdentity(key="child-b", label="Draft summary", session_id="child-b", order=2),
    )


def test_mark_counts_state_every_mark_in_its_stated_order() -> None:
    marks = {
        ("main", 1): ("failed", "repeated"),
        ("main", 2): ("failed",),
        ("child-a", 1): ("redirect",),
    }

    counts = mark_counts(marks)

    assert list(counts) == list(MARK_KINDS)
    assert counts == {"failed": 2, "degraded": 0, "repeated": 1, "redirect": 1}


def test_totals_count_the_whole_run_by_kind_and_its_tool_time() -> None:
    steps = [
        _step(second=0, kind="thinking", sequence=1),
        _step(second=1, kind="tool", sequence=2, duration_ms=500),
        # A Tool step the merge could not clock contributes no milliseconds rather than zero.
        _step(second=2, kind="tool", sequence=3),
        _step(second=3, kind="say", sequence=4),
        _step(second=4, kind="answer", sequence=5),
        _step(second=5, kind="instruction", sequence=6),
        _step(second=6, kind="delivery", sequence=7),
    ]
    marks = {
        ("main", 2): ("failed",),
        ("main", 3): ("repeated", "failed"),
    }

    totals = overview_totals(steps, marks)

    assert totals.steps == 7
    assert totals.kinds == {
        "thinking": 1,
        "tool": 2,
        "say": 1,
        "answer": 1,
        "instruction": 1,
        "delivery": 1,
    }
    # A thought is a step but never a decision: the count is the marked steps, not the marks.
    assert totals.decisions == 2
    assert totals.marks == {"failed": 2, "degraded": 0, "repeated": 1, "redirect": 0}
    assert totals.tool_time_ms == 500


def test_agent_rollups_cover_each_named_agent_in_roster_order() -> None:
    steps = [
        _step(second=1, sequence=2, subject="a.md", duration_ms=1500),
        _step(second=2, sequence=3, kind="say", name=None),
        _step(
            second=3,
            sequence=1,
            kind="answer",
            agent="child-a",
            agent_label="Look up filings",
            agent_order=1,
            session="child-a",
        ),
        # A Session the roster page does not name has steps in the merge, but no page of its own.
        _step(second=4, sequence=1, agent="ghost", session="ghost"),
    ]
    marks = {
        ("main", 2): ("failed",),
        ("main", 3): ("repeated", "failed"),
        ("ghost", 1): ("failed",),
    }

    rollups = agent_rollups(steps, marks, _agents())

    assert [(rollup.agent, rollup.label, rollup.order) for rollup in rollups] == [
        ("main", "main", 0),
        ("child-a", "Look up filings", 1),
        ("child-b", "Draft summary", 2),
    ]
    main, child, empty = rollups
    assert main.steps == 2
    assert main.tool_calls == 1
    assert main.marks == {"failed": 2, "degraded": 0, "repeated": 1, "redirect": 0}
    assert main.tool_time_ms == 1500
    assert (main.first_at, main.last_at) == (_at(1), _at(2))
    assert child.steps == 1
    assert child.tool_calls == 0
    assert child.marks == {"failed": 0, "degraded": 0, "repeated": 0, "redirect": 0}
    assert child.tool_time_ms == 0
    assert (child.first_at, child.last_at) == (_at(3), _at(3))
    # An agent the roster names but the Run holds no step for still gets its row, stated as empty.
    assert (empty.steps, empty.tool_calls, empty.tool_time_ms, empty.first_at, empty.last_at) == (
        0,
        0,
        0,
        None,
        None,
    )


def test_findings_are_marked_steps_and_delivery_steps_in_merge_order() -> None:
    steps = [
        _step(second=0, kind="thinking", sequence=1),
        _step(second=1, kind="say", sequence=1, name=None),
        _step(second=2, sequence=2, subject="a.md", failed=True),
        # An unmarked Tool step is not a finding, and neither is the answer the Run ended with.
        _step(second=3, sequence=3, subject="b.md"),
        _step(second=4, kind="delivery", sequence=4, name=None),
        _step(second=5, kind="answer", sequence=5, name=None),
    ]
    marks = {
        ("main", 1): ("repeated",),
        ("main", 2): ("failed", "repeated"),
    }

    findings, total = findings_of(steps, marks)

    assert total == 3
    # The thought shares its sequence with the marked say step: the marks name the step beside it,
    # and the thought never appears as a finding of its own.
    assert [finding.kind for finding in findings] == ["say", "tool", "delivery"]
    thought_shared = findings[0]
    assert (thought_shared.sequence, thought_shared.marks) == (1, ("repeated",))
    first = findings[1]
    assert (first.agent, first.sequence, first.at) == ("main", 2, _at(2))
    assert (first.name, first.subject, first.outcome) == ("read", "a.md", "failed")
    assert first.marks == ("failed", "repeated")
    assert first.session_id == _MAIN_SESSION
    # The excerpt is the one text the read fills after the gate; this module states no text.
    assert [finding.excerpt for finding in findings] == ["", "", ""]
    second = findings[2]
    assert (second.sequence, second.marks) == (4, ())


def test_findings_are_capped_and_the_total_states_the_rest() -> None:
    steps = [
        _step(second=index, sequence=index + 1, failed=True) for index in range(FINDINGS_MAX + 5)
    ]
    marks = {(step.agent, step.sequence): ("failed",) for step in steps}

    findings, total = findings_of(steps, marks)

    assert len(findings) == FINDINGS_MAX
    assert total == FINDINGS_MAX + 5
    # The cap keeps the oldest findings the merge reaches, not a severity-sorted subset: display
    # order is the panel's, and the wire states the merge's.
    assert [finding.sequence for finding in findings] == list(range(1, FINDINGS_MAX + 1))

    narrowed, narrowed_total = findings_of(steps, marks, 2)

    assert [finding.sequence for finding in narrowed] == [1, 2]
    assert narrowed_total == FINDINGS_MAX + 5


def test_an_excerpt_is_the_first_stripped_line_bounded() -> None:
    assert excerpt_of("  the first line  \n second line") == "the first line"
    assert excerpt_of("x" * 500) == "x" * FINDING_EXCERPT_MAX_CHARS
    assert excerpt_of("") == ""
    assert excerpt_of(None) == ""
    assert excerpt_of({"content": "not text"}) == ""


def test_usage_is_the_runs_own_record_or_none() -> None:
    triple: dict[str, Any] = {
        "usage_details": {"total_tokens": 15_000},
        "child_usage_details": {"total_tokens": 9_000},
        "inclusive_usage_details": {"total_tokens": 24_000},
    }

    assert run_usage(None) is None
    assert run_usage({}) is None
    # A record that states an empty usage states no usage: it is unknown, not free.
    assert run_usage({"usage": {}}) is None
    assert run_usage({"usage": triple}) == triple
    # A Fast Run's record may hold only the root counters; whatever it states is passed through.
    assert run_usage({"usage": {"usage_details": {"total_tokens": 5}}}) == {
        "usage_details": {"total_tokens": 5}
    }
