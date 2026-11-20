# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The marks a Run's timeline states: the four mechanical decisions and the order they carry.

The Run's own read judges a step against the whole Run before any page reaches a reader; these
tests fix the rules themselves, away from a database or a page, and the read's own tests fix the
step rows that feed them.
"""

import datetime

from dlightrag.application.answer_runs.timeline import RunTimelineStep, decision_marks

_T0 = datetime.datetime(2026, 8, 12, 12, 0, 0, tzinfo=datetime.UTC)


def _step(
    *,
    second: float,
    kind: str = "tool",
    agent: str = "main",
    sequence: int | None = None,
    name: str | None = "read",
    subject: str | None = None,
    failed: bool = False,
    degraded: bool = False,
) -> RunTimelineStep:
    """One step of the whole Run, as the mark rules read it."""
    return RunTimelineStep(
        agent=agent,
        agent_label=agent,
        agent_order=0 if agent == "main" else 1,
        session_id="00000000-0000-7000-8000-000000000001" if agent == "main" else agent,
        sequence=int(second * 10) + 1 if sequence is None else sequence,
        instant=_T0 + datetime.timedelta(seconds=second),
        at=(_T0 + datetime.timedelta(seconds=second)).isoformat(),
        kind=kind,
        name=None if kind != "tool" else name,
        subject=None if kind != "tool" else subject,
        failed=failed,
        degraded=degraded,
    )


def _child_answer(*, second: float, sequence: int | None = None) -> RunTimelineStep:
    return _step(second=second, kind="answer", agent="child-a", sequence=sequence)


def test_a_failed_degraded_repeated_call_and_a_redirect_are_the_four_decisions() -> None:
    steps = [
        _step(second=0, subject="a.md"),
        _step(second=1, subject="a.md"),
        _step(second=2, name="search_web", subject="penalty cap", degraded=True),
        _step(second=3, name="bash", subject="npm test", failed=True),
        _child_answer(second=4),
        _step(second=5, subject="b.md"),
    ]

    marks = decision_marks(steps)

    assert marks[(steps[0].agent, steps[0].sequence)] == ("repeated",)
    assert marks[(steps[1].agent, steps[1].sequence)] == ("repeated",)
    assert marks[(steps[2].agent, steps[2].sequence)] == ("degraded",)
    assert marks[(steps[3].agent, steps[3].sequence)] == ("failed",)
    assert (steps[4].agent, steps[4].sequence) not in marks, "a child answering carries no badge"
    assert marks[(steps[5].agent, steps[5].sequence)] == ("redirect",)


def test_one_step_can_carry_more_than_one_decision_in_a_stated_order() -> None:
    steps = [
        _step(second=0, name="search_web", subject="query", degraded=True),
        _step(
            second=1,
            name="search_web",
            subject="query",
            degraded=True,
            failed=True,
        ),
    ]

    marks = decision_marks(steps)

    assert marks[(steps[1].agent, steps[1].sequence)] == (
        "failed",
        "degraded",
        "repeated",
    )


def test_a_call_repeats_only_when_the_same_tool_acts_on_the_same_object() -> None:
    steps = [
        _step(second=0, subject="a.md"),
        _step(second=1, name="grep", subject="a.md"),
        _step(second=2, subject=None),
        _step(second=3, subject=None),
    ]

    assert decision_marks(steps) == {}


def test_a_redirect_needs_a_main_subject_to_change_from_and_a_new_one_to_change_to() -> None:
    child_answer = _child_answer(second=1)
    first = _step(second=0, subject="a.md")
    same = _step(second=2, subject="a.md")
    unseen = _step(second=2, subject="b.md")
    said = _step(second=1.5, kind="say", agent="child-a")

    # The first main call has nothing to be a change from.
    assert decision_marks([child_answer, unseen]) == {}
    # Looking at the same object is a repeat, never a change.
    assert decision_marks([first, child_answer, same])[(same.agent, same.sequence)] == ("repeated",)
    # A child step between the answer and the next main call does not use the signal up.
    assert decision_marks([first, child_answer, said, unseen])[(unseen.agent, unseen.sequence)] == (
        "redirect",
    )
    # A call with no object cannot redirect.
    assert decision_marks([first, child_answer, _step(second=2, subject=None)]) == {}


def test_the_main_agents_own_answer_does_not_make_a_redirect() -> None:
    first = _step(second=0, subject="a.md")
    answer = _step(second=1, kind="answer")
    next_call = _step(second=2, subject="b.md")

    marks = decision_marks([first, answer, next_call])

    assert (next_call.agent, next_call.sequence) not in marks
