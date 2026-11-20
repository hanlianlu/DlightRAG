# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Run-level timeline read: every agent's steps merged into the one order a reader follows.

A Run's durable transcript is per Session, so no one Session answers "what happened, in order".
The store keeps one clock across them — the entry timestamp every step carries — and this module
turns the Run's stored entries into the whole Run the overview counts and the order the step search
pages: one merge, read once, so a panel's totals and a search row cannot disagree about a step.

The read is step-grain. A step is one line a reader can draw, not the message behind it: an
assistant turn that only names its Tool calls is an invocation record, and the Tool result that
follows states that call. Nothing here reads a Session's messages page by page, and nothing here
re-derives a field from the model's own record: the store read is already reader-facing, and the
service layer is what applies the reader content gate once.
"""

from __future__ import annotations

import datetime
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

MAIN_AGENT = "main"
"""The Run's own agent: the identity the Inspector already names, and what its steps carry."""

TIMELINE_KINDS: tuple[str, ...] = ("thinking", "tool", "say", "answer", "instruction", "delivery")
"""The step kinds the Run timeline states, and the order its summary counts them in.

A new kind joins this tuple and the Inspector's wire schema in one change; a total cannot silently
count a kind the page has no shape for, and a kind cannot reach the wire uncounted. ``thinking``
leads because a turn's thought is stated immediately before the turn it produced.
"""

_ROLE_BY_ENTRY_TYPE: Mapping[str, str] = {
    "user_message": "user",
    "control_message": "user",
    "delivery_warning": "delivery",
    "assistant_message": "assistant",
    "tool_result": "tool",
}
"""The reader role each stored entry type takes in the step rule.

These are the entry types the transcript scope admits. A new type joins here, the store's
entry-type constraint, and the Inspector's kind set in one change, once its writer exists: reading
one early would state a step no producer writes.
"""


def step_kinds(
    *,
    entry_type: str,
    has_content: bool,
    has_reasoning: bool,
    tool_calls: int,
) -> tuple[str, ...]:
    """The step kinds one stored entry states, in the order the timeline states them.

    The step rule is written once here: the merge draws every row through it, so the order a
    search pages and the whole Run the overview counts cannot disagree about a step. An Assistant
    entry whose stored reasoning is non-empty states a ``thinking`` step before the turn it
    produced; a turn that names no Tool calls is the ``answer``, one that names calls beside its
    words is a ``say``, and a turn that only names its calls is an invocation record with no step of
    its own.
    """
    role = _ROLE_BY_ENTRY_TYPE.get(entry_type)
    if role is None:
        return ()
    if role == "user":
        return ("instruction",)
    if role == "delivery":
        return ("delivery",)
    if role == "tool":
        return ("tool",)
    kinds: list[str] = []
    if has_reasoning:
        kinds.append("thinking")
    if tool_calls == 0:
        kinds.append("answer")
    elif has_content:
        kinds.append("say")
    return tuple(kinds)


def position_of(steps: Sequence[RunTimelineStep], position: RunTimelinePosition) -> int | None:
    """Where the step a cursor names stands, or None when the Run holds no such step.

    A thought and the turn it produced share one agent and sequence, so the kind is what tells
    them apart.
    """
    for index, step in enumerate(steps):
        if (
            step.agent == position.agent
            and step.sequence == position.sequence
            and step.kind == position.kind
        ):
            return index
    return None


def parse_instant(value: Any) -> datetime.datetime | None:
    """The instant one projected ``at`` states, in UTC, or None when it states none."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=datetime.UTC)
    return parsed.astimezone(datetime.UTC)


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    """One agent of the Run, in the order the roster states it."""

    key: str
    """The identity a step carries on the wire: ``main`` or the Child Session's id."""
    label: str
    """What the roster calls the agent: ``main`` or the Child's stored objective."""
    session_id: str | None
    """The Session the agent's entries live in, when the roster states it.

    The Run's own agent has none: its identity is ``main`` before any step names a Session, and a
    step carries the Session its own row names.
    """
    order: int
    """Where the roster puts the agent, which orders the merge between equal instants."""


@dataclass(frozen=True, slots=True)
class RunTimelinePosition:
    """Where a page ended: the step it stopped in front of.

    The step is named by its agent, its sequence, and — since a thought and the turn it produced
    share one sequence — its kind, so a roster change cannot move it.
    """

    instant: datetime.datetime
    agent: str
    sequence: int
    kind: str


@dataclass(frozen=True, slots=True)
class RunTimelineStep:
    """One step of the Run, read once for the whole Run and the page being served.

    The identity and decision fields are read for every step, because the marks and totals are the
    Run's rather than a page's; ``content`` is read only for the page being served, after the reader
    content gate has been applied.
    """

    agent: str
    agent_label: str
    agent_order: int
    session_id: str
    sequence: int
    instant: datetime.datetime
    at: str
    kind: str
    name: str | None = None
    subject: str | None = None
    outcome: str | None = None
    duration_ms: int | None = None
    failed: bool = False
    degraded: bool = False
    content: str = ""

    def key(self) -> tuple[datetime.datetime, int, int]:
        """This step's position in the one order the whole Run is read in.

        This is the merge's order, not the cursor's identity: a page cursor names a step by its
        agent, sequence and kind (see :class:`RunTimelinePosition`), so a roster change cannot move
        it. A thought and the turn it produced share all three coordinates but the kind, and the
        stable sort keeps the thought in front — which is why the cursor names the kind too.
        """
        return (self.instant, self.agent_order, self.sequence)


@dataclass(frozen=True, slots=True)
class RunStepSearch:
    """One page of the Run's steps whose messages hold a search's words.

    The search-only read the Run panel walks: the matches are the whole Run's, the page is the
    newest matches older than the cursor, and thinking steps never appear, because a search result
    is the message's actionable step rather than the reasoning behind it. The marks are the whole
    Run's too, so a row states a decision whose other half may be several pages away.
    """

    page: tuple[RunTimelineStep, ...]
    """The matching steps of this page, oldest first, with their page content."""
    has_older: bool
    """Whether the Run holds matching steps older than the page."""
    matched: int
    """How many of the Run's searchable steps hold the words; a Run count, not the page's."""
    marks: Mapping[tuple[str, int], tuple[str, ...]]
    """The whole Run's marks, keyed by ``(agent, sequence)``, for the page's rows."""
    terms: tuple[str, ...] = ()
    """The words the search actually read (what a client marks)."""
    match: str = "all"
    """Whether the words as written answered, or the looser reading of them did."""


def agent_identities(children: Sequence[Mapping[str, Any]]) -> tuple[AgentIdentity, ...]:
    """The Run's agents in roster order: its own agent first, then each Child as the roster states."""
    agents = [AgentIdentity(key=MAIN_AGENT, label=MAIN_AGENT, session_id=None, order=0)]
    for child in children:
        session_id = str(child.get("child_session_id") or "")
        if not session_id:
            continue
        agents.append(
            AgentIdentity(
                key=session_id,
                label=str(child.get("objective") or ""),
                session_id=session_id,
                order=len(agents),
            )
        )
    return tuple(agents)


def _session_key(value: Any) -> str:
    return str(value or "").lower()


def merge_steps(
    *,
    agents: Sequence[AgentIdentity],
    rows: Sequence[Mapping[str, Any]],
    subjects: Mapping[str, str],
) -> tuple[RunTimelineStep, ...]:
    """The Run's steps in merge order, from the store's step rows and the roster it belongs to.

    An Assistant entry whose stored reasoning is non-empty states two steps: a ``thinking`` step
    before the ``say``/``answer`` it produced. Both carry the entry's identity and sequence, so the
    merge's stable order is what puts the thought first; the page read is what fills its text in.

    A Session the roster page does not name — a Child past the page the roster reads — keeps its
    steps in the merge, after the named agents, so the marks it causes are still the Run's. It is
    not a page the Inspector offers, only a cause the merge must account for.
    """
    if not agents:
        return ()
    by_session = {
        _session_key(agent.session_id): agent for agent in agents if agent.session_id is not None
    }
    unnamed = sorted(
        {
            _session_key(row.get("session_id"))
            for row in rows
            if not row.get("is_main") and _session_key(row.get("session_id")) not in by_session
        }
    )
    for position, session_id in enumerate(unnamed):
        by_session[session_id] = AgentIdentity(
            key=session_id,
            label="",
            session_id=session_id,
            order=len(agents) + position,
        )

    # The call's own clock, read from the Assistant entry that named it: a Tool result times itself
    # against its own Session's call, so two agents reusing "call_0" cannot swap clocks.
    calls: dict[tuple[str, str], list[tuple[int, datetime.datetime]]] = {}
    for row in rows:
        if row.get("entry_type") != "assistant_message":
            continue
        instant = parse_instant(row.get("at"))
        call_ids = row.get("tool_call_ids")
        if instant is None or not isinstance(call_ids, (list, tuple)):
            continue
        session = _session_key(row.get("session_id"))
        sequence = int(row.get("sequence") or 0)
        for call_id in call_ids:
            key = str(call_id or "")
            if key:
                calls.setdefault((session, key), []).append((sequence, instant))

    steps: list[RunTimelineStep] = []
    for row in rows:
        entry_type = str(row.get("entry_type") or "")
        role = _ROLE_BY_ENTRY_TYPE.get(entry_type)
        if role is None:
            continue
        instant = parse_instant(row.get("at"))
        if instant is None:
            continue
        session = _session_key(row.get("session_id"))
        if row.get("is_main"):
            agent = agents[0]
        else:
            agent = by_session.get(session)
        if agent is None:
            continue
        sequence = int(row.get("sequence") or 0)
        tool_calls = int(row.get("tool_calls") or 0) if role == "assistant" else 0
        kinds = step_kinds(
            entry_type=entry_type,
            has_content=bool(row.get("has_content")),
            has_reasoning=bool(row.get("has_reasoning")),
            tool_calls=tool_calls,
        )
        if not kinds:
            # An assistant turn that only names its calls is an invocation record; the Tool
            # result below states the same call, and the call map above already used its clock.
            continue
        tool = "tool" in kinds
        call_id = str(row.get("call_id") or "") if tool else ""
        # The latest call before the result: a retried turn reusing an id is timed from its own
        # attempt, not from whichever row the store happened to return first.
        call_instant = None
        if tool and call_id:
            preceding = [
                (call_sequence, call_at)
                for call_sequence, call_at in calls.get((session, call_id), ())
                if call_sequence < sequence
            ]
            if preceding:
                call_instant = max(preceding, key=lambda call: call[0])[1]
        duration_ms = (
            max(0, round((instant - call_instant).total_seconds() * 1000))
            if call_instant is not None
            else None
        )
        for kind in kinds:
            steps.append(
                RunTimelineStep(
                    agent=agent.key,
                    agent_label=agent.label,
                    agent_order=agent.order,
                    session_id=str(row.get("session_id") or ""),
                    sequence=sequence,
                    instant=instant,
                    at=str(row.get("at") or ""),
                    kind=kind,
                    name=(str(row.get("tool_name") or "") or None) if tool else None,
                    # The bounded Tool label, as the transcript page states it; the read states no
                    # more of it than that page does, so its masking is that page's masking too.
                    subject=(subjects.get(call_id) or None) if tool and call_id else None,
                    outcome=("failed" if row.get("failed") else "succeeded") if tool else None,
                    duration_ms=duration_ms if tool else None,
                    failed=bool(row.get("failed")) if tool else False,
                    degraded=bool(row.get("degraded")) if tool else False,
                )
            )
    steps.sort(key=RunTimelineStep.key)
    return tuple(steps)


def page_of(
    steps: Sequence[RunTimelineStep],
    *,
    limit: int,
    before: RunTimelinePosition | None,
    agents: Sequence[AgentIdentity],
) -> tuple[tuple[RunTimelineStep, ...], bool]:
    """The newest ``limit`` steps older than ``before``, and whether the Run holds older ones.

    Only the agents the roster states are pages a reader can open; a Session past the roster page
    is in the merge for its marks, not as a page. ``before`` is the opaque cursor position the
    previous page stated, so a page is strictly older than it.
    """
    if before is None:
        ordered = steps
    else:
        index = position_of(steps, before)
        ordered = (
            steps[:index]
            if index is not None
            else [step for step in steps if step.instant < before.instant]
        )
    named = {agent.key for agent in agents}
    candidates = [step for step in ordered if step.agent in named]
    size = max(1, limit)
    return tuple(candidates[-size:]), len(candidates) > size


def with_page_content(
    page: Sequence[RunTimelineStep],
    messages: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[RunTimelineStep, ...]:
    """The page steps with the words they show, as the masked transcript states them.

    A step's text is its message's content: the search row cuts its one-line excerpt from it, and
    the overview's finding excerpt does the same. The message behind a ``thinking`` step is the same
    one behind the ``say``/``answer`` it precedes, and a thinking step is never on a page either
    read walks — the search excludes it and the findings list never draws one — so the content is
    the step's own and no reasoning is read here.
    """
    steps: list[RunTimelineStep] = []
    for step in page:
        message = messages.get((_session_key(step.session_id), step.sequence))
        content = None if message is None else message.get("content")
        steps.append(
            replace(step, content=content if isinstance(content, str) else ""),
        )
    return tuple(steps)


__all__ = [
    "AgentIdentity",
    "MAIN_AGENT",
    "TIMELINE_KINDS",
    "RunStepSearch",
    "RunTimelinePosition",
    "RunTimelineStep",
    "agent_identities",
    "decision_marks",
    "merge_steps",
    "page_of",
    "parse_instant",
    "position_of",
    "step_kinds",
    "with_page_content",
]


def decision_marks(
    steps: Sequence[RunTimelineStep],
) -> dict[tuple[str, int], tuple[str, ...]]:
    """What the whole Run's steps state about its mechanical decisions, keyed by ``(agent, sequence)``.

    A step may carry more than one mark: a failed call that also repeats one is both. The four
    signals are the ones that are mechanical — nothing here guesses at intent:

    ① ``failed``: the call's outcome is failed.
    ② ``degraded``: the facts say the search degraded, or name an empty provider.
    ③ ``repeated``: the same tool called on the same subject at least twice in the Run.
    ④ ``redirect``: after a Child answered, the main agent's next call looks at something it has
       not looked at before. Only steps the main agent has already acted on can be redirected
       from, so the first main call after a Child's answer never marks one.
    """
    held: dict[tuple[str, int], list[str]] = {}

    def mark(step: RunTimelineStep, one: str) -> None:
        marks = held.setdefault((step.agent, step.sequence), [])
        if one not in marks:
            marks.append(one)

    identities: list[str | None] = []
    for step in steps:
        identity = None
        if step.kind == "tool" and step.name and step.subject:
            identity = f"{step.name}\u0000{step.subject}"
        identities.append(identity)
    seen = Counter(identity for identity in identities if identity is not None)

    main_subjects: set[str] = set()
    child_answered = False
    for step, identity in zip(steps, identities, strict=True):
        # A thinking step is the reasoning of a turn, not an action the Run took: it moves through
        # here without stating or clearing anything, and no mark is ever keyed to it.
        if step.kind == "thinking":
            continue
        if step.kind == "tool" and step.failed:
            mark(step, "failed")
        if step.kind == "tool" and step.degraded:
            mark(step, "degraded")
        if identity is not None and seen[identity] > 1:
            mark(step, "repeated")
        if step.kind == "answer" and step.agent != MAIN_AGENT:
            # A Child's conclusion is what the main agent may react to; the main agent's own answer
            # is the Run ending, not a decision to redirect from.
            child_answered = True
            continue
        if step.kind != "tool" or step.agent != MAIN_AGENT:
            continue
        subject = step.subject if step.subject not in (None, "") else None
        if child_answered:
            if subject is not None and main_subjects and subject not in main_subjects:
                mark(step, "redirect")
            child_answered = False
        if subject is not None:
            main_subjects.add(subject)
    return {key: tuple(marks) for key, marks in held.items()}
