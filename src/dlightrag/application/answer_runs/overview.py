# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Run overview: one Run's size, agents, and decisions stated as whole-Run facts.

The per-agent page is the Inspector's only step list; the Run panel states aggregates, structure,
and outcome facts, so nothing here is a row of the transcript. Every number is derived from the
same merged index the step list reads — its steps, their marks, and the roster's agents — which is
why a capped findings list can sit beside uncapped totals without the two disagreeing about the
Run.

This module is pure: it takes the merge, its marks, and the roster and returns the dataclasses the
Web route projects. Reading the stored entries behind a capped finding's excerpt, and applying the
reader content gate, is the service's job; nothing here reads a database or a Session.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .timeline import TIMELINE_KINDS, AgentIdentity, RunTimelineStep

MARK_KINDS: tuple[str, ...] = ("failed", "degraded", "repeated", "redirect")
"""The mechanical decisions a Run states, in the order the overview counts them.

The four are the ones the merged index marks (see :func:`~.timeline.decision_marks`), and a count
is a number of steps, never a guess at intent. The order is fixed so a mark total and the wire
shape that carries it cannot disagree about which marks exist.
"""

FINDINGS_MAX = 40
"""How many findings the overview states before the total stands in for the rest.

A finding is a row a reader can click; a Run can carry hundreds of marked steps, and the Run panel
is a dossier rather than the step list the search exists for. The cap bounds the projection and the
entries the read loads; ``findings_total`` remains the whole Run's count.
"""

FINDING_EXCERPT_MAX_CHARS = 160
"""The most text one finding states: the first line of its masked content, cut here.

The findings list scans a Run, so a row shows the start of what its step said or answered, not the
step. Slicing is the bound: no marker is appended, because a reader is not told a truncation the
list cannot show.
"""


@dataclass(frozen=True, slots=True)
class RunTotals:
    """The whole Run's size, stated once for the panel's summary line and composition bar."""

    steps: int
    """Every step of every agent, as the merged index orders them."""
    kinds: Mapping[str, int]
    """How many steps each kind states, keyed and ordered by :data:`TIMELINE_KINDS`."""
    decisions: int
    """How many steps state at least one mechanical decision (the existing meaning)."""
    marks: Mapping[str, int]
    """How many steps each mark names, keyed and ordered by :data:`MARK_KINDS`."""
    tool_time_ms: int
    """The Tool time the Run states, summed over every Tool step whose clock it knows."""


@dataclass(frozen=True, slots=True)
class RunAgentRollup:
    """One named agent's share of the whole Run, in roster order: its own row in the ledger."""

    agent: str
    """The identity a step carries on the wire: ``main`` or the Child Session's id."""
    label: str
    """What the roster calls the agent, already reader-gated by the read that built it."""
    order: int
    """Where the roster puts the agent: the Run's own agent is zero, then each Child."""
    steps: int
    tool_calls: int
    """How many of the agent's steps a Tool result settled, each one a call it made."""
    marks: Mapping[str, int]
    """How many of the agent's steps each mark names, keyed by :data:`MARK_KINDS`."""
    tool_time_ms: int
    first_at: str | None
    """The earliest instant the agent acts, when it acts at all; the Run clock otherwise."""
    last_at: str | None


@dataclass(frozen=True, slots=True)
class RunFinding:
    """One step a reader should look at, stated as the finding row the overview lists."""

    agent: str
    session_id: str
    """The Session the step's stored entry lives in, for the read that fills its excerpt.

    It is not a wire field: the finding names its agent on the wire, and the Session is what the
    read uses to load the one entry behind the excerpt.
    """
    sequence: int
    kind: str
    at: str
    name: str | None
    subject: str | None
    outcome: str | None
    marks: tuple[str, ...]
    excerpt: str = ""
    """The first line of the step's masked content, which the service fills after the gate."""


@dataclass(frozen=True, slots=True)
class RunOverview:
    """One Run's whole-Run facts, as the overview read states them to one reader."""

    run_id: str
    totals: RunTotals
    agents: tuple[RunAgentRollup, ...]
    """Every agent the merged index names, in roster order, the Run's own agent first."""
    findings: tuple[RunFinding, ...]
    """The newest-capped findings in merge order; ``findings_total`` is the uncapped count."""
    findings_total: int
    usage: Mapping[str, Any] | None
    """The Run result's usage record when the record states one, else None — never zeros."""
    version: str
    """The derived index version this overview was served from: the cache validator's material.

    It is not a wire field: the route derives its ``ETag`` from it, and the panel's numbers are a
    function of it, so a client that holds one has the whole overview it states.
    """


def mark_counts(marks: Mapping[tuple[str, int], Sequence[str]]) -> dict[str, int]:
    """How many steps the Run marks with each mechanical decision, keyed by :data:`MARK_KINDS`.

    ``marks`` is the merged index's ``(agent, sequence)`` mapping; a step may carry more than one
    mark, so the counts add up to more than ``decisions`` when they do. Every stated mark is
    counted even when no mark is there, so a wire shape can state all four without inventing one.
    """
    counts = dict.fromkeys(MARK_KINDS, 0)
    for held in marks.values():
        for mark in held:
            if mark in counts:
                counts[mark] += 1
    return counts


def overview_totals(
    steps: Sequence[RunTimelineStep],
    marks: Mapping[tuple[str, int], Sequence[str]],
) -> RunTotals:
    """The whole Run's size, counted over its merged steps rather than over any page of them.

    A thinking step counts in ``steps`` and under its kind like any other step, because it is a row
    the Run panel's composition bar sizes; it is never a mark, so it never counts in ``decisions``.
    A Tool step whose duration the merge could not time contributes no milliseconds: the total is
    the Tool time the Run states, not a duration guessed for the calls it cannot clock.
    """
    kinds = dict.fromkeys(TIMELINE_KINDS, 0)
    tool_time_ms = 0
    for step in steps:
        kinds[step.kind] += 1
        if step.kind == "tool" and step.duration_ms is not None:
            tool_time_ms += step.duration_ms
    return RunTotals(
        steps=len(steps),
        kinds=kinds,
        decisions=len(marks),
        marks=mark_counts(marks),
        tool_time_ms=tool_time_ms,
    )


def agent_rollups(
    steps: Sequence[RunTimelineStep],
    marks: Mapping[tuple[str, int], Sequence[str]],
    agents: Sequence[AgentIdentity],
) -> tuple[RunAgentRollup, ...]:
    """Each named agent's whole-Run rollup, in roster order, the Run's own agent first.

    Rollups cover the whole Run: they are read from the merged steps, not from a page, so the
    ledger's sums and the panel's totals are the same facts. An agent the roster names but the
    merge holds no step for still gets a row — stated as zeroes and no clock — because the ledger
    is the roster's structure; a Session the roster page does not name has steps in the merge for
    its marks but no row here, since it is not a page the Inspector offers.
    """
    by_agent: dict[str, list[RunTimelineStep]] = {agent.key: [] for agent in agents}
    for step in steps:
        owned = by_agent.get(step.agent)
        if owned is not None:
            owned.append(step)
    counted: dict[str, dict[str, int]] = {
        agent.key: dict.fromkeys(MARK_KINDS, 0) for agent in agents
    }
    for (agent, _sequence), held in marks.items():
        counts = counted.get(agent)
        if counts is None:
            continue
        for mark in held:
            if mark in counts:
                counts[mark] += 1
    rollups: list[RunAgentRollup] = []
    for agent in agents:
        owned = by_agent[agent.key]
        first = min(owned, key=lambda step: step.instant) if owned else None
        last = max(owned, key=lambda step: step.instant) if owned else None
        rollups.append(
            RunAgentRollup(
                agent=agent.key,
                label=agent.label,
                order=agent.order,
                steps=len(owned),
                tool_calls=sum(1 for step in owned if step.kind == "tool"),
                marks=counted[agent.key],
                tool_time_ms=sum(
                    step.duration_ms
                    for step in owned
                    if step.kind == "tool" and step.duration_ms is not None
                ),
                first_at=None if first is None else first.at,
                last_at=None if last is None else last.at,
            )
        )
    return tuple(rollups)


def findings_of(
    steps: Sequence[RunTimelineStep],
    marks: Mapping[tuple[str, int], Sequence[str]],
    limit: int = FINDINGS_MAX,
) -> tuple[tuple[RunFinding, ...], int]:
    """The Run's findings in merge order, capped at ``limit``, and the uncapped count.

    A finding is a step a reader should look at: one the marks name, or a delivery step — the
    warning the Run states about its own delivery, which is a finding even when nothing marks it.
    Thinking steps never appear: reasoning is how a turn was reached, not an action the Run took,
    and the Run panel does not ship reasoning to a list. The cap bounds what the read loads and the
    wire states; the total is the count a reader is told, so a truncated list cannot read as the
    Run's whole trouble.
    """
    selected: list[RunFinding] = []
    for step in steps:
        if step.kind == "thinking":
            continue
        held = marks.get((step.agent, step.sequence), ())
        if not held and step.kind != "delivery":
            continue
        selected.append(
            RunFinding(
                agent=step.agent,
                session_id=step.session_id,
                sequence=step.sequence,
                kind=step.kind,
                at=step.at,
                name=step.name,
                subject=step.subject,
                outcome=step.outcome,
                marks=tuple(held),
            )
        )
    return tuple(selected[: max(0, int(limit))]), len(selected)


def excerpt_of(content: Any) -> str:
    """The first line of a step's masked content, stripped and bounded for a findings row.

    The read hands the masked content in; this keeps a row one line and at most
    :data:`FINDING_EXCERPT_MAX_CHARS`, so a finding can preview a Tool result without shipping it.
    """
    if not isinstance(content, str):
        return ""
    return content.split("\n", 1)[0].strip()[:FINDING_EXCERPT_MAX_CHARS]


def run_usage(result: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The usage record one Run result states, or None when it states none.

    The record is the Run's own: the read neither recomputes it from the steps nor fills an absent
    value with zeroes, because a Run whose provider stated no usage is unknown, not free. A record
    that is present is passed through as stored, so the root, child, and inclusive counters stay
    the ones the Run wrote.
    """
    if not isinstance(result, Mapping):
        return None
    usage = result.get("usage")
    return dict(usage) if isinstance(usage, Mapping) and usage else None


__all__ = [
    "FINDING_EXCERPT_MAX_CHARS",
    "FINDINGS_MAX",
    "MARK_KINDS",
    "RunAgentRollup",
    "RunFinding",
    "RunOverview",
    "RunTotals",
    "agent_rollups",
    "excerpt_of",
    "findings_of",
    "mark_counts",
    "overview_totals",
    "run_usage",
]
