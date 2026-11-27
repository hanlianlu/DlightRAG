# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The durable objective a Session carries across Rounds (ADR 0060).

A goal is harness state, not Skill state: one goal per answer session, every mutation a
compare-and-set on ``revision``, and the transitions the reference shape prescribes
(DeepSeek Harness's goal package): ``edit`` changes only the objective and the round cap,
``pause`` and ``resume`` move between active and paused, ``complete`` is the model's call,
and ``blocked`` carries the condition it names.

The domain layer has no I/O, no clock and no names of Skills or Tools. Authority (a direct
human turn versus a goal round), durability (a session record replayed into this object), and
the "same condition for three rounds" rule belong to the application that drives it. Keeping
those out of this module is what lets goal mode be tested without a Skill installed and a
Skill be tested without a goal.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal, Self

#: When the Agent creates a goal without naming a round cap. A round is a whole Run, so this
#: product's default is deliberately small (16) while the reference's deployment default is 256.
#: The cap itself has no product ceiling: a caller may name any positive integer.
DEFAULT_MAX_ROUNDS = 16

#: Objective text is a task statement, not a document; the cap keeps a goal from becoming one.
MAX_OBJECTIVE_CHARS = 2_000

#: A blocking condition is a sentence or two, not a report.
MAX_BLOCK_REASON_CHARS = 2_000

#: A goal round may report a blocker only after this many rounds have started (the reference
#: shape's threshold); a direct human turn may block immediately. Both the tool description and
#: the application's authority check read this one number.
BLOCKED_AFTER_ROUNDS = 3

#: Stable codes the tools surface to the Agent; a caller branches on the code, never the text.
GOAL_INVALID_OBJECTIVE = "goal_invalid_objective"
GOAL_INVALID_BLOCK_REASON = "goal_invalid_block_reason"
GOAL_NOT_FOUND = "goal_not_found"
GOAL_INVALID_ROUNDS = "goal_invalid_rounds"
GOAL_STALE_REVISION = "goal_stale_revision"
GOAL_INVALID_TRANSITION = "goal_invalid_transition"
#: Completion was refused because the Run's loaded sealed Skill delivery is not attested.
GOAL_COMPLETION_UNATTESTED = "goal_completion_unattested"
GOAL_ROUNDS_EXHAUSTED = "goal_rounds_exhausted"

#: What the model may ask; every value has its own authority rule in the service.
GoalAction = Literal["edit", "pause", "resume", "complete", "blocked"]


class GoalPhase(StrEnum):
    """Where a goal is. A blocked goal waits for a human to resume; a complete one ends the work."""

    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    COMPLETE = "complete"


class GoalError(ValueError):
    """A refused goal mutation, with a stable code and an Agent-safe message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class Goal:
    """One session's objective and its compare-and-set revision."""

    id: str
    revision: int
    objective: str
    phase: GoalPhase
    max_rounds: int
    rounds_started: int = 0
    blocked_reason: str | None = None

    @classmethod
    def create(
        cls,
        objective: str,
        *,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        goal_id: str | None = None,
    ) -> Self:
        """A new goal at revision 1, active, with no round started.

        ``goal_id`` is for a caller that must recognise its own goal again after a retry; the
        default is a fresh random id.
        """
        text = _objective(objective)
        cap = _round_cap(max_rounds)
        return cls(
            id=goal_id or f"goal-{uuid.uuid4()}",
            revision=1,
            objective=text,
            phase=GoalPhase.ACTIVE,
            max_rounds=cap,
        )

    @property
    def rounds_left(self) -> int:
        """Rounds the driver may still start; zero once the cap is reached."""
        return max(0, self.max_rounds - self.rounds_started)

    def edit(
        self,
        *,
        revision: int,
        objective: str | None = None,
        max_rounds: int | None = None,
    ) -> Self:
        """Change the objective or the cap, never the phase. A completed goal is not editable."""
        self._expect(revision)
        if self.phase is GoalPhase.COMPLETE:
            raise GoalError(
                GOAL_INVALID_TRANSITION,
                "this goal is complete; start a new goal with `/goal <objective>` (or create_goal)",
            )
        if objective is None and max_rounds is None:
            raise GoalError(
                GOAL_INVALID_OBJECTIVE,
                "edit needs an objective or a max_rounds; it changes nothing else",
            )
        text = self.objective if objective is None else _objective(objective)
        cap = self.max_rounds if max_rounds is None else _round_cap(max_rounds)
        if cap < self.rounds_started:
            raise GoalError(
                GOAL_INVALID_ROUNDS,
                f"the cap cannot fall below the {self.rounds_started} rounds already started",
            )
        return replace(self, revision=self.revision + 1, objective=text, max_rounds=cap)

    def pause(self, *, revision: int) -> Self:
        """Stop the driver; a human resumes."""
        self._expect(revision)
        if self.phase is not GoalPhase.ACTIVE:
            raise GoalError(
                GOAL_INVALID_TRANSITION, f"pause needs an active goal; this one is {self.phase}"
            )
        return replace(self, revision=self.revision + 1, phase=GoalPhase.PAUSED)

    def resume(self, *, revision: int) -> Self:
        """Restart a paused or a blocked goal, clearing the condition that blocked it."""
        self._expect(revision)
        if self.phase not in {GoalPhase.PAUSED, GoalPhase.BLOCKED}:
            raise GoalError(
                GOAL_INVALID_TRANSITION,
                f"resume needs a paused or blocked goal; this one is {self.phase}",
            )
        if self.rounds_started >= self.max_rounds:
            raise GoalError(
                GOAL_ROUNDS_EXHAUSTED,
                f"all {self.max_rounds} rounds have started; raise max_rounds before resuming",
            )
        return replace(
            self, revision=self.revision + 1, phase=GoalPhase.ACTIVE, blocked_reason=None
        )

    def complete(self, *, revision: int) -> Self:
        """Record that the objective is achieved. The model's call, never machine-enforced."""
        self._expect(revision)
        if self.phase is GoalPhase.COMPLETE:
            raise GoalError(GOAL_INVALID_TRANSITION, "this goal is already complete")
        return replace(
            self, revision=self.revision + 1, phase=GoalPhase.COMPLETE, blocked_reason=None
        )

    def block(self, *, revision: int, reason: str) -> Self:
        """Record the concrete condition that stops progress. The application enforces the rounds."""
        self._expect(revision)
        if self.phase is not GoalPhase.ACTIVE:
            raise GoalError(
                GOAL_INVALID_TRANSITION, f"block needs an active goal; this one is {self.phase}"
            )
        condition = reason.strip()
        if not condition:
            raise GoalError(
                GOAL_INVALID_BLOCK_REASON, "block needs the concrete condition in blocked_reason"
            )
        if len(condition) > MAX_BLOCK_REASON_CHARS:
            raise GoalError(
                GOAL_INVALID_BLOCK_REASON,
                f"a blocking condition is at most {MAX_BLOCK_REASON_CHARS} characters",
            )
        return replace(
            self, revision=self.revision + 1, phase=GoalPhase.BLOCKED, blocked_reason=condition
        )

    def admit_round(self, *, revision: int) -> tuple[Self, int]:
        """Return the goal with one more round started, and that round's number.

        The caller records the admission durably; the stored row carries the count, so
        ``rounds_started`` counts admitted rounds rather than wall-clock attempts.
        """
        self._expect(revision)
        if self.phase is not GoalPhase.ACTIVE:
            raise GoalError(
                GOAL_INVALID_TRANSITION, f"a round needs an active goal; this one is {self.phase}"
            )
        if self.rounds_started >= self.max_rounds:
            raise GoalError(
                GOAL_ROUNDS_EXHAUSTED,
                f"all {self.max_rounds} rounds have started; the driver cannot start another",
            )
        started = replace(self, revision=self.revision + 1, rounds_started=self.rounds_started + 1)
        return started, started.rounds_started

    def release_round(self, *, revision: int) -> Self:
        """Give back one admitted round whose Run was never created; the driver's compensation."""
        self._expect(revision)
        if self.rounds_started == 0:
            raise GoalError(GOAL_INVALID_TRANSITION, "no round has been admitted to release")
        return replace(self, revision=self.revision + 1, rounds_started=self.rounds_started - 1)

    def _expect(self, revision: int) -> None:
        if revision != self.revision:
            raise GoalError(
                GOAL_STALE_REVISION,
                f"the goal is at revision {self.revision}, not {revision}; read it again",
            )


def _objective(objective: str) -> str:
    text = objective.strip()
    if not text:
        raise GoalError(GOAL_INVALID_OBJECTIVE, "an objective cannot be empty")
    if len(text) > MAX_OBJECTIVE_CHARS:
        raise GoalError(
            GOAL_INVALID_OBJECTIVE,
            f"an objective is at most {MAX_OBJECTIVE_CHARS} characters; "
            "put the plan in the session, not in the goal",
        )
    return text


def _round_cap(max_rounds: int) -> int:
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise GoalError(GOAL_INVALID_ROUNDS, "max_rounds must be a positive integer")
    return max_rounds
