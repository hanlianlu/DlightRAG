# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Continue an active goal after a Run settles (ADR 0060, ADR 0077).

The driver is application orchestration. It hears the coordinator's settlement observer, and
when a settled Answer Run leaves a Session whose durable goal is active, under its round cap and
with no continuation already pending, it starts the next Run in that same session with the
objective and the round number. A human message wins: if the Session already has another Run in
flight, no round starts. The durable phase is the only permission the driver needs, so any
process may continue the goal and a restart picks the work up.

Goal rounds settle differently from ordinary work, because the round driver is their observer:

- a succeeded round continues the goal (or is the one that reaches the cap);
- a succeeded round that ended on the model token limit blocks the goal, retryably;
- a failed round blocks the goal with the failure it observed;
- a cancelled round pauses the goal — the human stopped the work;
- a human or ordinary Run only continues on success, and never changes the goal otherwise.

The driver names no Skill and carries no delivery evidence: the Run's own pre-delivery checkpoint
handles that inside the turn, and the prompt it writes is harness prose about the goal.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.application.answer_runs.service import (
    ActiveRunLimitError,
    AnswerRequest,
    AnswerService,
)
from dlightrag.application.runs import (
    RunAdmissionLimitExceededError,
    RunService,
    RunView,
    SessionWorkBusyError,
)
from dlightrag.engine.answer.execution.input import AnswerRunRequest
from dlightrag.engine.answer.goal import Goal, GoalError, GoalPhase
from dlightrag.engine.answer.goal_service import ActiveGoal, AnswerGoalService
from dlightrag.engine.rag.retrieval.models import MetadataFilter
from dlightrag.engine.runtime.coordinator import SettledRun

logger = logging.getLogger(__name__)

#: The Run kind whose settlement can continue a goal; a retrieval or corpus run cannot.
_ANSWER = "answer"

#: A round is always research: the goal tools compose in that mode, and a round with no goal tools
#: could neither complete nor block. A human fast turn never turns the goal into a fast chain.
_ROUND_MODE = "research"

#: The trace stop reason that means the model ended at its output token limit. DlightRAG's
#: providers spell the condition ``length``; the reference shape calls it ``max-tokens``, and the
#: coordinator passes the raw trace value through, so both spellings are recognised. Any other
#: value is treated as a successful round.
_TOKEN_LIMIT_STOP_REASONS = frozenset({"length", "max-tokens", "max_tokens", "token_limit"})


@dataclass(frozen=True, slots=True)
class GoalRoundReservation:
    """The pending continuation one driver attempt reserved, as its own to withdraw.

    The request id and prompt digest are the row's identity: a withdrawal deletes exactly the
    reservation this attempt wrote, so a stale attempt that lost its compare-and-set cannot
    delete a newer attempt's row.
    """

    request_id: str
    prompt_digest: str


class GoalRoundTurns(Protocol):
    """A Web conversation that holds the goal continuation as its next turn.

    The driver requests the continuation; the one fenced delivery path admits it when the
    Session is at rest. A Run with no conversation (REST or MCP) falls back to the plain
    Answer acceptance, which is the only channel such a Run has. ``accept_goal_round_now``
    is the no-parent-Run path a human ``/goal`` command uses: it accepts the round turn
    immediately from the conversation's own seed. Every continuation carries the admitted
    goal's identity, so its settlement can tell which goal it served; ``continue_session_goal``
    is the no-Run path recovery uses for an active goal that never got a conversation turn.
    """

    async def request_goal_round(
        self,
        *,
        owner_id: str,
        parent_run_id: str,
        prompt: str,
        workspaces: tuple[str, ...] = (),
        goal_id: str,
        goal_revision: int,
        goal_round: int,
    ) -> GoalRoundReservation | None: ...

    async def goal_round_pending(self, *, owner_id: str, parent_run_id: str) -> bool: ...

    async def withdraw_goal_round(
        self,
        *,
        owner_id: str,
        parent_run_id: str,
        request_id: str,
        prompt_digest: str,
    ) -> bool: ...

    async def accept_goal_round_now(
        self,
        *,
        owner_id: str,
        conversation_id: str,
        prompt: str,
        workspaces: tuple[str, ...] = (),
        goal_id: str,
        goal_revision: int,
        goal_round: int,
    ) -> bool: ...

    async def continue_session_goal(self, *, owner_id: str, session_id: str) -> bool: ...


class GoalRoundDriver:
    """Continue a durable active goal after one Run settles; never retry a failed round."""

    def __init__(
        self,
        *,
        goals: AnswerGoalService,
        answers: AnswerService,
        runs: RunService,
        round_turns: GoalRoundTurns | None = None,
    ) -> None:
        self._goals = goals
        self._answers = answers
        self._runs = runs
        self._round_turns = round_turns
        self._deliver_pending: Callable[[str, str], Awaitable[bool]] | None = None

    def bind_pending_delivery(self, deliver: Callable[[str, str], Awaitable[bool]]) -> None:
        """Attach the one next-turn delivery path, so a resumed goal's round moves at once."""
        self._deliver_pending = deliver

    def bind_round_turns(self, turns: GoalRoundTurns) -> None:
        """Attach the Web conversations after both sides exist.

        A Web conversation needs this driver to continue a resumed goal, and the driver needs the
        conversation to accept a round as a turn. Composition creates the driver first and binds
        the turns second, instead of closing that cycle with a constructor argument.
        """
        self._round_turns = turns

    async def settle(self, settled: SettledRun) -> None:
        """Handle one settlement; a failure here never reaches the Run it is about."""
        try:
            view = await self._runs.get(owner_id=settled.owner_id, run_id=settled.run_id)
            if view is None or view.run_kind != _ANSWER:
                return
            if view.request.get("continuation_kind") == "goal_round":
                await self._settle_goal_round(settled, view)
            elif settled.status == "succeeded":
                await self._continue(settled.owner_id, view)
        except Exception:
            logger.warning(
                "Goal round driver failed after run %s settled", settled.run_id, exc_info=True
            )

    async def run_cancelled(self, view: RunView) -> None:
        """A queued Run cancelled before any coordinator saw it: a goal round pauses the goal."""
        if view.run_kind != _ANSWER:
            return
        if view.request.get("continuation_kind") != "goal_round":
            return
        try:
            await self._stop_cancelled(view)
        except Exception:
            logger.warning(
                "Goal round driver failed after run %s was cancelled", view.run_id, exc_info=True
            )

    async def resume_after_restart(self) -> None:
        """Take up every durable active goal on a recovery sweep (ADR 0077).

        A process holds no activation state; the row is the whole record. For a goal whose Session
        is between Runs this delivers a continuation that was already requested, or starts the
        next round, or records the cap; for a goal with no Run yet the Web conversation's
        immediate path starts it. For a goal whose Session already has a Run in flight it does
        nothing: that Run's settlement is the trigger, not a scan. Both the startup scan and the
        recovery process's periodic sweep call this; CAS and the pending-round row fence duplicate
        attempts. A failure for one goal never costs the others.
        """
        try:
            entries = await self._goals.active_goals()
        except Exception:
            logger.warning(
                "Goal round driver could not read the active goals at startup", exc_info=True
            )
            return
        if entries:
            logger.info("Goal recovery scan considered %s active goal(s)", len(entries))
        for entry in entries:
            try:
                await self._recover_entry(entry)
            except Exception:
                logger.warning(
                    "Active goal %s could not resume after a restart", entry.goal.id, exc_info=True
                )

    async def continue_created_goal(
        self,
        *,
        owner_id: str,
        session_id: str,
        conversation_id: str,
        workspaces: tuple[str, ...] = (),
    ) -> bool:
        """Start the next round of an active goal now, with no parent Run to continue from.

        The Web ``/goal`` create command calls this after the goal exists: the human asked for
        work immediately, and a fresh conversation has no Run whose settlement could start it.
        ``workspaces`` is the conversation's selected scope, so round 1 continues the same
        workspaces the reader chose. A failed acceptance releases the admission and leaves the
        goal active; the durable row is the record the next trigger retries from.
        """
        if self._round_turns is None:
            return False
        try:
            goal = await self._goals.get(owner_id=owner_id, session_id=session_id)
            if goal is None or goal.phase is not GoalPhase.ACTIVE or goal.rounds_left <= 0:
                return False
            started, round_number = await self._goals.admit_round(
                owner_id=owner_id,
                session_id=session_id,
                run_id="",
                goal_id=goal.id,
                revision=goal.revision,
            )
        except GoalError:
            return False
        except Exception:
            logger.warning("Goal round driver failed after a goal was created", exc_info=True)
            return False
        prompt = goal_round_prompt(goal=started, round_number=round_number)
        try:
            accepted = await self._round_turns.accept_goal_round_now(
                owner_id=owner_id,
                conversation_id=conversation_id,
                prompt=prompt,
                workspaces=workspaces,
                goal_id=started.id,
                goal_revision=started.revision,
                goal_round=round_number,
            )
        except Exception:
            await self._release(owner_id, session_id, "", started)
            logger.warning(
                "Goal round %s of %s could not be accepted after creation",
                round_number,
                started.id,
                exc_info=True,
            )
            return False
        if not accepted:
            await self._release(owner_id, session_id, "", started)
            return False
        return True

    async def continue_after_resume(
        self, *, owner_id: str, session_id: str, parent_run_id: str
    ) -> bool:
        """Start the next round when a human resume leaves an active goal with no Run advancing it.

        A Run already advancing the session keeps its priority: when it settles, the settlement
        path starts the round.
        """
        try:
            goal = await self._goals.get(owner_id=owner_id, session_id=session_id)
            if goal is None or goal.phase != GoalPhase.ACTIVE or goal.rounds_left <= 0:
                return False
            view = await self._runs.get(owner_id=owner_id, run_id=parent_run_id)
            if view is None or _session_id(view.request) != session_id:
                return False
            if view.run_kind != _ANSWER or view.access_scope_kind != "owner":
                return False
            if not view.terminal:
                return False
            return await self._start_round(
                owner_id=owner_id, view=view, session_id=session_id, goal=goal
            )
        except Exception:
            logger.warning("Goal round driver failed after a human resume", exc_info=True)
            return False

    # -- settlement paths --------------------------------------------------------------------

    async def _settle_goal_round(self, settled: SettledRun, view: RunView) -> None:
        """Settle one goal round: continue, pause or block the goal from what the round did."""
        session_id = _session_id(view.request)
        if not session_id:
            return
        goal = await self._goals.get(owner_id=view.submitted_by, session_id=session_id)
        if goal is None:
            return
        stopped = settled.status != "succeeded" or (
            settled.stop_reason in _TOKEN_LIMIT_STOP_REASONS
        )
        if stopped and not _round_serves_goal(view, goal):
            # A durable stop belongs only to the goal the round was admitted for. A round from a
            # goal that was cleared or replaced must not pause or block the replacement; a
            # successful one still continues the current active goal below.
            logger.info(
                "Ignoring the %s settlement of run %s: it did not serve goal %s",
                settled.status,
                view.run_id,
                goal.id,
            )
            return
        if settled.status == "cancelled":
            await self._goals.stop_round_cancelled(
                owner_id=view.submitted_by,
                session_id=session_id,
                run_id=view.run_id,
                goal_id=goal.id,
                revision=goal.revision,
            )
            return
        if settled.status == "failed":
            await self._goals.stop_round_failed(
                owner_id=view.submitted_by,
                session_id=session_id,
                run_id=view.run_id,
                goal_id=goal.id,
                revision=goal.revision,
                reason=(
                    f"the goal round failed ({settled.error_kind or 'unknown error'}); "
                    "resume to retry"
                ),
            )
            return
        if settled.stop_reason in _TOKEN_LIMIT_STOP_REASONS:
            await self._goals.stop_round_token_limit(
                owner_id=view.submitted_by,
                session_id=session_id,
                run_id=view.run_id,
                goal_id=goal.id,
                revision=goal.revision,
            )
            return
        await self._continue(settled.owner_id, view)

    async def _continue(self, owner_id: str, view: RunView) -> None:
        """Continue a successfully settled Run's active goal, if the Session is between Runs."""
        if view.access_scope_kind != "owner":
            # A workspace-scoped Run's caller policy is not the owner's to replay; leave it alone.
            return
        session_id = _session_id(view.request)
        if not session_id:
            return
        goal = await self._goals.get(owner_id=owner_id, session_id=session_id)
        if goal is None or goal.phase != GoalPhase.ACTIVE:
            return
        if goal.rounds_left <= 0:
            await self._block_at_cap(
                owner_id=owner_id, session_id=session_id, run_id=view.run_id, goal=goal
            )
            return
        await self._start_round(owner_id=owner_id, view=view, session_id=session_id, goal=goal)

    async def _recover_entry(self, entry: ActiveGoal) -> None:
        """Take up one durable active goal after a restart."""
        view: RunView | None = None
        if entry.parent_run_id is not None:
            candidate = await self._runs.get(owner_id=entry.owner_id, run_id=entry.parent_run_id)
            if candidate is not None and _session_id(candidate.request) == entry.session_id:
                view = candidate
        goal = await self._goals.get(owner_id=entry.owner_id, session_id=entry.session_id)
        if goal is None or goal.id != entry.goal.id or goal.phase is not GoalPhase.ACTIVE:
            return
        if view is not None and not view.terminal:
            # A Run is already advancing the Session; its settlement is the trigger, not a scan.
            return
        if view is None:
            # No answer Run has carried this Session yet: there is no settlement to trigger
            # the round, but a Web conversation that owns the Session can start it directly.
            if self._round_turns is not None:
                try:
                    await self._round_turns.continue_session_goal(
                        owner_id=entry.owner_id, session_id=entry.session_id
                    )
                except Exception:
                    logger.warning(
                        "Could not recover the goal of Session %s without a Run",
                        entry.session_id,
                        exc_info=True,
                    )
            return
        if view.run_kind != _ANSWER or view.access_scope_kind != "owner":
            return
        # A settlement observer's task can be lost with the process that owned it. When this
        # terminal Run's settlement never ran, replay the same durable stop the settlement would
        # have recorded; never start a new round past a failure, cancellation or token limit.
        is_goal_round = view.request.get("continuation_kind") == "goal_round"
        if is_goal_round and _round_serves_goal(view, goal):
            if await self._replay_terminal_stop(view, goal):
                return
        # A failed or cancelled human Run does not change the goal. Once the Session is idle the
        # active goal continues, exactly as a successful human turn would let it continue.
        if goal.rounds_left <= 0:
            await self._block_at_cap(
                owner_id=entry.owner_id,
                session_id=entry.session_id,
                run_id=entry.parent_run_id or "",
                goal=goal,
            )
            return
        if self._round_turns is not None:
            try:
                if await self._round_turns.goal_round_pending(
                    owner_id=entry.owner_id, parent_run_id=view.run_id
                ):
                    if self._deliver_pending is not None:
                        await self._deliver_pending(entry.owner_id, view.run_id)
                    return
            except Exception:
                logger.warning(
                    "Could not recover the pending goal round of %s", goal.id, exc_info=True
                )
                return
        await self._start_round(
            owner_id=entry.owner_id, view=view, session_id=entry.session_id, goal=goal
        )

    async def _replay_terminal_stop(self, view: RunView, goal: Goal) -> bool:
        """Record a lost settlement's durable stop; ``True`` when this Run must not continue.

        This maps a terminal goal round to the same outcome its settlement observer would have
        written: failed -> blocked (retryable), cancelled -> paused, succeeded at the model token
        limit -> blocked. A normally succeeded round returns ``False`` and the caller starts the
        next round.
        """
        if view.status == "cancelled":
            await self._goals.stop_round_cancelled(
                owner_id=view.submitted_by,
                session_id=_session_id(view.request),
                run_id=view.run_id,
                goal_id=goal.id,
                revision=goal.revision,
            )
            return True
        if view.status == "failed":
            await self._goals.stop_round_failed(
                owner_id=view.submitted_by,
                session_id=_session_id(view.request),
                run_id=view.run_id,
                goal_id=goal.id,
                revision=goal.revision,
                reason=(
                    f"the goal round failed ({view.error_kind or 'unknown error'}); resume to retry"
                ),
            )
            return True
        if _ended_on_token_limit(view):
            await self._goals.stop_round_token_limit(
                owner_id=view.submitted_by,
                session_id=_session_id(view.request),
                run_id=view.run_id,
                goal_id=goal.id,
                revision=goal.revision,
            )
            return True
        return False

    async def _stop_cancelled(self, view: RunView) -> None:
        """Record an active goal's cancelled round as a durable pause."""
        session_id = _session_id(view.request)
        if not session_id:
            return
        goal = await self._goals.get(owner_id=view.submitted_by, session_id=session_id)
        if goal is None or not _round_serves_goal(view, goal):
            return
        await self._goals.stop_round_cancelled(
            owner_id=view.submitted_by,
            session_id=session_id,
            run_id=view.run_id,
            goal_id=goal.id,
            revision=goal.revision,
        )

    # -- starting one round ------------------------------------------------------------------

    async def _start_round(
        self, *, owner_id: str, view: RunView, session_id: str, goal: Goal
    ) -> bool:
        """Reserve, admit and accept one round of an active, idle goal; False when it could not.

        The Web continuation is **reserved before the round is admitted**: a settlement or the
        periodic recovery scan that interleaves between the two sees the pending row and stops,
        so two attempts cannot each burn an admission. Delivery may find the reservation still
        unadmitted and admit it in its own transaction (the round it reserved); when admission
        fails here the reservation is withdrawn, and when this call is the loser to a delivery it
        leaves the round that delivery started alone.
        """
        current = await self._goals.get(owner_id=owner_id, session_id=session_id)
        if current is None or current.id != goal.id or current.phase is not GoalPhase.ACTIVE:
            return False
        if current.rounds_left <= 0:
            await self._block_at_cap(
                owner_id=owner_id, session_id=session_id, run_id=view.run_id, goal=current
            )
            return False
        round_number = current.rounds_started + 1
        workspaces = tuple(str(item) for item in (view.request.get("workspaces") or ()))
        reservation: GoalRoundReservation | None = None
        # 1. Reserve the continuation. A row already present is another attempt's work (or its
        #    undelivered continuation): this attempt must not admit a round beside it.
        if self._round_turns is not None:
            try:
                if await self._round_turns.goal_round_pending(
                    owner_id=owner_id, parent_run_id=view.run_id
                ):
                    return False
            except Exception:
                logger.warning("Could not check a pending goal continuation", exc_info=True)
                return False
            try:
                reservation = await self._round_turns.request_goal_round(
                    owner_id=owner_id,
                    parent_run_id=view.run_id,
                    prompt=goal_round_prompt(goal=current, round_number=round_number),
                    workspaces=workspaces,
                    goal_id=current.id,
                    goal_revision=current.revision + 1,
                    goal_round=round_number,
                )
            except Exception:
                logger.warning("Could not reserve a goal continuation", exc_info=True)
                return False
            if reservation is None:
                # Another reservation holds the row (it may be a legacy non-Web Run, which has
                # no row): only the absent row may fall through to the plain Run path.
                try:
                    if await self._round_turns.goal_round_pending(
                        owner_id=owner_id, parent_run_id=view.run_id
                    ):
                        return False
                except Exception:
                    logger.warning("Could not re-check a pending goal continuation", exc_info=True)
                    return False
        # 2. Admit the round the reservation names.
        try:
            started, admitted_number = await self._goals.admit_round(
                owner_id=owner_id,
                session_id=session_id,
                run_id=view.run_id,
                goal_id=current.id,
                revision=current.revision,
            )
        except GoalError:
            # A lost CAS is the normal way two attempts settle: if a delivery admitted and took
            # the reserved round its transaction consumed the row. Only a withdrawn row refunds.
            if reservation is not None:
                withdrawn = await self._withdraw_reservation(owner_id, view.run_id, reservation)
                if not withdrawn:
                    return True
            return False
        except Exception:
            logger.warning("Goal round admission failed", exc_info=True)
            if reservation is not None:
                await self._withdraw_reservation(owner_id, view.run_id, reservation)
            return False
        if admitted_number != round_number:
            await self._release(owner_id, session_id, view.run_id, started)
            if reservation is not None:
                await self._withdraw_reservation(owner_id, view.run_id, reservation)
            return False
        if reservation is not None:
            # 3. The reservation is the round's delivery request; admit it now when the Session
            #    is at rest. A busy Session leaves it pending for the next settlement.
            if self._deliver_pending is None:
                withdrawn = await self._withdraw_reservation(owner_id, view.run_id, reservation)
                if withdrawn:
                    await self._release(owner_id, session_id, view.run_id, started)
                return False
            try:
                await self._deliver_pending(owner_id, view.run_id)
            except Exception:
                withdrawn = await self._withdraw_reservation(owner_id, view.run_id, reservation)
                if withdrawn:
                    await self._release(owner_id, session_id, view.run_id, started)
                logger.warning(
                    "Could not deliver the queued goal round of %s", started.id, exc_info=True
                )
                return False
            return True
        # 4. No Web conversation: the Run itself is the round.
        prompt = goal_round_prompt(goal=started, round_number=admitted_number)
        # The Web conversation stores a submission id as a UUID; derive one deterministically so a
        # retry of the same round replays instead of creating a second turn.
        key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"dlightrag-goal:{started.id}:{admitted_number}"))
        try:
            await self._answers.create(
                request=_round_request(
                    request=view.request,
                    prompt=prompt,
                    goal=started,
                    round_number=admitted_number,
                    session_id=session_id,
                    parent_run_id=view.run_id,
                ),
                owner_id=owner_id,
                caller=None,
                idempotency_key=key,
            )
            return True
        except ActiveRunLimitError, RunAdmissionLimitExceededError, SessionWorkBusyError:
            # A transient refusal costs the goal its admission but not a round: give it back so
            # the next settlement (or a human turn) can try again from the durable row.
            await self._release(owner_id, session_id, view.run_id, started)
            logger.warning("Goal round %s of %s was refused admission", admitted_number, started.id)
            return False
        except Exception:
            await self._release(owner_id, session_id, view.run_id, started)
            logger.warning(
                "Goal round %s of %s could not be created",
                admitted_number,
                started.id,
                exc_info=True,
            )
            return False

    async def _withdraw_reservation(
        self, owner_id: str, parent_run_id: str, reservation: GoalRoundReservation
    ) -> bool:
        """Remove this attempt's pending reservation; False when a delivery already took it.

        The withdrawal names the reservation's own request id and digest, so only the row this
        attempt wrote can be deleted — never a newer attempt's reservation.
        """
        if self._round_turns is None:
            return False
        try:
            return await self._round_turns.withdraw_goal_round(
                owner_id=owner_id,
                parent_run_id=parent_run_id,
                request_id=reservation.request_id,
                prompt_digest=reservation.prompt_digest,
            )
        except Exception:
            logger.warning("Could not withdraw a goal continuation reservation", exc_info=True)
            return False

    async def _release(self, owner_id: str, session_id: str, run_id: str, goal: Goal) -> None:
        try:
            await self._goals.release_round(
                owner_id=owner_id,
                session_id=session_id,
                run_id=run_id,
                goal_id=goal.id,
                revision=goal.revision,
            )
        except Exception:
            logger.warning("Could not release an admitted goal round of %s", goal.id, exc_info=True)

    async def _block_at_cap(
        self, *, owner_id: str, session_id: str, run_id: str, goal: Goal
    ) -> None:
        """End an active goal at its round cap as blocked, so the reader sees why it stopped."""
        try:
            await self._goals.block_round_limit(
                owner_id=owner_id,
                session_id=session_id,
                run_id=run_id,
                goal_id=goal.id,
                revision=goal.revision,
            )
        except Exception:
            logger.warning(
                "Could not mark goal %s blocked at its round cap", goal.id, exc_info=True
            )


def _session_id(request: Mapping[str, Any]) -> str:
    return str(request.get("agent_session_id") or "")


def _ended_on_token_limit(view: RunView) -> bool:
    """Whether a terminal Run's stored result trace says the model hit its output token limit."""
    result = view.result
    if not isinstance(result, Mapping):
        return False
    trace = result.get("trace")
    if not isinstance(trace, Mapping):
        return False
    return trace.get("agent_stop_reason") in _TOKEN_LIMIT_STOP_REASONS


def _round_serves_goal(view: RunView, goal: Goal) -> bool:
    """Whether this goal round's observation still belongs to the goal the conversation carries.

    It names the same goal, and the goal has not moved past the revision the round was admitted
    at: a round admitted, then paused and resumed before it settled, is stale smoke — its
    cancellation must not re-pause the goal the human just resumed.
    """
    if str(view.request.get("goal_id") or "") != goal.id:
        return False
    admitted_revision = view.request.get("goal_revision")
    return not (isinstance(admitted_revision, int) and admitted_revision < goal.revision)


def _round_request(
    *,
    request: Mapping[str, Any],
    prompt: str,
    goal: Goal,
    round_number: int,
    session_id: str,
    parent_run_id: str,
) -> AnswerRequest:
    """The next Run asks for the same conversation, workspaces and retrieval as the one it continues.

    It carries the fields a round needs from the parent request — workspaces, retrieval controls,
    requested Skill, the caller's effort and the lane — plus the goal identity this round serves.
    A field the projection does not carry is deliberately not replayed; attachments and history
    stay behind because the Session holds the conversation and a round re-reads the workspace.
    """
    projected = AnswerRunRequest.from_request(request)
    return AnswerRequest(
        query=prompt,
        workspaces=tuple(str(workspace) for workspace in request.get("workspaces") or ()),
        episodic_summary=projected.episodic_summary,
        retrieval=projected.retrieval,
        filters=(
            MetadataFilter.model_validate(dict(projected.filters))
            if projected.filters is not None
            else None
        ),
        semantic_highlights=projected.semantic_highlights,
        mode=_ROUND_MODE,
        agent_session_id=session_id,
        agent_lane_id=projected.agent_lane_id,
        requested_skill=projected.requested_skill,
        effort=projected.effort,
        parent_run_id=parent_run_id,
        continuation_kind="goal_round",
        goal_id=goal.id,
        goal_revision=goal.revision,
        goal_round=round_number,
    )


def goal_round_prompt(*, goal: Goal, round_number: int) -> str:
    """The harness prose one continuation round starts with. It names no Skill."""
    lines = [
        "<goal_round>",
        f'Objective: "{goal.objective}"',
        f"Round: {round_number}/{goal.max_rounds}",
        "",
        "Continue working toward the objective. This round continues the same Session: the "
        "conversation history, the committed workspace and the session notes are the context an "
        "earlier round left behind, so a round is a budget boundary, not a fresh start. Inspect "
        "the current state before acting and do not redo work an earlier round already completed. "
        "Use only the workspace this Run is bound to: run `pwd` before file commands, use paths "
        "relative to that root, and never reuse an absolute path copied from earlier-round "
        "history, because a previous round's path may not exist in this Run. Read notes/goal.md "
        "when it exists; it is the handoff earlier rounds wrote for this one. If work remains "
        "when this round ends, write or refresh a terse notes/goal.md stating what "
        "is done, what is next, and any decision a later round must keep. If the objective cannot "
        "be finished, call update_goal blocked with the concrete condition; do not hand-write a "
        "deliverable a loaded Skill produces with its own tools, though a plain note "
        "(notes/goal.md or the answer text) is allowed. A Skill loaded in an "
        "earlier round is still context, but only a load_skill call in this Run plus the Skill's "
        "own verification passing in this Run makes a delivery count; reload and re-run it only "
        "when this Run itself must attach or attest the deliverable. Make concrete, "
        "verifiable progress. Call get_goal before updating or completing the goal, and complete "
        "only when the whole objective is achieved and verified; if work remains, leave the goal "
        "active so the next round continues.",
        "</goal_round>",
    ]
    return "\n".join(lines)


__all__ = ["GoalRoundDriver", "GoalRoundReservation", "GoalRoundTurns", "goal_round_prompt"]
