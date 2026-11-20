# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Session Memory Pass (ADR 0045): one Session's conversation, consolidated into memory.

Two actors write Profile Memory. The Agent — the model answering in a Run — holds
the memory tools. The pass is the other: it holds no tools, calls a model of its
own on the ``query`` role, and produces a verdict that the store applies. A
successful Answer Run schedules it; it reads the Session's own user and assistant
turns past its watermark, however few, and asks what is worth remembering. What is
longer than one model call takes is read in batches, one after another under the one
claim, until the pass has caught up with the Session, so no text is left waiting for
a later Run. Each call is also shown the turns just before its batch, as context to
read the batch by and nothing to remember on its own, and never text from before the
owner last turned Profile Memory back on: turning it on restarts every Session at its
latest entry, so nothing written while it was off is read at all. Zero records is the
ordinary outcome, and nothing a pass does can fail the Run that scheduled it.

A batch is two-phase so that a model's nondeterminism never meets the store's exact
idempotency: its verdict's plan is stored before any write applies, and a pass that
finds a stored plan resumes it rather than asking the model again, then reads on.

A pass is scheduled in this process and can be lost with it: a restart cancels the passes
it held, and a failed model call leaves its batch unread. So the database is also the queue.
A Session whose watermark is behind its own unread text needs a pass, and the runner looks
for those on a timer: shortly after it starts and every ``SESSION_SWEEP_INTERVAL_SECONDS``
after, it asks the store for a few of them and hands each to the same path a successful Run
takes, so the per-process bound, the per-Session coalescing and the cross-process claim apply
unchanged. A pass that fails counts the failure on the Session's row, and the sweep leaves
that Session alone for longer after each one; a Session that goes on being read clears it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from dlightrag_memory import Memory, MemoryProvenance, MemoryWriteRejectedError
from dlightrag_memory.consolidation import (
    CONTEXT_ROUNDS,
    ConsolidationVerdict,
    ConversationTurn,
    consolidation_request,
    plan_consolidation,
    planned_operation,
)

from dlightrag.engine.ai.scheduler import model_call_scope
from dlightrag.engine.ai.structured import StructuredOutput
from dlightrag.engine.ai.tokens import estimate_tokens
from dlightrag.engine.answer.memory import (
    MEMORY_INACTIVE,
    MemoryCapability,
    PendingSession,
    SessionMemoryPlan,
    SessionMemoryProgress,
    SessionTurn,
)
from dlightrag.engine.answer.runs.routing import RoutingRecord
from dlightrag.engine.runtime.coordinator import SettledRun

logger = logging.getLogger(__name__)

#: Passes one process runs at once; a burst of settling Runs cannot starve live
#: answers of model capacity, which the passes share through the model scheduler.
SESSION_PASS_CONCURRENCY = 2
#: How long a claim holds before another process may take the Session's pass over. A pass
#: renews it before every batch after its first, so it bounds one batch's work, not the pass.
SESSION_PASS_CLAIM_SECONDS = 300.0
#: How soon after it starts a process first looks for Sessions whose pass was lost: late enough that
#: the rest of startup is not competing with it, and soon enough that a deploy that cancelled passes
#: does not leave their Sessions waiting a whole interval.
SESSION_SWEEP_FIRST_SECONDS = 30.0
#: How often it looks after that. Low on purpose: a lost pass costs a conversation's tail a few
#: minutes of waiting, and the look itself reads every Session's counter.
SESSION_SWEEP_INTERVAL_SECONDS = 600.0
#: The most Sessions one look hands to passes. Several processes sweep, and each may be offered the
#: same Sessions; the claim lets one pass run and sends the others away, so a small batch keeps the
#: waste small, and the rest are offered at the next look.
SESSION_SWEEP_BATCH = 20
#: How long a sweep leaves a Session alone after its pass first fails, and the longest it ever does:
#: the wait doubles with each failure and stops at the ceiling. A pass a successful Run schedules
#: ignores it, since a new Run is new evidence; only a sweep, which has none, waits.
SESSION_BACKOFF_FIRST_SECONDS = 600.0
SESSION_BACKOFF_MAX_SECONDS = 6 * 60 * 60.0
# Turns fetched past the watermark for one batch, before the token bound applies.
_TURN_FETCH_LIMIT = 2_000
# Batches one pass reads before it stops and leaves the rest to the next Run's pass, which
# continues from the watermark. A safety valve against a loop that never catches up, not a
# limit a conversation meets: one batch is up to ``max_batch_tokens`` of user and assistant text.
_MAX_BATCHES_PER_PASS = 25
# Candidate records the pass shows the model to compare the new material against.
_CANDIDATE_LIMIT = 20
# The tail of the context and the new material, together, that the candidate search is run on:
# a reply as short as "yes, that" is searched by what it answers.
_SEARCH_QUERY_CHARS = 2_000
# Every pass waits as this one scheduling owner, so passes together take a fair
# share of model capacity beside the owners whose answers are live.
_SCHEDULING_OWNER = "dlightrag-session-memory-pass"
_VERDICT = StructuredOutput(name="session_memory_verdict", schema=ConsolidationVerdict)

#: How a pass ended. ``applied``: it read and applied at least one batch of new text.
#: ``resumed``: it only settled a plan an earlier pass had stored. ``nothing_new``: the Session
#: held no text past the watermark. ``disabled``: the owner's Profile Memory is off, so nothing
#: more was read. ``model_failed``: a batch's model call failed, so that batch is left for the
#: next Run's pass or a sweep once the Session's wait is over (the batches before it stand).
#: ``superseded``: the claim was lost to another process. ``busy``: another process holds the claim.
PassOutcome = Literal[
    "busy", "disabled", "nothing_new", "model_failed", "applied", "resumed", "superseded"
]


class SessionMemoryStore(Protocol):
    """The Run store's view of Sessions a pass needs: routing, the transcript, progress."""

    async def load_routing(self, *, owner_id: str, run_id: str) -> RoutingRecord | None: ...

    async def claim_session_memory(
        self, *, owner_id: str, session_id: str, holder: str, lease_seconds: float
    ) -> SessionMemoryProgress | None: ...

    async def renew_session_memory(
        self, *, owner_id: str, session_id: str, holder: str, lease_seconds: float
    ) -> bool: ...

    async def release_session_memory(
        self, *, owner_id: str, session_id: str, holder: str
    ) -> None: ...

    async def store_session_memory_plan(
        self, *, owner_id: str, session_id: str, holder: str, plan: SessionMemoryPlan
    ) -> bool: ...

    async def advance_session_memory(
        self, *, owner_id: str, session_id: str, holder: str, through: int
    ) -> int | None: ...

    async def session_turns_after(
        self, *, owner_id: str, session_id: str, after: int, limit: int
    ) -> Sequence[SessionTurn]: ...

    async def session_turns_before(
        self, *, owner_id: str, session_id: str, before: int, rounds: int
    ) -> Sequence[SessionTurn]: ...

    async def pending_session_memory(self, *, limit: int) -> Sequence[PendingSession]: ...

    async def back_off_session_memory(
        self,
        *,
        owner_id: str,
        session_id: str,
        holder: str,
        first_seconds: float,
        max_seconds: float,
    ) -> None: ...


class MemoryCapabilityGate(Protocol):
    """An owner's Profile Memory switch, read up front and rechecked inside each write."""

    async def capability(self, *, owner_id: str) -> MemoryCapability: ...

    async def capability_current(
        self, *, owner_id: str, epoch: int, settlement: object | None = None
    ) -> bool: ...


class VerdictModel(Protocol):
    """The model a pass asks, owned by the pass: called with a structured output, then closed."""

    async def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SessionPassSettings:
    max_batch_tokens: int
    session_record_limit: int


class SessionMemoryPass:
    """Run one Session's pass to completion under a claim."""

    def __init__(
        self,
        *,
        memory: Memory,
        store: SessionMemoryStore,
        gate: MemoryCapabilityGate,
        model: VerdictModel,
        settings: SessionPassSettings,
        holder: str,
    ) -> None:
        self._memory = memory
        self._store = store
        self._gate = gate
        self._model = model
        self._settings = settings
        self._holder = holder

    async def run(self, *, owner_id: str, session_id: str, run_id: str) -> PassOutcome:
        """Run one pass; what it writes is attributed to ``run_id``, the Run that asked for it."""
        progress = await self._store.claim_session_memory(
            owner_id=owner_id,
            session_id=session_id,
            holder=self._holder,
            lease_seconds=SESSION_PASS_CLAIM_SECONDS,
        )
        if progress is None:
            return "busy"
        try:
            outcome = await self._run_claimed(owner_id, session_id, run_id, progress)
            if outcome == "superseded":
                # Rare, and the one outcome that means a pass outran its lease or was taken over:
                # named with its Session so a deployment where it recurs can be diagnosed.
                logger.warning(
                    "Session Memory Pass lost its claim and stopped",
                    extra={"session_id": str(session_id)},
                )
            return outcome
        finally:
            await self._store.release_session_memory(
                owner_id=owner_id, session_id=session_id, holder=self._holder
            )

    async def aclose(self) -> None:
        await self._model.aclose()

    async def _run_claimed(
        self, owner_id: str, session_id: str, run_id: str, progress: SessionMemoryProgress
    ) -> PassOutcome:
        """The claimed pass, with a failure counted against the Session before the claim is let go.

        A model call that failed and an error nothing here expected both leave text unread without
        moving the watermark, so a sweep must not offer the Session straight back. Losing the
        claim, finding nothing to read and finding memory off are no failure of this Session's.
        """
        try:
            outcome = await self._claimed(owner_id, session_id, run_id, progress)
        except Exception:
            await self._back_off(owner_id, session_id)
            raise
        if outcome == "model_failed":
            await self._back_off(owner_id, session_id)
        return outcome

    async def _back_off(self, owner_id: str, session_id: str) -> None:
        """Count a failure on the Session's row. Never raises: it runs where one already has."""
        try:
            await self._store.back_off_session_memory(
                owner_id=owner_id,
                session_id=session_id,
                holder=self._holder,
                first_seconds=SESSION_BACKOFF_FIRST_SECONDS,
                max_seconds=SESSION_BACKOFF_MAX_SECONDS,
            )
        except Exception:
            logger.warning(
                "Session Memory Pass could not record its failure",
                extra={"session_id": str(session_id)},
                exc_info=True,
            )

    async def _claimed(
        self, owner_id: str, session_id: str, run_id: str, progress: SessionMemoryProgress
    ) -> PassOutcome:
        """Settle a stored plan, then read batch after batch until the Session is caught up.

        The watermark each step reads from is the one the store answers with, never one the pass
        counted for itself: a restart may have moved it past what the pass last applied.
        """
        outcome: PassOutcome = "nothing_new"
        watermark = progress.read_through
        if progress.plan is not None:
            # A stored plan was read by a pass that is gone, and the epoch it was read under was
            # not stored with it, so it is guarded by the epoch as it stands now.
            resumed_under = await self._gate.capability(owner_id=owner_id)
            moved = await self._apply(owner_id, session_id, progress.plan, resumed_under)
            if moved is None:
                return "superseded"
            watermark = moved
            outcome = "resumed"

        batches = 0
        while True:
            # Taken before every batch is read. An owner who switches memory off mid-pass is not
            # read on: nothing is advanced, because turning memory back on restarts every Session
            # at its latest entry. And the epoch in it is the one the batch's writes are guarded
            # by: if the owner turns memory off and on, or clears it, while the model thinks, the
            # epoch has moved and every write is refused, yet the watermark still moves past the
            # batch. Text read under a stale epoch is never read again, as text held while memory
            # was off is never mined.
            state = await self._gate.capability(owner_id=owner_id)
            if not state.enabled:
                return "disabled"

            turns = await self._store.session_turns_after(
                owner_id=owner_id,
                session_id=session_id,
                after=watermark,
                limit=_TURN_FETCH_LIMIT,
            )
            batch, through = _batch(turns, self._settings.max_batch_tokens)
            if not batch:
                return outcome
            if batches == _MAX_BATCHES_PER_PASS:
                logger.warning(
                    "Session Memory Pass stopped at its batch bound; the next pass continues "
                    "from the watermark",
                    extra={"session_id": str(session_id), "batches": batches},
                )
                return outcome
            if batches and not await self._store.renew_session_memory(
                owner_id=owner_id,
                session_id=session_id,
                holder=self._holder,
                lease_seconds=SESSION_PASS_CLAIM_SECONDS,
            ):
                return "superseded"

            plan = await self._judge(
                owner_id, session_id, run_id, batch, read_from=watermark + 1, through=through
            )
            if plan is None:
                return "model_failed"
            if not await self._store.store_session_memory_plan(
                owner_id=owner_id, session_id=session_id, holder=self._holder, plan=plan
            ):
                return "superseded"
            moved = await self._apply(owner_id, session_id, plan, state)
            if moved is None:
                return "superseded"
            watermark = moved
            batches += 1
            outcome = "applied"

    async def _judge(
        self,
        owner_id: str,
        session_id: str,
        run_id: str,
        batch: tuple[ConversationTurn, ...],
        *,
        read_from: int,
        through: int,
    ) -> SessionMemoryPlan | None:
        """The plan the model's verdict on one batch becomes, or None when the model fails.

        The batch covers the entries from ``read_from`` through ``through``. The model is also
        shown the last ``CONTEXT_ROUNDS`` rounds before ``read_from``, as context to read the
        batch by, and the search for the memories it compares the batch with reads that context as well.
        """
        earlier = await self._store.session_turns_before(
            owner_id=owner_id, session_id=session_id, before=read_from, rounds=CONTEXT_ROUNDS
        )
        # Searched by the end of the context and the batch together, so that a reply too short
        # to name what it means still finds the memories about what it answers.
        query = "\n".join(turn.text for turn in (*earlier, *batch))[-_SEARCH_QUERY_CHARS:]
        candidates, own = await asyncio.gather(
            self._memory.search(owner_id=owner_id, query=query, limit=_CANDIDATE_LIMIT),
            self._memory.session_records(
                owner_id=owner_id, session_id=session_id, origin_kind="session_pass"
            ),
        )
        request = consolidation_request(
            turns=batch,
            context_turns=[ConversationTurn(role=turn.role, text=turn.text) for turn in earlier],
            candidates=candidates,
            session_records=own,
            record_limit=self._settings.session_record_limit,
        )
        try:
            with model_call_scope(_SCHEDULING_OWNER):
                raw = await self._model(list(request.messages), structured_output=_VERDICT)
            verdict = cast(ConsolidationVerdict, _VERDICT.parse(raw))
        except Exception:
            # The watermark stays: the next successful Run asks again over this range. The
            # session is named so a deployment whose pass keeps failing is diagnosable from
            # one line rather than from a watermark that never moves.
            logger.warning(
                "Session Memory Pass model call failed",
                extra={"session_id": str(session_id)},
                exc_info=True,
            )
            return None
        return SessionMemoryPlan(
            read_from=read_from,
            read_through=through,
            writes=plan_consolidation(
                verdict,
                handles=request.handles,
                record_limit=self._settings.session_record_limit,
                owner_words=request.owner_words,
            ),
            run_id=run_id,
        )

    async def _apply(
        self, owner_id: str, session_id: str, plan: SessionMemoryPlan, state: MemoryCapability
    ) -> int | None:
        """Apply a stored plan under ``state`` and move the watermark past it.

        ``state`` is the capability the plan was made under, whose epoch guards every write:
        a write is refused once the owner's switch has gone off or the epoch has moved, and
        a refused write never stops the rest, nor the watermark. Answers where the watermark
        stands, or None once the claim is gone.
        """
        span = f"{session_id}:{plan.read_from}-{plan.read_through}"
        # The plan's own Run, not the caller's: a plan resumed by a pass another Run asked for
        # applies under the Run it was stored for, which keeps every write's key and fingerprint.
        provenance = MemoryProvenance(
            origin_kind="session_pass", origin_id=span, run_id=plan.run_id, session_id=session_id
        )

        async def guard(settlement: object | None) -> None:
            if not await self._gate.capability_current(
                owner_id=owner_id, epoch=state.epoch, settlement=settlement
            ):
                raise MemoryWriteRejectedError(MEMORY_INACTIVE)

        if state.enabled:
            for index, write in enumerate(plan.writes):
                operation = planned_operation(
                    write,
                    index=index,
                    owner_id=owner_id,
                    provenance=provenance,
                    pass_key=f"pass:{span}",
                )
                try:
                    receipt = await self._memory.apply(operation, guard=guard)
                except (MemoryWriteRejectedError, ValueError) as exc:
                    # One refused write never stops the rest, nor the watermark.
                    logger.info("Session Memory Pass write refused: %s", exc)
                    continue
                if receipt.outcome == "conflict":
                    logger.info("Session Memory Pass write conflicted; the profile moved")
        return await self._advance(owner_id, session_id, plan.read_through)

    async def _advance(self, owner_id: str, session_id: str, through: int) -> int | None:
        return await self._store.advance_session_memory(
            owner_id=owner_id, session_id=session_id, holder=self._holder, through=through
        )


def _batch(
    turns: Sequence[SessionTurn], max_tokens: int
) -> tuple[tuple[ConversationTurn, ...], int]:
    """The turns one model call reads, and the sequence it reads through.

    The first turn is always taken, so one turn larger than the bound still moves
    the watermark.
    """
    batch: list[ConversationTurn] = []
    through = 0
    tokens = 0
    for turn in turns:
        cost = estimate_tokens(turn.text)
        if batch and tokens + cost > max_tokens:
            break
        batch.append(ConversationTurn(role=turn.role, text=turn.text))
        through = turn.sequence
        tokens += cost
    return tuple(batch), through


class SessionMemoryPassRunner:
    """Schedule passes in this process: off successful Answer Runs, and off a timer.

    The coordinator calls ``run_settled`` and returns at once; the pass runs
    detached and never holds Run settlement open. One Session runs one pass at a
    time here, and a Run that settles while its Session's pass is running asks
    for one more pass afterwards, so its turns are not left waiting for another
    Run; the Runs that asked while it ran become that one pass, attributed to the
    latest of them. Across processes, the claim on the progress row is what serializes.

    A pass scheduled this way can be lost with the process, or end with its model call failed. So
    ``start`` also begins a sweep: the store is asked, shortly after and then every
    ``SESSION_SWEEP_INTERVAL_SECONDS``, for a few Sessions whose watermark is behind their own
    unread text, and each is handed to a pass attributed to its latest succeeded Run. A swept
    pass is an ordinary one: it takes a slot of the same bound and the same claim, and a Session
    already running here is left alone, since that pass reads to the end by itself. The runner
    owns its pass, and closing it stops the sweep and closes the pass's model.
    """

    def __init__(
        self,
        *,
        session_pass: SessionMemoryPass,
        store: SessionMemoryStore,
        concurrency: int = SESSION_PASS_CONCURRENCY,
    ) -> None:
        self._pass = session_pass
        self._store = store
        self._slots = asyncio.Semaphore(concurrency)
        self._tasks: set[asyncio.Task[None]] = set()
        self._running: set[tuple[str, str]] = set()
        # The latest Run that asked for another pass of a Session while its pass was running.
        self._again: dict[tuple[str, str], str] = {}
        self._sweeper: asyncio.Task[None] | None = None
        self._closing = False

    def run_settled(self, run: SettledRun) -> None:
        # Only a succeeded Answer Run has an Agent Session to consolidate.
        if self._closing or run.status != "succeeded" or run.run_kind != "answer":
            return
        self._schedule(self._for_run, run.owner_id, run.run_id)

    def start(self) -> None:
        """Begin sweeping for lost passes. Once started, it goes on until the runner is closed."""
        if self._closing or self._sweeper is not None:
            return
        self._sweeper = asyncio.create_task(
            self._sweep_forever(), name="dlightrag-session-memory-sweep"
        )

    async def sweep(self) -> int:
        """Hand the Sessions the store finds behind to passes, and answer how many it handed.

        Each is attributed to its latest succeeded Run, as if that Run had just succeeded. One
        already running here is skipped without asking for another pass after it.
        """
        scheduled = 0
        for pending in await self._store.pending_session_memory(limit=SESSION_SWEEP_BATCH):
            if self._closing:
                break
            if (pending.owner_id, pending.session_id) in self._running:
                continue
            self._schedule(self._swept, pending)
            scheduled += 1
        if scheduled:
            logger.info("Session Memory sweep handed %d Session(s) to a pass", scheduled)
        return scheduled

    async def _sweep_forever(self) -> None:
        delay = SESSION_SWEEP_FIRST_SECONDS
        while True:
            await asyncio.sleep(delay)
            delay = SESSION_SWEEP_INTERVAL_SECONDS
            try:
                await self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Session Memory sweep failed", exc_info=True)

    def _schedule[**P](
        self, work: Callable[P, Coroutine[Any, Any, None]], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        task = asyncio.create_task(
            self._detached(work, *args, **kwargs), name="dlightrag-session-memory-pass"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _detached[**P](
        self, work: Callable[P, Coroutine[Any, Any, None]], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        """Run scheduled work to its end: what it raises is logged, never propagated."""
        try:
            await work(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Session Memory Pass failed", exc_info=True)

    async def _for_run(self, owner_id: str, run_id: str) -> None:
        routing = await self._store.load_routing(owner_id=owner_id, run_id=run_id)
        if routing is None:
            return
        await self._pass_session(owner_id, routing.agent_session_id, run_id, follow_up=True)

    async def _swept(self, pending: PendingSession) -> None:
        await self._pass_session(
            pending.owner_id, pending.session_id, pending.run_id, follow_up=False
        )

    async def _pass_session(
        self, owner_id: str, session_id: str, run_id: str, *, follow_up: bool
    ) -> None:
        """Run one pass of the Session, unless one is running here already.

        A Run that asks while the Session's pass is running becomes one more pass after it
        (``follow_up``), attributed to the latest Run that asked; a sweep, whose request is not
        new text, is simply dropped, for the running pass reads to the end by itself.
        """
        key = (owner_id, session_id)
        if key in self._running:
            if follow_up:
                self._again[key] = run_id
            return
        self._running.add(key)
        try:
            asking = run_id
            while True:
                self._again.pop(key, None)
                async with self._slots:
                    await self._pass.run(owner_id=owner_id, session_id=session_id, run_id=asking)
                later = self._again.get(key)
                if later is None or self._closing:
                    break
                asking = later
        finally:
            self._running.discard(key)

    async def aclose(self) -> None:
        self._closing = True
        sweeper, self._sweeper = self._sweeper, None
        tasks = tuple(self._tasks) if sweeper is None else (sweeper, *self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._pass.aclose()


__all__ = [
    "SESSION_BACKOFF_FIRST_SECONDS",
    "SESSION_BACKOFF_MAX_SECONDS",
    "SESSION_PASS_CONCURRENCY",
    "SESSION_SWEEP_BATCH",
    "SESSION_SWEEP_FIRST_SECONDS",
    "SESSION_SWEEP_INTERVAL_SECONDS",
    "MemoryCapabilityGate",
    "PassOutcome",
    "SessionMemoryPass",
    "SessionMemoryPassRunner",
    "SessionMemoryStore",
    "SessionPassSettings",
    "VerdictModel",
]
