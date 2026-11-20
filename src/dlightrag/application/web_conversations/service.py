# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Application-owned persistence lifecycle for durable Web Conversations."""

import asyncio
import datetime
import hashlib
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from dlightrag.application.access.principal import (
    UserContext,
    owner_id_from_user,
)
from dlightrag.application.answer.goal_rounds import GoalRoundReservation
from dlightrag.application.answer_runs.service import (
    AnswerHistoryResource,
    AnswerRequest,
    AnswerRunAcceptor,
    AnswerService,
    RunResourceDescriptor,
)
from dlightrag.application.errors import ApplicationInputError
from dlightrag.application.runs import RunView
from dlightrag.engine.ai.media import thumbnail_bytes
from dlightrag.engine.answer.client_contracts import AnswerEffort
from dlightrag.engine.answer.execution.connection_binding import RunConnectionBinding
from dlightrag.engine.answer.goal import (
    DEFAULT_MAX_ROUNDS,
    GOAL_NOT_FOUND,
    Goal,
    GoalAction,
    GoalError,
    GoalPhase,
)
from dlightrag.engine.answer.goal_service import AnswerGoalService
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.answer.runs.routing import RoutingAcceptance
from dlightrag.engine.runtime.contracts import RunKind
from dlightrag.engine.runtime.records import (
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    SessionWorkBusyError,
    parse_run_id,
    run_request_fingerprint,
)

from .models import (
    RUN_ACTIVITY_DEFAULT_LIMIT,
    RUN_ACTIVITY_MAX_LIMIT,
    AgentspaceFile,
    AgentspaceFileVersion,
    AgentspaceHead,
    AgentspaceSnapshot,
    AnswerTurnCreation,
    CommandTone,
    ConversationCommand,
    ConversationCursor,
    ConversationCursorCodec,
    ConversationCursorError,
    ConversationGoal,
    ConversationHead,
    ConversationHistoryCursor,
    ConversationHistoryCursorCodec,
    ConversationHistoryPage,
    ConversationHistoryPageRequest,
    ConversationPage,
    ConversationPageRequest,
    ConversationSessionBusy,
    ConversationSummary,
    GoalRoundGuard,
    GoalRoundRequest,
    LinkedTurn,
    LiveAnswerRun,
    QueuedDeliveryGuard,
    QueuedMessage,
    SessionHeadReclaimer,
    SubmissionSeed,
    UnknownConversationCommand,
    WebConversationStore,
    WebRunActivity,
    queued_delivery_digest,
)

logger = logging.getLogger(__name__)


GoalCommandAction = Literal["create", "edit", "pause", "resume", "clear"]

#: The server-owned commands this revision registers. One entry is the whole catalog.
WEB_COMMAND_NAMES = frozenset({"goal"})

#: What a bare ``/goal`` says when there is no goal to show.
_GOAL_STATUS_NONE = "No goal in this conversation. /goal <objective> creates one."
_GOAL_CLEARED = "Goal cleared."
_GOAL_FAILED = "The goal command failed."
#: Pause stops immediately: the phase change stands and the running goal round is cancelled.
_GOAL_PAUSE_NOTE = (
    "Paused. The goal round in progress was cancelled; no next round will start. "
    "Your own running turn is not stopped."
)
_GOAL_COMMAND_HELP = (
    "Commands: /goal edit <objective>, /goal cap N, /goal pause, /goal resume, /goal clear"
)
_GOAL_CAP_NUMBER = re.compile(r"[1-9][0-9]*")
#: JavaScript's ``Number.isSafeInteger`` bound: the parser the composer used to run agreed
#: with the browser on which caps exist, so the server keeps the same admission rule.
_JS_MAX_SAFE_INTEGER = 2**53 - 1


class WebAttachment(Protocol):
    """Admitted browser attachment fields consumed by Application acceptance."""

    @property
    def filename(self) -> str: ...

    @property
    def mime_type(self) -> str: ...

    @property
    def ordinal(self) -> int: ...

    @property
    def byte_size(self) -> int: ...

    @property
    def content_sha256(self) -> str: ...

    @property
    def attachment_bytes(self) -> bytes: ...


class GoalRoundContinuation(Protocol):
    """The round driver, from the Web's side: the two human-control entry points it asks.

    Both return whether a round was started. A refusal is the driver's (the goal is not active, the
    Session is busy, the cap is reached); the Web command reports the stored goal either way.
    """

    async def continue_after_resume(
        self, *, owner_id: str, session_id: str, parent_run_id: str
    ) -> bool: ...

    async def continue_created_goal(
        self,
        *,
        owner_id: str,
        session_id: str,
        conversation_id: str,
        workspaces: tuple[str, ...] = (),
    ) -> bool: ...


class GoalRoundCanceller(Protocol):
    """The Web pause's one runtime action: ask the Run service to cancel a running goal round.

    The return value is advice (a cancellation may be pending until the worker observes it);
    the command only needs the request to have been made. Implementations must be owner-scoped.
    """

    async def cancel(self, *, owner_id: str, run_id: str) -> object: ...


_HISTORY_THUMBNAIL_MAX_PX = 320
_HISTORY_THUMBNAIL_MAX_BYTES = 128 * 1024
_HISTORY_THUMBNAIL_QUALITY = 82
_HISTORY_THUMBNAIL_MIN_QUALITY = 50
_HISTORY_THUMBNAIL_MIN_PX = 64
_PRUNE_INTERVAL_SECONDS = 60 * 60
#: The Inspector's snapshot page: enough lineage to inspect a goal, bounded like other pages.
_WORKSPACE_SNAPSHOT_PAGE = 50
_PRUNE_BATCH_SIZE = 500
_NEW_CONVERSATION_NAMESPACE = UUID("9c0e62a5-a12c-45b2-8aeb-474fc2237cdf")


def _is_image_mime(mime_type: str | None) -> bool:
    return bool(mime_type) and mime_type.lower().startswith("image/")


@dataclass(frozen=True, slots=True)
class WebAnswerSubmission:
    """The run and conversation entry one accepted browser submission created."""

    run: RunView
    turn_id: str
    turn_number: int
    conversation: ConversationSummary
    submission_id: str = ""
    created_at: datetime.datetime | None = None


@dataclass(frozen=True, slots=True)
class WebAnswerDelivery:
    """A human send delivered to the live Run instead of starting a new one.

    ``mode`` is what the caller asked for and the server could honour: ``steer`` injected
    the words into the live Run, ``queue`` holds them for one next-turn prompt after it
    settles. A queued delivery carries the durable row the queue strip reads back; a
    steered delivery names the Run and the control sequence it was given.
    """

    mode: Literal["queue", "steer"]
    queued: QueuedMessage | None = None
    steered_run_id: str | None = None
    control_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class _PreparedSubmission:
    """One browser submission projected into the Answer service contract."""

    request: AnswerRequest


@dataclass(frozen=True, slots=True)
class _ParsedGoalCommand:
    """One ``/goal`` argument line, parsed as the composer's parser parsed it."""

    kind: Literal["status", "command", "invalid"]
    action: GoalCommandAction | None = None
    objective: str | None = None
    max_rounds: int | None = None
    invalid_reason: Literal["cap", "objective"] | None = None
    error_text: str | None = None


@dataclass(frozen=True, slots=True)
class _WebAnswerAcceptor(AnswerRunAcceptor[WebAnswerSubmission]):
    """Atomically link AnswerService acceptance to one browser conversation."""

    store: WebConversationStore
    conversation_id: str
    title_hint: str | None
    create_conversation: bool = False
    forked_from_conversation_id: str | None = None
    queue_guard: QueuedDeliveryGuard | None = None
    goal_round_guard: GoalRoundGuard | None = None

    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        run_kind: RunKind,
    ) -> WebAnswerSubmission | None:
        if run_kind != "answer":
            raise ValueError("Web Answer replay requires an Answer run kind")
        creation = await self.store.replay_answer_turn(
            principal_id=owner_id,
            conversation_id=self.conversation_id,
            submission_id=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
        )
        return None if creation is None else _submission(creation)

    async def accept_run(
        self,
        *,
        envelope: PreparedRunEnvelope,
        run_id: str,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
        connection_bindings: tuple[RunConnectionBinding, ...] = (),
    ) -> WebAnswerSubmission | None:
        creation = await self.store.create_answer_turn(
            principal_id=envelope.submitted_by,
            conversation_id=self.conversation_id,
            submission_id=envelope.submission_key,
            envelope=envelope,
            run_id=run_id,
            artifacts=artifacts,
            references=references,
            title_hint=self.title_hint,
            routing=routing,
            connection_bindings=connection_bindings,
            create_conversation=self.create_conversation,
            forked_from_conversation_id=self.forked_from_conversation_id,
            queue_guard=self.queue_guard,
            goal_round_guard=self.goal_round_guard,
        )
        return None if creation is None else _submission(creation)


class WebConversationService:
    """Map authenticated browser operations onto the scoped persistence store."""

    def __init__(
        self,
        *,
        store: WebConversationStore,
        answers: AnswerService,
        max_attachments: int,
        cursor_secret: bytes,
        goals: AnswerGoalService | None = None,
        goal_rounds: GoalRoundContinuation | None = None,
        goal_round_canceller: GoalRoundCanceller | None = None,
        workspace_heads: SessionHeadReclaimer | None = None,
    ) -> None:
        self._store = store
        self._answers = answers
        self._goals = goals
        self._goal_rounds = goal_rounds
        self._goal_round_canceller = goal_round_canceller
        self._workspace_heads = workspace_heads
        self._max_attachments = max_attachments
        self._cursor_codec = ConversationCursorCodec(cursor_secret)
        self._history_cursor_codec = ConversationHistoryCursorCodec(cursor_secret)
        self._prune_task: asyncio.Task[None] | None = None

    @property
    def cursor_codec(self) -> ConversationCursorCodec:
        """Return the codec shared by this service and its HTTP adapter."""
        return self._cursor_codec

    @property
    def history_cursor_codec(self) -> ConversationHistoryCursorCodec:
        """Return the owned HTTP adapter's conversation-turn cursor codec."""
        return self._history_cursor_codec

    async def start_retention(self) -> None:
        """Start the empty-conversation sweep after schema composition."""
        await self._prune_empty_batch()
        if self._prune_task is None:
            self._prune_task = asyncio.create_task(self._prune_empty_loop())

    async def _prune_empty_batch(self) -> int:
        try:
            return await self._store.prune_empty_conversations(batch_size=_PRUNE_BATCH_SIZE)
        except Exception:
            logger.exception("Failed to prune empty Web conversations")
            return 0

    async def _prune_empty_loop(self) -> None:
        while True:
            await asyncio.sleep(_PRUNE_INTERVAL_SECONDS)
            while await self._prune_empty_batch() >= _PRUNE_BATCH_SIZE:
                await asyncio.sleep(0)

    async def aclose(self) -> None:
        """Stop periodic retention. Safe to call more than once."""
        task = self._prune_task
        self._prune_task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    # ------------------------------------------------------------------
    # Conversation lifecycle
    # ------------------------------------------------------------------

    async def create(self, user: UserContext | None) -> ConversationSummary:
        principal_id = owner_id_from_user(user)
        row = await self._store.create_conversation(principal_id)
        return _conversation_summary(row)

    async def list(
        self,
        user: UserContext | None,
        *,
        page: ConversationPageRequest | None = None,
    ) -> ConversationPage:
        """Return one newest-first keyset page scoped to the authenticated principal.

        This is standard keyset pagination rather than a snapshot: conversations
        created or touched above the supplied cursor after an earlier page are not
        pulled into the older traversal. Already returned rows never repeat.
        """
        principal_id = owner_id_from_user(user)
        requested_page = page or ConversationPageRequest()
        result = await self._store.list_conversations(principal_id, page=requested_page)
        items = tuple(_conversation_summary(row) for row in result.items)
        next_cursor = None
        if result.has_more:
            if not items:
                raise RuntimeError("conversation store reported more rows after an empty page")
            last = items[-1]
            next_cursor = ConversationCursor(
                updated_at=last.updated_at,
                conversation_id=UUID(last.conversation_id),
            )
        return ConversationPage(
            items=items,
            next_cursor=next_cursor,
            fetched_rows=result.fetched_rows,
        )

    async def history(
        self,
        user: UserContext | None,
        conversation_id: str,
        *,
        page: ConversationHistoryPageRequest | None = None,
    ) -> ConversationHistoryPage | None:
        """Return one chronological recent or older presentation page."""
        principal_id = owner_id_from_user(user)
        requested = page or ConversationHistoryPageRequest()
        if (
            requested.cursor is not None
            and str(requested.cursor.conversation_id) != conversation_id
        ):
            raise ConversationCursorError(
                "conversation history cursor belongs to another conversation"
            )
        result = await self._store.history_page(
            principal_id,
            conversation_id,
            page=requested,
        )
        if result is None:
            return None
        next_cursor = None
        if result.next_cursor is not None:
            next_cursor = ConversationHistoryCursor(
                conversation_id=UUID(conversation_id),
                before_sequence=result.next_cursor.before_sequence,
            )
        return ConversationHistoryPage(
            conversation=result.conversation,
            turns=result.turns,
            commands=result.commands,
            next_cursor=next_cursor,
            fetched_rows=result.fetched_rows,
        )

    async def rename(
        self,
        user: UserContext | None,
        conversation_id: str,
        title: str,
    ) -> ConversationSummary | None:
        principal_id = owner_id_from_user(user)
        row = await self._store.rename_conversation(
            principal_id,
            conversation_id,
            title=title,
        )
        return _conversation_summary(row) if row is not None else None

    async def agentspace(
        self, user: UserContext | None, conversation_id: str
    ) -> tuple[tuple[AgentspaceHead, ...], tuple[AgentspaceSnapshot, ...]] | None:
        """The conversation's branch heads and its newest snapshots, for the Inspector."""
        principal_id = owner_id_from_user(user)
        heads = await self._store.agentspace_heads(principal_id, conversation_id)
        if heads is None:
            return None
        snapshots = await self._store.agentspace_snapshots(
            principal_id, conversation_id, limit=_WORKSPACE_SNAPSHOT_PAGE
        )
        if snapshots is None:
            return None
        return heads, snapshots

    async def agentspace_snapshot(
        self, user: UserContext | None, conversation_id: str, run_id: str
    ) -> tuple[AgentspaceFile, ...] | None:
        """The files one snapshot carried; None when conversation or snapshot is absent."""
        principal_id = owner_id_from_user(user)
        return await self._store.agentspace_snapshot_files(principal_id, conversation_id, run_id)

    async def agentspace_file(
        self, user: UserContext | None, conversation_id: str, relative_path: str
    ) -> tuple[AgentspaceFileVersion, ...] | None:
        """Every snapshot that carried a path, oldest first, with which one changed it."""
        principal_id = owner_id_from_user(user)
        return await self._store.agentspace_file_versions(
            principal_id, conversation_id, relative_path
        )

    async def reset_agentspace(self, user: UserContext | None, conversation_id: str) -> bool:
        """Forget the conversation's workspace state; refused while the Session works."""
        principal_id = owner_id_from_user(user)
        try:
            session_id = await self._store.reset_session_agentspace(principal_id, conversation_id)
        except SessionWorkBusyError as exc:
            raise ConversationSessionBusy(exc.active_run_id) from exc
        if session_id is None:
            return False
        heads = self._workspace_heads
        if heads is not None:
            await self._reclaim_heads(
                lambda: heads.reclaim_session(owner_id=principal_id, session_id=session_id)
            )
        return True

    async def delete(self, user: UserContext | None, conversation_id: str) -> bool:
        principal_id = owner_id_from_user(user)
        # The Session's head trees are disk state beside the row; the row cascades with
        # the Session, so the id is read while the conversation still exists.
        session_id = await self._store.conversation_agent_session(principal_id, conversation_id)
        deleted = await self._store.delete_conversation(principal_id, conversation_id)
        heads = self._workspace_heads
        if deleted and session_id is not None and heads is not None:
            await self._reclaim_heads(
                lambda: heads.reclaim_session(owner_id=principal_id, session_id=session_id)
            )
        return deleted

    async def delete_all(self, user: UserContext | None) -> int:
        principal_id = owner_id_from_user(user)
        sessions = await self._store.agent_sessions(principal_id)
        deleted = await self._store.delete_all_conversations(principal_id)
        heads = self._workspace_heads
        if deleted and sessions and heads is not None:
            await self._reclaim_heads(lambda: heads.reclaim_owner(owner_id=principal_id))
        return deleted

    async def _reclaim_heads(self, operation: Callable[[], Awaitable[None]]) -> None:
        """Best-effort disk cleanup: the conversation deletion already committed."""
        try:
            await operation()
        except Exception:
            logger.warning("Could not reclaim deleted Session workspace heads", exc_info=True)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def run_activity(
        self,
        user: UserContext | None,
        *,
        limit: int = RUN_ACTIVITY_DEFAULT_LIMIT,
    ) -> tuple[WebRunActivity, ...]:
        """Return the owner's newest conversation-linked Answer runs, newest first.

        The list is one hard-bounded page with no continuation: it answers "what ran
        recently" and a later poll replaces it whole. Ownership is derived from
        authentication on every read. ``limit`` is validated here for an internal
        caller; the route's ``Query`` is the public bound.
        """
        principal_id = owner_id_from_user(user)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("run activity limit must be an integer")
        if not 1 <= limit <= RUN_ACTIVITY_MAX_LIMIT:
            raise ValueError(f"run activity limit must be between 1 and {RUN_ACTIVITY_MAX_LIMIT}")
        return await self._store.recent_answer_runs(principal_id, limit=limit)

    async def turn_for_run(self, user: UserContext | None, run_id: str) -> LinkedTurn | None:
        """Return the conversation entry an owned run belongs to, if any.

        An unparseable identifier is simply unknown here: every run route reads
        through this projection, so a malformed id is the same opaque miss as a
        pruned or foreign one and never reaches storage.
        """
        principal_id = owner_id_from_user(user)
        if parse_run_id(run_id) is None:
            return None
        return await self._store.find_turn_by_run(principal_id, run_id)

    async def submission(
        self, user: UserContext | None, submission_id: str
    ) -> WebAnswerSubmission | None:
        """Return one owner-scoped accepted browser submission, if it exists."""
        principal_id = owner_id_from_user(user)
        try:
            UUID(submission_id)
        except ValueError:
            return None
        creation = await self._store.find_answer_turn_by_submission(principal_id, submission_id)
        return None if creation is None else _submission(creation)

    async def run_resource(
        self,
        user: UserContext | None,
        run_id: str,
        resource_id: str,
    ) -> tuple[RunResourceDescriptor, bytes] | None:
        """Load one owned run's stored resource bytes by its resource id."""
        principal_id = owner_id_from_user(user)
        if await self.turn_for_run(user, run_id) is None:
            return None
        return await self._answers.read_run_resource(
            owner_id=principal_id,
            run_id=run_id,
            resource_id=resource_id,
        )

    async def run_external_sources(
        self, user: UserContext | None, run_id: str
    ) -> Mapping[str, str]:
        """Return each external URL this owned run holds stored bytes for."""
        return await self._answers.run_external_source_map(
            owner_id=owner_id_from_user(user),
            run_id=run_id,
        )

    async def thumbnail(
        self,
        user: UserContext | None,
        run_id: str,
        resource_id: str,
    ) -> tuple[bytes, str] | None:
        """Derive one bounded UI thumbnail for one stored image resource."""
        stored = await self.run_resource(user, run_id, resource_id)
        if stored is None or not _is_image_mime(stored[0].mime_type):
            return None
        try:
            payload, mime_type = await asyncio.to_thread(
                thumbnail_bytes,
                stored[1],
                max_px=_HISTORY_THUMBNAIL_MAX_PX,
                max_bytes=_HISTORY_THUMBNAIL_MAX_BYTES,
                quality=_HISTORY_THUMBNAIL_QUALITY,
                min_quality=_HISTORY_THUMBNAIL_MIN_QUALITY,
                min_px=_HISTORY_THUMBNAIL_MIN_PX,
            )
        except Exception:
            logger.warning("Failed to derive Web conversation thumbnail", exc_info=True)
            return None
        return payload, mime_type

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def start_answer(
        self,
        user: UserContext | None,
        *,
        conversation_id: str | None,
        submission_id: str,
        query: str,
        workspaces: Sequence[str],
        attachments: Sequence[WebAttachment] = (),
        mode: str | None = None,
        requested_skill: str | None = None,
        effort: AnswerEffort | None = None,
        delivery: Literal["queue", "steer"] | None = None,
    ) -> WebAnswerSubmission | WebAnswerDelivery | None:
        """Create or replay one submission's run and its conversation entry.

        An existing conversation is validated and locked. When ``conversation_id``
        is absent, a stable server-generated id is derived from this owner-wide
        submission key and the conversation is inserted in the same transaction
        as its first run, uploaded bytes, and linked turn.
        """
        principal_id = owner_id_from_user(user)
        create_conversation = conversation_id is None
        if create_conversation:
            conversation_id = _new_conversation_id(principal_id, submission_id)
        idempotency_fingerprint = _web_answer_request_fingerprint(
            conversation_id=conversation_id,
            query=query,
            workspaces=workspaces,
            attachments=attachments,
            mode=mode,
            requested_skill=requested_skill,
            effort=effort,
        )
        replay = await self._store.replay_answer_turn(
            principal_id=principal_id,
            conversation_id=conversation_id,
            submission_id=submission_id,
            idempotency_fingerprint=idempotency_fingerprint,
        )
        if replay is not None:
            return _submission(replay)

        if create_conversation:
            seed = SubmissionSeed(head=_empty_head(principal_id, conversation_id))
        else:
            seed = await self._store.submission_seed(
                principal_id,
                conversation_id,
                attachment_limit=max(0, self._max_attachments - len(attachments)),
            )
            if seed is None:
                return None
        if not create_conversation:
            live = await self._store.live_answer_run(principal_id, conversation_id)
            if live is not None:
                return await self._deliver_to_live_run(
                    principal_id,
                    conversation_id,
                    live=live,
                    query=query,
                    attachments=attachments,
                    delivery=delivery,
                )
        prepared = _prepare_submission(
            query=query,
            workspaces=workspaces,
            seed=seed,
            attachments=attachments,
            mode=mode,
            requested_skill=requested_skill,
            effort=effort,
        )

        try:
            return await self._answers.accept(
                request=prepared.request,
                owner_id=principal_id,
                caller=user,
                idempotency_key=submission_id,
                idempotency_fingerprint=idempotency_fingerprint,
                acceptor=_WebAnswerAcceptor(
                    store=self._store,
                    conversation_id=conversation_id,
                    title_hint=_auto_title(query),
                    create_conversation=create_conversation,
                ),
            )
        except SessionWorkBusyError as exc:
            # The front door refused because another Run holds the Session; nothing was
            # accepted for this submission.
            raise ConversationSessionBusy(exc.active_run_id) from exc
        except asyncio.CancelledError:
            raise
        except Exception:
            # A concurrent identical acceptance may win while recovery is being
            # projected. Never turn that durable replay into a recovery failure.
            replay = await self._store.replay_answer_turn(
                principal_id=principal_id,
                conversation_id=conversation_id,
                submission_id=submission_id,
                idempotency_fingerprint=idempotency_fingerprint,
            )
            if replay is not None:
                return _submission(replay)
            raise

    async def goal(self, user: UserContext | None, conversation_id: str) -> ConversationGoal | None:
        """The conversation's current goal, or None when it carries none."""
        session_id = await self._goal_session(owner_id_from_user(user), conversation_id)
        if session_id is None or self._goals is None:
            return None
        goal = await self._goals.get(owner_id=owner_id_from_user(user), session_id=session_id)
        if goal is None:
            return None
        return self._project_goal(conversation_id, goal)

    async def goal_command(
        self,
        user: UserContext | None,
        conversation_id: str,
        *,
        action: GoalCommandAction,
        objective: str | None = None,
        max_rounds: int | None = None,
        workspaces: Sequence[str] | None = None,
    ) -> ConversationGoal | None:
        """Run one human goal command for this conversation (create/edit/pause/resume/clear).

        ``create`` starts the new goal's first round at once, through the driver: the command is the
        reader asking for work, not a plan to be resumed later, so a fresh conversation starts its
        round here too. A driver failure leaves the goal active — the next trigger retries — and
        never fails the command whose goal already exists. ``pause`` stops immediately: the phase
        changes, the continuation already waiting is dropped and its admission returned, and a live
        goal round is asked to cancel (a live human turn is left running); ``clear`` drops the
        waiting continuation before the goal row goes away. ``resume`` is the human control a paused
        goal needs, then the driver starts the next round when the Session is idle.
        """
        owner_id = owner_id_from_user(user)
        session_id = await self._goal_session(owner_id, conversation_id)
        if session_id is None:
            return None
        if self._goals is None:
            return None
        if action == "create":
            goal = await self._goals.create(
                owner_id=owner_id,
                session_id=session_id,
                run_id="",
                objective=objective or "",
                max_rounds=max_rounds or DEFAULT_MAX_ROUNDS,
                goal_round=None,
            )
            goal = await self._continue_created_goal(
                owner_id,
                conversation_id,
                session_id,
                goal,
                await self._round_workspaces(owner_id, conversation_id, workspaces),
            )
        elif action == "clear":
            await self._drop_pending_goal_round(owner_id, conversation_id, session_id)
            await self._goals.clear(owner_id=owner_id, session_id=session_id, run_id="")
            return None
        else:
            current = await self._goals.get(owner_id=owner_id, session_id=session_id)
            if current is None:
                raise GoalError(GOAL_NOT_FOUND, "this conversation carries no goal")
            goal = await self._goals.update(
                owner_id=owner_id,
                session_id=session_id,
                run_id="",
                goal_id=current.id,
                revision=current.revision,
                action=cast(GoalAction, action),
                objective=objective,
                max_rounds=max_rounds,
                goal_round=None,
                human_control=action == "resume",
            )
            if action == "resume":
                goal = await self._continue_resumed_goal(
                    owner_id, conversation_id, session_id, goal
                )
            elif action in ("pause", "edit"):
                # A continuation admitted under the old phase or objective must not run: an edit
                # changes what the next round asks, and a pause drops the next round before the
                # running one is stopped.
                await self._drop_pending_goal_round(owner_id, conversation_id, session_id)
            if action == "pause":
                # The owner's pause stops immediately: the round in progress is cancelled. A
                # human turn is the reader's own work and is never cancelled by a goal command.
                await self._cancel_live_goal_round(owner_id, conversation_id)
        return self._project_goal(conversation_id, goal)

    async def _continue_created_goal(
        self,
        owner_id: str,
        conversation_id: str,
        session_id: str,
        goal: Goal,
        workspaces: tuple[str, ...],
    ) -> Goal:
        """Start the first round of a goal a human just created, and return the goal it moved.

        The driver admits the round, builds its prompt and asks this service for an immediate Web
        turn; a False or an exception means no round started, which leaves the stored goal active
        for the next trigger. The command's goal creation is not undone by that. ``workspaces``
        are the conversation's selected ones, so the round continues the reader's chosen scope.
        """
        if self._goal_rounds is None:
            return goal
        try:
            await self._goal_rounds.continue_created_goal(
                owner_id=owner_id,
                session_id=session_id,
                conversation_id=conversation_id,
                workspaces=workspaces,
            )
        except Exception:
            logger.warning(
                "Could not start the first round of the goal of conversation %s",
                conversation_id,
                exc_info=True,
            )
            return goal
        if self._goals is None:
            return goal
        current = await self._goals.get(owner_id=owner_id, session_id=session_id)
        return current if current is not None else goal

    async def _round_workspaces(
        self,
        owner_id: str,
        conversation_id: str,
        selected: Sequence[str] | None,
    ) -> tuple[str, ...]:
        """The workspaces a first goal round continues.

        The composer's live selection wins, because the reader may have changed it since the last
        turn; otherwise the newest answer turn's selection stands; a conversation with no turn and
        no selection falls back to the product's stated default.
        """
        chosen = tuple(str(item) for item in (selected or ()) if str(item).strip())
        if chosen:
            return chosen
        return await self._conversation_workspaces(owner_id, conversation_id)

    async def _conversation_workspaces(
        self, owner_id: str, conversation_id: str
    ) -> tuple[str, ...]:
        """The workspaces the conversation's newest turn selected; the only fallback is default.

        A goal round continues the same Session, so it continues the scope the reader last chose.
        A conversation with no turn yet has no selection to carry; ``default`` is then the
        product's stated default and the round says so through the Run it starts.
        """
        selected = await self._store.newest_answer_workspaces(owner_id, conversation_id)
        return ("default",) if selected is None else selected

    async def continue_session_goal(self, *, owner_id: str, session_id: str) -> bool:
        """Start a round for an active goal whose Session has no Run to continue from.

        The round driver's startup/periodic recovery calls this after a crash between ``/goal``
        creation and its first round: the goal is durable and active, but there is no settlement
        to trigger it. The conversation that owns the Session supplies the workspace scope; a
        Session with no conversation is not a Web goal and returns False.
        """
        if self._goal_rounds is None:
            return False
        conversation_id = await self._store.conversation_for_session(owner_id, session_id)
        if conversation_id is None:
            return False
        workspaces = await self._conversation_workspaces(owner_id, conversation_id)
        try:
            return await self._goal_rounds.continue_created_goal(
                owner_id=owner_id,
                session_id=session_id,
                conversation_id=conversation_id,
                workspaces=workspaces,
            )
        except Exception:
            logger.warning(
                "Could not recover the goal of Session %s in conversation %s",
                session_id,
                conversation_id,
                exc_info=True,
            )
            return False

    async def _drop_pending_goal_round(
        self, owner_id: str, conversation_id: str, session_id: str
    ) -> None:
        """Drop a continuation waiting to become the next turn and return its admission.

        A pending row is a round the driver already admitted but no Run has started yet: dropping
        it for a pause, edit or clear must therefore give the admission back, or a continuation
        that never ran still counts against the cap. Best effort — the phase change already stands.
        """
        try:
            pending = await self._store.pending_goal_round(owner_id, conversation_id)
            if pending is None:
                return
            await self._refund_dropped_round(owner_id, conversation_id, session_id, pending)
        except Exception:
            logger.warning(
                "Could not drop the pending goal round of conversation %s",
                conversation_id,
                exc_info=True,
            )

    async def _refund_dropped_round(
        self,
        owner_id: str,
        conversation_id: str,
        session_id: str | None,
        pending: GoalRoundRequest,
        goal: Goal | None = None,
    ) -> bool:
        """Drop a continuation and return its admission only when this call deleted it.

        The clear reports deletion, and a turn committed for the same submission means the round
        was delivered rather than dropped — the two race a pause, so neither alone may decide a
        refund. When both say dropped, only the same goal and the same admitted round qualify:
        a replaced or cleared goal has no admission to return.
        """
        deleted = await self._store.clear_goal_round(
            owner_id,
            conversation_id,
            request_id=pending.request_id,
            prompt_digest=pending.prompt_digest,
        )
        if not deleted:
            return False
        if await self._store.submission_exists(
            owner_id,
            conversation_id,
            _goal_delivery_submission_id(conversation_id, pending.request_id),
        ):
            return False
        if self._goals is None or session_id is None:
            return False
        if goal is None:
            goal = await self._goals.get(owner_id=owner_id, session_id=session_id)
        if goal is None or goal.id != pending.goal_id:
            return False
        if goal.rounds_started != pending.goal_round:
            return False
        try:
            await self._goals.release_round(
                owner_id=owner_id,
                session_id=session_id,
                run_id="",
                goal_id=goal.id,
                revision=goal.revision,
            )
        except Exception:
            logger.warning(
                "Could not release the dropped round %s of goal %s",
                pending.goal_round,
                goal.id,
                exc_info=True,
            )
            return False
        return True

    async def _cancel_live_goal_round(self, owner_id: str, conversation_id: str) -> None:
        """Ask the Run service to cancel the conversation's running goal round, if any.

        Best effort and phase-independent: the goal is already ``paused``, so a cancellation that
        is pending, refused or already finished changes nothing — the round's own settlement
        cannot resume a paused goal, and no next round starts. A live human turn (or any
        continuation that is not a goal round) is deliberately left running.
        """
        if self._goal_round_canceller is None:
            return
        try:
            live = await self._store.live_answer_run(owner_id, conversation_id)
            if live is None or live.continuation_kind != "goal_round":
                return
            await self._goal_round_canceller.cancel(owner_id=owner_id, run_id=live.run_id)
        except Exception:
            logger.warning(
                "Could not cancel the live goal round of conversation %s",
                conversation_id,
                exc_info=True,
            )

    async def _continue_resumed_goal(
        self, owner_id: str, conversation_id: str, session_id: str, goal: Goal
    ) -> Goal:
        """Turn an idle human resume into the goal's next round, and return the goal it moved.

        A resumed goal with a parent Run continues from that Run's conversation turn; a goal with
        no prior Run (created then paused before anything ran) starts immediately through the
        no-parent entry point. A failure here leaves the goal active: the next settlement or
        trigger still starts its round.
        """
        if self._goal_rounds is None:
            return goal
        seed = await self._store.submission_seed(owner_id, conversation_id, attachment_limit=0)
        try:
            if seed is not None and seed.parent_run_id is not None:
                await self._goal_rounds.continue_after_resume(
                    owner_id=owner_id,
                    session_id=session_id,
                    parent_run_id=seed.parent_run_id,
                )
            else:
                await self._goal_rounds.continue_created_goal(
                    owner_id=owner_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    workspaces=await self._conversation_workspaces(owner_id, conversation_id),
                )
        except Exception:
            logger.warning(
                "Could not continue the resumed goal of conversation %s",
                conversation_id,
                exc_info=True,
            )
            return goal
        if self._goals is None:
            return goal
        current = await self._goals.get(owner_id=owner_id, session_id=session_id)
        return current if current is not None else goal

    async def command(
        self,
        user: UserContext | None,
        conversation_id: str,
        name: str,
        input: str,
        *,
        workspaces: Sequence[str] | None = None,
    ) -> ConversationCommand | None:
        """Execute one registered server-owned command for an owned conversation.

        The conversation (and therefore its Agent Session) is validated before anything
        runs, so an unknown conversation is the same opaque miss as every other owned
        read. Execution returns the reader-facing canonical text and its tone; a refused
        command is a result too, persisted exactly like a successful one. ``input`` is
        stored exactly as the caller sent it. ``None`` means the conversation is not this
        principal's.
        """
        if name not in WEB_COMMAND_NAMES:
            raise UnknownConversationCommand(name)
        owner_id = owner_id_from_user(user)
        session_id = await self._goal_session(owner_id, conversation_id)
        if session_id is None:
            return None
        result, tone, result_data = await self._execute_goal_command(
            user, conversation_id, input, workspaces
        )
        return await self._store.append_command(
            owner_id,
            conversation_id,
            name=name,
            command_input=input,
            result=result,
            tone=tone,
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Messages queued behind a live Run
    # ------------------------------------------------------------------

    async def queue(
        self,
        user: UserContext | None,
        conversation_id: str,
    ) -> tuple[QueuedMessage, ...] | None:
        """Return the conversation's queued messages in delivery order.

        ``None`` means the conversation is not this principal's, the same opaque miss as
        every other owned read. An empty tuple means the conversation exists and nothing
        waits behind its live Run.
        """
        return await self._store.queue_messages(owner_id_from_user(user), conversation_id)

    async def enqueue(
        self,
        user: UserContext | None,
        conversation_id: str,
        content: str,
    ) -> QueuedMessage | None:
        """Append one message to the conversation's queue; None when it is not owned.

        The queue is the holding place for a send that arrives while a Run is live: the
        message is not a Run, it is delivered as one next-turn prompt after that Run
        settles.
        """
        text = content.strip()
        if not text:
            raise ApplicationInputError("a queued message must not be blank")
        return await self._store.append_queued_message(
            owner_id_from_user(user), conversation_id, content=text
        )

    async def edit_queued(
        self,
        user: UserContext | None,
        conversation_id: str,
        message_id: str,
        content: str,
    ) -> QueuedMessage | None:
        """Rewrite one queued message's words; None when it or the conversation is gone."""
        text = content.strip()
        if not text:
            raise ApplicationInputError("a queued message must not be blank")
        return await self._store.edit_queued_message(
            owner_id_from_user(user), conversation_id, message_id, content=text
        )

    async def remove_queued(
        self,
        user: UserContext | None,
        conversation_id: str,
        message_id: str,
    ) -> bool:
        """Delete one queued message; False when it or the conversation is gone."""
        return await self._store.remove_queued_message(
            owner_id_from_user(user), conversation_id, message_id
        )

    async def reorder_queue(
        self,
        user: UserContext | None,
        conversation_id: str,
        message_ids: Sequence[str],
    ) -> tuple[QueuedMessage, ...] | None:
        """Rewrite the queue order to ``message_ids``; None when the conversation is not owned.

        The store owns the set check because it holds the conversation's row lock: a tab
        that edited the queue while another was reordering gets a conflict rather than a
        reordered set that silently dropped a message.
        """
        return await self._store.reorder_queued_messages(
            owner_id_from_user(user), conversation_id, message_ids=message_ids
        )

    async def _deliver_to_live_run(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        live: LiveAnswerRun,
        query: str,
        attachments: Sequence[WebAttachment],
        delivery: Literal["queue", "steer"] | None,
    ) -> WebAnswerDelivery | None:
        """Deliver one human send to the live Run; None when the conversation vanished first.

        The default is the queue: a send that arrives while work is in progress is held for
        the next turn, not turned into a competing Run. ``steer`` is the explicit ask and is
        best-effort: a Run that has stopped taking steers falls back to the queue, because a
        refusal must never drop the reader's words.
        """
        if attachments:
            raise ApplicationInputError(
                "attachments cannot be delivered to a Run that is already live"
            )
        if delivery == "steer":
            receipt = await self._answers.steer(
                owner_id=principal_id,
                run_id=live.run_id,
                instruction=query,
            )
            if receipt is not None:
                return WebAnswerDelivery(
                    mode="steer",
                    steered_run_id=receipt.run_id,
                    control_sequence=receipt.control_sequence,
                )
        queued = await self._store.append_queued_message(
            principal_id, conversation_id, content=query
        )
        if queued is None:
            return None
        return WebAnswerDelivery(mode="queue", queued=queued)

    async def _drop_or_deliver_goal_round(
        self, owner_id: str, conversation_id: str, parent_run_id: str
    ) -> bool:
        """Deliver the pending goal continuation under the same fenced acceptance path.

        Human queue entries win: this runs only when the human queue is empty. A goal that is no
        longer active drops its request instead of continuing a line the human stopped; the
        durable phase is the whole continuation permission, so no process activation is consulted.
        """
        pending = await self._store.pending_goal_round(owner_id, conversation_id)
        if pending is None:
            return False
        if self._goals is None:
            await self._store.clear_goal_round(
                owner_id,
                conversation_id,
                request_id=pending.request_id,
                prompt_digest=pending.prompt_digest,
            )
            return False
        session_id = await self._goal_session(owner_id, conversation_id)
        goal = (
            None
            if session_id is None
            else await self._goals.get(owner_id=owner_id, session_id=session_id)
        )
        if (
            session_id is None
            or goal is None
            or goal.phase != GoalPhase.ACTIVE
            or goal.id != pending.goal_id
            or goal.rounds_started not in (pending.goal_round, pending.goal_round - 1)
        ):
            # The continuation belongs to another goal or another round: drop it, and return the
            # admission it consumed so a dropped continuation does not burn the round cap.
            await self._refund_dropped_round(owner_id, conversation_id, session_id, pending, goal)
            return False
        seed = await self._store.submission_seed(owner_id, conversation_id, attachment_limit=0)
        if seed is None:
            return False
        prepared = _prepare_submission(
            query=pending.content,
            workspaces=pending.workspaces or ("default",),
            seed=seed,
            attachments=(),
            mode="research",
            continuation_kind="goal_round",
            goal_id=pending.goal_id,
            goal_revision=pending.goal_revision,
            goal_round=pending.goal_round,
        )
        accepted = await self._answers.accept(
            request=prepared.request,
            owner_id=owner_id,
            caller=None,
            idempotency_key=_goal_delivery_submission_id(conversation_id, pending.request_id),
            idempotency_fingerprint=_web_answer_request_fingerprint(
                conversation_id=conversation_id,
                query=pending.content,
                workspaces=prepared.request.workspaces,
                attachments=(),
                mode="research",
            ),
            acceptor=_WebAnswerAcceptor(
                store=self._store,
                conversation_id=conversation_id,
                title_hint=None,
                create_conversation=False,
                goal_round_guard=GoalRoundGuard(
                    request_id=pending.request_id,
                    prompt_digest=pending.prompt_digest,
                    goal_id=pending.goal_id,
                    goal_revision=pending.goal_revision,
                    goal_round=pending.goal_round,
                ),
            ),
        )
        if accepted is None:
            # The fenced transaction refused (paused/cleared/replaced/moved goal, or another
            # delivery won): drop the continuation and return its admission only if this call
            # deleted it and no turn committed for its submission.
            if self._goals is not None:
                await self._refund_dropped_round(owner_id, conversation_id, session_id, pending)
            else:
                await self._store.clear_goal_round(
                    owner_id,
                    conversation_id,
                    request_id=pending.request_id,
                    prompt_digest=pending.prompt_digest,
                )
        return accepted is not None

    async def accept_queued_turn(
        self,
        owner_id: str,
        *,
        parent_run_id: str,
        workspaces: Sequence[str],
    ) -> bool:
        """Deliver the conversation's queued messages as one next-turn Run.

        Called by the settlement driver, never by a request: the parent Run has ended, the
        queue is the reader's work that was waiting behind it, and the new Run is a normal
        conversation turn parented to that Run. The submission id is derived from the
        conversation and the oldest queued message, so a retried delivery replays instead of
        starting a second turn. The accept's own transaction removes the delivered rows by id, so
        a message that arrived in the meantime keeps its place for the next turn.
        """
        turn = await self._store.find_turn_by_run(owner_id, parent_run_id)
        if turn is None or not turn.conversation_id:
            return False
        conversation_id = turn.conversation_id
        messages = await self._store.queue_messages(owner_id, conversation_id)
        if not messages:
            return await self._drop_or_deliver_goal_round(owner_id, conversation_id, parent_run_id)
        seed = await self._store.submission_seed(owner_id, conversation_id, attachment_limit=0)
        if seed is None:
            return False
        query = _coalesced_queue_prompt(messages)
        submission_id = str(
            uuid5(
                NAMESPACE_URL,
                f"dlightrag-queue:{conversation_id}:{messages[0].message_id}",
            )
        )
        prepared = _prepare_submission(
            query=query,
            workspaces=workspaces,
            seed=seed,
            attachments=(),
        )
        fingerprint = _web_answer_request_fingerprint(
            conversation_id=conversation_id,
            query=query,
            workspaces=workspaces,
            attachments=(),
        )
        accepted = await self._answers.accept(
            request=prepared.request,
            owner_id=owner_id,
            caller=None,
            idempotency_key=submission_id,
            idempotency_fingerprint=fingerprint,
            acceptor=_WebAnswerAcceptor(
                store=self._store,
                conversation_id=conversation_id,
                title_hint=None,
                create_conversation=False,
                queue_guard=QueuedDeliveryGuard(
                    message_ids=tuple(message.message_id for message in messages),
                    digest=queued_delivery_digest(messages),
                ),
            ),
        )
        return accepted is not None

    async def _execute_goal_command(
        self,
        user: UserContext | None,
        conversation_id: str,
        raw_input: str,
        workspaces: Sequence[str] | None = None,
    ) -> tuple[str, CommandTone, dict[str, Any] | None]:
        """Run one parsed ``goal`` command and return its text, tone, and structured payload.

        The text is the canonical English fallback the transcript audits; the structured
        payload repeats the same outcome as data so the browser can localize it. A refused
        command is a result too: a domain refusal keeps its stable ``GoalError.code``.
        """
        parsed = _parse_goal_command(raw_input)
        if parsed.kind == "status":
            current = await self.goal(user, conversation_id)
            return (
                _goal_status_text(current),
                "success",
                _goal_status_result_data(current),
            )
        if parsed.kind == "invalid":
            return (
                parsed.error_text or _GOAL_FAILED,
                "error",
                {"kind": "goal_invalid", "reason": parsed.invalid_reason or "objective"},
            )
        try:
            goal = await self.goal_command(
                user,
                conversation_id,
                action=parsed.action or "edit",
                objective=parsed.objective,
                max_rounds=parsed.max_rounds,
                workspaces=workspaces,
            )
        except GoalError as error:
            return (
                str(error),
                "error",
                {"kind": "goal_error", "code": error.code, "message": str(error)},
            )
        if parsed.action == "clear":
            return _GOAL_CLEARED, "success", {"kind": "goal_cleared"}
        if goal is None:
            # Unreachable when the goal service is composed; the text remains the fallback
            # and a reader without a structured payload simply shows it.
            return _GOAL_FAILED, "error", None
        return _goal_status_text(goal), "success", _goal_status_result_data(goal)

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
    ) -> GoalRoundReservation | None:
        """Reserve the goal continuation as the conversation's next turn; None when not Web.

        The driver no longer submits a Run: it asks the conversation to continue after its
        current turn, and the one fenced delivery path admits the turn when the Session is at
        rest. The row names the goal and admitted round it serves, so a continuation whose goal
        was paused, edited, cleared or replaced is dropped instead of continuing stale work. The
        returned reservation carries the row's own request identity, which is the only thing the
        driver may later withdraw.
        """
        linked = await self._store.find_turn_by_run(owner_id, parent_run_id)
        if linked is None or not linked.conversation_id:
            return None
        content = prompt.strip()
        if not content:
            return None
        request_id = str(uuid4())
        prompt_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        reserved = await self._store.enqueue_goal_round(
            owner_id,
            linked.conversation_id,
            request_id=request_id,
            content=content,
            prompt_digest=prompt_digest,
            workspaces=workspaces,
            goal_id=goal_id,
            goal_revision=goal_revision,
            goal_round=goal_round,
        )
        if not reserved:
            return None
        return GoalRoundReservation(request_id=request_id, prompt_digest=prompt_digest)

    async def goal_round_pending(self, *, owner_id: str, parent_run_id: str) -> bool:
        """Whether the conversation already holds an undelivered goal continuation."""
        linked = await self._store.find_turn_by_run(owner_id, parent_run_id)
        if linked is None or not linked.conversation_id:
            return False
        pending = await self._store.pending_goal_round(owner_id, linked.conversation_id)
        return pending is not None

    async def withdraw_goal_round(
        self,
        *,
        owner_id: str,
        parent_run_id: str,
        request_id: str,
        prompt_digest: str,
    ) -> bool:
        """Withdraw exactly the continuation this driver attempt reserved.

        True means this attempt's own row was deleted (its reservation is gone); False means it
        was already gone — consumed by a delivery or replaced by a newer reservation — so the
        round it names may be running and must not be refunded here. The request identity is part
        of the delete, so a stale attempt can never remove a newer attempt's row.
        """
        linked = await self._store.find_turn_by_run(owner_id, parent_run_id)
        if linked is None or not linked.conversation_id:
            return False
        return await self._store.clear_goal_round(
            owner_id,
            linked.conversation_id,
            request_id=request_id,
            prompt_digest=prompt_digest,
        )

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
    ) -> bool:
        """Start a goal round as this conversation's next turn, with no parent Run required.

        This is the Web driver's first-round entry point: a conversation is its Agent Session from
        creation, so a goal created in a fresh conversation's round is an ordinary research turn
        parented to nothing. The round arrives as a Web turn immediately (no queue, no live Run to
        deliver behind), and its submission id is derived from the round prompt, so a retried
        delivery replays instead of opening a second round. ``False`` means no turn was created:
        the conversation vanished, the prompt was blank, another Run holds the Session, or the
        acceptance found the turn already stored for this submission.
        """
        content = prompt.strip()
        if not content:
            return False
        seed = await self._store.submission_seed(owner_id, conversation_id, attachment_limit=0)
        if seed is None:
            return False
        prepared = _prepare_submission(
            query=content,
            workspaces=workspaces or ("default",),
            seed=seed,
            attachments=(),
            mode="research",
            continuation_kind="goal_round",
            goal_id=goal_id,
            goal_revision=goal_revision,
            goal_round=goal_round,
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        try:
            accepted = await self._answers.accept(
                request=prepared.request,
                owner_id=owner_id,
                caller=None,
                idempotency_key=str(
                    uuid5(
                        NAMESPACE_URL,
                        "dlightrag-goal-round-now:"
                        f"{conversation_id}:{goal_id}:{goal_revision}:{goal_round}:{digest}",
                    )
                ),
                idempotency_fingerprint=_web_answer_request_fingerprint(
                    conversation_id=conversation_id,
                    query=content,
                    workspaces=prepared.request.workspaces,
                    attachments=(),
                    mode="research",
                ),
                acceptor=_WebAnswerAcceptor(
                    store=self._store,
                    conversation_id=conversation_id,
                    title_hint=None,
                    create_conversation=False,
                ),
            )
        except SessionWorkBusyError:
            # The driver treats a False as "no round started" and releases its admission; the
            # Session's live Run is its next trigger, not a reason to fail the command.
            return False
        return accepted is not None

    async def _goal_session(self, owner_id: str, conversation_id: str) -> str | None:
        seed = await self._store.submission_seed(owner_id, conversation_id, attachment_limit=0)
        return None if seed is None else seed.head.agent_session_id

    def _project_goal(self, conversation_id: str, goal: Goal) -> ConversationGoal:
        """One stored goal as the browser reads it: phase, objective, round, blocker.

        The phase is the whole continuation state; there is no process-local activation to project
        (ADR 0077), so two readers of the same row are told the same thing.
        """
        return ConversationGoal(
            conversation_id=conversation_id,
            goal_id=goal.id,
            revision=goal.revision,
            objective=goal.objective,
            phase=str(goal.phase),
            max_rounds=goal.max_rounds,
            rounds_started=goal.rounds_started,
            blocked_reason=goal.blocked_reason,
        )

    async def fork_answer(
        self,
        user: UserContext | None,
        *,
        parent_run_id: str,
        submission_id: str,
        query: str,
        authorized_workspaces: Sequence[str] | None,
    ) -> WebAnswerSubmission | None:
        """Open a new conversation branch at the state the named Run settled at."""
        principal_id = owner_id_from_user(user)
        parent = await self.turn_for_run(user, parent_run_id)
        if parent is None or not parent.conversation_id:
            return None
        request = await self._answers.continuation_request(
            owner_id=principal_id,
            run_id=parent_run_id,
            query=query,
            include_answer=False,
            authorized_workspaces=authorized_workspaces,
        )
        if request is None:
            return None
        conversation_id = _new_conversation_id(principal_id, submission_id)
        fingerprint = run_request_fingerprint(
            {
                "conversation_id": conversation_id,
                "parent_run_id": parent_run_id,
                "continuation_kind": "fork",
                "query": query.strip(),
            }
        )
        return await self._answers.accept(
            request=request,
            owner_id=principal_id,
            caller=user,
            idempotency_key=submission_id,
            idempotency_fingerprint=fingerprint,
            acceptor=_WebAnswerAcceptor(
                store=self._store,
                conversation_id=conversation_id,
                title_hint=_auto_title(query),
                create_conversation=True,
                forked_from_conversation_id=parent.conversation_id,
            ),
        )


def _new_conversation_id(principal_id: str, submission_id: str) -> str:
    """Return the stable server-owned conversation id for a first submission."""
    return str(
        uuid5(
            _NEW_CONVERSATION_NAMESPACE,
            f"{principal_id}\0{submission_id}",
        )
    )


def _empty_head(principal_id: str, conversation_id: str) -> ConversationHead:
    """Supply execution identity while the first row awaits atomic acceptance."""
    now = datetime.datetime.now(datetime.UTC)
    return ConversationHead(
        principal_id=principal_id,
        conversation_id=conversation_id,
        content_revision=0,
        title=None,
        created_at=now,
        updated_at=now,
        agent_session_id=conversation_id,
        agent_lane_id="main",
    )


def _auto_title(query: str) -> str | None:
    return " ".join(query.split())[:120] or None


def _submission(creation: AnswerTurnCreation) -> WebAnswerSubmission:
    return WebAnswerSubmission(
        run=creation.turn.run,
        turn_id=creation.turn.turn_id,
        turn_number=creation.turn.turn_number,
        submission_id=creation.turn.submission_id,
        created_at=creation.turn.created_at,
        conversation=_conversation_summary(creation.summary),
    )


#: The one line that tells the model where a coalesced prompt came from. It is prompt
#: prose, not a transcript event: the queued messages themselves are the reader's words.
_QUEUED_MESSAGES_PREAMBLE = (
    "The reader queued these messages while a Run was working; answer them together, in order."
)


def _coalesced_queue_prompt(messages: Sequence[QueuedMessage]) -> str:
    """Turn the queued messages into one next-turn prompt, oldest first.

    Each message keeps its own time so the model can see the order and the pauses it was
    given; the content is the reader's, untouched. The format is deterministic, so a
    replayed delivery hashes to the same request.
    """
    lines = [_QUEUED_MESSAGES_PREAMBLE, ""]
    for message in messages:
        stamp = message.created_at.astimezone(datetime.UTC).strftime("%Y-%m-%d %H:%M")
        lines.extend((f"[{stamp} UTC] {message.content}", ""))
    return "\n".join(lines).rstrip()


def _goal_delivery_submission_id(conversation_id: str, request_id: str) -> str:
    """The submission key one pending continuation delivers under: deterministic per request."""
    return str(
        uuid5(
            NAMESPACE_URL,
            f"dlightrag-goal-delivery:{conversation_id}:{request_id}",
        )
    )


def _prepare_submission(
    *,
    query: str,
    workspaces: Sequence[str],
    seed: SubmissionSeed,
    attachments: Sequence[WebAttachment],
    mode: str | None = None,
    requested_skill: str | None = None,
    effort: AnswerEffort | None = None,
    continuation_kind: str | None = None,
    goal_id: str | None = None,
    goal_revision: int | None = None,
    goal_round: int | None = None,
) -> _PreparedSubmission:
    """Normalize a browser submission without coupling it to a UI history page.

    The submission names the conversation's newest Run as its parent: a Web turn is a
    line append, and recording that once is cheaper than leaving lineage implied by
    turn order. Memory does not travel on it — the Session owns that (ADR 0022) — so
    the submission keeps its own mode, effort, attachments, and workspaces rather than
    inheriting the parent's.
    """
    request = AnswerRequest(
        query=query,
        parent_run_id=seed.parent_run_id,
        continuation_kind=(
            continuation_kind
            if continuation_kind is not None
            else ("follow_up" if seed.parent_run_id else None)
        ),
        workspaces=tuple(workspaces),
        history=(),
        semantic_highlights=True,
        mode=mode,
        requested_skill=requested_skill,
        effort=effort,
        resources=tuple(
            ResourceInput(
                filename=attachment.filename,
                declared_mime=attachment.mime_type,
                content=attachment.attachment_bytes,
            )
            for attachment in attachments
        ),
        history_resources=tuple(
            AnswerHistoryResource(
                run_id=item.run_id,
                source_ordinal=item.source_ordinal,
                digest=item.digest,
                filename=item.filename,
                mime_type=item.mime_type,
                byte_size=item.byte_size,
            )
            for item in seed.attachments
        ),
        agent_session_id=seed.head.agent_session_id,
        agent_lane_id=seed.head.agent_lane_id,
        goal_id=goal_id,
        goal_revision=goal_revision,
        goal_round=goal_round,
    )
    return _PreparedSubmission(request=request)


def _web_answer_request_fingerprint(
    *,
    conversation_id: str,
    query: str,
    workspaces: Sequence[str],
    attachments: Sequence[WebAttachment],
    mode: str | None = None,
    requested_skill: str | None = None,
    effort: AnswerEffort | None = None,
) -> str:
    """Hash only the stable browser submission, before conversation enrichment."""
    return run_request_fingerprint(
        {
            "conversation_id": conversation_id,
            "query": query,
            "mode": mode or "auto",
            "requested_skill": requested_skill or "",
            "effort": effort or "",
            "workspaces": list(workspaces),
            "attachments": [
                {
                    "digest": attachment.content_sha256,
                    "filename": attachment.filename,
                    "mime_type": attachment.mime_type,
                    "ordinal": attachment.ordinal,
                    "byte_size": attachment.byte_size,
                }
                for attachment in attachments
            ],
        }
    )


def _parse_goal_command(raw_input: str) -> _ParsedGoalCommand:
    """Parse the words after ``/goal`` with the grammar the composer used to accept.

    The composer forwards the reader's argument text untouched; the server now owns the
    parse. The rules, and their order, are the browser parser's: an empty line reads the
    goal, the three control words are their own commands, ``cap <n>`` is an edit of the
    round cap, ``edit <objective>`` edits the objective, and every other non-empty text
    creates a goal with that objective. ``cap``/``edit`` without an argument are refused
    as invalid input rather than becoming an objective.
    """
    rest = raw_input.strip()
    if rest == "":
        return _ParsedGoalCommand(kind="status")
    if rest in {"pause", "resume", "clear"}:
        return _ParsedGoalCommand(kind="command", action=cast(GoalCommandAction, rest))
    if rest == "cap":
        return _ParsedGoalCommand(
            kind="invalid",
            invalid_reason="cap",
            error_text="Add a positive number: /goal cap 5",
        )
    if rest.startswith("cap "):
        number = rest[len("cap ") :].strip()
        if _GOAL_CAP_NUMBER.fullmatch(number) is None or int(number) > _JS_MAX_SAFE_INTEGER:
            return _ParsedGoalCommand(
                kind="invalid",
                invalid_reason="cap",
                error_text="Add a positive number: /goal cap 5",
            )
        return _ParsedGoalCommand(kind="command", action="edit", max_rounds=int(number))
    if rest == "edit":
        return _ParsedGoalCommand(
            kind="invalid",
            invalid_reason="objective",
            error_text="Add an objective: /goal <objective>",
        )
    if rest.startswith("edit "):
        objective = rest[len("edit ") :].strip()
        if objective == "":
            return _ParsedGoalCommand(
                kind="invalid",
                invalid_reason="objective",
                error_text="Add an objective: /goal <objective>",
            )
        return _ParsedGoalCommand(kind="command", action="edit", objective=objective)
    return _ParsedGoalCommand(kind="command", action="create", objective=rest)


_GOAL_PHASE_LABELS = {
    "paused": "Paused goal",
    "blocked": "Blocked goal",
    "complete": "Complete goal",
}


def _goal_phase_label(goal: ConversationGoal) -> str:
    if goal.phase == "active":
        return "Ongoing goal"
    return _GOAL_PHASE_LABELS.get(goal.phase, "Complete goal")


def _goal_status_result_data(goal: ConversationGoal | None) -> dict[str, Any]:
    """The structured snapshot a browser localizes; the English text below is the fallback."""
    if goal is None:
        return {"kind": "goal_status", "goal": None}
    return {
        "kind": "goal_status",
        "goal": {
            "objective": goal.objective,
            "phase": goal.phase,
            "max_rounds": goal.max_rounds,
            "rounds_started": goal.rounds_started,
            "blocked_reason": goal.blocked_reason,
        },
    }


def _goal_status_text(goal: ConversationGoal | None) -> str:
    """The canonical English text of a goal's current state, as the result line shows it."""
    if goal is None:
        return _GOAL_STATUS_NONE
    lines = [
        _goal_phase_label(goal),
        f"Objective: {goal.objective}",
        f"Round: {goal.rounds_started}/{goal.max_rounds}",
    ]
    if goal.blocked_reason is not None:
        lines.append(f"Blocked: {goal.blocked_reason}")
    if goal.phase == "paused":
        lines.append(_GOAL_PAUSE_NOTE)
    lines.append(_GOAL_COMMAND_HELP)
    return "\n".join(lines)


def _conversation_summary(row: Mapping[str, Any]) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=str(row["conversation_id"]),
        title=(str(row["title"]) if row.get("title") is not None else None),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        forked_from_conversation_id=(
            str(row["forked_from_conversation_id"])
            if row.get("forked_from_conversation_id") is not None
            else None
        ),
        forked_from_title=(
            str(row["forked_from_title"]) if row.get("forked_from_title") is not None else None
        ),
    )


__all__ = ["WEB_COMMAND_NAMES", "WebAnswerSubmission", "WebAttachment", "WebConversationService"]
