# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Web Conversation records and persistence port."""

import datetime
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import UUID

from dlightrag.application.errors import (
    ApplicationConflictError,
    ApplicationInputError,
    ApplicationUnavailableError,
)
from dlightrag.application.opaque_cursor import (
    OpaqueCursorEnvelope,
    format_utc_timestamp,
    parse_canonical_uuid,
    parse_utc_timestamp,
)
from dlightrag.application.runs import RunView
from dlightrag.engine.answer.errors import SESSION_BUSY_MESSAGE
from dlightrag.engine.answer.execution.connection_binding import RunConnectionBinding
from dlightrag.engine.answer.feedback import Judgement
from dlightrag.engine.answer.runs.routing import RoutingAcceptance
from dlightrag.engine.runtime.contracts import RunPhase, RunStatus
from dlightrag.engine.runtime.records import (
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
)


@dataclass(frozen=True, slots=True)
class LinkedTurn:
    """One conversation entry and the authoritative run state behind it."""

    turn_id: str
    turn_number: int
    submission_id: str
    created_at: datetime.datetime
    run: RunView
    conversation_id: str = ""
    #: How many Child Sessions this turn's Run has spawned, so a settled turn can
    #: tell whether it has children to open. A turn just accepted has none.
    child_count: int = 0
    #: The Session ``sequence`` of the message the Run answered with, which is what a reader
    #: names to judge it: the Run's last assistant message when that carries no tool calls and
    #: says something. None until the Run has answered, and for a turn just accepted.
    answer_sequence: int | None = None
    #: The owner's judgement of that message, None while they have given none.
    feedback: Judgement | None = None
    steering_messages: tuple[str, ...] = ()

    @property
    def answer_run_id(self) -> str:
        return self.run.run_id


@dataclass(frozen=True, slots=True)
class ConversationHead:
    """Owned conversation identity and durable execution mapping, without turns."""

    principal_id: str
    conversation_id: str
    content_revision: int
    title: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    agent_session_id: str
    agent_lane_id: str


@dataclass(frozen=True, slots=True)
class CarriedAttachment:
    """Bounded metadata for one successful prior current attachment."""

    run_id: str
    source_ordinal: int
    digest: str
    filename: str
    mime_type: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class ConversationGoal:
    """A conversation's current goal, projected for the browser's goal bar.

    The phase is the whole continuation state: an ``active`` goal is one the platform continues on
    its own, a ``paused`` or ``blocked`` goal waits for a human, and ``complete`` is terminal. There
    is no process-local activation to project.
    """

    conversation_id: str
    goal_id: str
    revision: int
    objective: str
    phase: str
    max_rounds: int
    rounds_started: int
    blocked_reason: str | None

    @property
    def rounds_left(self) -> int:
        return max(0, self.max_rounds - self.rounds_started)


#: How one server-owned command's result is presented to the reader.
CommandTone = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class ConversationCommand:
    """One command the server executed for a conversation, as the transcript shows it.

    A command is an append-only, human-initiated Web event: it carries the one ordering
    ``sequence`` the conversation's turns also allocate from, so a reader interleaves both
    streams by that number. ``input`` is the words the caller sent after the command name,
    unmodified; ``result`` is the server's canonical English text for this execution,
    successful or refused. ``result_data`` is the same execution's structured payload,
    which lets a browser localize the result line while the stored text stays the audit
    fallback; it is data, never reader prose.
    """

    command_id: str
    sequence: int
    name: str
    input: str
    result: str
    tone: CommandTone
    created_at: datetime.datetime
    result_data: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class LiveAnswerRun:
    """One non-terminal Answer Run the conversation's Session lane is executing.

    It is the one fact admission needs: an ``answer`` Run linked to this conversation whose
    status is ``queued`` or ``running``. ``continuation_kind`` states what the Run continues:
    a ``goal_round`` is the pause command's to cancel when the reader pauses the goal, while a
    human turn (or any other continuation) is the reader's own work and is left running.
    """

    run_id: str
    status: str
    continuation_kind: str | None = None


@dataclass(frozen=True, slots=True)
class QueuedMessage:
    """One human message waiting behind the conversation's live Run.

    A queue is not a transcript event: the message has not started a Run and carries
    no answer. ``position`` is the conversation's queue order (1-based), allocated when
    the message arrives and rewritten by a reorder; ``content`` is the caller's own
    words, unmodified.
    """

    message_id: str
    content: str
    position: int
    created_at: datetime.datetime


@dataclass(frozen=True, slots=True)
class SubmissionSeed:
    """Conversation mapping plus bounded attachment carry-forward metadata."""

    head: ConversationHead
    attachments: tuple[CarriedAttachment, ...] = ()
    #: The conversation's newest turn's Run: what this submission follows in its
    #: conversation. It is lineage, not a memory transport (ADR 0022).
    parent_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    """Application projection of one durable conversation row."""

    conversation_id: str
    title: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    forked_from_conversation_id: str | None = None
    forked_from_title: str | None = None


CONVERSATION_PAGE_DEFAULT_LIMIT = 50
CONVERSATION_PAGE_MAX_LIMIT = 100
CONVERSATION_HISTORY_PAGE_DEFAULT_LIMIT = 40
CONVERSATION_HISTORY_PAGE_MAX_LIMIT = 100
#: One hard-bounded page of the owner's recent conversation-linked Answer runs.
RUN_ACTIVITY_DEFAULT_LIMIT = 30
RUN_ACTIVITY_MAX_LIMIT = 100
#: The newest steering messages one turn carries in history; the Web draws no more.
STEERING_MESSAGES_PER_TURN = 50


@dataclass(frozen=True, slots=True)
class WebRunActivity:
    """One conversation-linked Answer run, as the Web's run-activity read projects it.

    The row is a read-only projection of the durable Run and the turn that links it to a
    conversation; it carries no execution internals. ``status`` is the Run's lifecycle
    state and ``phase`` the executor-owned string, both authoritative on every poll.
    ``goal_round`` states the one continuation fact the page needs: this Run continues a
    goal, so its settlement is activity rather than a per-answer completion notice.
    """

    run_id: str
    conversation_id: str
    conversation_title: str | None
    status: RunStatus
    phase: RunPhase | None
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None
    goal_round: bool


class ConversationCursorError(ApplicationInputError):
    """An opaque conversation page cursor is malformed or fails integrity checking."""


@dataclass(frozen=True, slots=True)
class ConversationCursor:
    """The complete ordering key for the next newest-first page."""

    updated_at: datetime.datetime
    conversation_id: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.updated_at, datetime.datetime):
            raise ValueError("conversation cursor timestamp must be a datetime")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("conversation cursor timestamp must include a timezone")
        if not isinstance(self.conversation_id, UUID):
            raise ValueError("conversation cursor id must be a UUID")


@dataclass(frozen=True, slots=True)
class ConversationPageRequest:
    """One hard-bounded application request for conversation summaries."""

    limit: int = CONVERSATION_PAGE_DEFAULT_LIMIT
    cursor: ConversationCursor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("conversation page limit must be an integer")
        if not 1 <= self.limit <= CONVERSATION_PAGE_MAX_LIMIT:
            raise ValueError(
                f"conversation page limit must be between 1 and {CONVERSATION_PAGE_MAX_LIMIT}"
            )
        if self.cursor is not None and not isinstance(self.cursor, ConversationCursor):
            raise ValueError("conversation cursor must contain paired ordering fields")


@dataclass(frozen=True, slots=True)
class ConversationRowPage:
    """Bounded persistence result, including the measured physical fetch size."""

    items: tuple[Mapping[str, Any], ...]
    has_more: bool
    fetched_rows: int


@dataclass(frozen=True, slots=True)
class ConversationPage:
    """Newest-first application page with a paired continuation key."""

    items: tuple[ConversationSummary, ...]
    next_cursor: ConversationCursor | None
    fetched_rows: int


@dataclass(frozen=True, slots=True)
class ConversationHistoryCursor:
    """Immutable boundary for an older page in one conversation.

    The boundary is the conversation's shared ordering counter: a page returns the turns
    and commands whose ``sequence`` is strictly older than ``before_sequence``.
    """

    conversation_id: UUID
    before_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, UUID):
            raise ValueError("conversation history cursor id must be a UUID")
        if isinstance(self.before_sequence, bool) or not isinstance(self.before_sequence, int):
            raise ValueError("conversation history cursor sequence must be an integer")
        if self.before_sequence < 1:
            raise ValueError("conversation history cursor sequence must be positive")


@dataclass(frozen=True, slots=True)
class ConversationHistoryPageRequest:
    """One hard-bounded recent or older turn-page request."""

    limit: int = CONVERSATION_HISTORY_PAGE_DEFAULT_LIMIT
    cursor: ConversationHistoryCursor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("conversation history page limit must be an integer")
        if not 1 <= self.limit <= CONVERSATION_HISTORY_PAGE_MAX_LIMIT:
            raise ValueError(
                "conversation history page limit must be between 1 and "
                f"{CONVERSATION_HISTORY_PAGE_MAX_LIMIT}"
            )
        if self.cursor is not None and not isinstance(self.cursor, ConversationHistoryCursor):
            raise ValueError("conversation history cursor is invalid")


@dataclass(frozen=True, slots=True)
class ConversationHistoryPage:
    """Chronological presentation page with an older-page continuation.

    ``turns`` and ``commands`` are each in chronological (oldest-first) order; their shared
    ``sequence`` numbers order them against each other. A page fetches both streams before
    its cursor, so the continuation is the oldest included sequence across the two.
    """

    conversation: ConversationHead
    turns: tuple[LinkedTurn, ...]
    next_cursor: ConversationHistoryCursor | None
    fetched_rows: int
    #: Commands are a newer durable event than turns; a page without them is compatible.
    commands: tuple[ConversationCommand, ...] = ()


class ConversationCursorCodec:
    """Encode paired ordering facts as an opaque, integrity-checked token.

    The cursor deliberately carries no principal. Ownership remains a mandatory
    query predicate derived from authentication on every page. Pre-governance
    tokens without scope/version pins are intentionally invalidated because
    continuation cursors are short-lived state.
    """

    def __init__(self, secret: bytes) -> None:
        self._envelope = OpaqueCursorEnvelope(
            secret,
            domain="conversation-list",
            scope="conversation-list",
            fields={"conversation_id", "updated_at"},
        )

    def encode(self, cursor: ConversationCursor) -> str:
        return self._envelope.encode(
            {
                "conversation_id": str(cursor.conversation_id),
                "updated_at": format_utc_timestamp(cursor.updated_at),
            }
        )

    def decode(self, token: str) -> ConversationCursor:
        try:
            decoded = self._envelope.decode(token)
            return ConversationCursor(
                updated_at=parse_utc_timestamp(decoded["updated_at"]),
                conversation_id=parse_canonical_uuid(decoded["conversation_id"]),
            )
        except ValueError as exc:
            raise ConversationCursorError("invalid conversation page cursor") from exc


class ConversationHistoryCursorCodec:
    """Sealed, opaque, conversation-bound ordering cursor with canonical decoding."""

    def __init__(self, secret: bytes) -> None:
        self._envelope = OpaqueCursorEnvelope(
            secret,
            domain="conversation-history",
            scope="conversation-history",
            fields={"before_sequence", "conversation_id"},
        )

    def encode(self, cursor: ConversationHistoryCursor) -> str:
        return self._envelope.encode(
            {
                "before_sequence": cursor.before_sequence,
                "conversation_id": str(cursor.conversation_id),
            }
        )

    def decode(self, token: str) -> ConversationHistoryCursor:
        try:
            decoded = self._envelope.decode(token)
            return ConversationHistoryCursor(
                conversation_id=parse_canonical_uuid(decoded["conversation_id"]),
                before_sequence=decoded["before_sequence"],
            )
        except ValueError as exc:
            raise ConversationCursorError("invalid conversation history cursor") from exc


@dataclass(frozen=True, slots=True)
class AnswerTurnCreation:
    """The run and conversation entry one submission durably created."""

    turn: LinkedTurn
    summary: dict[str, Any]
    replayed: bool


@dataclass(frozen=True, slots=True)
class GoalRoundRequest:
    """One goal continuation waiting to become the conversation's next turn.

    The identity bytes name the goal and the admitted round this continuation belongs to: the
    delivery path compares them with the durable goal, so a continuation from a cleared, replaced
    or superseded goal is dropped instead of being started as the new goal's work.
    """

    request_id: str
    content: str
    prompt_digest: str
    workspaces: tuple[str, ...] = ()
    goal_id: str = ""
    goal_revision: int = 0
    goal_round: int = 0
    created_at: datetime.datetime | None = None


@dataclass(frozen=True, slots=True)
class GoalRoundGuard:
    """The continuation a delivery read, as the transaction must still find it.

    The identity bytes make the fenced check content, not sequencing: the delivery's own
    transaction re-reads the goal row and refuses a continuation whose goal was paused,
    cleared, replaced or moved past the round it reserved.
    """

    request_id: str
    prompt_digest: str
    goal_id: str
    goal_revision: int
    goal_round: int


@dataclass(frozen=True, slots=True)
class QueuedDeliveryGuard:
    """The queue rows one delivery read, as the transaction must still find them.

    The coalesced prompt is built outside the transaction; the digest re-reads the queue
    under the conversation lock, so a message edited or dropped in between makes the
    acceptance refuse instead of delivering words the reader changed.
    """

    message_ids: tuple[str, ...]
    digest: str


def queued_delivery_digest(messages: Sequence[QueuedMessage]) -> str:
    """The digest of exactly what a delivery read: ids, order and content."""
    payload = [[message.message_id, message.position, message.content] for message in messages]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AgentspaceHead:
    """One branch's committed workspace pointer, as the Inspector reads it."""

    lane_id: str
    head_run_id: str
    manifest_digest: str
    revision: int
    committed_at: datetime.datetime


#: Which way one snapshot moved a path, compared with the snapshot it was built on.
AgentspaceChange = Literal["added", "changed", "unchanged"]


@dataclass(frozen=True, slots=True)
class AgentspaceSnapshotChanges:
    """How one snapshot differs from its parent snapshot, counted per path.

    ``added`` counts paths the parent manifest did not carry, ``changed`` counts paths both
    carried with a different content digest, and ``removed`` counts paths the parent carried
    that this snapshot no longer does. A parent that is no longer retained (never kept, or
    pruned out of the history bound) is read as an empty manifest, so every file is ``added``
    and nothing is ``removed``: the read reports the difference the retained rows still show
    instead of inventing one against a snapshot nobody can restore.
    """

    added: int
    changed: int
    removed: int


@dataclass(frozen=True, slots=True)
class AgentspaceSnapshot:
    """One settled Run's restorable snapshot, summarized for a list."""

    run_id: str
    parent_run_id: str | None
    manifest_digest: str
    file_count: int
    created_at: datetime.datetime
    #: How this snapshot differs from its parent, zeroed against a parent it cannot see.
    changes: AgentspaceSnapshotChanges


@dataclass(frozen=True, slots=True)
class AgentspaceFile:
    """One carried file in a snapshot, and how that snapshot got it.

    ``change`` compares this snapshot's manifest with its parent's: ``added`` when the parent
    did not carry the path, ``changed`` when both carried it with a different digest, and
    ``unchanged`` when the digest is the same. Parsing a manifest alone cannot know any of
    that, so the parser starts every file at ``unchanged`` and the projector's one diff helper
    annotates the real value before a reader sees the file.
    """

    relative_path: str
    content_digest: str
    size_bytes: int
    change: AgentspaceChange = "unchanged"


@dataclass(frozen=True, slots=True)
class AgentspaceFileVersion:
    """One snapshot's version of a file, and whether that snapshot changed it."""

    run_id: str
    content_digest: str
    size_bytes: int
    created_at: datetime.datetime
    change: AgentspaceChange


class ConversationSubmissionConflict(ApplicationConflictError):
    """One principal reused a submission id for different accepted input."""


class ConversationSessionBusy(ApplicationConflictError):
    """The Session's one work slot already holds a live Run.

    The refusal happens at the front door, in the transaction that would have linked the
    turn, so nothing was accepted. It names the Run that holds the slot; a send retried
    while that Run is live belongs in the conversation queue instead.
    """

    error_kind = "session_busy"

    def __init__(self, active_run_id: str | None = None) -> None:
        super().__init__(SESSION_BUSY_MESSAGE)
        self.active_run_id = active_run_id


class ConversationQueueConflict(ApplicationConflictError):
    """A queue reorder named a set of messages that is no longer the queued set.

    The caller reads the queue, edits it, and writes the new order back: another tab or
    the delivery path may have changed it between the read and the write, and silently
    reordering a different set would lose or invent a message.
    """

    def __init__(self) -> None:
        super().__init__("the conversation queue changed before the reorder was applied")


class UnknownConversationCommand(ApplicationConflictError):
    """The caller named a command this revision does not register."""

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown conversation command: {name}")


class WebConversationUnavailableError(ApplicationUnavailableError):
    """Durable Web Conversation storage cannot currently be reached."""

    def __init__(self) -> None:
        super().__init__("Web conversation storage is unavailable")


class WebConversationSchemaError(RuntimeError):
    """The durable Web Conversation schema is incompatible with this revision."""


class SessionHeadReclaimer(Protocol):
    """Removes a Session's committed workspace trees after its conversation is gone.

    The pointer row cascades with the Session; the trees on disk need an owner that
    knows the layout. Reclamation is best-effort: a failed deletion of files must not
    make the conversation deletion fail, and the next sweep retries it.
    """

    async def reclaim_session(self, *, owner_id: str, session_id: str) -> None: ...

    async def reclaim_owner(self, *, owner_id: str) -> None: ...


class WebConversationStore(Protocol):
    """Persistence operations owned by Web Conversations."""

    async def prune_empty_conversations(self, *, batch_size: int = 500) -> int: ...
    async def create_conversation(self, principal_id: str) -> dict[str, Any]: ...
    async def list_conversations(
        self,
        principal_id: str,
        *,
        page: ConversationPageRequest,
    ) -> ConversationRowPage: ...
    async def recent_answer_runs(
        self, principal_id: str, *, limit: int
    ) -> tuple[WebRunActivity, ...]: ...
    async def rename_conversation(
        self, principal_id: str, conversation_id: str, *, title: str
    ) -> dict[str, Any] | None: ...
    async def agentspace_heads(
        self, principal_id: str, conversation_id: str
    ) -> tuple[AgentspaceHead, ...] | None: ...
    async def agentspace_snapshots(
        self, principal_id: str, conversation_id: str, *, limit: int
    ) -> tuple[AgentspaceSnapshot, ...] | None: ...
    async def agentspace_snapshot_files(
        self, principal_id: str, conversation_id: str, run_id: str
    ) -> tuple[AgentspaceFile, ...] | None: ...
    async def agentspace_file_versions(
        self, principal_id: str, conversation_id: str, relative_path: str
    ) -> tuple[AgentspaceFileVersion, ...] | None: ...
    async def reset_session_agentspace(
        self, principal_id: str, conversation_id: str
    ) -> str | None: ...
    async def conversation_agent_session(
        self, principal_id: str, conversation_id: str
    ) -> str | None: ...
    async def conversation_for_session(self, principal_id: str, session_id: str) -> str | None: ...
    async def newest_answer_workspaces(
        self, principal_id: str, conversation_id: str
    ) -> tuple[str, ...] | None: ...
    async def agent_sessions(self, principal_id: str) -> tuple[str, ...]: ...
    async def delete_conversation(self, principal_id: str, conversation_id: str) -> bool: ...
    async def delete_all_conversations(self, principal_id: str) -> int: ...
    async def history_page(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        page: ConversationHistoryPageRequest,
    ) -> ConversationHistoryPage | None: ...
    async def append_command(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        name: str,
        command_input: str,
        result: str,
        tone: CommandTone,
        result_data: Mapping[str, Any] | None = None,
    ) -> ConversationCommand | None: ...
    async def live_answer_run(
        self, principal_id: str, conversation_id: str
    ) -> LiveAnswerRun | None: ...
    async def queue_messages(
        self, principal_id: str, conversation_id: str
    ) -> tuple[QueuedMessage, ...] | None: ...
    async def append_queued_message(
        self, principal_id: str, conversation_id: str, *, content: str
    ) -> QueuedMessage | None: ...
    async def edit_queued_message(
        self,
        principal_id: str,
        conversation_id: str,
        message_id: str,
        *,
        content: str,
    ) -> QueuedMessage | None: ...
    async def remove_queued_message(
        self, principal_id: str, conversation_id: str, message_id: str
    ) -> bool: ...
    async def reorder_queued_messages(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        message_ids: Sequence[str],
    ) -> tuple[QueuedMessage, ...] | None: ...
    async def submission_seed(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        attachment_limit: int,
    ) -> SubmissionSeed | None: ...
    async def find_turn_by_run(self, principal_id: str, run_id: str) -> LinkedTurn | None: ...
    async def find_answer_turn_by_submission(
        self, principal_id: str, submission_id: str
    ) -> AnswerTurnCreation | None: ...
    async def replay_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        idempotency_fingerprint: str,
    ) -> AnswerTurnCreation | None: ...
    async def enqueue_goal_round(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        request_id: str,
        content: str,
        prompt_digest: str,
        workspaces: Sequence[str] = (),
        goal_id: str,
        goal_revision: int,
        goal_round: int,
    ) -> bool: ...
    async def pending_goal_round(
        self, principal_id: str, conversation_id: str
    ) -> GoalRoundRequest | None: ...
    async def clear_goal_round(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        request_id: str,
        prompt_digest: str,
    ) -> bool: ...
    async def submission_exists(
        self, principal_id: str, conversation_id: str, submission_id: str
    ) -> bool: ...
    async def create_answer_turn(
        self,
        *,
        principal_id: str,
        conversation_id: str,
        submission_id: str,
        envelope: PreparedRunEnvelope,
        run_id: str,
        artifacts: Sequence[PendingArtifact],
        references: Sequence[PendingArtifactReference],
        title_hint: str | None,
        routing: RoutingAcceptance | None = None,
        connection_bindings: tuple[RunConnectionBinding, ...] = (),
        create_conversation: bool = False,
        forked_from_conversation_id: str | None = None,
        queue_guard: QueuedDeliveryGuard | None = None,
        goal_round_guard: GoalRoundGuard | None = None,
    ) -> AnswerTurnCreation | None: ...


__all__ = [
    "AnswerTurnCreation",
    "CarriedAttachment",
    "CONVERSATION_HISTORY_PAGE_DEFAULT_LIMIT",
    "CONVERSATION_HISTORY_PAGE_MAX_LIMIT",
    "CONVERSATION_PAGE_DEFAULT_LIMIT",
    "CONVERSATION_PAGE_MAX_LIMIT",
    "RUN_ACTIVITY_DEFAULT_LIMIT",
    "RUN_ACTIVITY_MAX_LIMIT",
    "STEERING_MESSAGES_PER_TURN",
    "CommandTone",
    "ConversationCommand",
    "ConversationCursor",
    "ConversationCursorCodec",
    "ConversationCursorError",
    "ConversationGoal",
    "ConversationHead",
    "ConversationHistoryCursor",
    "ConversationHistoryCursorCodec",
    "ConversationHistoryPage",
    "ConversationHistoryPageRequest",
    "ConversationPage",
    "ConversationPageRequest",
    "ConversationQueueConflict",
    "ConversationSessionBusy",
    "ConversationRowPage",
    "ConversationSummary",
    "ConversationSubmissionConflict",
    "LinkedTurn",
    "LiveAnswerRun",
    "QueuedDeliveryGuard",
    "QueuedMessage",
    "SessionHeadReclaimer",
    "SubmissionSeed",
    "UnknownConversationCommand",
    "WebConversationSchemaError",
    "WebConversationStore",
    "WebConversationUnavailableError",
    "WebRunActivity",
    "AgentspaceChange",
    "AgentspaceFile",
    "AgentspaceFileVersion",
    "AgentspaceHead",
    "AgentspaceSnapshot",
    "AgentspaceSnapshotChanges",
]
