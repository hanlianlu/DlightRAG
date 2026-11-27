# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser Pydantic presentation models for Web Conversations."""

import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, field_validator

from dlightrag.adapters.http.browser.presentation import AnswerPresentation
from dlightrag.application.runs import RunStatus
from dlightrag.application.web_conversations.models import AgentspaceChange, CommandTone
from dlightrag.engine.answer.client_contracts import ClientContractModel
from dlightrag.engine.answer.feedback import FeedbackValue


class ConversationSchedule(ClientContractModel):
    """The Schedule whose Fire wrote a conversation: the list marks it with the Schedule's name."""

    schedule_id: str
    name: str


class ConversationSummary(ClientContractModel):
    conversation_id: str
    title: str | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    forked_from_conversation_id: str | None = None
    forked_from_title: str | None = None
    schedule: ConversationSchedule | None = None


class ConversationPage(ClientContractModel):
    items: list[ConversationSummary]
    next_cursor: str | None = None


class WebRunActivity(ClientContractModel):
    """One recent conversation-linked Answer run, as the sidebar's activity read shows it.

    ``phase`` is the executor-owned string (or null), and ``finished_at`` stays null
    until the run is terminal; a terminal run's ``status`` is authoritative.
    ``goal_round`` is true when the run continues a goal: the page keeps the row as
    activity but never announces its settlement as a per-answer completion.
    """

    run_id: str
    conversation_id: str
    conversation_title: str | None = None
    status: RunStatus
    phase: str | None = None
    created_at: datetime.datetime
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    goal_round: bool


class WebRunActivityPage(ClientContractModel):
    items: list[WebRunActivity]


class ConversationAttachmentReference(ClientContractModel):
    attachment_id: str
    ordinal: int
    kind: str
    filename: str
    mime_type: str
    byte_size: int
    url: str
    thumbnail_url: str | None = None
    label: str


class TurnFeedback(ClientContractModel):
    """What the owner judged the turn's answer message to be: the vote, and why."""

    value: FeedbackValue
    reasons: list[str] = Field(default_factory=list)
    note: str | None = None


class ConversationTurn(ClientContractModel):
    turn_id: str
    turn_number: int
    answer_run_id: str
    submission_id: str
    status: RunStatus
    cancel_requested: bool = False
    user_text: str
    assistant_text: str
    user_attachments: list[ConversationAttachmentReference] = Field(default_factory=list)
    steering_messages: list[str] = Field(default_factory=list)
    presentation: AnswerPresentation | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    error_kind: str | None = None
    error_message: str | None = None
    child_count: int = 0
    # The Session `sequence` of the message the Run answered with, which the feedback routes name
    # to judge it. Null until the Run has answered.
    answer_sequence: int | None = None
    # The owner's judgement of that message. Null while they have given none.
    feedback: TurnFeedback | None = None
    created_at: datetime.datetime


class AcceptedAnswer(ClientContractModel):
    """Authoritative conversation and turn created by one accepted command."""

    kind: Literal["accepted"] = "accepted"
    conversation: ConversationSummary
    turn: ConversationTurn


class AnswerDeliveryReceipt(ClientContractModel):
    """A send delivered to the live Run; no conversation turn was created.

    ``queue`` names the durable message a reader can still edit or drop before it is
    delivered as the next turn; ``steer`` names the Run it was injected into and the
    control record that says so.
    """

    kind: Literal["queue", "steer"]
    message: QueuedMessage | None = None
    run_id: str | None = None
    control_sequence: int | None = None


AnswerSubmissionResponse = Annotated[
    AcceptedAnswer | AnswerDeliveryReceipt, Field(discriminator="kind")
]


WebCommandErrorKind = Literal[
    "invalid_request",
    "attachment_rejected",
    "scope_forbidden",
    "conversation_missing",
    "submission_conflict",
    "service_unavailable",
]


class WebCommandError(ClientContractModel):
    """Stable browser answer-command error payload.

    ``error_kind`` names the stable answer error kind when admission rejected the
    request's input, so the browser can localize the reason.
    """

    kind: WebCommandErrorKind
    message: str
    error_kind: str | None = None


class ConversationCommand(ClientContractModel):
    """One server-owned command event, ordered against turns by its sequence.

    ``result`` is the canonical English fallback; ``result_data`` is the same outcome as
    structured data, which the browser localizes. Both are the persisted command's, never
    a re-rendering of it.
    """

    command_id: str
    sequence: int
    name: str
    input: str
    result: str
    tone: CommandTone
    created_at: datetime.datetime
    result_data: dict[str, Any] | None = None


class ConversationCommandResponse(ClientContractModel):
    """The command one invocation executed and persisted."""

    command: ConversationCommand


class ConversationCommandRequest(ClientContractModel):
    """One command invocation: the registered name and the words after it."""

    name: str = Field(min_length=1, max_length=64)
    input: str = Field(max_length=8_000)
    #: The composer's current workspaces; a command that starts a goal round continues them.
    workspaces: list[str] | None = None


class ConversationHistory(ClientContractModel):
    conversation: ConversationSummary
    turns: list[ConversationTurn]
    commands: list[ConversationCommand] = Field(default_factory=list)
    next_cursor: str | None = None


class ConversationGoalState(ClientContractModel):
    """The conversation's current goal, as the goal bar reads it."""

    conversation_id: str
    goal_id: str
    revision: int
    objective: str
    phase: str
    max_rounds: int
    rounds_started: int
    blocked_reason: str | None = None


class GoalCommandRequest(ClientContractModel):
    """One human goal command from the goal bar or the `/goal` command."""

    action: Literal["create", "edit", "pause", "resume", "clear"]
    objective: str | None = Field(default=None, max_length=2000)
    max_goal_rounds: int | None = Field(default=None, ge=1)
    #: The composer's current workspaces: a create's first round continues this selection.
    workspaces: list[str] | None = None


class QueuedMessage(ClientContractModel):
    """One message waiting behind the conversation's live Run, in queue order."""

    message_id: str
    content: str
    position: int
    created_at: datetime.datetime


class AgentspaceHeadView(ClientContractModel):
    """One branch's committed workspace pointer."""

    lane_id: str
    head_run_id: str
    manifest_digest: str
    revision: int
    committed_at: datetime.datetime


class AgentspaceSnapshotChangesView(ClientContractModel):
    """How one snapshot differs from the snapshot it was built on, counted per path.

    A snapshot whose parent is no longer retained is read against an empty parent, so all
    of its files count as added and none as removed.
    """

    added: int
    changed: int
    removed: int


class AgentspaceSnapshotView(ClientContractModel):
    """One settled Run's restorable snapshot, summarized."""

    run_id: str
    parent_run_id: str | None
    manifest_digest: str
    file_count: int
    changes: AgentspaceSnapshotChangesView
    created_at: datetime.datetime


class ConversationAgentspace(ClientContractModel):
    """The conversation's workspace, as the Inspector reads it."""

    heads: list[AgentspaceHeadView] = Field(default_factory=list)
    snapshots: list[AgentspaceSnapshotView] = Field(default_factory=list)


class AgentspaceFileView(ClientContractModel):
    """One file a snapshot carried, and how that snapshot changed it."""

    relative_path: str
    content_digest: str
    size_bytes: int
    change: AgentspaceChange


class AgentspaceSnapshotDetail(ClientContractModel):
    """One snapshot and its files."""

    snapshot: AgentspaceSnapshotView
    files: list[AgentspaceFileView] = Field(default_factory=list)


class AgentspaceFileVersionView(ClientContractModel):
    """One snapshot's version of a file, and whether it changed it."""

    run_id: str
    content_digest: str
    size_bytes: int
    created_at: datetime.datetime
    change: AgentspaceChange


class AgentspaceFileHistory(ClientContractModel):
    """Which Run wrote a file, in snapshot order."""

    relative_path: str
    versions: list[AgentspaceFileVersionView] = Field(default_factory=list)


class ConversationQueue(ClientContractModel):
    """The conversation's queued messages, oldest first."""

    items: list[QueuedMessage] = Field(default_factory=list)


class QueueMessageRequest(ClientContractModel):
    """One message to hold behind the live Run; the service trims and refuses a blank."""

    content: str = Field(min_length=1, max_length=20_000)


class QueueMessageResponse(ClientContractModel):
    """The queued message one call appended or rewrote."""

    message: QueuedMessage


class ReorderQueueRequest(ClientContractModel):
    """The queue's complete new order; every current message exactly once."""

    message_ids: list[UUID] = Field(min_length=1)


class RenameConversationRequest(ClientContractModel):
    title: str = Field(min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("title must not be blank")
        return normalized


__all__ = [
    "AcceptedAnswer",
    "AnswerDeliveryReceipt",
    "AnswerSubmissionResponse",
    "ConversationAttachmentReference",
    "ConversationCommand",
    "ConversationCommandRequest",
    "ConversationCommandResponse",
    "ConversationGoalState",
    "ConversationHistory",
    "ConversationPage",
    "ConversationQueue",
    "ConversationAgentspace",
    "ConversationSummary",
    "ConversationTurn",
    "QueuedMessage",
    "QueueMessageRequest",
    "QueueMessageResponse",
    "ReorderQueueRequest",
    "GoalCommandRequest",
    "RenameConversationRequest",
    "AgentspaceFileHistory",
    "AgentspaceFileVersionView",
    "AgentspaceSnapshotChangesView",
    "AgentspaceSnapshotDetail",
    "TurnFeedback",
    "WebCommandError",
    "WebCommandErrorKind",
    "WebRunActivity",
    "WebRunActivityPage",
]
