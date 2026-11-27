# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser presentation for Application-owned Web Conversation records."""

import re
from collections.abc import Callable, Mapping
from typing import Any

from dlightrag.adapters.http.browser.conversation_models import (
    AnswerDeliveryReceipt,
    ConversationAttachmentReference,
    ConversationCommand,
    ConversationGoalState,
    ConversationHistory,
    ConversationSchedule,
    ConversationSummary,
    ConversationTurn,
    QueuedMessage,
    TurnFeedback,
    WebRunActivity,
)
from dlightrag.adapters.http.browser.presentation import build_answer_presentation
from dlightrag.adapters.http.browser.run_resources import (
    RUN_RESOURCE_URL_BASE,
    run_resource_url,
)
from dlightrag.application.answer.provenance import (
    ArtifactAttestation,
    stamp_when_publishing,
)
from dlightrag.application.web_conversations.models import (
    ConversationCommand as ApplicationConversationCommand,
)
from dlightrag.application.web_conversations.models import (
    ConversationGoal,
    ConversationHead,
    ConversationHistoryPage,
    LinkedTurn,
)
from dlightrag.application.web_conversations.models import (
    ConversationSummary as ApplicationConversationSummary,
)
from dlightrag.application.web_conversations.models import (
    QueuedMessage as ApplicationQueuedMessage,
)
from dlightrag.application.web_conversations.models import (
    WebRunActivity as ApplicationWebRunActivity,
)
from dlightrag.application.web_conversations.service import (
    WebAnswerDelivery as ApplicationAnswerDelivery,
)
from dlightrag.engine.answer.citations.sources import SourceDownloadLinkBuilder
from dlightrag.engine.answer.execution.input import AnswerRunRequest, AttachmentReference
from dlightrag.engine.answer.results import project_answer_result

WEB_SOURCE_DOWNLOAD_BASE = "/web/api/files/raw"
WEB_IMAGE_URL_BASE = "/web/api/images"


def _is_image_mime(mime_type: str | None) -> bool:
    return bool(mime_type) and mime_type.lower().startswith("image/")


def project_answer_delivery(
    delivery: ApplicationAnswerDelivery,
    *,
    mask_reader_text: Callable[[str], str] | None = None,
) -> AnswerDeliveryReceipt:
    """Project a send the server delivered to the live Run instead of a new turn."""
    return AnswerDeliveryReceipt(
        kind=delivery.mode,
        message=(
            None
            if delivery.queued is None
            else project_queued_message(delivery.queued, mask_reader_text=mask_reader_text)
        ),
        run_id=delivery.steered_run_id,
        control_sequence=delivery.control_sequence,
    )


def project_queued_message(
    message: ApplicationQueuedMessage,
    *,
    mask_reader_text: Callable[[str], str] | None = None,
) -> QueuedMessage:
    """Project one queued message into the browser model.

    ``content`` is the caller's own words; it passes the same protected-text gate a
    command's input does, because a reader may have pasted Skill text into it.
    """
    content = message.content
    if mask_reader_text is not None:
        content = mask_reader_text(content)
    return QueuedMessage(
        message_id=message.message_id,
        content=content,
        position=message.position,
        created_at=message.created_at,
    )


def project_conversation_goal(
    goal: ConversationGoal,
    *,
    mask_reader_text: Callable[[str], str] | None = None,
) -> ConversationGoalState:
    """Project an Application goal into its browser model.

    ``mask_reader_text`` is the reader's content gate: a goal's objective is human prose and
    its blocked reason is the model's own sentence about why it stopped, so both are
    reader-facing text like an answer and pass the same gate (ADR 0061).
    """
    objective = goal.objective
    blocked_reason = goal.blocked_reason
    if mask_reader_text is not None:
        objective = mask_reader_text(objective)
        if blocked_reason is not None:
            blocked_reason = mask_reader_text(blocked_reason)
    return ConversationGoalState(
        conversation_id=goal.conversation_id,
        goal_id=goal.goal_id,
        revision=goal.revision,
        objective=objective,
        phase=goal.phase,
        max_rounds=goal.max_rounds,
        rounds_started=goal.rounds_started,
        blocked_reason=blocked_reason,
    )


def project_conversation_summary(
    summary: ApplicationConversationSummary | Mapping[str, Any],
) -> ConversationSummary:
    """Project an Application conversation summary into its browser model."""
    if isinstance(summary, Mapping):
        return ConversationSummary.model_validate(summary)
    return ConversationSummary(
        conversation_id=summary.conversation_id,
        title=summary.title,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        forked_from_conversation_id=summary.forked_from_conversation_id,
        forked_from_title=summary.forked_from_title,
        schedule=(
            ConversationSchedule(
                schedule_id=summary.schedule.schedule_id, name=summary.schedule.name
            )
            if summary.schedule is not None
            else None
        ),
    )


_GOAL_ROUND_PROMPT = re.compile(
    r'^<goal_round>\s*Objective: "(?P<objective>.*?)"\s*Round: (?P<round>\d+)/(?P<rounds>\d+)',
    re.DOTALL,
)


def reader_visible_query(query: str) -> str:
    """The words a linked turn shows the reader, which are not always the words its Run read.

    A goal round's prompt carries the objective and the round inside the harness wrapper the model
    needs; the transcript names the same facts without showing that markup.
    """
    match = _GOAL_ROUND_PROMPT.match(query)
    if match is None:
        return query
    objective = " ".join(match.group("objective").split())
    return f"Goal round {match.group('round')}/{match.group('rounds')}: {objective}"


def project_conversation_history(
    page: ConversationHistoryPage,
    *,
    next_cursor: str | None = None,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
    image_rewrites: Mapping[str, Mapping[str, str]] | None = None,
    attestations: Mapping[str, ArtifactAttestation] | None = None,
    mask_reader_text: Callable[[str], str] | None = None,
) -> ConversationHistory:
    """Project one bounded durable page into browser presentation models.

    ``image_rewrites`` carries, per run id, the finished same-origin address for
    each external image the run stored bytes for; a run absent from it renders
    exactly what its answer wrote. ``attestations`` carries the derived Artifact record per run id,
    for the same reason and with the same rule: a run absent from it renders exactly what its answer
    wrote, which is what every run did before the record existed. ``mask_reader_text`` is the
    reader's content gate, applied to every answer, context and source passage on the page
    (ADR 0061).
    """
    rewrites = image_rewrites or {}
    records = attestations or {}
    return ConversationHistory(
        conversation=_head_summary(page.conversation),
        turns=[
            project_conversation_turn(
                turn,
                downloadable_workspaces=downloadable_workspaces,
                visual_workspaces=visual_workspaces,
                image_rewrites=rewrites.get(turn.run.run_id),
                attestation=records.get(turn.run.run_id),
                mask_reader_text=mask_reader_text,
            )
            for turn in page.turns
        ],
        commands=[
            project_conversation_command(command, mask_reader_text=mask_reader_text)
            for command in getattr(page, "commands", ())
        ],
        next_cursor=next_cursor,
    )


def project_conversation_command(
    command: ApplicationConversationCommand,
    *,
    mask_reader_text: Callable[[str], str] | None = None,
) -> ConversationCommand:
    """Project one persisted command event into its browser model.

    ``input`` is the caller's own words and ``result`` is the server's canonical text;
    both are reader-facing prose, so both pass the same content gate an answer does
    (ADR 0061). The stored row keeps them unmasked. ``result_data`` carries the facts the
    browser localizes, and its prose fields (a goal objective, a blocked reason) can repeat
    protected text, so every string value except the fixed enum fields passes the same
    gate. The browser can then draw the localized result without reopening a leak.
    """
    command_input = command.input
    result = command.result
    if mask_reader_text is not None:
        command_input = mask_reader_text(command_input)
        result = mask_reader_text(result)
    result_data = (
        None
        if command.result_data is None
        else _mask_command_result_data(command.result_data, mask_reader_text)
    )
    return ConversationCommand(
        command_id=command.command_id,
        sequence=command.sequence,
        name=command.name,
        input=command_input,
        result=result,
        tone=command.tone,
        created_at=command.created_at,
        result_data=result_data,
    )


#: Result-data strings that are protocol values the browser matches, not prose: masking one
#: would turn a known outcome into an unknown payload. Every other string is reader-facing.
_RESULT_DATA_ENUM_KEYS = frozenset({"kind", "phase", "code", "reason"})


def _mask_command_result_data(
    value: Mapping[str, Any],
    mask_reader_text: Callable[[str], str] | None,
) -> dict[str, Any]:
    """Apply the reader content gate to the prose inside a structured command result."""

    def _mask(item: Any, *, key: str | None = None) -> Any:
        if mask_reader_text is not None and isinstance(item, str):
            return item if key in _RESULT_DATA_ENUM_KEYS else mask_reader_text(item)
        if isinstance(item, Mapping):
            return {
                str(child_key): _mask(child, key=str(child_key))
                for child_key, child in item.items()
            }
        if isinstance(item, list):
            return [_mask(child) for child in item]
        return item

    return _mask(value)


def project_conversation_turn(
    turn: LinkedTurn,
    *,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
    image_rewrites: Mapping[str, str] | None = None,
    attestation: ArtifactAttestation | None = None,
    mask_reader_text: Callable[[str], str] | None = None,
) -> ConversationTurn:
    """Render one linked turn from the authoritative state of its run.

    A queued or running turn is a pending entry carrying its run id and
    cancellation state, so a reloaded browser resubscribes without remembering
    the original 202 response. A failed or cancelled turn stays visible with its
    public terminal state until its run prunes. Only a succeeded turn carries an
    answer, and that answer is projected from the run's canonical result.

    ``attestation`` is the record the caller derived for this turn's Run -- the same record the live
    Answer published, stamped here by the same helper, so a reader who reopens a conversation sees
    what the reader who watched it saw. The derivation is asynchronous and this projection is not,
    which is why it arrives as an argument. ``mask_reader_text`` is the reader's content gate: the
    answer, the contexts and the source evidence this turn hands a reader pass it, and nothing
    else does (ADR 0061). A turn draws no retrieval contexts (only the answer, its parts and its
    sources), so the projection skips them instead of paying the gate for text no reader sees.
    """

    run = turn.run
    request = AnswerRunRequest.from_request(run.request_input())
    answer = ""
    presentation = None
    if run.status == "succeeded" and run.result is not None:
        projected = project_answer_result(
            run.result,
            source_link_builder=SourceDownloadLinkBuilder(base_url=WEB_SOURCE_DOWNLOAD_BASE),
            downloadable_workspaces=downloadable_workspaces,
            visual_workspaces=visual_workspaces,
            image_url_prefix=WEB_IMAGE_URL_BASE,
            run_id=run.run_id,
            artifact_url_prefix="/web/api/answer",
            run_resource_prefix=RUN_RESOURCE_URL_BASE,
            mask_reader_text=mask_reader_text,
            include_contexts=False,
        )
        projected = stamp_when_publishing(projected, attestation=attestation)
        answer = str(projected["answer"])
        presentation = build_answer_presentation(
            answer=answer,
            sources=projected["sources"],
            evidence_images=projected["evidence_images"],
            link_cards=projected.get("link_cards") or [],
            artifacts=projected["artifacts"],
            artifact_outcome=projected["artifact_outcome"],
            artifact_bindings=projected["artifact_bindings"],
            image_rewrites=image_rewrites,
        )
    return ConversationTurn(
        turn_id=turn.turn_id,
        turn_number=turn.turn_number,
        answer_run_id=run.run_id,
        submission_id=turn.submission_id,
        status=run.status,
        cancel_requested=run.cancel_requested,
        user_text=reader_visible_query(request.query),
        assistant_text=answer,
        user_attachments=[
            _attachment_reference(run.run_id, turn.turn_number, attachment)
            for attachment in request.attachments
        ],
        steering_messages=list(turn.steering_messages),
        presentation=presentation,
        usage=dict((run.result or {}).get("usage") or {}),
        error_kind=run.error_kind,
        error_message=run.error_message,
        child_count=turn.child_count,
        answer_sequence=turn.answer_sequence,
        feedback=(
            None if turn.feedback is None else TurnFeedback.model_validate(turn.feedback.payload())
        ),
        created_at=turn.created_at,
    )


def project_run_activity(activity: ApplicationWebRunActivity) -> WebRunActivity:
    """Project one Application run-activity row into its browser model.

    The projection carries the store's facts unchanged — the durable Run's lifecycle,
    the conversation's title, and the Run's goal-continuation fact: it adds no derived
    state, so a poll reads exactly what the store holds.
    """
    return WebRunActivity(
        run_id=activity.run_id,
        conversation_id=activity.conversation_id,
        conversation_title=activity.conversation_title,
        status=activity.status,
        phase=activity.phase,
        created_at=activity.created_at,
        started_at=activity.started_at,
        finished_at=activity.finished_at,
        goal_round=activity.goal_round,
    )


def _head_summary(head: ConversationHead) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=head.conversation_id,
        title=head.title,
        created_at=head.created_at,
        updated_at=head.updated_at,
    )


def _attachment_reference(
    run_id: str,
    turn_number: int,
    attachment: AttachmentReference,
) -> ConversationAttachmentReference:
    is_image = _is_image_mime(attachment.mime_type)
    url = run_resource_url(run_id, attachment.resource_id)
    return ConversationAttachmentReference(
        attachment_id=f"{run_id}:{attachment.ordinal}",
        ordinal=attachment.ordinal,
        kind="image" if is_image else "document",
        filename=attachment.filename,
        mime_type=attachment.mime_type,
        byte_size=attachment.byte_size,
        url=url,
        thumbnail_url=(url + "/thumbnail") if is_image else None,
        label=f"Turn {turn_number}, attachment {attachment.ordinal}",
    )


__all__ = [
    "WEB_IMAGE_URL_BASE",
    "WEB_SOURCE_DOWNLOAD_BASE",
    "project_conversation_command",
    "project_conversation_history",
    "project_conversation_summary",
    "project_conversation_turn",
    "project_run_activity",
]
