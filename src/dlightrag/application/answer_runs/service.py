# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable answer runs over already-authorized canonical workspaces."""

import asyncio
import re
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import aclosing, asynccontextmanager
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Any, Literal, Protocol, cast
from uuid import UUID

from dlightrag.application.connections.models import (
    BoundConnectionSurface,
    BoundResearchConnections,
)
from dlightrag.application.errors import (
    ApplicationConflictError,
    ApplicationInputError,
    ApplicationUnavailableError,
)
from dlightrag.application.runs import (
    RunAdmission,
    RunAdmissionScheduler,
    RunCancelledError,
    RunCreation,
    RunEvent,
    RunFailedError,
    RunReplayer,
    RunRuntimeUnavailableError,
)
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.skill_protection import MASK_PLACEHOLDER, mask_reader_strings
from dlightrag.engine.agent.skills import load_skill_document_pair
from dlightrag.engine.agent.tools.contracts import ToolDeclaration
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
from dlightrag.engine.ai.reasoning import request_under_ceiling
from dlightrag.engine.ai.settings import CHAT_MODEL_SELECTORS, ChatModelSelector, ModelSettings
from dlightrag.engine.answer.agent_browser.accounts import (
    MAY_REGISTER_PIN,
)
from dlightrag.engine.answer.capabilities import AnswerCapabilities, RequestModelContext
from dlightrag.engine.answer.client_contracts import (
    ANSWER_EFFORT_LEVELS,
    AnswerEffort,
    offered_answer_efforts,
)
from dlightrag.engine.answer.errors import ACTIVE_RUN_LIMIT, ACTIVE_RUN_LIMIT_MESSAGE
from dlightrag.engine.answer.execution.acceptance import (
    AnswerHistoryBudget,
    ResearchSeed,
    accept_history,
    reserved_memory_text,
    routing_resources,
)
from dlightrag.engine.answer.execution.connection_binding import (
    ConnectionSurface,
    RunConnectionBinding,
    StaleConnectionBindingError,
    decode_connection_surface,
)
from dlightrag.engine.answer.execution.input import (
    AnswerRunInput,
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    PinnedModelProfile,
    build_current_answer_resources,
    child_model_guidance,
    in_memory_attachment_loader,
    model_reasoning_settings,
    new_resource_identity,
)
from dlightrag.engine.answer.execution.resources import ResolvedAnswerResources
from dlightrag.engine.answer.feedback import AnswerFeedback, normalize_feedback
from dlightrag.engine.answer.image_capability import AnswerImageCapability
from dlightrag.engine.answer.images import AnswerImagePolicy
from dlightrag.engine.answer.mode import (
    AnswerMode,
    ModeCapability,
    ModeResource,
    ResolvedMode,
    canonical_answer_mode,
    require_supported_mode,
    resource_role,
    valid_modes,
)
from dlightrag.engine.answer.resources.admission import require_readable_attachment
from dlightrag.engine.answer.resources.images import QueryImageDescriber, prepare_query_images
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.answer.results import AnswerResult, restore_answer_result
from dlightrag.engine.answer.runs.envelope import accepted_input_envelope
from dlightrag.engine.answer.runs.routing import RoutingAcceptance
from dlightrag.engine.answer.tools.composition import (
    ResearchToolDeclarations,
    research_tool_declarations,
)
from dlightrag.engine.network_admission import public_http_url_identity
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename
from dlightrag.engine.rag.retrieval.models import (
    MetadataFilter,
    RetrievalOptions,
)
from dlightrag.engine.rag.retrieval.planner import RetrievalPlanner
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id
from dlightrag.engine.runtime.records import (
    ActiveRunLimitExceededError,
    ArtifactReferenceKind,
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    RunAccessScope,
    RunArtifactReference,
    RunFetchedResource,
    RunRecord,
    artifact_digest,
    run_request_fingerprint,
)
from dlightrag.engine.runtime.records import RunCreation as RuntimeRunCreation
from dlightrag.engine.runtime.records import RunEvent as RuntimeRunEvent

from .artifacts import published_artifact_descriptor
from .child_roster import (
    ChildRosterCursor,
    ChildRosterCursorCodec,
    ChildRosterCursorError,
    ChildRosterPage,
    ChildRosterPageRequest,
    ChildRosterRowPage,
    child_result_lineage,
    public_child_status,
    stamp,
)
from .overview import (
    FINDINGS_MAX,
    RunOverview,
    agent_rollups,
    excerpt_of,
    findings_of,
    overview_totals,
    run_usage,
)
from .timeline import (
    MAIN_AGENT,
    RunStepSearch,
    RunTimelinePosition,
    RunTimelineStep,
    agent_identities,
    merge_steps,
    page_of,
    with_page_content,
)
from .timeline_index import (
    RunTimelineToken,
    TimelineIndex,
    TimelineIndexCache,
)

#: Accepted input uploads, in the precedence one ordinal resolves against.
_INPUT_REFERENCE_KINDS: tuple[ArtifactReferenceKind, ...] = (
    "current_attachment",
    "history_attachment",
)


def _resource_source_urls(resource: RunFetchedResource) -> tuple[str, ...]:
    """Every spelling of the external URL one stored resource answers for."""
    recorded: list[str] = []
    if resource.source_locator:
        recorded.append(resource.source_locator.decode("utf-8", "replace"))
    aliases = resource.capabilities.get("resource_aliases")
    if isinstance(aliases, list):
        recorded.extend(str(alias) for alias in aliases if alias)
    return tuple(url for url in recorded if url)


_AGENT_CONTROL_CONTENT_LIMIT = 20_000
CHILD_CONTROL_SUCCESS_OUTCOMES: frozenset[str] = frozenset(
    {"queued", "consumed", "accepted", "cancellation_requested"}
)


def child_control_succeeded(outcome: str) -> bool:
    """Return whether one owner Child control was durably applied."""
    return outcome in CHILD_CONTROL_SUCCESS_OUTCOMES


_HAN = re.compile(r"[0-9A-Za-z_]+|[\u3400-\u9fff]")


def split_terms(text: str, *, loose: bool = False) -> tuple[str, ...]:
    """The words a search is for: a quoted phrase stays whole, and Han is read whole too.

    Chinese is written without spaces, so reading a run of Han as one word is what the reader
    typed; ``loose`` reads it per character instead, which asks for those characters in any order
    and is the second pass a reader gets when the words as written found nothing.
    """
    terms: list[str] = []
    for token in re.findall(r'"[^"]*"|\S+', text):
        phrase = token[1:-1].strip() if token.startswith('"') and token.endswith('"') else ""
        if phrase:
            terms.append(phrase)
        elif loose:
            terms.extend(re.findall(r"[0-9A-Za-z_]+|[\u3400-\u9fff]", token))
        else:
            terms.extend(re.findall(r"[0-9A-Za-z_]+|[\u3400-\u9fff]+", token))
    return tuple(dict.fromkeys(term for term in terms if term))[:8]


def _masked_skill_result(content: str, pair: tuple[str, str]) -> str:
    """One protected ``load_skill`` result a reader may read: framing kept, document body gone.

    The banner and the ``--- name/path ---`` marker are the fact the trace exists to state —
    which document the Run read — and are not the author's own words. The body becomes
    ``[...]`` whatever its size: the fingerprint gate needs a long enough passage, and a
    short Skill document reads whole through this tool.
    """
    marker = f"--- {pair[0]}/{pair[1]} ---"
    end = content.index(marker) + len(marker)
    return f"{content[:end]}\n{MASK_PLACEHOLDER}"


def _masked_argument_value(value: Any, mask_reader_text: Callable[[str], str] | None) -> Any:
    """One stored Tool argument with every string inside it passed through the reader gate.

    An argument is the text the model itself wrote to a Tool — a ``write_to`` body, a heredoc, a
    document it asks a Tool to publish — so a protected document's words travel through
    arguments exactly as they travel through results. A JSON string is masked whole; a decoded
    object keeps its shape and only its string leaves change.
    """
    if mask_reader_text is None:
        return value
    return mask_reader_strings(value, mask_reader_text)


def _masked_tool_call(
    call: Mapping[str, Any], mask_reader_text: Callable[[str], str]
) -> dict[str, Any]:
    """One stored Tool call a reader may read: the call itself, its arguments masked."""
    projected = dict(call)
    for key in ("arguments", "argument_error"):
        if key in projected:
            projected[key] = _masked_argument_value(projected[key], mask_reader_text)
    return projected


def _masked_store_message(
    message: Mapping[str, Any],
    *,
    protected: frozenset[tuple[str, str]],
    mask_reader_text: Callable[[str], str] | None = None,
    mask_tool_arguments: bool = True,
) -> dict[str, Any]:
    """One store-projected message, shape untouched, with protected Skill text replaced.

    A ``load_skill`` result naming a protected document takes the header fast path: the banner
    and the marker line stay readable and only the body is replaced. Every other string goes
    through the fingerprint gate, which catches the same bytes arriving through a ``bash cat``,
    a copied file read back, or a child agent's conclusion — the tool name cannot tell them
    apart. A Tool call's arguments are text the Run wrote, so they are gated too: that is the
    road a ``write_to`` body, a heredoc, or a ``publish_skill`` document takes to a reader.

    ``mask_tool_arguments=False`` is for the projections that never emit what a call was asked
    to do (they state its id, name and bounded subject alone): gating a field the reader cannot
    receive is work the reader pays for and never sees, and it is the largest share of a timeline
    page's masking. A projection that keeps the stored shape must leave this on.
    """
    projected = dict(message)
    body = projected.get("content")
    if (
        projected.get("role") == "tool"
        and projected.get("name") == "load_skill"
        and isinstance(body, str)
    ):
        pair = load_skill_document_pair(body)
        if pair is not None and pair in protected:
            projected["content"] = _masked_skill_result(body, pair)
    if mask_reader_text is not None:
        body = projected.get("content")
        if isinstance(body, str):
            projected["content"] = mask_reader_text(body)
        calls = projected.get("tool_calls")
        if (
            mask_tool_arguments
            and isinstance(calls, Sequence)
            and not isinstance(calls, (str, bytes))
        ):
            projected["tool_calls"] = [
                _masked_tool_call(call, mask_reader_text) if isinstance(call, Mapping) else call
                for call in calls
            ]
        facts = projected.get("facts")
        if facts is not None:
            projected["facts"] = mask_reader_strings(facts, mask_reader_text)
    return projected


def _without_reasoning(message: Mapping[str, Any]) -> dict[str, Any]:
    """One store-projected message with the private reasoning field left behind.

    The store projects an assistant entry's reasoning so the Inspector can draw it; a transcript
    tail is not that surface, and ``_masked_store_message`` keeps a message's shape untouched, so
    the field is dropped here rather than travelling to a reader unmasked.
    """
    return {key: value for key, value in message.items() if key != "reasoning"}


def _public_search_hit(
    hit: Mapping[str, Any],
    *,
    protected: frozenset[tuple[str, str]],
    mask_reader_text: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """One search hit as a reader sees it: the words in their neighbourhood, or a placeholder.

    A hit inside a ``load_skill`` result names the document, so a protected one is replaced by
    its readable framing; any other hit keeps the neighbourhood the search measured and is
    masked only where the words are a protected document's own text.
    """
    snippet = str(hit["snippet"])
    if hit.get("name") == "load_skill":
        content = str(hit.get("content") or "")
        pair = load_skill_document_pair(content)
        if pair is not None and pair in protected:
            snippet = _masked_skill_result(content, pair)
    if mask_reader_text is not None:
        snippet = mask_reader_text(snippet)
    return {
        "sequence": int(hit["sequence"]),
        "role": str(hit["role"]),
        "snippet": snippet,
        # The words this message holds, so the pane that shows the matches has something to
        # show and something to mark. It is the neighbourhood the search already measured;
        # the whole message would be megabytes a Run.
        "content": snippet,
        **({"agent": str(hit["agent"])} if hit.get("agent") else {}),
        **({"agent_label": str(hit["agent_label"])} if hit.get("agent_label") else {}),
    }


#: The most reasoning text one public transcript message states.
#: The Inspector draws a step's thinking beside its words, not as a page of its own; a provider's
#: reasoning can run far longer than the answer it produces, so the reader edge bounds it. Slicing
#: is the bound: no marker is appended, because the reader is not told a truncation it cannot see.
PUBLIC_REASONING_MAX_CHARS = 4000


def _public_transcript_message(
    message: Mapping[str, Any],
    *,
    protected_skill_documents: frozenset[tuple[str, str]],
    tool_subjects: Mapping[str, str] | None = None,
    mask_reader_text: Callable[[str], str] | None = None,
    include_reasoning: bool = False,
) -> dict[str, Any]:
    """Keep useful conversation/tool lineage; the model's reasoning only when asked for.

    A protected Skill document reads as a placeholder: the trace says what the Run read without
    handing over its author's text (ADR 0061). The model's own transcript is untouched.

    ``include_reasoning`` is the Inspector activity page's surface alone: every other reader —
    child observation, the transcript tail, search, the step-search excerpt — states no reasoning.
    When it is asked for, the text is gated through the reader's masker (reasoning can quote a
    protected document the Run read) and bounded by ``PUBLIC_REASONING_MAX_CHARS``.
    """
    message = _masked_store_message(
        message,
        protected=protected_skill_documents,
        mask_reader_text=mask_reader_text,
        # This projection states a call's id, name and bounded subject; what the call was asked
        # to do never leaves it, so gating the stored arguments would mask text no reader receives.
        mask_tool_arguments=False,
    )
    role = str(message.get("role") or "")
    projected: dict[str, Any] = {"role": role, "content": message.get("content") or ""}
    if message.get("agent") is not None:
        projected["agent"] = str(message["agent"])
    if message.get("agent_label") is not None:
        projected["agent_label"] = str(message["agent_label"])
    if "sequence" in message:
        projected["sequence"] = int(message["sequence"])
    # When the entry behind this message was written, which is what a client times a step by: a
    # tool's own time is the gap between the message that called it and the result's. A message
    # built from a request rather than a stored entry has no clock and says so by leaving it out.
    if message.get("at") is not None:
        projected["at"] = str(message["at"])
    if role == "assistant":
        tool_calls: list[dict[str, Any]] = []
        for call in message.get("tool_calls") or ():
            if not isinstance(call, Mapping):
                continue
            call_id = call.get("id")
            entry: dict[str, Any] = {
                "id": call_id,
                "name": call.get("name")
                or (
                    call.get("function", {}).get("name")
                    if isinstance(call.get("function"), Mapping)
                    else None
                ),
            }
            # The bounded label the Tool derived from one of its inputs (a command, query or path),
            # from the same event log; it is the Tool's own subject, never its raw argument or
            # result payload.
            subject = tool_subjects.get(str(call_id)) if tool_subjects and call_id else None
            if subject:
                entry["subject"] = subject
            tool_calls.append(entry)
        projected["tool_calls"] = tool_calls
        if include_reasoning:
            reasoning = str(message.get("reasoning") or "")
            if reasoning.strip():
                text = mask_reader_text(reasoning) if mask_reader_text is not None else reasoning
                projected["reasoning"] = text[:PUBLIC_REASONING_MAX_CHARS]
    if role == "tool":
        projected["tool_call_id"] = str(message.get("tool_call_id") or "")
        projected["name"] = str(message.get("name") or "")
        projected["is_error"] = bool(message.get("is_error"))
        # The facts a Tool reported are the bounded structure a step card states the call
        # with, so they travel beside the result text rather than inside it. Shape is kept
        # as stored: this projection only decides which message fields a reader sees.
        facts = message.get("facts")
        if isinstance(facts, Mapping):
            projected["facts"] = (
                mask_reader_strings(dict(facts), mask_reader_text)
                if mask_reader_text is not None
                else dict(facts)
            )
    return projected


def _project_timeline_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    protected: frozenset[tuple[str, str]],
    mask_reader_text: Callable[[str], str] | None,
) -> dict[tuple[str, int], dict[str, Any]]:
    """One Run read's entries projected for the reader, keyed by ``(Session, sequence)``.

    The projection states the content a search row's excerpt and a finding's excerpt read, and no
    reasoning: the step search is the only step list a Run read draws, and a list states no
    reasoning (the Inspector activity page is that surface). A cold page's gate walks the loaded
    text, which is work the event loop must not carry while it also runs the Run being watched: the
    callers hand this to a thread. The projection itself is pure and touches no shared state beyond
    the masker's own thread-safe cache.
    """
    return {
        (str(message.get("session_id") or "").lower(), int(message.get("sequence") or 0)): (
            _public_transcript_message(
                message,
                protected_skill_documents=protected,
                mask_reader_text=mask_reader_text,
            )
        )
        for message in messages
    }


def _project_activity_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    protected: frozenset[tuple[str, str]],
    tool_subjects: Mapping[str, str],
    mask_reader_text: Callable[[str], str] | None,
) -> tuple[dict[str, Any], ...]:
    """One agent transcript page projected for the reader, oldest first.

    The activity page is the one transcript surface that states reasoning; it runs off the event
    loop for the same reason the Run read's projection does.
    """
    return tuple(
        _public_transcript_message(
            message,
            protected_skill_documents=protected,
            tool_subjects=tool_subjects,
            mask_reader_text=mask_reader_text,
            include_reasoning=True,
        )
        for message in messages
    )


def _tool_call_ids(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    """The tool call ids one transcript page names, in order, deduped.

    Only ids the event log can answer for; the subjects join works on ids, so nothing else is needed.
    """
    ids: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            if isinstance(call, Mapping) and call.get("id"):
                call_id = str(call["id"])
                if call_id not in ids:
                    ids.append(call_id)
    return ids


def _public_control_record(
    row: Mapping[str, Any], *, mask_reader_text: Callable[[str], str] | None = None
) -> dict[str, Any]:
    """One Child control a reader may read; its content is text a Run or a reader wrote."""
    content = str(row.get("content") or "")
    if mask_reader_text is not None:
        content = mask_reader_text(content)
    consumed_at = row.get("consumed_at")
    return {
        "control_sequence": int(row.get("control_sequence") or 0),
        "kind": str(row.get("kind") or ""),
        "content": content,
        "origin": str(row.get("origin") or ""),
        "consumed": bool(row.get("consumed") if "consumed" in row else consumed_at),
        "consumed_at": stamp(consumed_at),
        "created_at": stamp(row.get("created_at")),
        "operation_id": (str(row["operation_id"]) if row.get("operation_id") else None),
    }


def _public_question_record(
    row: Mapping[str, Any], *, mask_reader_text: Callable[[str], str] | None = None
) -> dict[str, Any]:
    """One Child guidance question and its reply, gated like every other reader text."""
    question = str(row.get("question") or "")
    reply = row.get("reply")
    if mask_reader_text is not None:
        question = mask_reader_text(question)
        if isinstance(reply, str):
            reply = mask_reader_text(reply)
    return {
        "request_id": str(row.get("request_id") or ""),
        "question": question,
        "status": str(row.get("status") or ""),
        "reply": reply,
        "reply_origin": row.get("reply_origin"),
        "expires_at": stamp(row.get("expires_at")),
        "created_at": stamp(row.get("created_at")),
    }


@dataclass(frozen=True, slots=True)
class AnswerHistoryResource:
    """One accepted upload carried from an owned prior run into this request."""

    run_id: str
    source_ordinal: int
    digest: str
    filename: str
    mime_type: str
    byte_size: int
    reference_kind: ArtifactReferenceKind = "current_attachment"


@dataclass(frozen=True, slots=True)
class AgentEffortOffer:
    """The agent efforts one deployment applies for one caller, and the level that caller's
    Answers run at when they choose none."""

    levels: tuple[AnswerEffort, ...]
    default: AnswerEffort | None


class Caller(Protocol):
    """Whoever submits an Answer: the verified facts a Runtime Policy may name them by.

    It has the shape of the Application's ``Principal``, which every transport's identity
    satisfies. It is stated here because Answer Runs import nothing of access: who may
    submit was decided before a submission reaches them.
    """

    @property
    def user_id(self) -> str: ...

    @property
    def auth_mode(self) -> str: ...

    @property
    def claims(self) -> Mapping[str, object]: ...


class RunPolicies(Protocol):
    """What the Runtime Policies decide about an Answer, read for whoever submits it.

    Every call reads the current document, so a change holds for the next submission.
    A document that can no longer be read raises, and a submission is then refused rather
    than admitted under a default.
    """

    async def effort_ceiling(self, caller: Caller | None, /) -> AnswerEffort: ...

    async def active_run_limit(self, caller: Caller | None, /) -> int | None: ...


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    """One authorized answer request over concrete canonical workspaces.

    Authorization happens before this contract exists: the workspace set is the
    already-expanded canonical result, never a policy wildcard, a token claim,
    or a user-visible display name.
    """

    query: str
    workspaces: tuple[str, ...]
    history: tuple[Mapping[str, Any], ...] = ()
    episodic_summary: str = ""
    retrieval: RetrievalOptions = RetrievalOptions()
    filters: MetadataFilter | None = None
    semantic_highlights: bool = False
    resources: tuple[ResourceInput, ...] = ()
    history_resources: tuple[AnswerHistoryResource, ...] = ()
    mode: str | None = None
    parent_run_id: str | None = None
    continuation_kind: str | None = None
    #: The goal identity of a goal round, so its settlement can tell which goal it served.
    goal_id: str | None = None
    goal_revision: int | None = None
    goal_round: int | None = None
    agent_session_id: str = ""
    agent_lane_id: str = "main"
    source_lane_id: str | None = None
    requested_skill: str | None = None
    #: The caller's own agent effort for this run, when one was chosen.
    effort: AnswerEffort | None = None


@dataclass(frozen=True, slots=True)
class AnswerInputArtifact:
    """One accepted run input upload, read back with its stored bytes."""

    reference_kind: ArtifactReferenceKind
    ordinal: int
    filename: str
    mime_type: str
    digest: str
    content: bytes


@dataclass(frozen=True, slots=True)
class RunResourceDescriptor:
    """One run-scoped byte surface, resolved from whichever registry owns it.

    A run records bytes in two registries: the accepted input artifacts and
    publications of ``dlightrag_answer_run_artifacts``, and the resources a worker
    fetched, rendered, or adopted during the run. Both name the same
    content-addressed blob plane, so one id resolves to one digest and one read
    path serves them all.
    """

    resource_id: str
    registry: Literal["artifact", "resource"]
    reference_kind: ArtifactReferenceKind | None
    ordinal: int
    filename: str
    mime_type: str
    digest: str


@dataclass(frozen=True, slots=True)
class AgentControlReceipt:
    """One ordered control accepted for a live Research session."""

    run_id: str
    control_sequence: int
    kind: str


@dataclass(frozen=True, slots=True)
class ChildControlReceipt:
    """Durable result of one owner-scoped Child intervention."""

    run_id: str
    action: str
    outcome: str
    child_session_id: str | None = None
    request_id: str | None = None
    operation_id: str | None = None
    operation_sequence: int | None = None
    control_sequence: int | None = None
    consumed_at: Any | None = None
    status: str | None = None

    def payload(self) -> dict[str, Any]:
        """Return the transport-neutral control receipt document."""
        return child_control_receipt_payload(self)


def child_control_receipt_payload(receipt: Any) -> dict[str, Any]:
    """Project one Child control receipt for REST, MCP, and Web."""
    consumed = getattr(receipt, "consumed_at", None)
    isoformat = getattr(consumed, "isoformat", None)
    child_session_id = getattr(receipt, "child_session_id", None)
    operation_id = getattr(receipt, "operation_id", None)
    return {
        "run_id": receipt.run_id,
        "child_session_id": child_session_id or None,
        "request_id": getattr(receipt, "request_id", None),
        "action": receipt.action,
        "outcome": receipt.outcome,
        "operation_id": operation_id or None,
        "operation_sequence": getattr(receipt, "operation_sequence", None),
        "control_sequence": getattr(receipt, "control_sequence", None),
        "consumed_at": isoformat() if callable(isoformat) else consumed,
        **({"status": receipt.status} if getattr(receipt, "status", None) is not None else {}),
    }


@dataclass(frozen=True, slots=True)
class AgentTranscriptTail:
    """Bounded application projection shared by every transport."""

    run_id: str
    status: str
    messages: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ChildObservation:
    """Bounded child transcript, control, question, and result lineage."""

    run_id: str
    child: Mapping[str, Any]
    transcript: tuple[Mapping[str, Any], ...]
    controls: tuple[Mapping[str, Any], ...]
    questions: tuple[Mapping[str, Any], ...]
    result: Mapping[str, Any] | None

    def payload(self) -> dict[str, Any]:
        """Return the transport-neutral observation document."""
        return {
            "run_id": self.run_id,
            "child": dict(self.child),
            "transcript": [dict(item) for item in self.transcript],
            "controls": [dict(item) for item in self.controls],
            "questions": [dict(item) for item in self.questions],
            "result": dict(self.result) if self.result is not None else None,
        }


#: One Activity page holds this many messages unless the caller asks for fewer.
# A search answers with every match in one Run, and a Run is bounded: the ceiling is a guard
# against a single character matching thousands of messages, not a page size. The answer states
# how many matches there are, so a reader is never shown a silently shortened list.
ACTIVITY_PAGE_DEFAULT_LIMIT = 30
ACTIVITY_PAGE_MAX_LIMIT = 50


@dataclass(frozen=True, slots=True)
class ActivityPage:
    """One page of an Agent Session's public transcript, oldest message first.

    The page holds the newest messages older than the caller's cursor. ``next_before`` is the
    cursor of the next older page, or None when this page reaches the start. ``running`` says
    whether the Session's agent is still working, so a call without a result is in flight.
    """

    messages: tuple[Mapping[str, Any], ...]
    next_before: int | None
    running: bool

    def payload(self) -> dict[str, Any]:
        return {
            "messages": [dict(item) for item in self.messages],
            "next_before": self.next_before,
            "running": self.running,
        }


class AnswerRequestError(ApplicationInputError):
    """A steering, child-control, continuation, or feedback request is invalid as given."""


class AnswerConnectionsChangedError(ApplicationConflictError):
    """The owner's Connections kept changing while an Answer was being accepted."""

    def __init__(self) -> None:
        super().__init__("Connections changed; submit the Answer again")


class ActiveRunLimitError(ApplicationUnavailableError):
    """The owner already has as many Answers in progress as the policy lets them.

    It is the caller's own limit rather than the deployment's, so transports answer it as a
    refusal to retry once one of their Answers has ended (HTTP 429), under a kind of its own.
    """

    error_kind = ACTIVE_RUN_LIMIT

    def __init__(self, limit: int) -> None:
        super().__init__(ACTIVE_RUN_LIMIT_MESSAGE)
        self.limit = limit


class AnswerRunAcceptor[T](RunReplayer[T], Protocol):
    """Persist or replay one prepared run, optionally with an atomic domain link."""

    async def accept_run(
        self,
        *,
        envelope: PreparedRunEnvelope,
        run_id: str,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
        connection_bindings: tuple[RunConnectionBinding, ...] = (),
    ) -> T | None: ...


TranscriptScope = Literal["run", "lane"]
"""Which messages a transcript read asks for: the named Run's, or a Session's lane across Runs.

A conversation's agent is one Session across the Runs that continue it, so both questions are
asked of the same Session and only the caller knows which one it means.
"""


class _AnswerRunRepository(AnswerRunAcceptor[RuntimeRunCreation], Protocol):
    """The owner-scoped durable operations this service performs."""

    async def get_run(self, *, owner_id: str, run_id: str) -> RunRecord | None: ...

    async def enqueue_agent_control(
        self, *, owner_id: str, run_id: str, kind: str, content: str
    ) -> Mapping[str, Any] | None: ...

    async def list_child_sessions(
        self, *, owner_id: str, run_id: str
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def list_child_sessions_page(
        self,
        *,
        owner_id: str,
        run_id: str,
        page: ChildRosterPageRequest,
    ) -> ChildRosterRowPage: ...

    async def load_agent_transcript(
        self,
        *,
        owner_id: str,
        run_id: str,
        session_id: str | None,
        limit: int,
        before: int | None = None,
        scope: TranscriptScope = ...,
        since: int | None = None,
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def search_run_transcript(
        self,
        *,
        owner_id: str,
        run_id: str,
        terms: Sequence[str],
        require_all: bool,
        limit: int = ...,
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def load_run_timeline_steps(
        self, *, owner_id: str, run_id: str
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def load_run_timeline_token(
        self, *, owner_id: str, run_id: str
    ) -> Mapping[str, Any] | None: ...

    async def load_timeline_entries(
        self,
        *,
        owner_id: str,
        run_id: str,
        keys: Sequence[tuple[str, int]],
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def load_tool_subjects(
        self, *, owner_id: str, run_id: str, call_ids: Sequence[str]
    ) -> Mapping[str, str]: ...

    async def put_answer_feedback(
        self,
        *,
        owner_id: str,
        run_id: str,
        agent: str,
        sequence: int,
        value: str,
        reasons: Sequence[str],
        note: str | None,
    ) -> AnswerFeedback | None: ...

    async def delete_answer_feedback(
        self, *, owner_id: str, run_id: str, agent: str, sequence: int
    ) -> None: ...

    async def list_answer_feedback(
        self, *, owner_id: str, run_id: str
    ) -> tuple[AnswerFeedback, ...]: ...

    async def load_child_session(
        self, *, owner_id: str, run_id: str, child_session_id: str
    ) -> Mapping[str, Any] | None: ...

    async def list_child_controls(
        self, *, owner_id: str, run_id: str, child_session_id: str, limit: int
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def list_child_guidance(
        self, *, owner_id: str, run_id: str, child_session_id: str, limit: int
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def enqueue_child_control(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
        worker_id: str | None = None,
        fencing_epoch: int | None = None,
    ) -> Mapping[str, Any] | bool: ...

    async def continue_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
        reauthorize_user_cancelled: bool = False,
        worker_id: str | None = None,
        fencing_epoch: int | None = None,
    ) -> Mapping[str, Any] | bool: ...

    async def cancel_child_session_by_owner(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        submission_key: str,
        parent_session_id: str | None = None,
    ) -> Mapping[str, Any]: ...

    async def reply_child_guidance(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
        worker_id: str | None = None,
        fencing_epoch: int | None = None,
    ) -> Mapping[str, Any] | bool: ...

    async def list_run_artifacts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunArtifactReference, ...]: ...

    async def list_fetched_resources(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunFetchedResource, ...]: ...

    async def read_run_resource_row(
        self, *, owner_id: str, run_id: str, resource_id: str
    ) -> RunFetchedResource | None: ...

    async def latest_run_attachment(
        self, *, owner_id: str, run_id: str, safe_name: str
    ) -> RunFetchedResource | None: ...


class _RunBlobReader(Protocol):
    """The opaque bytes seam; Answer metadata never owns blob persistence."""

    def stream(
        self,
        *,
        owner_id: str,
        digest: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes]: ...

    async def size(self, *, owner_id: str, digest: str) -> int | None: ...


class _RunScheduler(RunAdmissionScheduler, Protocol):
    """The started coordinator accepted runs execute and stream through."""

    def cancel_local(self, owner_id: str, run_id: str) -> None: ...

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncGenerator[RuntimeRunEvent]: ...


class _RetrievalPlanning(Protocol):
    """The retrieval facts acceptance needs before a run is scheduled."""

    def planner_for(self, model_profile: ModelProfile | None = None) -> RetrievalPlanner: ...

    def warm(self, workspaces: Sequence[str]) -> None: ...

    async def schema_for(self, workspaces: Sequence[str]) -> dict[str, Any]: ...


class _AnswerCapabilityReader(Protocol):
    """The public immutable capability snapshot callers may read."""

    async def read(self) -> AnswerCapabilities: ...


class _AnswerCapabilityPlanner(Protocol):
    """Role profiles and image policy one acceptance pins itself against."""

    async def refresh_vlm(self) -> AnswerCapabilities: ...

    def current_profiles(self) -> dict[ChatModelSelector, ModelProfile]: ...

    def request_model_context(
        self, pinned: Mapping[ChatModelSelector, ModelProfile] | None, /
    ) -> RequestModelContext: ...

    def answer_image_policy(self, profile: ModelProfile, /) -> AnswerImagePolicy: ...

    async def confirmed_live_answer_context(
        self, models: RequestModelContext, /
    ) -> tuple[RequestModelContext, AnswerImageCapability | None]: ...


class _QueryImageRuntime(Protocol):
    """The model runtime acceptance describes current-turn images with."""

    def query_image_describer(self) -> QueryImageDescriber: ...

    def model_settings(self, role: ChatModelSelector) -> ModelSettings: ...


class _AnswerResourcePreparer(Protocol):
    """Resource materialization and resolution shared with run execution."""

    async def pin_current_image_links(
        self, request: AnswerRunRequest, attachment_bytes: Sequence[bytes], /
    ) -> tuple[AnswerRunRequest, list[bytes]]: ...

    async def resolve(
        self,
        resources: list[ResourceInput] | None,
        /,
        *,
        models: RequestModelContext,
        confirm_image_context: Callable[
            [RequestModelContext],
            Awaitable[tuple[RequestModelContext, AnswerImageCapability | None]],
        ],
        resolved_mode: ResolvedMode,
        resource_identity: str | None = None,
    ) -> ResolvedAnswerResources: ...


@dataclass(frozen=True, slots=True)
class _AcceptanceProjection:
    history: tuple[Mapping[str, Any], ...]
    episodic_summary: str
    image_descriptions: tuple[str, ...]
    pinned_models: tuple[PinnedModelProfile, ...]
    agent_run_plan: AgentRunPlan | None
    valid_modes: frozenset[ResolvedMode]


def _attachment_bytes(resources: Sequence[ResourceInput]) -> list[bytes]:
    """Return the inline bytes an accepted run must persist with its input."""
    return [resource.content for resource in resources if resource.content is not None]


def _prepared_input_payload(run_input: Any, *, requested_mode: str) -> dict[str, Any]:
    """Encode one accepted run with its canonical Session/Lane mapping."""
    payload = dict(run_input.as_request())
    payload["mode"] = requested_mode
    return payload


def _require_readable_uploads(uploads: Sequence[tuple[str | None, str | None, bytes]]) -> None:
    """One admission rule for every transport: the Engine decides what a Run reads."""
    for filename, declared_mime, content in uploads:
        require_readable_attachment(filename, declared_mime, content)


def _normalized_request(request: AnswerRequest) -> AnswerRunRequest:
    """Project one public request into durable acceptance input, without I/O."""
    if not request.workspaces:
        raise AnswerRequestError("at least one canonical workspace is required")
    try:
        workspaces = tuple(
            require_canonical_workspace_id(workspace) for workspace in request.workspaces
        )
    except ValueError:
        raise AnswerRequestError("workspaces must be canonical workspace ids") from None
    links: list[LinkReference] = []
    attachments: list[AttachmentReference] = []
    for resource in request.resources:
        if resource.url is not None:
            links.append(
                LinkReference(
                    url=resource.url,
                    filename=resource.filename,
                    ordinal=len(links),
                    mime_type=resource.declared_mime,
                )
            )
            continue
        if resource.content is None:
            raise ValueError("durable answer resources need inline bytes or an HTTPS link")
        attachments.append(
            AttachmentReference(
                digest=artifact_digest(resource.content),
                filename=safe_source_filename(resource.filename),
                mime_type=resource.declared_mime or "application/octet-stream",
                ordinal=len(attachments),
                byte_size=len(resource.content),
            )
        )
    return AnswerRunRequest(
        query=request.query,
        workspaces=workspaces,
        history=tuple(dict(message) for message in request.history),
        episodic_summary=request.episodic_summary,
        retrieval=request.retrieval,
        filters=(
            request.filters.model_dump(exclude_none=True, mode="json") if request.filters else None
        ),
        semantic_highlights=request.semantic_highlights,
        links=tuple(links),
        attachments=tuple(attachments),
        mode=request.mode or "auto",
        parent_run_id=request.parent_run_id,
        continuation_kind=request.continuation_kind,
        goal_id=request.goal_id,
        goal_revision=request.goal_revision,
        goal_round=request.goal_round,
        agent_session_id=request.agent_session_id,
        agent_lane_id=request.agent_lane_id,
        source_lane_id=request.source_lane_id,
        requested_skill=request.requested_skill,
        effort=request.effort,
        history_attachments=tuple(
            AttachmentReference(
                digest=resource.digest,
                filename=safe_source_filename(resource.filename),
                mime_type=resource.mime_type,
                ordinal=ordinal,
                byte_size=resource.byte_size,
            )
            for ordinal, resource in enumerate(request.history_resources)
        ),
    )


def _artifact_references(request: AnswerRunInput) -> list[PendingArtifactReference]:
    """Describe every accepted input upload's durable replay slot."""
    references = [
        PendingArtifactReference(
            resource_id=attachment.resource_id,
            reference_kind="current_attachment",
            ordinal=attachment.ordinal,
            digest=attachment.digest,
            filename=attachment.filename,
            mime_type=attachment.mime_type,
        )
        for attachment in request.attachments
    ]
    references.extend(
        PendingArtifactReference(
            resource_id=attachment.history_resource_id,
            reference_kind="history_attachment",
            ordinal=attachment.ordinal,
            digest=attachment.digest,
            filename=attachment.filename,
            mime_type=attachment.mime_type,
        )
        for attachment in request.history_attachments
    )
    return references


class AnswerService:
    """Accept, follow, and read back this deployment's durable answer runs."""

    def __init__(
        self,
        *,
        store: _AnswerRunRepository,
        blob_store: _RunBlobReader,
        coordinator: _RunScheduler,
        retrieval: _RetrievalPlanning,
        capabilities: _AnswerCapabilityPlanner,
        capability_view: _AnswerCapabilityReader,
        models: _QueryImageRuntime,
        resources: _AnswerResourcePreparer,
        model_invocation_fingerprint_for_role: Callable[
            [ChatModelSelector], ModelInvocationFingerprint
        ],
        child_roster_cursor_secret: bytes,
        agent_may_register: Callable[..., Awaitable[bool]],
        policies: RunPolicies,
        research_tool_declarations: ResearchToolDeclarations | None = None,
        bind_research: Callable[..., Awaitable[BoundResearchConnections]] | None = None,
        memory_capability: Callable[..., Awaitable[tuple[bool, int]]] | None = None,
        protected_skill_documents: Callable[[], frozenset[tuple[str, str]]],
        protected_text: Callable[[str], str] | None = None,
        run_retention_seconds: int = 365 * 24 * 3600,
    ) -> None:
        self._store = store
        self._blob_store = blob_store
        self._coordinator = coordinator
        self._retrieval = retrieval
        self._capabilities = capabilities
        self._capability_view = capability_view
        self._models = models
        self._resources = resources
        self._model_invocation_fingerprint_for_role = model_invocation_fingerprint_for_role
        self._research_tool_declarations = research_tool_declarations
        self._bind_research = bind_research
        self._memory_capability = memory_capability
        self._protected_skill_documents = protected_skill_documents
        # The content gate behind every reader-facing transcript surface. None in a
        # process composed without one (tests, a tool-only slice): the header fast path
        # still masks a load_skill result, and nothing else is replaced.
        self._protected_text = protected_text
        self._agent_may_register = agent_may_register
        self._policies = policies
        self._admission = RunAdmission(
            run_kind="answer",
            lane="query",
            runtime="Answer",
            retention_seconds=int(run_retention_seconds),
            input_error=AnswerRequestError,
            scheduler=coordinator,
        )
        self._child_roster_codec = ChildRosterCursorCodec(child_roster_cursor_secret)
        # The Run's derived index is built once per Run version here and shared by the overview
        # and the step search; the cache is bounded and disposable, and a page pays for its own
        # content only.
        self._timeline_indexes = TimelineIndexCache()

    @staticmethod
    def _continuation_fingerprint(request: AnswerRequest) -> str:
        """Hash one continuation as the caller submitted it.

        The admission fingerprint decides whether a keyed submission is a replay or a
        conflict, so it has to describe what the caller asked for. A continuation's
        Session is its parent's and a Fork mints its own Lane per attempt; neither is
        the caller's input, and leaving the Lane in makes every retry of an identical
        submission look like changed input.
        """
        fields = dict(_normalized_request(request).as_request())
        fields.pop("agent_session_id", None)
        fields.pop("agent_lane_id", None)
        return run_request_fingerprint(fields)

    async def create(
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        caller: Caller | None,
        idempotency_key: str | None = None,
        idempotency_fingerprint: str | None = None,
    ) -> RunCreation:
        """Accept one durable run and return its descriptor without waiting.

        A keyed replay is resolved before any link is materialized or any input
        is prepared, so a retried submission never repeats acceptance work. The
        accepted run outlives this call and is read back through :meth:`get`,
        :meth:`subscribe`, and :meth:`cancel`.
        """
        creation = await self.accept(
            request=request,
            owner_id=owner_id,
            caller=caller,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
            acceptor=self._store,
        )
        if creation is None:
            raise RuntimeError("Answer run acceptance returned no descriptor")
        return RunCreation.from_runtime(creation)

    async def accept[T](
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        caller: Caller | None,
        acceptor: AnswerRunAcceptor[T],
        idempotency_key: str | None = None,
        idempotency_fingerprint: str | None = None,
    ) -> T | None:
        """Accept one durable run through ``acceptor``, which may link it atomically.

        The Run store accepts the run alone; a Web conversation's acceptor also
        writes its turn in the same transaction. Both go through one pipeline.

        ``caller`` is who submits, which the Runtime Policies are read for. They are read once,
        here, after a replay has been answered: the Run is accepted under what they said and
        carries it, so a change to a policy never reaches a Run already accepted.
        """
        run_request = _normalized_request(request)
        fingerprint = idempotency_fingerprint or run_request_fingerprint(run_request.as_request())
        # Canonical mode syntax is capability-independent and may be checked
        # before replay. Live capability validation must wait until after the
        # answer/VLM refreshes below because an unknown probe can recover.
        requested_mode = canonical_answer_mode(run_request.mode)
        replay = await self._admission.replay(
            acceptor,
            submitted_by=owner_id,
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
        )
        if replay is not None:
            return replay
        uploads = tuple(
            (resource.filename, resource.declared_mime, resource.content)
            for resource in request.resources
            if resource.content is not None
        )
        if uploads:
            # Deciding an upload by its bytes decodes it, so it runs off the event loop.
            await asyncio.to_thread(_require_readable_uploads, uploads)
        self._admission.require_runtime()
        run_request, attachment_bytes = await self._resources.pin_current_image_links(
            run_request,
            _attachment_bytes(request.resources),
        )
        if run_request.links or run_request.attachments or run_request.history_attachments:
            await self._capabilities.refresh_vlm()
        # Pinning may narrow the live query profile, and the VLM refresh may
        # remove resource viewing. This is the authoritative Valid Mode Set persisted
        # with the run; the earlier check only avoids needless acceptance I/O.
        requested_mode, allowed_modes = self._reject_unsupported_mode(run_request)
        acceptance_resources = await build_current_answer_resources(
            links=run_request.links,
            attachments=run_request.attachments,
            attachment_loaders=[
                in_memory_attachment_loader(content) for content in attachment_bytes
            ],
        )
        acceptance_resources.extend(
            self._history_resource_input(owner_id, resource)
            for resource in request.history_resources
        )
        memory_enabled, memory_epoch = True, 0
        if self._memory_capability is not None:
            memory_enabled, memory_epoch = await self._memory_capability(owner_id=owner_id)
        # Read once, like the memory capability: the Run's tools are what it was accepted with.
        agent_may_register = await self._agent_may_register(owner_id=owner_id)
        # The Runtime Policies are read once too, and the Run is pinned to what they said. A
        # document that cannot be read refuses the submission: a default would admit it
        # without the limit or the ceiling the document sets for this caller.
        effort_ceiling = await self._policies.effort_ceiling(caller)
        active_run_limit = await self._policies.active_run_limit(caller)
        async with self._prepare_input(
            run_request,
            resources=acceptance_resources or None,
            idempotency_fingerprint=fingerprint,
            requested_mode=requested_mode,
            allowed_modes=allowed_modes,
            memory_enabled=memory_enabled,
            agent_may_register=agent_may_register,
            effort_ceiling=effort_ceiling,
        ) as prepare:
            for attempt in range(2):
                bound = (
                    await self._bind_research(owner_id=owner_id)
                    if self._bind_research is not None
                    and requested_mode != "fast"
                    and "research" in allowed_modes
                    else BoundResearchConnections()
                )
                run_input, effective_modes = await prepare(
                    bound.tools, bound.surface, bound.resource_tools_available
                )
                run_input = replace(run_input, run_connection_bindings=bound.bindings)
                prepared_input = _prepared_input_payload(run_input, requested_mode=requested_mode)
                prepared_input["profile_memory_enabled"] = memory_enabled
                prepared_input["profile_memory_epoch"] = memory_epoch
                prepared_input[MAY_REGISTER_PIN] = agent_may_register
                try:
                    return await self._admission.admit(
                        partial(
                            acceptor.accept_run,
                            artifacts=[
                                PendingArtifact(content=content) for content in attachment_bytes
                            ],
                            references=_artifact_references(run_input),
                            connection_bindings=bound.bindings,
                            routing=RoutingAcceptance(
                                requested_mode=requested_mode,
                                valid_modes=tuple(sorted(effective_modes)),
                                context_policy_revision=CONTEXT_POLICY_REVISION,
                                model_fingerprints={
                                    item.role: item.fingerprint.as_json()
                                    for item in run_input.pinned_models
                                },
                                agent_session_id=run_input.agent_session_id,
                                agent_lane_id=run_input.agent_lane_id,
                                source_lane_id=run_input.source_lane_id,
                            ),
                        ),
                        submitted_by=owner_id,
                        access_scope=RunAccessScope(kind="owner", scope_id=owner_id),
                        idempotency_key=idempotency_key,
                        fingerprint=fingerprint,
                        payload=prepared_input,
                        accepted_input=accepted_input_envelope(prepared_input),
                        active_run_limit=active_run_limit,
                    )
                except ActiveRunLimitExceededError as exc:
                    raise ActiveRunLimitError(exc.limit) from exc
                except StaleConnectionBindingError as exc:
                    replay = await self._admission.replay(
                        acceptor,
                        submitted_by=owner_id,
                        idempotency_key=idempotency_key,
                        fingerprint=fingerprint,
                    )
                    if replay is not None:
                        return replay
                    if attempt == 1:
                        raise AnswerConnectionsChangedError() from exc
        raise RuntimeError("Answer acceptance exhausted its bounded attempts")

    def _reject_unsupported_mode(
        self, request: AnswerRunRequest
    ) -> tuple[AnswerMode, frozenset[ResolvedMode]]:
        """Fail closed before a run row exists when the requested mode cannot resolve."""
        profiles = self._capabilities.current_profiles()
        query = profiles["query"]
        resources: list[ModeResource] = []
        for attachment in (*request.attachments, *request.history_attachments):
            resources.append(
                ModeResource(
                    role=resource_role(filename=attachment.filename, mime_type=attachment.mime_type)
                )
            )
        for link in request.links:
            role = resource_role(filename=link.filename or link.url, mime_type=link.mime_type)
            if role == "other":
                role = "document"
            resources.append(ModeResource(role=role))
        allowed = valid_modes(
            resources=tuple(resources),
            capability=ModeCapability(
                query_supports_images=query.supports_images,
            ),
        )
        requested = require_supported_mode(requested=request.mode, valid=allowed)
        return requested, allowed

    def _history_resource_input(
        self,
        owner_id: str,
        resource: AnswerHistoryResource,
    ) -> ResourceInput:
        async def load() -> bytes:
            artifact = await self.read_input_artifact(
                owner_id=owner_id,
                run_id=resource.run_id,
                ordinal=resource.source_ordinal,
                reference_kind=resource.reference_kind,
            )
            if artifact is None:
                raise RunRuntimeUnavailableError("Accepted answer input artifact is unavailable")
            return artifact.content

        return ResourceInput(
            filename=resource.filename,
            declared_mime=resource.mime_type,
            loader=load,
        )

    async def list_artifacts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunArtifactReference, ...] | None:
        """List artifacts for one owned Answer; every other id is unknown."""
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return None
        return await self._store.list_run_artifacts(owner_id=owner_id, run_id=run_id)

    async def published_artifact(
        self, *, owner_id: str, run_id: str, resource_id: str
    ) -> Mapping[str, Any] | None:
        """Return one owned Answer's available Published Artifact descriptor.

        Input uploads and fetched resources share the resource id space, so the
        artifact readers below serve an id only while this names it.
        """
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None:
            return None
        return published_artifact_descriptor(record.result, resource_id)

    async def read_artifact(
        self,
        *,
        owner_id: str,
        run_id: str,
        resource_id: str,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes | None:
        """Read a bounded Published Artifact; every other id returns ``None``."""
        stream = await self.open_artifact(
            owner_id=owner_id,
            run_id=run_id,
            resource_id=resource_id,
            offset=offset,
            length=length,
        )
        if stream is None:
            return None
        pieces = [piece async for piece in stream]
        return b"".join(pieces)

    async def run_resource(
        self, *, owner_id: str, run_id: str, resource_id: str
    ) -> RunResourceDescriptor | None:
        """Resolve one run-scoped resource id for this owner.

        Ids are unique per owner and run, so the same lookup serves an accepted
        input upload, a publication, and a resource a worker fetched, rendered, or
        adopted. An id another owner or another run registered resolves to ``None``
        rather than leaking its existence.
        """
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return None
        return await self._resolve_run_resource(
            owner_id=owner_id, run_id=run_id, resource_id=resource_id
        )

    async def open_run_resource(
        self,
        *,
        owner_id: str,
        run_id: str,
        resource_id: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes] | None:
        """Open one resolved run resource; an unknown id returns ``None``.

        Callers that serve large bytes stream these chunks; no complete-blob
        materialization happens on this path.
        """
        descriptor = await self.run_resource(
            owner_id=owner_id, run_id=run_id, resource_id=resource_id
        )
        if descriptor is None:
            return None
        return self._blob_store.stream(
            owner_id=owner_id,
            digest=descriptor.digest,
            offset=max(0, offset),
            length=length,
        )

    async def run_resource_size(
        self, *, owner_id: str, run_id: str, resource_id: str
    ) -> int | None:
        """Return one resolved run resource's size; an unknown id returns ``None``."""
        descriptor = await self.run_resource(
            owner_id=owner_id, run_id=run_id, resource_id=resource_id
        )
        if descriptor is None:
            return None
        return await self._blob_store.size(owner_id=owner_id, digest=descriptor.digest)

    async def latest_run_attachment(
        self, *, owner_id: str, run_id: str, safe_name: str
    ) -> tuple[RunResourceDescriptor, bytes] | None:
        """Read the newest attachment a Run stored under ``safe_name``, whole.

        A tool attachment is durable Run state like any other resource: this names the most
        recent one by the name the tool gave it, which is what a reader shows when there is
        nothing newer left to show. An unknown Run, or one that stored none, answers None.
        """
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return None
        row = await self._store.latest_run_attachment(
            owner_id=owner_id, run_id=run_id, safe_name=safe_name
        )
        if row is None:
            return None
        return (
            RunResourceDescriptor(
                resource_id=row.resource_id,
                registry="resource",
                reference_kind=None,
                ordinal=row.ordinal,
                filename=row.filename,
                mime_type=row.mime_type,
                digest=row.digest,
            ),
            await self._read_digest(owner_id=owner_id, digest=row.digest),
        )

    async def run_external_source_map(self, *, owner_id: str, run_id: str) -> dict[str, str]:
        """Map each external URL this run holds bytes for to its resource id.

        One surface decides what a same-origin reader may offer: the row that
        recorded the bytes. Matching normalizes both sides the way admission did,
        so a URL the answer text spells slightly differently still resolves to the
        copy this run fetched; a URL with no stored bytes is simply absent.
        """
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return {}
        sources: dict[str, str] = {}
        for resource in await self._store.list_fetched_resources(owner_id=owner_id, run_id=run_id):
            for candidate in _resource_source_urls(resource):
                identity = public_http_url_identity(candidate)
                if identity is not None:
                    sources.setdefault(identity, resource.resource_id)
        return sources

    async def read_run_resource(
        self, *, owner_id: str, run_id: str, resource_id: str
    ) -> tuple[RunResourceDescriptor, bytes] | None:
        """Read one bounded run resource whole; an unknown id returns ``None``."""
        descriptor = await self.run_resource(
            owner_id=owner_id, run_id=run_id, resource_id=resource_id
        )
        if descriptor is None:
            return None
        return descriptor, await self._read_digest(owner_id=owner_id, digest=descriptor.digest)

    async def _read_digest(self, *, owner_id: str, digest: str) -> bytes:
        return b"".join(
            [piece async for piece in self._blob_store.stream(owner_id=owner_id, digest=digest)]
        )

    async def _resolve_run_resource(
        self, *, owner_id: str, run_id: str, resource_id: str
    ) -> RunResourceDescriptor | None:
        """Resolve an id across both registries, input before run state.

        A current-turn upload wins over a carried-forward one sharing its ordinal,
        which is the precedence the acceptance path already applies; a registered
        row is reached only when no accepted artifact claims the id, and every
        kind of row counts including an adopted entry attachment.
        """
        if not resource_id:
            return None
        references = await self.list_artifacts(owner_id=owner_id, run_id=run_id)
        if references:
            ordered = sorted(
                (item for item in references if item.resource_id == resource_id),
                key=lambda item: (
                    _INPUT_REFERENCE_KINDS.index(item.reference_kind)
                    if item.reference_kind in _INPUT_REFERENCE_KINDS
                    else len(_INPUT_REFERENCE_KINDS)
                ),
            )
            if ordered:
                match = ordered[0]
                return RunResourceDescriptor(
                    resource_id=match.resource_id,
                    registry="artifact",
                    reference_kind=match.reference_kind,
                    ordinal=match.ordinal,
                    filename=match.filename,
                    mime_type=match.mime_type,
                    digest=match.digest,
                )
        resource = await self._store.read_run_resource_row(
            owner_id=owner_id, run_id=run_id, resource_id=resource_id
        )
        if resource is None:
            return None
        return RunResourceDescriptor(
            resource_id=resource.resource_id,
            registry="resource",
            reference_kind=None,
            ordinal=resource.ordinal,
            filename=resource.filename,
            mime_type=resource.mime_type,
            digest=resource.digest,
        )

    async def open_artifact(
        self,
        *,
        owner_id: str,
        run_id: str,
        resource_id: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes] | None:
        """Open one Published Artifact through the shared run-resource reader.

        An input upload or fetched resource id is unknown here, as is an artifact
        not (or no longer) available, whichever transport asks.
        """
        if (
            await self.published_artifact(owner_id=owner_id, run_id=run_id, resource_id=resource_id)
            is None
        ):
            return None
        return await self.open_run_resource(
            owner_id=owner_id,
            run_id=run_id,
            resource_id=resource_id,
            offset=offset,
            length=length,
        )

    async def artifact_size(self, *, owner_id: str, run_id: str, resource_id: str) -> int | None:
        """Return one Published Artifact's size; every other id returns ``None``."""
        if (
            await self.published_artifact(owner_id=owner_id, run_id=run_id, resource_id=resource_id)
            is None
        ):
            return None
        return await self.run_resource_size(
            owner_id=owner_id, run_id=run_id, resource_id=resource_id
        )

    async def _get_answer_run(self, *, owner_id: str, run_id: str) -> RunRecord | None:
        """Return one owned Answer; unknown, foreign, and wrong-kind ids are identical."""
        record = await self._store.get_run(owner_id=owner_id, run_id=run_id)
        return record if record is not None and record.run_kind == "answer" else None

    async def steer(
        self, *, owner_id: str, run_id: str, instruction: str
    ) -> AgentControlReceipt | None:
        """Queue one ordered steering instruction for a live Research session."""
        text = instruction.strip()
        if not text:
            raise AnswerRequestError("steer instruction cannot be empty")
        if len(text) > _AGENT_CONTROL_CONTENT_LIMIT:
            raise AnswerRequestError("steer instruction exceeds 20000 characters")
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return None
        row = await self._store.enqueue_agent_control(
            owner_id=owner_id,
            run_id=run_id,
            kind="steer",
            content=text,
        )
        if row is None:
            return None
        self._coordinator.wake()
        return AgentControlReceipt(
            run_id=run_id,
            control_sequence=int(row["control_sequence"]),
            kind=str(row["kind"]),
        )

    async def control_child(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        action: str,
        content: str = "",
        idempotency_key: str = "",
        reauthorize_user_cancelled: bool = False,
    ) -> ChildControlReceipt | None:
        """Apply one owner-scoped durable steer, continuation, or cancellation."""
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None:
            return None
        parent_session_id = str(record.request_input().get("agent_session_id") or "") or None
        key = idempotency_key.strip()
        if not key or len(key) > 200:
            raise AnswerRequestError(
                "Child control idempotency key must be between 1 and 200 characters"
            )
        if action == "cancel":
            row = await self._store.cancel_child_session_by_owner(
                owner_id=owner_id,
                run_id=run_id,
                child_session_id=child_session_id,
                submission_key=key,
                parent_session_id=parent_session_id,
            )
        else:
            text = content.strip()
            if not text or len(text) > _AGENT_CONTROL_CONTENT_LIMIT:
                raise AnswerRequestError(
                    "Child control content must be non-empty and at most 20000 characters"
                )
            if action == "steer":
                row = await self._store.enqueue_child_control(
                    owner_id=owner_id,
                    run_id=run_id,
                    child_session_id=child_session_id,
                    parent_session_id=parent_session_id,
                    content=text,
                    submission_key=key,
                    origin="user",
                )
            elif action == "continue":
                row = await self._store.continue_child_session(
                    owner_id=owner_id,
                    run_id=run_id,
                    child_session_id=child_session_id,
                    parent_session_id=parent_session_id,
                    content=text,
                    submission_key=key,
                    origin="user",
                    reauthorize_user_cancelled=reauthorize_user_cancelled,
                )
            else:
                raise AnswerRequestError("unknown Child control action")
        if not isinstance(row, Mapping):
            return None
        outcome = str(row.get("outcome") or "unknown_child")
        if outcome == "unknown_child":
            return None
        if outcome in {"queued", "accepted", "cancellation_requested"}:
            self._coordinator.wake()
        return ChildControlReceipt(
            run_id=run_id,
            action=action,
            outcome=outcome,
            status=(str(row["status"]) if action == "cancel" and row.get("status") else None),
            child_session_id=child_session_id or None,
            operation_id=(str(row["operation_id"]) if row.get("operation_id") else None),
            operation_sequence=(
                int(row["operation_sequence"])
                if row.get("operation_sequence") is not None
                else None
            ),
            control_sequence=(
                int(row["control_sequence"]) if row.get("control_sequence") is not None else None
            ),
            consumed_at=row.get("consumed_at"),
        )

    async def reply_to_child(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        content: str,
        idempotency_key: str,
    ) -> ChildControlReceipt | None:
        """Reply to one owned correlated Child guidance request."""
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None:
            return None
        parent_session_id = str(record.request_input().get("agent_session_id") or "") or None
        row = await self._store.reply_child_guidance(
            owner_id=owner_id,
            run_id=run_id,
            request_id=request_id,
            parent_session_id=parent_session_id,
            content=content,
            submission_key=idempotency_key,
            origin="user",
        )
        if not isinstance(row, Mapping):
            return None
        outcome = str(row.get("outcome") or "unknown_request")
        if outcome == "unknown_request":
            return None
        if outcome == "replied":
            self._coordinator.wake()
        child_id = str(row.get("child_session_id") or "") or None
        return ChildControlReceipt(
            run_id=run_id,
            action="reply",
            outcome=outcome,
            child_session_id=child_id,
            request_id=request_id,
        )

    async def children(
        self,
        *,
        owner_id: str,
        run_id: str,
        page: ChildRosterPageRequest | None = None,
    ) -> ChildRosterPage | None:
        """Return one bounded newest-first child-roster page, or None if unknown."""
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return None
        requested = page or ChildRosterPageRequest()
        if requested.cursor is not None and str(requested.cursor.run_id) != run_id:
            raise ChildRosterCursorError("child-roster cursor belongs to another run")
        result = await self._store.list_child_sessions_page(
            owner_id=owner_id,
            run_id=run_id,
            page=requested,
        )
        next_cursor = None
        if result.has_more:
            if not result.children:
                raise RuntimeError("child-roster store reported more rows after an empty page")
            last = result.children[-1]
            next_cursor = ChildRosterCursor(
                run_id=UUID(run_id),
                created_at=last["created_at"],
                child_session_id=UUID(str(last["child_session_id"])),
            )
        return ChildRosterPage(
            children=tuple(
                public_child_status(row, mask_reader_text=self._protected_text)
                for row in result.children
            ),
            next_cursor=next_cursor,
            fetched_rows=result.fetched_rows,
        )

    async def observe_child(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        limit: int = 20,
        with_transcript: bool = True,
    ) -> ChildObservation | None:
        """Return one bounded child observation, or None if unknown.

        A caller that reads the transcript by page asks for none here.
        """
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return None
        row = await self._store.load_child_session(
            owner_id=owner_id, run_id=run_id, child_session_id=child_session_id
        )
        if row is None:
            return None
        cap = max(1, min(int(limit), 100))
        transcript = (
            await self._store.load_agent_transcript(
                owner_id=owner_id,
                run_id=run_id,
                session_id=child_session_id,
                limit=cap,
            )
            if with_transcript
            else ()
        )
        controls = await self._store.list_child_controls(
            owner_id=owner_id,
            run_id=run_id,
            child_session_id=child_session_id,
            limit=cap,
        )
        questions = await self._store.list_child_guidance(
            owner_id=owner_id,
            run_id=run_id,
            child_session_id=child_session_id,
            limit=cap,
        )
        child_subjects = await self._tool_subjects(
            owner_id=owner_id, run_id=run_id, messages=transcript
        )
        # Once per page, not once per message: discovery is a directory read, and a page of
        # thirty messages would otherwise pay for thirty of them on every follow poll.
        protected = self._protected_skill_documents()
        return ChildObservation(
            run_id=run_id,
            child=public_child_status(row, mask_reader_text=self._protected_text),
            transcript=tuple(
                _public_transcript_message(
                    message,
                    protected_skill_documents=protected,
                    tool_subjects=child_subjects,
                    mask_reader_text=self._protected_text,
                )
                for message in transcript
            ),
            controls=tuple(
                _public_control_record(item, mask_reader_text=self._protected_text)
                for item in controls
            ),
            questions=tuple(
                _public_question_record(item, mask_reader_text=self._protected_text)
                for item in questions
            ),
            result=child_result_lineage(row, mask_reader_text=self._protected_text),
        )

    async def _tool_subjects(
        self, *, owner_id: str, run_id: str, messages: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, str]:
        """The bounded Tool subjects for one transcript page, joined from the Run's durable events."""
        call_ids = _tool_call_ids(messages)
        if not call_ids:
            return {}
        return await self._store.load_tool_subjects(
            owner_id=owner_id, run_id=run_id, call_ids=call_ids
        )

    async def _session_of(
        self,
        *,
        owner_id: str,
        run_id: str,
        record: RunRecord,
        child_session_id: str | None,
    ) -> tuple[str, bool] | None:
        """The Session a transcript request is about, and whether its agent is still working."""
        if child_session_id is None:
            session_id = str(record.request_input().get("agent_session_id") or "")
            return session_id, not record.terminal
        row = await self._store.load_child_session(
            owner_id=owner_id, run_id=run_id, child_session_id=child_session_id
        )
        if row is None:
            return None
        return child_session_id, public_child_status(row)["status"] == "running"

    async def activity_page(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str | None = None,
        before: int | None = None,
        limit: int = ACTIVITY_PAGE_DEFAULT_LIMIT,
        since: int | None = None,
    ) -> ActivityPage | None:
        """One page of a Run's main Session or of a Child Session, or None if unknown.

        ``before`` reads the page older than a sequence; ``since`` reads the messages written after
        one, oldest first, which is how a reader that already holds a transcript appends to it.
        """
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None:
            return None
        bound = await self._session_of(
            owner_id=owner_id, run_id=run_id, record=record, child_session_id=child_session_id
        )
        if bound is None:
            return None
        session_id, running = bound
        cap = max(1, min(int(limit), ACTIVITY_PAGE_MAX_LIMIT))
        # One message beyond the page says whether an older page exists.
        fetched = (
            list(
                await self._store.load_agent_transcript(
                    owner_id=owner_id,
                    run_id=run_id,
                    session_id=session_id,
                    limit=cap + 1,
                    before=before,
                    since=since,
                )
            )
            if session_id
            else []
        )
        # A page walks back from the newest and states the cursor that reaches the one before it. A
        # delta keeps the oldest of what it found, so a reader asking again from the last message it
        # holds never skips one, and it states no cursor: the older pages are not its business.
        delta = since is not None
        shortened = len(fetched) > cap
        messages = fetched[:cap] if delta else (fetched[1:] if shortened else fetched)
        tool_subjects = await self._tool_subjects(
            owner_id=owner_id, run_id=run_id, messages=messages
        )
        # One discovery per page; see observe_child.
        protected = self._protected_skill_documents()
        # The Inspector draws a step's thinking in its activity pane; this is the one reader
        # surface that states it, masked and bounded at the projection. The gate walks the page's
        # whole text, so the projection runs off the event loop.
        projected = await asyncio.to_thread(
            _project_activity_messages,
            messages,
            protected=protected,
            tool_subjects=tool_subjects,
            mask_reader_text=self._protected_text,
        )
        return ActivityPage(
            messages=projected,
            next_before=None if delta or not shortened else int(messages[0]["sequence"]),
            running=running,
        )

    async def _timeline_index(self, *, owner_id: str, run_id: str) -> TimelineIndex | None:
        """The Run's whole timeline index, rebuilt only when the Run moved since it was derived.

        None when the Run is unknown: the version row is the owner-scoped existence check. Every
        page and the overview read from this one derivation, so they cannot disagree about which
        steps the Run has, and a page costs its own entries rather than the whole Run.
        """
        token_row = await self._store.load_run_timeline_token(owner_id=owner_id, run_id=run_id)
        if token_row is None:
            return None
        token = RunTimelineToken.from_store(token_row)
        held = self._timeline_indexes.get(owner_id, run_id, token)
        if held is not None:
            return held
        agents = agent_identities(
            tuple(
                {"child_session_id": child_session_id, "objective": objective}
                for child_session_id, objective in token.roster
            )
        )
        rows = await self._store.load_run_timeline_steps(owner_id=owner_id, run_id=run_id)
        call_ids = tuple(
            dict.fromkeys(str(row.get("call_id") or "") for row in rows if row.get("call_id"))
        )
        subjects = (
            await self._store.load_tool_subjects(
                owner_id=owner_id,
                run_id=run_id,
                call_ids=call_ids,
            )
            if call_ids
            else {}
        )
        index = TimelineIndex.build(
            token=token,
            agents=agents,
            steps=merge_steps(agents=agents, rows=rows, subjects=subjects),
        )
        self._timeline_indexes.put(owner_id, run_id, index)
        return index

    def _labelled(self, page: Sequence[RunTimelineStep]) -> tuple[RunTimelineStep, ...]:
        """The page's steps with each agent's roster label gated for the reader."""
        if self._protected_text is None:
            return tuple(page)
        return tuple(
            replace(step, agent_label=self._protected_text(step.agent_label)) for step in page
        )

    async def run_overview(self, *, owner_id: str, run_id: str) -> RunOverview | None:
        """One Run's whole-Run overview: its totals, its agents, and its capped findings.

        The read reuses the derived index the step list reads, so the panel's numbers are the same
        facts the steps state, and every number is the whole Run's rather than a page's. Only the
        capped findings' entries are loaded — the excerpt is the one text a finding carries — and
        they are projected through the same reader content gate every transcript surface applies;
        the roster's labels are gated at emission too, so a Skill root that becomes protected
        reaches a cached label.

        ``usage`` is the Run result's own record and stays None when the record states none; no
        number here is recomputed from the steps. ``version`` is the derived index's, which the
        route turns into the panel's cache validator.

        None when the Run is unknown, so a caller can answer 404.
        """
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None:
            return None
        index = await self._timeline_index(owner_id=owner_id, run_id=run_id)
        if index is None:
            return None
        agents = (
            tuple(index.agents)
            if self._protected_text is None
            else tuple(
                replace(agent, label=self._protected_text(agent.label)) for agent in index.agents
            )
        )
        findings, findings_total = findings_of(index.steps, index.marks, FINDINGS_MAX)
        if findings:
            messages = await self._store.load_timeline_entries(
                owner_id=owner_id,
                run_id=run_id,
                keys=tuple((finding.session_id, finding.sequence) for finding in findings),
            )
            # One discovery per read; the same gate the transcript surfaces apply, so a dossier
            # states no more of a protected document than an agent's own page does. A cold read's
            # gate walks the loaded text, so the projection runs off the event loop.
            protected = self._protected_skill_documents()
            projected = await asyncio.to_thread(
                _project_timeline_messages,
                messages,
                protected=protected,
                mask_reader_text=self._protected_text,
            )
            # The projection keys a message by its lower-cased Session; a step carries the
            # Session as its row stated it.
            excerpts = {
                key: excerpt_of(message.get("content")) for key, message in projected.items()
            }
            findings = tuple(
                replace(
                    finding,
                    excerpt=excerpts.get((finding.session_id.lower(), finding.sequence), ""),
                )
                for finding in findings
            )
        return RunOverview(
            run_id=run_id,
            totals=overview_totals(index.steps, index.marks),
            agents=agent_rollups(index.steps, index.marks, agents),
            findings=findings,
            findings_total=findings_total,
            usage=run_usage(record.result),
            version=index.version,
        )

    async def search_run_steps(
        self,
        *,
        owner_id: str,
        run_id: str,
        query: str,
        limit: int,
        before: RunTimelinePosition | None = None,
    ) -> RunStepSearch | None:
        """The Run's steps whose messages hold a search's words, as a paged list read.

        The successor of the timeline read's search mode: the same whole-Run transcript search and
        the same merged index, narrowed to the steps the words found. Thinking steps are excluded —
        a search result is the message's actionable step, and a list states no reasoning — so a
        thought never arrives without the turn it produced. ``before`` is the step the previous
        page stopped in front of, so a page is strictly older than it and walks matches newest page
        first, in the merge's order within the page.

        ``matched`` counts the Run's searchable matches, so a page never reads as the whole result;
        the marks are the whole Run's, so a repeated call's other half does not have to be on this
        page for the row to state it. A query with no words in it matches nothing: it is a search
        that read no words, not a request for the Run.

        None when the Run is unknown, so a caller can answer 404.
        """
        index = await self._timeline_index(owner_id=owner_id, run_id=run_id)
        if index is None:
            return None
        words = query.strip()
        terms: tuple[str, ...] = ()
        matched = 0
        match = "all"
        shown: tuple[RunTimelineStep, ...] = ()
        if words:
            searched = await self.search_run_traces(owner_id=owner_id, run_id=run_id, query=words)
            if searched is None:
                return None
            terms, hits, _total, match = searched
            keys = {(str(hit.get("agent") or MAIN_AGENT), int(hit["sequence"])) for hit in hits}
            shown = tuple(
                step
                for step in index.steps
                if step.kind != "thinking" and (step.agent, step.sequence) in keys
            )
            matched = len(shown)
        page, has_older = page_of(shown, limit=limit, before=before, agents=index.agents)
        # The roster's objective is reader text like any other: it is gated at emission, not when
        # the index was built, so a Skill root that becomes protected reaches a cached label too.
        page = self._labelled(page)
        if page:
            messages = await self._store.load_timeline_entries(
                owner_id=owner_id,
                run_id=run_id,
                keys=tuple((step.session_id, step.sequence) for step in page),
            )
            # One discovery per read; the same gate the transcript surfaces apply, so a search
            # row states no more than an agent's own page does. A cold page's gate walks the
            # loaded text, so the projection runs off the event loop.
            protected = self._protected_skill_documents()
            projected = await asyncio.to_thread(
                _project_timeline_messages,
                messages,
                protected=protected,
                mask_reader_text=self._protected_text,
            )
            page = with_page_content(page, projected)
        return RunStepSearch(
            page=page,
            has_older=has_older,
            matched=matched,
            marks=index.marks,
            terms=terms,
            match=match,
        )

    async def search_run_traces(
        self,
        *,
        owner_id: str,
        run_id: str,
        query: str,
    ) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...], int, str] | None:
        """Every message of every agent of one Run that holds the words of ``query``.

        Oldest first, each carrying the words in their neighbourhood, naming its agent. Answers the
        words the search read (which is what a client marks), the matches, how many the Run holds
        before the per-agent bound shortened the list, and whether the words as written answered or
        the looser pass did. None when the Run is unknown, so a caller can answer 404.
        """
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None:
            return None
        # The words as written first, then the same characters loose: a reader who typed two
        # Chinese words gets those words, and only if they match nothing does the looser pass run.
        for loose in (False, True):
            terms = split_terms(query, loose=loose)
            if not terms:
                return ((), (), 0, "all")
            found = await self._store.search_run_transcript(
                owner_id=owner_id,
                run_id=run_id,
                terms=terms,
                require_all=not loose,
            )
            if found:
                break
        total = int(found[0].get("total") or len(found)) if found else 0
        protected = self._protected_skill_documents()
        return (
            terms,
            tuple(
                _public_search_hit(
                    hit,
                    protected=protected,
                    mask_reader_text=self._protected_text,
                )
                for hit in found
            ),
            total,
            "any" if loose else "all",
        )

    async def record_feedback(
        self,
        *,
        owner_id: str,
        run_id: str,
        agent: str,
        sequence: int,
        value: str,
        reasons: Sequence[str],
        note: str | None,
    ) -> AnswerFeedback | None:
        """Keep the reader's judgement of one message of one agent of a Run, replacing an earlier.

        ``agent`` is ``main`` or a Child Session's id, and ``sequence`` the message's own place in
        that agent's Session, which a transcript states on every message. The judgement is shaped
        before anything is stored, so one that cannot be kept is refused as the request it is.
        None says the Run, the agent, or the message is not there to judge, so a caller can
        answer 404.
        """
        try:
            kept_value, kept_reasons, kept_note = normalize_feedback(value, reasons, note)
        except ValueError as exc:
            raise AnswerRequestError(str(exc)) from None
        return await self._store.put_answer_feedback(
            owner_id=owner_id,
            run_id=run_id,
            agent=agent,
            sequence=sequence,
            value=kept_value,
            reasons=kept_reasons,
            note=kept_note,
        )

    async def retract_feedback(
        self, *, owner_id: str, run_id: str, agent: str, sequence: int
    ) -> None:
        """Take the reader's judgement of one message back. One never given is no error."""
        await self._store.delete_answer_feedback(
            owner_id=owner_id, run_id=run_id, agent=agent, sequence=sequence
        )

    async def answer_feedback(
        self, *, owner_id: str, run_id: str
    ) -> tuple[AnswerFeedback, ...] | None:
        """The judgements the reader gave the messages of one Run, or None if the Run is unknown.

        The Run's own agent comes first, then each Child, and within an agent the oldest message
        first: the order the transcript search answers in.
        """
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return None
        return await self._store.list_answer_feedback(owner_id=owner_id, run_id=run_id)

    @property
    def child_roster_cursor_codec(self) -> ChildRosterCursorCodec:
        """Return the codec shared with the answer-runs HTTP adapters."""
        return self._child_roster_codec

    async def transcript_tail(
        self, *, owner_id: str, run_id: str, limit: int = 20
    ) -> AgentTranscriptTail | None:
        """Return a bounded transport-neutral transcript projection."""
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None:
            return None
        request = record.request_input()
        cap = max(1, min(int(limit), 100))
        session_id = str(request.get("agent_session_id") or "")
        if session_id:
            canonical = await self._store.load_agent_transcript(
                owner_id=owner_id,
                run_id=run_id,
                session_id=session_id,
                limit=cap,
                # A conversation's projection is its lane, which spans the Runs that continued it.
                scope="lane",
            )
            if canonical:
                protected = self._protected_skill_documents()
                return AgentTranscriptTail(
                    run_id=run_id,
                    status=record.status,
                    messages=tuple(
                        _masked_store_message(
                            # A tail states no reasoning: the field is the activity page's alone.
                            _without_reasoning(message),
                            protected=protected,
                            mask_reader_text=self._protected_text,
                        )
                        for message in canonical
                    ),
                )
        # A Run whose Session holds none of its turns yet (still queued, or failed
        # before its question was accepted) projects its accepted invocation and
        # final result through the same transport-neutral message shape.
        messages: list[Mapping[str, Any]] = [
            message for message in request.get("history") or () if isinstance(message, Mapping)
        ]
        query = str(request.get("query") or "")
        if query:
            messages.append({"role": "user", "content": query})
        result = record.result or {}
        answer = str(result.get("answer") or "")
        if answer:
            messages.append({"role": "assistant", "content": answer})
        protected = self._protected_skill_documents()
        return AgentTranscriptTail(
            run_id=run_id,
            status=record.status,
            messages=tuple(
                _masked_store_message(
                    # An accepted invocation's history is a stored message too: no reasoning here.
                    _without_reasoning(message),
                    protected=protected,
                    mask_reader_text=self._protected_text,
                )
                for message in messages[-cap:]
            ),
        )

    async def continuation_workspaces(
        self, *, owner_id: str, run_id: str
    ) -> tuple[str, ...] | None:
        """Return a terminal Answer's workspace set for current authorization."""
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None or not record.terminal:
            return None
        return tuple(str(item) for item in record.request_input().get("workspaces") or ())

    async def follow_up(
        self,
        *,
        owner_id: str,
        caller: Caller | None,
        run_id: str,
        query: str,
        idempotency_key: str | None = None,
        authorized_workspaces: Sequence[str] | None,
    ) -> RunCreation | None:
        """Append one turn to the selected run's Lane tip through normal acceptance."""
        request = await self.continuation_request(
            owner_id=owner_id,
            run_id=run_id,
            query=query,
            include_answer=True,
            authorized_workspaces=authorized_workspaces,
        )
        if request is None:
            return None
        return await self.create(
            request=request,
            owner_id=owner_id,
            caller=caller,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=self._continuation_fingerprint(request),
        )

    async def fork(
        self,
        *,
        owner_id: str,
        caller: Caller | None,
        run_id: str,
        query: str,
        idempotency_key: str | None = None,
        authorized_workspaces: Sequence[str] | None,
    ) -> RunCreation | None:
        """Start a sibling branch from the state the selected run settled at."""
        request = await self.continuation_request(
            owner_id=owner_id,
            run_id=run_id,
            query=query,
            include_answer=False,
            authorized_workspaces=authorized_workspaces,
        )
        if request is None:
            return None
        return await self.create(
            request=request,
            owner_id=owner_id,
            caller=caller,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=self._continuation_fingerprint(request),
        )

    async def continuation_request(
        self,
        *,
        owner_id: str,
        run_id: str,
        query: str,
        include_answer: bool,
        authorized_workspaces: Sequence[str] | None,
    ) -> AnswerRequest | None:
        """Build one continuation's request after transport authorization.

        History is derived from the branch point, so a continuation injects none:
        every accepted Run records an Agent Session, and the fold at that point is
        the context. It is the caller's arrival at this endpoint, not history, that
        says whether the Run continues the Lane or branches from a Fork Point:
        ``include_answer`` is that choice of kind (a Follow-Up appends to the Lane
        tip, a Fork opens at the recorded Fork Point).
        """
        text = query.strip()
        if not text:
            raise AnswerRequestError("continuation query cannot be empty")
        if len(text) > _AGENT_CONTROL_CONTENT_LIMIT:
            raise AnswerRequestError("continuation query exceeds 20000 characters")
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None or not record.terminal:
            return None
        if authorized_workspaces is None:
            raise ValueError("continuation requires a currently authorized workspace set")
        accepted = record.request_input()
        session_id = str(accepted.get("agent_session_id") or "")
        if not session_id:
            # Every accepted Run records its Session. A parent without one was written by
            # an older release, and acceptance would otherwise mint a fresh Session and
            # answer the continuation with no history at all.
            raise AnswerRequestError("the parent Run has no Agent Session to continue")
        history_resources: list[AnswerHistoryResource] = []
        for reference_kind, items in (
            ("history_attachment", accepted.get("history_attachments") or ()),
            ("current_attachment", accepted.get("attachments") or ()),
        ):
            history_resources.extend(
                AnswerHistoryResource(
                    run_id=run_id,
                    source_ordinal=int(item.get("ordinal") or 0),
                    digest=str(item.get("digest") or ""),
                    filename=str(item.get("filename") or "attachment"),
                    mime_type=str(item.get("mime_type") or "application/octet-stream"),
                    byte_size=int(item.get("byte_size") or 0),
                    reference_kind=reference_kind,  # type: ignore[arg-type]
                )
                for item in items
                if isinstance(item, Mapping) and item.get("digest")
            )
        link_resources = tuple(
            ResourceInput(
                url=str(item.get("url") or ""),
                filename=(str(item["filename"]) if item.get("filename") else None),
                declared_mime=(str(item["mime_type"]) if item.get("mime_type") else None),
            )
            for item in accepted.get("links") or ()
            if isinstance(item, Mapping) and item.get("url")
        )
        filters = accepted.get("filters")
        parent_lane_id = str(accepted.get("agent_lane_id") or LaneId.main().value)
        continuation_kind = "follow_up" if include_answer else "fork"
        agent_lane_id = parent_lane_id if include_answer else LaneId.new().value
        return AnswerRequest(
            query=text,
            workspaces=tuple(str(item) for item in authorized_workspaces),
            episodic_summary=str(accepted.get("episodic_summary") or ""),
            retrieval=RetrievalOptions(
                top_k=(int(accepted["top_k"]) if accepted.get("top_k") is not None else None),
                chunk_top_k=(
                    int(accepted["chunk_top_k"])
                    if accepted.get("chunk_top_k") is not None
                    else None
                ),
                federated_rerank=bool(accepted.get("federated_rerank")),
            ),
            filters=(
                MetadataFilter.model_validate(filters) if isinstance(filters, Mapping) else None
            ),
            semantic_highlights=bool(accepted.get("semantic_highlights")),
            resources=link_resources,
            history_resources=tuple(history_resources),
            mode=str(accepted.get("mode") or "auto"),
            parent_run_id=run_id,
            continuation_kind=continuation_kind,
            agent_session_id=session_id,
            agent_lane_id=agent_lane_id,
            source_lane_id=(parent_lane_id if not include_answer else None),
        )

    async def wait(self, *, owner_id: str, run_id: str) -> AnswerResult:
        """Follow one owned run to its terminal state and project its result.

        Cancelling this wait detaches this observer only; use the common Run
        service to stop the run itself.
        """
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            raise RunFailedError(
                "answer_run_missing",
                "Answer run disappeared before it finished.",
            )
        async with aclosing(
            self._coordinator.subscribe(owner_id=owner_id, run_id=run_id)
        ) as events:
            async for _event in events:
                pass
        final = await self._store.get_run(owner_id=owner_id, run_id=run_id)
        if final is None:
            raise RunFailedError(
                "answer_run_missing",
                "Answer run disappeared before it finished.",
            )
        if final.status == "succeeded":
            return restore_answer_result(final.result or {})
        if final.status == "cancelled":
            raise RunCancelledError(final.run_id)
        raise RunFailedError(
            final.error_kind or "answer_stream_failed",
            final.error_message or "Answer run failed.",
        )

    async def answer(
        self,
        request: AnswerRequest,
        *,
        owner_id: str,
        caller: Caller | None,
        idempotency_key: str | None = None,
    ) -> AnswerResult:
        """Create one durable answer run and wait for its canonical result."""
        creation = await self.create(
            request=request,
            owner_id=owner_id,
            caller=caller,
            idempotency_key=idempotency_key,
        )
        return await self.wait(owner_id=owner_id, run_id=creation.run.run_id)

    async def answer_stream(
        self,
        request: AnswerRequest,
        *,
        owner_id: str,
        caller: Caller | None,
        idempotency_key: str | None = None,
    ) -> AsyncGenerator[RunEvent]:
        """Create one durable answer run and follow its events until it ends."""
        creation = await self.create(
            request=request,
            owner_id=owner_id,
            caller=caller,
            idempotency_key=idempotency_key,
        )
        run = creation.run
        async with aclosing(
            self._coordinator.subscribe(owner_id=owner_id, run_id=run.run_id)
        ) as events:
            async for event in events:
                yield RunEvent.from_runtime(event)

    async def capabilities(self) -> AnswerCapabilities:
        """Return the public image-capability snapshot after its allowed re-probe."""
        return await self._capability_view.read()

    def answering_model_profile(self) -> ModelProfile:
        """Return the answering role's resolved profile.

        One fact about this deployment is stated by a transport (the efforts the
        browser may offer) and enforced by the run (the level it applies), so both read
        the same resolution the run itself will use.
        """
        return self._capabilities.current_profiles()["query"]

    async def agent_effort_offer(self, caller: Caller | None) -> AgentEffortOffer:
        """Return the agent efforts this deployment applies for ``caller``, and the level
        their Answers run at when they choose none.

        One call states both, so a control can neither advertise a level the run would
        ignore nor mark a default it is not offering. A role that owns its reasoning
        through raw model kwargs applies no typed level at all, so it offers none; an
        uncatalogued model offers all three because its best-effort profile maps
        every level. The caller's ceiling takes the levels above it away, and the default
        is the level the Run will run at: the deployment's own, lowered to the ceiling
        where it is above it. A ceiling policy that cannot be read raises.
        """
        settings = self._models.model_settings("query")
        if settings.raw_agentic_reasoning_keys:
            return AgentEffortOffer(levels=(), default=None)
        ceiling = await self._policies.effort_ceiling(caller)
        profile = self.answering_model_profile()
        top = ANSWER_EFFORT_LEVELS.index(ceiling)
        levels: tuple[AnswerEffort, ...] = tuple(
            level
            for level in offered_answer_efforts(profile)
            if ANSWER_EFFORT_LEVELS.index(level) <= top
        )
        running = request_under_ceiling(
            profile.reasoning, settings.effective_agentic_reasoning, ceiling
        )
        default = cast(AnswerEffort, running) if running is not None and running in levels else None
        return AgentEffortOffer(levels=levels, default=default)

    async def read_input_artifact(
        self,
        *,
        owner_id: str,
        run_id: str,
        ordinal: int,
        reference_kind: ArtifactReferenceKind | None = None,
    ) -> AnswerInputArtifact | None:
        """Read one owned run's accepted input upload by its ordinal.

        Resources a worker fetched mid-run are run state, never accepted input,
        so they are not readable here; a current-turn upload takes precedence
        over a history upload sharing its ordinal.
        """
        references = await self.list_artifacts(owner_id=owner_id, run_id=run_id)
        if references is None:
            return None
        reference = next(
            (
                item
                for kind in (
                    (reference_kind,) if reference_kind is not None else _INPUT_REFERENCE_KINDS
                )
                for item in references
                if item.reference_kind == kind and item.ordinal == ordinal
            ),
            None,
        )
        if reference is None:
            return None
        pieces = [
            piece
            async for piece in self._blob_store.stream(owner_id=owner_id, digest=reference.digest)
        ]
        if not pieces:
            return None
        content = b"".join(pieces)
        return AnswerInputArtifact(
            reference_kind=reference.reference_kind,
            ordinal=reference.ordinal,
            filename=reference.filename,
            mime_type=reference.mime_type,
            digest=reference.digest,
            content=content,
        )

    @asynccontextmanager
    async def _prepare_input(
        self,
        request: AnswerRunRequest,
        *,
        resources: list[ResourceInput] | None,
        idempotency_fingerprint: str,
        requested_mode: AnswerMode,
        allowed_modes: frozenset[ResolvedMode],
        agent_may_register: bool,
        effort_ceiling: AnswerEffort,
        memory_enabled: bool = True,
    ) -> AsyncIterator[
        Callable[
            [Sequence[ToolDeclaration], Sequence[BoundConnectionSurface], bool],
            Awaitable[tuple[AnswerRunInput, frozenset[ResolvedMode]]],
        ]
    ]:
        """Resolve one normalized request and its capacity-narrowed mode set."""
        # Drawn once per accepted Run, so a retried binding attempt keeps the handles
        # acceptance measured with, and every resume mints them again.
        resource_identity = new_resource_identity()
        async with self._project_acceptance(
            request,
            resources=resources,
            requested_mode=requested_mode,
            allowed_modes=allowed_modes,
            memory_enabled=memory_enabled,
            agent_may_register=agent_may_register,
            effort_ceiling=effort_ceiling,
            resource_identity=resource_identity,
        ) as project:

            async def prepare(
                connection_tools: Sequence[ToolDeclaration],
                surface_entries: Sequence[BoundConnectionSurface],
                resource_tools_available: bool,
            ) -> tuple[AnswerRunInput, frozenset[ResolvedMode]]:
                try:
                    connection_surface = decode_connection_surface(
                        [
                            (
                                ConnectionSurface(
                                    connection_id=entry.connection_id,
                                    token=entry.token,
                                    label=entry.label,
                                    instructions=entry.instructions,
                                )
                            ).as_json()
                            for entry in surface_entries
                        ]
                    )
                except ValueError:
                    raise AnswerRequestError(
                        "This Run's Connection instructions exceed what a Run may pin"
                    ) from None
                projection = await project(
                    connection_tools, connection_surface, resource_tools_available
                )
                return AnswerRunInput(
                    query=request.query,
                    workspaces=request.workspaces,
                    history=projection.history,
                    episodic_summary=projection.episodic_summary,
                    retrieval=request.retrieval,
                    filters=request.filters,
                    semantic_highlights=request.semantic_highlights,
                    links=request.links,
                    attachments=request.attachments,
                    history_attachments=request.history_attachments,
                    pinned_models=projection.pinned_models,
                    context_policy_revision=CONTEXT_POLICY_REVISION,
                    model_catalog_revision=current_model_catalog_revision(),
                    idempotency_fingerprint=idempotency_fingerprint,
                    resource_identity=resource_identity,
                    agent_run_plan=projection.agent_run_plan,
                    connection_surface=connection_surface,
                    image_descriptions=projection.image_descriptions,
                    parent_run_id=request.parent_run_id,
                    continuation_kind=request.continuation_kind,
                    goal_id=request.goal_id,
                    goal_revision=request.goal_revision,
                    goal_round=request.goal_round,
                    agent_session_id=request.agent_session_id or SessionId.new().value,
                    agent_lane_id=request.agent_lane_id,
                    source_lane_id=request.source_lane_id,
                    effort=request.effort,
                    effort_ceiling=effort_ceiling,
                ), projection.valid_modes

            yield prepare

    @asynccontextmanager
    async def _project_acceptance(
        self,
        request: AnswerRunRequest,
        *,
        resources: list[ResourceInput] | None,
        requested_mode: AnswerMode,
        allowed_modes: frozenset[ResolvedMode],
        agent_may_register: bool,
        effort_ceiling: AnswerEffort,
        memory_enabled: bool = True,
        resource_identity: str,
    ) -> AsyncIterator[
        Callable[
            [Sequence[ToolDeclaration], Sequence[ConnectionSurface], bool],
            Awaitable[_AcceptanceProjection],
        ]
    ]:
        """Resolve the exact shared-history envelopes without building the run rig."""
        model_profiles = self._capabilities.current_profiles()
        resolved = await self._resources.resolve(
            resources,
            models=self._capabilities.request_model_context(model_profiles),
            confirm_image_context=self._capabilities.confirmed_live_answer_context,
            resolved_mode=("research" if "research" in allowed_modes else "fast"),
            resource_identity=resource_identity,
        )
        try:
            workspaces = list(request.workspaces)
            self._retrieval.warm(workspaces)
            models = resolved.models
            model_profiles["extract"] = models.extract
            model_profiles["query"] = models.query
            model_profiles["vlm"] = models.vlm
            image_descriptions = tuple(
                await prepare_query_images(
                    query_images=resolved.current_images,
                    describer=self._models.query_image_describer(),
                )
                if resolved.current_images
                else ()
            )
            budget = AnswerHistoryBudget(
                query=request.query,
                profiles=model_profiles,
                planner_for=self._retrieval.planner_for,
                schema=await self._retrieval.schema_for(workspaces),
                answer_image_policy=self._capabilities.answer_image_policy,
                image_descriptions=image_descriptions,
                current_images=resolved.current_images,
                memory_text=reserved_memory_text(enabled=memory_enabled),
                episodic_summary=request.episodic_summary,
            )
            web_search = resolved.web_sources is not None and resolved.web_sources.search_enabled

            async def project(
                connection_tools: Sequence[ToolDeclaration],
                connection_surface: Sequence[ConnectionSurface],
                resource_tools_available: bool,
            ) -> _AcceptanceProjection:
                pinned_models = self._pin_model_profiles(model_profiles)
                agent_run_plan: AgentRunPlan | None = None
                research: ResearchSeed | None = None
                if "research" in allowed_modes:
                    tools = (
                        self._research_tool_declarations(
                            web_search=web_search,
                            memory=memory_enabled,
                            agent_may_register=agent_may_register,
                            model_guidance=child_model_guidance(
                                pinned_models, effort_ceiling=effort_ceiling
                            ),
                            injected=connection_tools,
                            mcp_resources=resource_tools_available and bool(connection_surface),
                        )
                        if self._research_tool_declarations is not None
                        else research_tool_declarations(
                            web_search=web_search,
                            injected=connection_tools,
                            mcp_resources=resource_tools_available and bool(connection_surface),
                        )
                    )
                    agent_run_plan = AgentRunPlan.from_tools(
                        tools,
                        model_role="query",
                        context_policy_revision=CONTEXT_POLICY_REVISION,
                        model_identity=asdict(self._model_invocation_fingerprint_for_role("query")),
                        model_profile=asdict(models.query),
                    )
                    research = ResearchSeed(
                        tools=tools,
                        query_images=resolved.query_images,
                        resource_manifest=resolved.resource_manifest,
                        image_budget=resolved.image_budget,
                        connection_surface=tuple(connection_surface),
                    )
                accepted = accept_history(
                    budget,
                    history=request.history,
                    requested_mode=requested_mode,
                    allowed_modes=allowed_modes,
                    research=research,
                    mode_resources=routing_resources(
                        request.attachments, request.history_attachments
                    ),
                    web_search=web_search,
                )
                return _AcceptanceProjection(
                    history=accepted.history,
                    episodic_summary=accepted.episodic_summary,
                    image_descriptions=image_descriptions,
                    pinned_models=pinned_models,
                    agent_run_plan=agent_run_plan,
                    valid_modes=accepted.valid_modes,
                )

            yield project
        finally:
            if resolved.registry is not None:
                await resolved.registry.aclose()

    def _pin_model_profiles(
        self,
        profiles: Mapping[ChatModelSelector, ModelProfile],
    ) -> tuple[PinnedModelProfile, ...]:
        return tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=self._model_invocation_fingerprint_for_role(role),
                profile=profiles[role],
                reasoning_settings=model_reasoning_settings(self._models.model_settings(role)),
            )
            for role in CHAT_MODEL_SELECTORS
        )


__all__ = [
    "ActiveRunLimitError",
    "AnswerConnectionsChangedError",
    "AnswerRequestError",
    "AnswerHistoryResource",
    "AnswerInputArtifact",
    "AnswerRequest",
    "AnswerRunAcceptor",
    "AnswerService",
]
