# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable answer runs over already-authorized canonical workspaces."""

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, aclosing, asynccontextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Protocol, cast
from uuid import UUID, uuid7

from dlightrag.application.connections import BoundResearchConnections, ConnectionsError
from dlightrag.application.runs import (
    IdempotencyKeyConflict,
    RunAdmissionLimitExceededError,
    RunCancelledError,
    RunCreation,
    RunEvent,
    RunFailedError,
    RunRuntimeUnavailableError,
)
from dlightrag.engine.agent.session.fold import PriorTurns
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.tools import AgentTool
from dlightrag.engine.agent.tools.registry import DuplicateToolError, ToolRegistry
from dlightrag.engine.ai.capacity import (
    CONTEXT_POLICY,
    CONTEXT_POLICY_REVISION,
    ModelProfile,
)
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.settings import CHAT_MODEL_SELECTORS, ChatModelSelector, ModelSettings
from dlightrag.engine.answer.capabilities import AnswerCapabilities, RequestModelContext
from dlightrag.engine.answer.client_contracts import AnswerEffort, offered_answer_efforts
from dlightrag.engine.answer.errors import (
    AnswerInputOverflowError,
    InvalidToolConfigurationError,
    UnsupportedAnswerModeError,
)
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.execution import (
    ResolvedAnswerResources,
    research_history_input_measure,
)
from dlightrag.engine.answer.execution.connection_binding import (
    RunConnectionBinding,
    StaleConnectionBindingError,
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
)
from dlightrag.engine.answer.history import (
    HistoryProjectionOverflowError,
    HistoryProjectionTarget,
    project_history,
)
from dlightrag.engine.answer.image_capability import AnswerImageCapability
from dlightrag.engine.answer.images import AnswerImagePolicy
from dlightrag.engine.answer.memory import memory_owner_allowed, standing_memory_for_acceptance
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
from dlightrag.engine.answer.resources.images import QueryImageDescriber, prepare_query_images
from dlightrag.engine.answer.resources.models import ResourceInput
from dlightrag.engine.answer.results import AnswerResult, restore_answer_result
from dlightrag.engine.answer.runs.envelope import accepted_input_envelope
from dlightrag.engine.answer.runs.routing import RoutingAcceptance
from dlightrag.engine.answer.synthesizer import AnswerSynthesizer
from dlightrag.engine.answer.tools import compose_research_tools
from dlightrag.engine.answer.tools.subagents import SubagentHost, subagent_tools
from dlightrag.engine.network_admission import public_http_url_identity
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename
from dlightrag.engine.rag.retrieval import MetadataFilter, RetrievalOptions, RetrievalResult
from dlightrag.engine.rag.retrieval.planner import RetrievalPlanner
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id
from dlightrag.engine.runtime.contracts import RunKind
from dlightrag.engine.runtime.records import (
    ArtifactReferenceKind,
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    RunAccessScope,
    RunArtifactReference,
    RunFetchedResource,
    RunRecord,
    artifact_digest,
    require_prepared_input_bounds,
    run_request_fingerprint,
)
from dlightrag.engine.runtime.records import (
    IdempotencyKeyConflict as RuntimeIdempotencyKeyConflict,
)
from dlightrag.engine.runtime.records import (
    RunAdmissionLimitExceededError as RuntimeRunAdmissionLimitExceededError,
)
from dlightrag.engine.runtime.records import RunCreation as RuntimeRunCreation
from dlightrag.engine.runtime.records import RunEvent as RuntimeRunEvent

from .child_roster import (
    ChildRosterCursor,
    ChildRosterCursorCodec,
    ChildRosterPage,
    ChildRosterPageRequest,
    ChildRosterRowPage,
    child_result_lineage,
    public_child_status,
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


def _stamp(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat()).replace("+00:00", "Z")
    text = str(value).strip()
    return text or None


def _public_transcript_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Keep useful conversation/tool lineage and drop private reasoning fields."""
    role = str(message.get("role") or "")
    projected: dict[str, Any] = {"role": role, "content": message.get("content") or ""}
    if role == "assistant":
        projected["tool_calls"] = [
            {
                "id": call.get("id"),
                "name": call.get("name")
                or (
                    call.get("function", {}).get("name")
                    if isinstance(call.get("function"), Mapping)
                    else None
                ),
            }
            for call in message.get("tool_calls") or ()
            if isinstance(call, Mapping)
        ]
    if role == "tool":
        projected["tool_call_id"] = str(message.get("tool_call_id") or "")
        projected["name"] = str(message.get("name") or "")
        projected["is_error"] = bool(message.get("is_error"))
    return projected


def _public_control_record(row: Mapping[str, Any]) -> dict[str, Any]:
    consumed_at = row.get("consumed_at")
    return {
        "control_sequence": int(row.get("control_sequence") or 0),
        "kind": str(row.get("kind") or ""),
        "content": str(row.get("content") or ""),
        "origin": str(row.get("origin") or ""),
        "consumed": bool(row.get("consumed") if "consumed" in row else consumed_at),
        "consumed_at": _stamp(consumed_at),
        "created_at": _stamp(row.get("created_at")),
        "operation_id": (str(row["operation_id"]) if row.get("operation_id") else None),
    }


def _public_question_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(row.get("request_id") or ""),
        "question": str(row.get("question") or ""),
        "status": str(row.get("status") or ""),
        "reply": row.get("reply"),
        "reply_origin": row.get("reply_origin"),
        "expires_at": _stamp(row.get("expires_at")),
        "created_at": _stamp(row.get("created_at")),
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
    """The agent efforts one deployment applies, and its own configured level."""

    levels: tuple[AnswerEffort, ...]
    default: AnswerEffort | None


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


class HistoryResolver(Protocol):
    """In-process durable history projection invoked after exact targets exist."""

    def __call__(self, targets: Sequence[HistoryProjectionTarget]) -> Awaitable[PriorTurns]: ...


class AnswerRuntimeUnavailableError(RunRuntimeUnavailableError):
    """Answer-specific compatibility name for common runtime unavailability."""


class AnswerRunAcceptor[T](Protocol):
    """Persist or replay one prepared run, optionally with an atomic domain link."""

    async def create_run(
        self,
        *,
        envelope: PreparedRunEnvelope,
        run_id: str,
        resources: Sequence[Mapping[str, Any]] = (),
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
        connection_bindings: tuple[RunConnectionBinding, ...] = (),
    ) -> T | None: ...

    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        run_kind: RunKind,
    ) -> T | None: ...


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
        self, *, owner_id: str, run_id: str, session_id: str, limit: int
    ) -> tuple[Mapping[str, Any], ...]: ...

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


class _RunScheduler(Protocol):
    """The started coordinator accepted runs execute and stream through."""

    @property
    def is_started(self) -> bool: ...

    def admission(self) -> AbstractAsyncContextManager[bool]: ...

    def wake(self) -> None: ...

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


def _prepared_input_payload(
    run_input: Any, *, requested_mode: str, auth_mode: str = "none"
) -> dict[str, Any]:
    """Encode one accepted run with its canonical Session/Lane mapping."""
    payload = dict(run_input.as_request())
    payload["auth_mode"] = auth_mode
    payload["mode"] = requested_mode
    return payload


def _accepted_resource_payloads(
    run_input: Any, *, attachment_bytes: Sequence[bytes]
) -> list[dict[str, Any]]:
    import hashlib

    payloads: list[dict[str, Any]] = []
    for ordinal, attachment in enumerate(run_input.attachments):
        content = attachment_bytes[ordinal] if ordinal < len(attachment_bytes) else b""
        payloads.append(
            {
                "resource_id": attachment.resource_id,
                "safe_name": attachment.filename,
                "media_type": attachment.mime_type or "application/octet-stream",
                "capabilities": {},
                "ordinal": ordinal,
                "blob_digest": hashlib.sha256(content).hexdigest(),
            }
        )
    return payloads


def _normalized_request(request: AnswerRequest) -> AnswerRunRequest:
    """Project one public request into durable acceptance input, without I/O."""
    if not request.workspaces:
        raise ValueError("at least one canonical workspace is required")
    workspaces = tuple(
        require_canonical_workspace_id(workspace) for workspace in request.workspaces
    )
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
        model_fingerprint_for_role: Callable[[ChatModelSelector], ModelFingerprint],
        child_roster_cursor_secret: bytes,
        research_tool_supplements: Callable[[], Sequence[AgentTool]] | None = None,
        bind_research: Callable[..., Awaitable[BoundResearchConnections]] | None = None,
        memory_capability: Callable[..., Awaitable[tuple[bool, int]]] | None = None,
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
        self._model_fingerprint_for_role = model_fingerprint_for_role
        self._research_tool_supplements = research_tool_supplements or (lambda: ())
        self._bind_research = bind_research
        self._memory_capability = memory_capability
        self._run_retention_seconds = int(run_retention_seconds)
        self._child_roster_codec = ChildRosterCursorCodec(child_roster_cursor_secret)

    async def create(
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        idempotency_key: str | None = None,
        auth_mode: str = "none",
    ) -> RunCreation:
        """Accept one durable run and return its descriptor without waiting.

        A keyed replay is resolved before any link is materialized or any input
        is prepared, so a retried submission never repeats acceptance work. The
        accepted run outlives this call and is read back through :meth:`get`,
        :meth:`subscribe`, and :meth:`cancel`.
        """
        try:
            creation = await self._accept(
                request=request,
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=None,
                acceptor=self._store,
                auth_mode=auth_mode,
            )
        except RuntimeIdempotencyKeyConflict as exc:
            raise IdempotencyKeyConflict(str(exc)) from exc
        except RuntimeRunAdmissionLimitExceededError as exc:
            raise RunAdmissionLimitExceededError(str(exc)) from exc
        if creation is None:
            raise RuntimeError("Answer run acceptance returned no descriptor")
        return RunCreation.from_runtime(creation)

    async def accept[T](
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        acceptor: AnswerRunAcceptor[T],
        auth_mode: str = "none",
        history_resolver: HistoryResolver | None = None,
    ) -> T | None:
        """Accept through a typed atomic linker while preserving one run pipeline."""
        try:
            return await self._accept(
                request=request,
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=idempotency_fingerprint,
                acceptor=acceptor,
                auth_mode=auth_mode,
                history_resolver=history_resolver,
            )
        except RuntimeIdempotencyKeyConflict as exc:
            raise IdempotencyKeyConflict(str(exc)) from exc
        except RuntimeRunAdmissionLimitExceededError as exc:
            raise RunAdmissionLimitExceededError(str(exc)) from exc

    async def _accept[T](
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        idempotency_key: str | None,
        idempotency_fingerprint: str | None,
        acceptor: AnswerRunAcceptor[T],
        auth_mode: str = "none",
        history_resolver: HistoryResolver | None = None,
    ) -> T | None:
        run_request = _normalized_request(request)
        fingerprint = idempotency_fingerprint or run_request_fingerprint(run_request.as_request())
        # Canonical mode syntax is capability-independent and may be checked
        # before replay. Live capability validation must wait until after the
        # answer/VLM refreshes below because an unknown probe can recover.
        requested_mode = canonical_answer_mode(run_request.mode)
        if idempotency_key is not None:
            replay = await acceptor.replay_run(
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=fingerprint,
                run_kind="answer",
            )
            if replay is not None:
                return replay
        if not self._coordinator.is_started:
            raise AnswerRuntimeUnavailableError("Answer runtime is unavailable")
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
        memory_enabled = memory_owner_allowed(auth_mode)
        memory_epoch = 0
        if memory_enabled and self._memory_capability is not None:
            memory_enabled, memory_epoch = await self._memory_capability(owner_id=owner_id)
        async with self._prepare_input(
            run_request,
            resources=acceptance_resources or None,
            idempotency_fingerprint=fingerprint,
            requested_mode=requested_mode,
            allowed_modes=allowed_modes,
            auth_mode=auth_mode,
            memory_enabled=memory_enabled,
            history_resolver=history_resolver,
        ) as prepare:
            for attempt in range(2):
                bound = (
                    await self._bind_research(owner_id=owner_id, auth_mode=auth_mode)
                    if self._bind_research is not None
                    and requested_mode != "fast"
                    and "research" in allowed_modes
                    else BoundResearchConnections()
                )
                run_input, effective_modes = await prepare(bound.tools)
                run_input = replace(run_input, run_connection_bindings=bound.bindings)
                prepared_input = _prepared_input_payload(
                    run_input, requested_mode=requested_mode, auth_mode=auth_mode
                )
                prepared_input["profile_memory_enabled"] = memory_enabled
                prepared_input["profile_memory_epoch"] = memory_epoch
                require_prepared_input_bounds(prepared_input)
                resources_payload = _accepted_resource_payloads(
                    run_input, attachment_bytes=attachment_bytes
                )
                try:
                    async with self._coordinator.admission() as runtime_available:
                        if not runtime_available:
                            raise AnswerRuntimeUnavailableError("Answer runtime is unavailable")
                        run_id = str(uuid7())
                        accepted = await acceptor.create_run(
                            envelope=PreparedRunEnvelope(
                                run_kind="answer",
                                lane="query",
                                submitted_by=owner_id,
                                access_scope=RunAccessScope(kind="owner", scope_id=owner_id),
                                submission_key=idempotency_key or run_id,
                                request_fingerprint=fingerprint,
                                payload=prepared_input,
                                accepted_input=accepted_input_envelope(prepared_input),
                                retention_seconds=self._run_retention_seconds,
                            ),
                            run_id=run_id,
                            resources=resources_payload,
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
                                    item.role: {
                                        "provider": item.fingerprint.provider,
                                        "model": item.fingerprint.model,
                                        "endpoint_fingerprint": item.fingerprint.endpoint_fingerprint,
                                    }
                                    for item in run_input.pinned_models
                                },
                                agent_session_id=run_input.agent_session_id,
                                agent_lane_id=run_input.agent_lane_id,
                                source_lane_id=run_input.source_lane_id,
                            ),
                        )
                        if accepted is not None:
                            self._coordinator.wake()
                except StaleConnectionBindingError as exc:
                    if idempotency_key is not None:
                        replay = await acceptor.replay_run(
                            owner_id=owner_id,
                            idempotency_key=idempotency_key,
                            idempotency_fingerprint=fingerprint,
                            run_kind="answer",
                        )
                        if replay is not None:
                            return replay
                    if attempt == 1:
                        raise ConnectionsError(
                            "Connections changed repeatedly; submit the Answer again"
                        ) from exc
                    continue
                return accepted
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
                raise AnswerRuntimeUnavailableError("Accepted answer input artifact is unavailable")
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

    async def read_artifact(
        self,
        *,
        owner_id: str,
        run_id: str,
        resource_id: str,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes | None:
        """Read a bounded Answer artifact; every other id returns ``None``."""
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
        """Open one published artifact through the shared run-resource reader."""
        return await self.open_run_resource(
            owner_id=owner_id,
            run_id=run_id,
            resource_id=resource_id,
            offset=offset,
            length=length,
        )

    async def artifact_size(self, *, owner_id: str, run_id: str, resource_id: str) -> int | None:
        """Return one published artifact size through the shared reader."""
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
            raise ValueError("steer instruction cannot be empty")
        if len(text) > _AGENT_CONTROL_CONTENT_LIMIT:
            raise ValueError("steer instruction exceeds 20000 characters")
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
            raise ValueError("Child control idempotency key must be between 1 and 200 characters")
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
                raise ValueError(
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
                raise ValueError("unknown Child control action")
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
            raise ValueError("child-roster cursor belongs to another run")
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
            children=tuple(public_child_status(row) for row in result.children),
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
    ) -> ChildObservation | None:
        """Return one bounded child observation, or None if unknown."""
        if await self._get_answer_run(owner_id=owner_id, run_id=run_id) is None:
            return None
        row = await self._store.load_child_session(
            owner_id=owner_id, run_id=run_id, child_session_id=child_session_id
        )
        if row is None:
            return None
        cap = max(1, min(int(limit), 100))
        transcript = await self._store.load_agent_transcript(
            owner_id=owner_id,
            run_id=run_id,
            session_id=child_session_id,
            limit=cap,
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
        return ChildObservation(
            run_id=run_id,
            child=public_child_status(row),
            transcript=tuple(_public_transcript_message(message) for message in transcript),
            controls=tuple(_public_control_record(item) for item in controls),
            questions=tuple(_public_question_record(item) for item in questions),
            result=child_result_lineage(row),
        )

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
        load_transcript = getattr(self._store, "load_agent_transcript", None)
        if session_id and callable(load_transcript):
            loader = cast(
                Callable[..., Awaitable[Sequence[Mapping[str, Any]]]],
                load_transcript,
            )
            canonical = await loader(
                owner_id=owner_id,
                run_id=run_id,
                session_id=session_id,
                limit=cap,
            )
            if canonical:
                return AgentTranscriptTail(
                    run_id=run_id,
                    status=record.status,
                    messages=tuple(dict(message) for message in canonical),
                )
        # Fast has no Agent Session. Project its accepted invocation and final
        # result through the same transport-neutral message shape.
        messages = [
            dict(message)
            for message in request.get("history") or ()
            if isinstance(message, Mapping)
        ]
        query = str(request.get("query") or "")
        if query:
            messages.append({"role": "user", "content": query})
        result = record.result or {}
        answer = str(result.get("answer") or "")
        if answer:
            messages.append({"role": "assistant", "content": answer})
        return AgentTranscriptTail(
            run_id=run_id,
            status=record.status,
            messages=tuple(messages[-cap:]),
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
        run_id: str,
        query: str,
        idempotency_key: str | None = None,
        auth_mode: str = "none",
        authorized_workspaces: Sequence[str] | None,
    ) -> RunCreation | None:
        """Start a continuation from one terminal result through normal acceptance."""
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
            idempotency_key=idempotency_key,
            auth_mode=auth_mode,
        )

    async def fork(
        self,
        *,
        owner_id: str,
        run_id: str,
        query: str,
        idempotency_key: str | None = None,
        auth_mode: str = "none",
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
            idempotency_key=idempotency_key,
            auth_mode=auth_mode,
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

        History is derived from the branch point. A continuation whose parent
        recorded an Agent Session injects none: the fold at that point is the
        context, and it is the caller's arrival at this endpoint — not history —
        that says whether the run continues the Lane or branches from a Fork Point.
        ``include_answer`` is the endpoint's own choice of kind (a Follow-Up
        appends to the Lane tip, a Fork opens at the recorded Fork Point) and, for
        a caller with no Session branch point, it also decides whether the parent's
        answer joins the history that only such a caller receives.
        """
        text = query.strip()
        if not text:
            raise ValueError("continuation query cannot be empty")
        if len(text) > _AGENT_CONTROL_CONTENT_LIMIT:
            raise ValueError("continuation query exceeds 20000 characters")
        record = await self._get_answer_run(owner_id=owner_id, run_id=run_id)
        if record is None or not record.terminal:
            return None
        if authorized_workspaces is None:
            raise ValueError("continuation requires a currently authorized workspace set")
        accepted = record.request_input()
        parent_session_id = str(accepted.get("agent_session_id") or "")
        history: list[Mapping[str, Any]] = []
        if not parent_session_id:
            history = [
                dict(message)
                for message in accepted.get("history") or ()
                if isinstance(message, Mapping)
            ]
            parent_query = str(accepted.get("query") or "")
            if parent_query:
                history.append({"role": "user", "content": parent_query})
            if include_answer:
                parent_answer = str((record.result or {}).get("answer") or "")
                if parent_answer:
                    history.append({"role": "assistant", "content": parent_answer})

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
        agent_session_id = parent_session_id or SessionId.new().value
        parent_lane_id = str(accepted.get("agent_lane_id") or LaneId.main().value)
        continuation_kind = "follow_up" if include_answer else "fork"
        agent_lane_id = parent_lane_id if include_answer else LaneId.new().value
        return AnswerRequest(
            query=text,
            workspaces=tuple(str(item) for item in authorized_workspaces),
            history=tuple(history),
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
            agent_session_id=agent_session_id,
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
        idempotency_key: str | None = None,
    ) -> AnswerResult:
        """Create one durable answer run and wait for its canonical result."""
        creation = await self.create(
            request=request,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
        )
        return await self.wait(owner_id=owner_id, run_id=creation.run.run_id)

    async def answer_stream(
        self,
        request: AnswerRequest,
        *,
        owner_id: str,
        idempotency_key: str | None = None,
    ) -> AsyncGenerator[RunEvent]:
        """Create one durable answer run and follow its events until it ends."""
        creation = await self.create(
            request=request,
            owner_id=owner_id,
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

    def agent_effort_offer(self) -> AgentEffortOffer:
        """Return the agent efforts this deployment applies, and its own level among them.

        One call states both, so a control can neither advertise a level the run would
        ignore nor mark a default it is not offering. A role that owns its reasoning
        through raw model kwargs applies no typed level at all, so it offers none; an
        uncatalogued model offers all three because its best-effort profile maps
        every level.
        """
        settings = self._models.model_settings("query")
        levels: tuple[AnswerEffort, ...] = ()
        if not settings.raw_agentic_reasoning_keys:
            levels = offered_answer_efforts(self.answering_model_profile())
        configured = settings.effective_agentic_reasoning
        default: AnswerEffort | None = None
        if configured is not None and configured in levels:
            default = cast(AnswerEffort, configured)
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
        auth_mode: str = "none",
        memory_enabled: bool = True,
        history_resolver: HistoryResolver | None = None,
    ) -> AsyncIterator[
        Callable[[Sequence[AgentTool]], Awaitable[tuple[AnswerRunInput, frozenset[ResolvedMode]]]]
    ]:
        """Resolve one normalized request and its capacity-narrowed mode set."""
        async with self._project_acceptance(
            request,
            resources=resources,
            requested_mode=requested_mode,
            allowed_modes=allowed_modes,
            auth_mode=auth_mode,
            memory_enabled=memory_enabled,
            history_resolver=history_resolver,
        ) as project:

            async def prepare(
                connection_tools: Sequence[AgentTool],
            ) -> tuple[AnswerRunInput, frozenset[ResolvedMode]]:
                projection = await project(connection_tools)
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
                    agent_run_plan=projection.agent_run_plan,
                    image_descriptions=projection.image_descriptions,
                    parent_run_id=request.parent_run_id,
                    continuation_kind=request.continuation_kind,
                    agent_session_id=request.agent_session_id or SessionId.new().value,
                    agent_lane_id=request.agent_lane_id,
                    source_lane_id=request.source_lane_id,
                    effort=request.effort,
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
        auth_mode: str = "none",
        memory_enabled: bool = True,
        history_resolver: HistoryResolver | None = None,
    ) -> AsyncIterator[Callable[[Sequence[AgentTool]], Awaitable[_AcceptanceProjection]]]:
        """Resolve the exact shared-history envelopes without building the run rig."""
        model_profiles = self._capabilities.current_profiles()
        models = self._capabilities.request_model_context(model_profiles)
        planner = self._retrieval.planner_for(models.extract)
        resolved = await self._resources.resolve(
            resources,
            models=models,
            confirm_image_context=self._capabilities.confirmed_live_answer_context,
            resolved_mode=("research" if "research" in allowed_modes else "fast"),
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
            schema = await self._retrieval.schema_for(workspaces)

            async def project(connection_tools: Sequence[AgentTool]) -> _AcceptanceProjection:
                agent_run_plan: AgentRunPlan | None = None
                memory_text = standing_memory_for_acceptance(auth_mode) if memory_enabled else ""
                effective_modes = allowed_modes
                fast_targets: list[HistoryProjectionTarget] = []
                if "fast" in effective_modes:
                    fast_targets.append(
                        HistoryProjectionTarget(
                            "fast_planner",
                            models.extract,
                            planner.history_input_measure(
                                request.query,
                                schema=schema,
                                current_image_descriptions=list(image_descriptions) or None,
                                preserve_query=None,
                            ),
                            proactive_compaction=True,
                            require_full_dynamic_reserve=True,
                        )
                    )
                    synthesizer = AnswerSynthesizer(
                        image_policy=self._capabilities.answer_image_policy(models.query),
                        model_profile=models.query,
                        context_policy=CONTEXT_POLICY,
                        model_func=None,
                    )
                    fast_generation_measure = (
                        synthesizer.history_input_measure(
                            request.query,
                            memory_text=memory_text,
                            episodic_summary=request.episodic_summary,
                            current_images=resolved.current_images,
                        )
                        if resolved.current_images
                        else synthesizer.history_input_measure(
                            request.query,
                            memory_text=memory_text,
                            episodic_summary=request.episodic_summary,
                        )
                    )
                    fast_targets.append(
                        HistoryProjectionTarget(
                            "fast_generation",
                            models.query,
                            fast_generation_measure,
                            proactive_compaction=True,
                            require_full_dynamic_reserve=True,
                        )
                    )
                    try:
                        project_history([], targets=fast_targets)
                    except HistoryProjectionOverflowError as exc:
                        if requested_mode == "fast":
                            raise AnswerInputOverflowError(str(exc)) from exc
                        effective_modes = cast(
                            frozenset[ResolvedMode],
                            frozenset(mode for mode in effective_modes if mode != "fast"),
                        )
                        if not effective_modes:
                            raise UnsupportedAnswerModeError(requested_mode) from exc

                pinned_models = self._pin_model_profiles(model_profiles)
                targets: list[HistoryProjectionTarget] = []
                if "research" in effective_modes:
                    targets.append(
                        HistoryProjectionTarget(
                            "research_planner",
                            models.extract,
                            planner.history_input_measure(
                                request.query,
                                schema=schema,
                                current_image_descriptions=list(image_descriptions) or None,
                                preserve_query=True,
                            ),
                        )
                    )
                    evidence = EvidenceLedger(image_budget=resolved.image_budget)

                    async def unused_retrieve(_query: str) -> RetrievalResult:
                        raise RuntimeError("acceptance tool definitions are never executed")

                    tools = compose_research_tools(
                        evidence=evidence,
                        trace={},
                        retrieve_knowledge_base=unused_retrieve,
                        search_web=(
                            resolved.web_sources.search
                            if resolved.web_sources is not None
                            and resolved.web_sources.search_enabled
                            else None
                        ),
                        injected_tools=[],
                        register_web_source=(
                            resolved.registry.register_discovered_link
                            if resolved.registry is not None and resolved.web_sources is not None
                            else None
                        ),
                    )
                    supplements = [*self._research_tool_supplements(), *connection_tools]
                    child_definitions = {
                        tool.name: tool
                        for tool in subagent_tools(
                            host=SubagentHost(model_guidance=child_model_guidance(pinned_models))
                        )
                    }
                    supplements = [
                        replace(
                            tool,
                            description=child_definitions[tool.name].description,
                            input_model=child_definitions[tool.name].input_model,
                            contract_version=child_definitions[tool.name].contract_version,
                        )
                        if tool.name in child_definitions
                        else tool
                        for tool in supplements
                    ]
                    if not memory_enabled:
                        supplements = [
                            tool
                            for tool in supplements
                            if tool.name not in {"remember", "forget", "recall_memory"}
                        ]
                    try:
                        tools = list(ToolRegistry([*tools, *supplements]).resolve())
                    except DuplicateToolError as exc:
                        raise InvalidToolConfigurationError(exc.names) from exc
                    agent_run_plan = AgentRunPlan.from_tools(
                        tools,
                        model_role="query",
                        context_policy_revision=CONTEXT_POLICY_REVISION,
                        model_identity=asdict(self._model_fingerprint_for_role("query")),
                        model_profile=asdict(models.query),
                    )
                    measure = research_history_input_measure(
                        model_profile=models.query,
                        context_policy=CONTEXT_POLICY,
                        query=request.query,
                        query_images=resolved.query_images,
                        resource_manifest=resolved.resource_manifest,
                        image_budget=resolved.image_budget,
                        tools=tools,
                        memory_text=memory_text,
                        episodic_summary=request.episodic_summary,
                    )
                    targets.append(
                        HistoryProjectionTarget(
                            "research_seed",
                            models.query,
                            measure,
                            proactive_compaction=True,
                        )
                    )
                if "fast" in effective_modes:
                    targets.extend(fast_targets)
                if requested_mode == "auto" and effective_modes >= {"fast", "research"}:
                    from dlightrag.engine.answer.router import AnswerModeRouter

                    async def _unused_router(**_kwargs: Any) -> str:
                        raise RuntimeError("acceptance router measure never calls the model")

                    router = AnswerModeRouter(_unused_router)
                    mode_resources = tuple(
                        ModeResource(
                            role=resource_role(filename=item.filename, mime_type=item.mime_type)
                        )
                        for item in (*request.attachments, *request.history_attachments)
                    )
                    targets.append(
                        HistoryProjectionTarget(
                            "router",
                            models.query,
                            router.history_input_measure(
                                request.query,
                                resources=mode_resources,
                                valid_modes=tuple(sorted(effective_modes)),
                            ),
                        )
                    )
                try:
                    history = (
                        await history_resolver(targets)
                        if history_resolver is not None
                        else project_history(
                            [dict(message) for message in request.history],
                            targets=targets,
                        )
                    )
                except HistoryProjectionOverflowError as exc:
                    if exc.target == "router":
                        raise UnsupportedAnswerModeError("auto") from exc
                    raise AnswerInputOverflowError(str(exc)) from exc
                episodic_parts = [
                    item.strip()
                    for item in (request.episodic_summary, history.episodic_summary)
                    if item.strip()
                ]
                return _AcceptanceProjection(
                    history=tuple(dict(message) for message in history.messages),
                    episodic_summary="\n\n".join(dict.fromkeys(episodic_parts)),
                    image_descriptions=image_descriptions,
                    pinned_models=pinned_models,
                    agent_run_plan=agent_run_plan,
                    valid_modes=effective_modes,
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
                fingerprint=self._model_fingerprint_for_role(role),
                profile=profiles[role],
                reasoning_settings=model_reasoning_settings(self._models.model_settings(role)),
            )
            for role in CHAT_MODEL_SELECTORS
        )


__all__ = [
    "AnswerHistoryResource",
    "AnswerInputArtifact",
    "HistoryResolver",
    "AnswerRequest",
    "AnswerRunAcceptor",
    "AnswerRuntimeUnavailableError",
    "AnswerService",
]
