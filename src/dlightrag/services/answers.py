# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable answer runs over already-authorized canonical workspaces."""

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, aclosing
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag_ai.capacity import (
    CONTEXT_POLICY,
    CONTEXT_POLICY_REVISION,
    ModelProfile,
)
from dlightrag_ai.catalog import MODEL_CATALOG_REVISION
from dlightrag_ai.fingerprints import ModelFingerprint
from dlightrag_ai.settings import MODEL_ROLE_NAMES, ModelRole
from dlightrag_rag.retrieval import MetadataFilter, RetrievalResult
from dlightrag_rag.retrieval.planner import RetrievalPlanner
from dlightrag_rag.sourcing.source_contract import safe_source_filename
from dlightrag_rag.workspaces import require_canonical_workspace_id

from dlightrag.answer.agent.orchestrator import research_history_input_measure
from dlightrag.answer.capabilities import AnswerCapabilities, RequestModelContext
from dlightrag.answer.capability import AnswerImageCapability
from dlightrag.answer.errors import AnswerInputOverflowError
from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.executor import ResolvedAnswerResources
from dlightrag.answer.history import (
    HistoryProjectionOverflowError,
    HistoryProjectionTarget,
    project_history,
)
from dlightrag.answer.images import AnswerImagePolicy
from dlightrag.answer.resources.images import QueryImageDescriber, prepare_query_images
from dlightrag.answer.resources.models import ResourceInput, TextWindowBudget
from dlightrag.answer.runs.execution import (
    AnswerRunInput,
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    PinnedModelProfile,
    build_current_answer_resources,
    in_memory_attachment_loader,
)
from dlightrag.answer.runs.results import AnswerResult, restore_answer_result
from dlightrag.answer.synthesizer import AnswerSynthesizer
from dlightrag.answer.tools import compose_research_tools
from dlightrag.runtime import (
    AnswerRunCancelledError,
    AnswerRunEvent,
    AnswerRunFailedError,
    AnswerRunRecord,
    ArtifactReferenceKind,
    CancellationOutcome,
    PendingArtifact,
    PendingArtifactReference,
    RunArtifactReference,
    RunCreation,
    answer_run_request_fingerprint,
    artifact_digest,
)

#: Accepted input uploads, in the precedence one ordinal resolves against.
_INPUT_REFERENCE_KINDS: tuple[ArtifactReferenceKind, ...] = (
    "current_attachment",
    "history_attachment",
)


@dataclass(frozen=True, slots=True)
class AnswerHistoryResource:
    """One accepted upload carried from an owned prior run into this request."""

    run_id: str
    source_ordinal: int
    digest: str
    filename: str
    mime_type: str
    byte_size: int


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
    top_k: int | None = None
    chunk_top_k: int | None = None
    filters: MetadataFilter | None = None
    semantic_highlights: bool = False
    resources: tuple[ResourceInput, ...] = ()
    history_resources: tuple[AnswerHistoryResource, ...] = ()


@dataclass(frozen=True, slots=True)
class AnswerInputArtifact:
    """One accepted run input upload, read back with its stored bytes."""

    reference_kind: ArtifactReferenceKind
    ordinal: int
    filename: str
    mime_type: str
    digest: str
    content: bytes


class AnswerRuntimeUnavailableError(RuntimeError):
    """Raised before acceptance when no local durable-run scheduler is active."""


class AnswerRunAcceptor[T](Protocol):
    """Persist or replay one prepared run, optionally with an atomic domain link."""

    async def create_run(
        self,
        *,
        owner_id: str,
        prepared_input: Mapping[str, Any],
        idempotency_fingerprint: str,
        idempotency_key: str | None = None,
        resources: Sequence[Mapping[str, Any]] = (),
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
    ) -> T | None: ...

    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
    ) -> T | None: ...


class _AnswerRunRepository(AnswerRunAcceptor[RunCreation], Protocol):
    """The owner-scoped durable operations this service performs."""

    async def get_run(self, *, owner_id: str, run_id: str) -> AnswerRunRecord | None: ...

    async def list_runs(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[AnswerRunRecord, ...]: ...

    async def request_cancellation(self, *, owner_id: str, run_id: str) -> CancellationOutcome: ...

    async def list_run_artifacts(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunArtifactReference, ...]: ...

    async def load_artifact(self, *, owner_id: str, digest: str) -> bytes | None: ...


class _RunScheduler(Protocol):
    """The started coordinator accepted runs execute and stream through."""

    @property
    def is_started(self) -> bool: ...

    def admission(self) -> AbstractAsyncContextManager[bool]: ...

    def wake(self) -> None: ...

    def cancel_local(self, owner_id: str, run_id: str) -> None: ...

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncGenerator[AnswerRunEvent]: ...


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

    def current_profiles(self) -> dict[ModelRole, ModelProfile]: ...

    def request_model_context(
        self, pinned: Mapping[ModelRole, ModelProfile] | None, /
    ) -> RequestModelContext: ...

    def answer_image_policy(self, profile: ModelProfile, /) -> AnswerImagePolicy: ...

    async def confirmed_live_answer_context(
        self, models: RequestModelContext, /
    ) -> tuple[RequestModelContext, AnswerImageCapability | None]: ...


class _QueryImageRuntime(Protocol):
    """The model runtime acceptance describes current-turn images with."""

    def query_image_describer(self) -> QueryImageDescriber: ...


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
        text_window_budget: TextWindowBudget,
        confirm_image_context: Callable[
            [RequestModelContext],
            Awaitable[tuple[RequestModelContext, AnswerImageCapability | None]],
        ],
    ) -> ResolvedAnswerResources: ...


@dataclass(frozen=True, slots=True)
class _AcceptanceProjection:
    history: tuple[Mapping[str, Any], ...]
    image_descriptions: tuple[str, ...]
    pinned_models: tuple[PinnedModelProfile, ...]


def _attachment_bytes(resources: Sequence[ResourceInput]) -> list[bytes]:
    """Return the inline bytes an accepted run must persist with its input."""
    return [resource.content for resource in resources if resource.content is not None]


def _prepared_input_payload(run_input: Any) -> dict[str, Any]:
    """Encode the M3 prepared input: pinned session id plus the run input."""
    from dlightrag_agent.session.ids import SessionId

    return {
        "session_id": SessionId.new().value,
        **run_input.as_request(),
    }


def _require_prepared_input_bounds(prepared_input: Mapping[str, Any]) -> None:
    from dlightrag_agent.session.effects import canonical_json

    from dlightrag.answer.prepared_input import MAX_PREPARED_INPUT_BYTES, PreparedInputTooLargeError

    encoded = canonical_json(dict(prepared_input)).encode("utf-8")
    if len(encoded) > MAX_PREPARED_INPUT_BYTES:
        raise PreparedInputTooLargeError(encoded_bytes=len(encoded))


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
        top_k=request.top_k,
        chunk_top_k=request.chunk_top_k,
        filters=(
            request.filters.model_dump(exclude_none=True, mode="json") if request.filters else None
        ),
        semantic_highlights=request.semantic_highlights,
        links=tuple(links),
        attachments=tuple(attachments),
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
        coordinator: _RunScheduler,
        retrieval: _RetrievalPlanning,
        capabilities: _AnswerCapabilityPlanner,
        capability_view: _AnswerCapabilityReader,
        models: _QueryImageRuntime,
        resources: _AnswerResourcePreparer,
        model_fingerprint_for_role: Callable[[ModelRole], ModelFingerprint],
    ) -> None:
        self._store = store
        self._coordinator = coordinator
        self._retrieval = retrieval
        self._capabilities = capabilities
        self._capability_view = capability_view
        self._models = models
        self._resources = resources
        self._model_fingerprint_for_role = model_fingerprint_for_role

    async def create(
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        idempotency_key: str | None = None,
    ) -> RunCreation:
        """Accept one durable run and return its descriptor without waiting.

        A keyed replay is resolved before any link is materialized or any input
        is prepared, so a retried submission never repeats acceptance work. The
        accepted run outlives this call and is read back through :meth:`get`,
        :meth:`subscribe`, and :meth:`cancel`.
        """
        creation = await self._accept(
            request=request,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=None,
            acceptor=self._store,
        )
        if creation is None:
            raise RuntimeError("Answer run acceptance returned no descriptor")
        return creation

    async def accept[T](
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        acceptor: AnswerRunAcceptor[T],
    ) -> T | None:
        """Accept through a typed atomic linker while preserving one run pipeline."""
        return await self._accept(
            request=request,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
            acceptor=acceptor,
        )

    async def _accept[T](
        self,
        *,
        request: AnswerRequest,
        owner_id: str,
        idempotency_key: str | None,
        idempotency_fingerprint: str | None,
        acceptor: AnswerRunAcceptor[T],
    ) -> T | None:
        run_request = _normalized_request(request)
        fingerprint = idempotency_fingerprint or answer_run_request_fingerprint(
            run_request.as_request()
        )
        if idempotency_key is not None:
            replay = await acceptor.replay_run(
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=fingerprint,
            )
            if replay is not None:
                return replay
        if not self._coordinator.is_started:
            raise AnswerRuntimeUnavailableError("Answer runtime is unavailable")
        run_request, attachment_bytes = await self._resources.pin_current_image_links(
            run_request,
            _attachment_bytes(request.resources),
        )
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
        run_input = await self._prepare_input(
            run_request,
            resources=acceptance_resources or None,
            idempotency_fingerprint=fingerprint,
        )
        prepared_input = _prepared_input_payload(run_input)
        _require_prepared_input_bounds(prepared_input)
        resources_payload = _accepted_resource_payloads(
            run_input, attachment_bytes=attachment_bytes
        )
        async with self._coordinator.admission() as runtime_available:
            if not runtime_available:
                raise AnswerRuntimeUnavailableError("Answer runtime is unavailable")
            accepted = await acceptor.create_run(
                owner_id=owner_id,
                prepared_input=prepared_input,
                idempotency_fingerprint=fingerprint,
                idempotency_key=idempotency_key,
                resources=resources_payload,
                artifacts=[PendingArtifact(content=content) for content in attachment_bytes],
                references=_artifact_references(run_input),
            )
            if accepted is not None:
                self._coordinator.wake()
        return accepted

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
            )
            if artifact is None:
                raise AnswerRuntimeUnavailableError("Accepted answer input artifact is unavailable")
            return artifact.content

        return ResourceInput(
            filename=resource.filename,
            declared_mime=resource.mime_type,
            loader=load,
        )

    async def list(
        self, *, owner_id: str, after_run_id: str | None = None, limit: int = 50
    ) -> tuple[AnswerRunRecord, ...]:
        """List this owner's runs oldest-first after an optional cursor."""
        return await self._store.list_runs(
            owner_id=owner_id, after_run_id=after_run_id, limit=limit
        )

    async def list_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]:
        """List artifact descriptors for one owned run."""
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
        """Read a published artifact; omit length to return the remainder."""
        refs = await self._store.list_run_artifacts(owner_id=owner_id, run_id=run_id)
        match = next((item for item in refs if item.resource_id == resource_id), None)
        if match is None:
            return None
        blob = await self._store.load_artifact(owner_id=owner_id, digest=match.digest)
        if blob is None:
            return None
        start = max(0, offset)
        if length is None:
            return blob[start:]
        return blob[start : start + max(0, length)]

    async def get(self, *, owner_id: str, run_id: str) -> AnswerRunRecord | None:
        """Read one owned run; unknown and foreign identifiers both return ``None``."""
        return await self._store.get_run(owner_id=owner_id, run_id=run_id)

    async def cancel(self, *, owner_id: str, run_id: str) -> CancellationOutcome:
        """Request cancellation; only this mutates a run on a caller's behalf.

        A pending running cancellation also signals this process's local task
        immediately after the durable commit, so the owner observes the stop
        before the next heartbeat (Task 5 same-process signal).
        """
        outcome = await self._store.request_cancellation(owner_id=owner_id, run_id=run_id)
        if outcome.outcome == "pending":
            self._coordinator.cancel_local(owner_id, run_id)
        return outcome

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncGenerator[AnswerRunEvent]:
        """Follow one owned run's durable events; detaching never cancels it."""
        return self._coordinator.subscribe(
            owner_id=owner_id, run_id=run_id, after_sequence=after_sequence
        )

    async def wait(self, *, owner_id: str, run_id: str) -> AnswerResult:
        """Follow one owned run to its terminal state and project its result.

        Cancelling this wait detaches this observer only; use :meth:`cancel` to
        stop the run itself.
        """
        async with aclosing(self.subscribe(owner_id=owner_id, run_id=run_id)) as events:
            async for _event in events:
                pass
        final = await self._store.get_run(owner_id=owner_id, run_id=run_id)
        if final is None:
            raise AnswerRunFailedError(
                "answer_run_missing",
                "Answer run disappeared before it finished.",
            )
        if final.status == "succeeded":
            return restore_answer_result(final.result or {})
        if final.status == "cancelled":
            raise AnswerRunCancelledError(final.run_id)
        raise AnswerRunFailedError(
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
        return await self.wait(owner_id=creation.run.owner_id, run_id=creation.run.run_id)

    async def answer_stream(
        self,
        request: AnswerRequest,
        *,
        owner_id: str,
        idempotency_key: str | None = None,
    ) -> AsyncGenerator[AnswerRunEvent]:
        """Create one durable answer run and follow its events until it ends."""
        creation = await self.create(
            request=request,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
        )
        run = creation.run
        async with aclosing(self.subscribe(owner_id=run.owner_id, run_id=run.run_id)) as events:
            async for event in events:
                yield event

    async def capabilities(self) -> AnswerCapabilities:
        """Return the public image-capability snapshot after its allowed re-probe."""
        return await self._capability_view.read()

    async def read_input_artifact(
        self, *, owner_id: str, run_id: str, ordinal: int
    ) -> AnswerInputArtifact | None:
        """Read one owned run's accepted input upload by its ordinal.

        Resources a worker fetched mid-run are run state, never accepted input,
        so they are not readable here; a current-turn upload takes precedence
        over a history upload sharing its ordinal.
        """
        references = await self._store.list_run_artifacts(owner_id=owner_id, run_id=run_id)
        reference = next(
            (
                item
                for kind in _INPUT_REFERENCE_KINDS
                for item in references
                if item.reference_kind == kind and item.ordinal == ordinal
            ),
            None,
        )
        if reference is None:
            return None
        content = await self._store.load_artifact(owner_id=owner_id, digest=reference.digest)
        if content is None:
            return None
        return AnswerInputArtifact(
            reference_kind=reference.reference_kind,
            ordinal=reference.ordinal,
            filename=reference.filename,
            mime_type=reference.mime_type,
            digest=reference.digest,
            content=content,
        )

    async def _prepare_input(
        self,
        request: AnswerRunRequest,
        *,
        resources: list[ResourceInput] | None,
        idempotency_fingerprint: str,
    ) -> AnswerRunInput:
        """Resolve one normalized request into immutable durable run input."""
        projection = await self._project_acceptance(request, resources=resources)
        return AnswerRunInput(
            query=request.query,
            workspaces=request.workspaces,
            history=projection.history,
            top_k=request.top_k,
            chunk_top_k=request.chunk_top_k,
            filters=request.filters,
            semantic_highlights=request.semantic_highlights,
            links=request.links,
            attachments=request.attachments,
            history_attachments=request.history_attachments,
            pinned_models=projection.pinned_models,
            context_policy_revision=CONTEXT_POLICY_REVISION,
            model_catalog_revision=MODEL_CATALOG_REVISION,
            idempotency_fingerprint=idempotency_fingerprint,
            image_descriptions=projection.image_descriptions,
        )

    async def _project_acceptance(
        self,
        request: AnswerRunRequest,
        *,
        resources: list[ResourceInput] | None,
    ) -> _AcceptanceProjection:
        """Resolve the exact shared-history envelopes without building the run rig."""
        if resources:
            await self._capabilities.refresh_vlm()
        model_profiles = self._capabilities.current_profiles()
        models = self._capabilities.request_model_context(model_profiles)
        planner = self._retrieval.planner_for(models.extract)
        text_window_budget = TextWindowBudget(CONTEXT_POLICY.hard_input_limit(models.query))
        resolved = await self._resources.resolve(
            resources,
            models=models,
            text_window_budget=text_window_budget,
            confirm_image_context=self._capabilities.confirmed_live_answer_context,
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
            targets = [
                HistoryProjectionTarget(
                    "planner",
                    models.extract,
                    planner.history_input_measure(
                        request.query,
                        schema=schema,
                        current_image_descriptions=list(image_descriptions) or None,
                        preserve_query=True if resolved.research else None,
                    ),
                )
            ]
            if resolved.research:
                evidence = EvidenceLedger(image_budget=resolved.image_budget)

                async def unused_retrieve(_query: str) -> RetrievalResult:
                    raise RuntimeError("acceptance tool definitions are never executed")

                tools = compose_research_tools(
                    evidence=evidence,
                    trace={},
                    retrieve_knowledge_base=unused_retrieve,
                    search_web=(
                        resolved.web_search.search if resolved.web_search is not None else None
                    ),
                    resource_tools=resolved.resource_tools,
                    register_web_source=(
                        resolved.registry.register_discovered_link
                        if resolved.registry is not None and resolved.web_search is not None
                        else None
                    ),
                )
                measure = research_history_input_measure(
                    model_profile=models.query,
                    context_policy=CONTEXT_POLICY,
                    query=request.query,
                    query_images=resolved.query_images,
                    resource_manifest=resolved.resource_manifest,
                    image_budget=resolved.image_budget,
                    tools=tools,
                    retained_tail_tokens=CONTEXT_POLICY.retained_tail_target(models.query),
                )
                targets.append(
                    HistoryProjectionTarget(
                        "research_seed",
                        models.query,
                        measure,
                        proactive_compaction=True,
                    )
                )
            else:
                synthesizer = AnswerSynthesizer(
                    image_policy=self._capabilities.answer_image_policy(models.query),
                    model_profile=models.query,
                    context_policy=CONTEXT_POLICY,
                    model_func=None,
                )
                targets.append(
                    HistoryProjectionTarget(
                        "fast_generation",
                        models.query,
                        synthesizer.history_input_measure(request.query),
                    )
                )
            try:
                history = project_history(
                    [dict(message) for message in request.history],
                    targets=targets,
                )
            except HistoryProjectionOverflowError as exc:
                raise AnswerInputOverflowError(str(exc)) from exc
            return _AcceptanceProjection(
                history=tuple(dict(message) for message in history.messages),
                image_descriptions=image_descriptions,
                pinned_models=self._pin_model_profiles(model_profiles),
            )
        finally:
            if resolved.registry is not None:
                await resolved.registry.aclose()

    def _pin_model_profiles(
        self,
        profiles: Mapping[ModelRole, ModelProfile],
    ) -> tuple[PinnedModelProfile, ...]:
        return tuple(
            PinnedModelProfile(
                role=role,
                fingerprint=self._model_fingerprint_for_role(role),
                profile=profiles[role],
            )
            for role in MODEL_ROLE_NAMES
        )


__all__ = [
    "AnswerHistoryResource",
    "AnswerInputArtifact",
    "AnswerRequest",
    "AnswerRunAcceptor",
    "AnswerRuntimeUnavailableError",
    "AnswerService",
]
