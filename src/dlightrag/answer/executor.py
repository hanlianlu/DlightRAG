# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer execution over durable Runtime sessions and RAG workspaces."""

import asyncio
import base64
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from dlightrag_agent.session.effects import (
    EffectIntent,
    EffectSettlement,
    ToolResultEntry,
    canonical_json,
)
from dlightrag_agent.session.entries import (
    AssistantMessageEntry,
    EffectIntentEntry,
    EffectResultEntry,
    ProfileFactEntry,
    SessionEntry,
    UserMessageEntry,
)
from dlightrag_agent.session.fold import PriorTurns
from dlightrag_agent.session.ids import EntryId, ProjectionId, SessionId, StageIntentId
from dlightrag_agent.session.projection import ContextProjection, TokenAnchor
from dlightrag_agent.session.store import (
    AgentSessionStore,
    EffectAlreadySettled,
    EffectCommit,
    EffectContractChanged,
    EffectMissing,
    EvidenceConflict,
    LeaseLost,
    SessionCommit,
    SettleCommit,
    VersionConflict,
)
from dlightrag_agent.tools import AgentTool, ExecutedTurn
from dlightrag_ai.capacity import CONTEXT_POLICY, CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag_ai.scheduler import model_call_scope
from dlightrag_ai.settings import MODEL_ROLE_NAMES, ModelRole
from dlightrag_ai.telemetry import Telemetry, safe_log_text
from dlightrag_rag.lifecycle import defer_cancellation
from dlightrag_rag.pool import WorkspacePool
from dlightrag_rag.retrieval import (
    MetadataFilter,
    RetrievalContexts,
    RetrievalResult,
)
from dlightrag_rag.sourcing.source_contract import safe_source_filename
from dlightrag_rag.sourcing.url import afetch_public_https_bytes, avalidate_public_https_url

from dlightrag.answer.agent.orchestrator import AnswerOrchestrator
from dlightrag.answer.capabilities import AnswerCapabilityCoordinator, RequestModelContext
from dlightrag.answer.capability import AnswerImageCapability, check_answer_image_capability
from dlightrag.answer.citations.finalization import finalize_answer
from dlightrag.answer.citations.streaming import aclose_answer_stream
from dlightrag.answer.errors import (
    AnswerInputError,
    AnswerModelCapabilityError,
    AnswerResourceAdmissionError,
    CurrentImagePayloadError,
    InvalidToolConfigurationError,
    classify_answer_error,
)
from dlightrag.answer.highlights import SemanticHighlightSettings, enrich_semantic_highlights
from dlightrag.answer.images import AnswerImageBudget
from dlightrag.answer.media import answer_images_from_sources
from dlightrag.answer.mode import ModeResource, resource_role
from dlightrag.answer.model_runtime import AnswerModelRuntime
from dlightrag.answer.publication import is_empty_answer
from dlightrag.answer.resources import ResourceInput, ResourceRegistry
from dlightrag.answer.resources.models import (
    ResourceManifestEntry,
    ResourceRegistryError,
    TextWindowBudget,
)
from dlightrag.answer.resources.registry import FetchedBytesSink, FetchedResourceBytes
from dlightrag.answer.resources.visual import ResourceInspector
from dlightrag.answer.router import AnswerModeRouter, RoutingFailedError
from dlightrag.answer.routing import decide_resolved_mode
from dlightrag.answer.runs.execution import (
    AnswerRunInput,
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    build_current_answer_resources,
)
from dlightrag.answer.runs.results import store_answer_result
from dlightrag.answer.sources import project_contexts_for_client
from dlightrag.answer.tools.resources import build_resource_tools, make_resource_reader
from dlightrag.answer.tools.web import ExaSearch
from dlightrag.answer.workspace import (
    WorkspaceIntegrityError,
    WorkspaceRecoveryFailed,
    bind_run_workspace,
)
from dlightrag.runtime import (
    LeaseLostError,
    RunCancelledError,
    RunExecutionError,
    RunSession,
)
from dlightrag.runtime.blob_chunks import blob_digest, plan_blob
from dlightrag.runtime.progress import RunProgressStore, StageCommit
from dlightrag.runtime.records import PendingPublication, artifact_digest
from dlightrag.runtime.settlements import (
    CompleteBlobDescriptor,
    EvidenceSettlementUpdate,
    FetchedResourceSettlementUpdate,
    M3HostUpdate,
    OpaqueEvidenceWrite,
    OpaqueFetchedResourceWrite,
)

logger = logging.getLogger(__name__)


class ArtifactReader(Protocol):
    """Read one owner-scoped complete blob by digest (executor store surface)."""

    async def load_artifact(self, *, owner_id: str, digest: str) -> bytes | None: ...
    async def list_run_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]: ...


class RawRetrieval(Protocol):
    async def __call__(
        self,
        query: str,
        *,
        workspaces: Sequence[str],
        conversation_history: Sequence[Mapping[str, object]] | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        bm25_query: str | None = None,
        filters: MetadataFilter | None = None,
        query_images: Sequence[Mapping[str, Any]] = (),
        image_descriptions: Sequence[str] = (),
        preserve_query: bool | None = None,
        model_profile: ModelProfile | None = None,
    ) -> RetrievalResult: ...


@dataclass(frozen=True, slots=True)
class AnswerResourceSettings:
    max_attachments: int
    max_attachment_bytes: int
    max_total_attachment_bytes: int
    image_max_bytes: int
    image_max_pixels: int


@dataclass(frozen=True, slots=True)
class AnswerExecutorSettings:
    default_top_k: int
    default_chunk_top_k: int
    semantic_highlights: SemanticHighlightSettings


@dataclass(frozen=True)
class OrchestratorRun:
    """One durable request resolved into an orchestrator and its exact inputs."""

    orchestrator: AnswerOrchestrator
    image_descriptions: list[str]
    query_images: list[dict[str, Any]] | None
    history: PriorTurns
    current_image_count: int
    workspaces: list[str]
    registry: ResourceRegistry | None


@dataclass(frozen=True, slots=True)
class ResolvedAnswerResources:
    models: RequestModelContext
    web_search: ExaSearch | None
    registry: ResourceRegistry | None
    resource_tools: list[AgentTool]
    resource_manifest: tuple[ResourceManifestEntry, ...]
    current_images: list[dict[str, Any]]
    current_image_count: int
    research: bool
    image_budget: AnswerImageBudget | None
    query_images: list[dict[str, Any]] | None


class IncompatibleActiveRunError(RuntimeError):
    """An accepted run cannot execute under this binary's Answer contract."""


class AnswerResourceResolver:
    """Resolve request resources, visual policy, and peer tools exactly once."""

    def __init__(
        self,
        *,
        settings: AnswerResourceSettings,
        models: AnswerModelRuntime,
        capabilities: AnswerCapabilityCoordinator,
    ) -> None:
        self._settings = settings
        self._models = models
        self._capabilities = capabilities

    async def pin_current_image_links(
        self,
        request: AnswerRunRequest,
        attachment_bytes: Sequence[bytes],
    ) -> tuple[AnswerRunRequest, list[bytes]]:
        """Materialize declared image links once for acceptance and durable replay."""
        if len(request.attachments) != len(attachment_bytes):
            raise ValueError("current attachment references and bytes must have equal length")
        image_count = sum(
            1
            for attachment in request.attachments
            if attachment.mime_type.strip().casefold().startswith("image/")
        ) + sum(
            1
            for link in request.links
            if (link.mime_type or "").strip().casefold().startswith("image/")
        )
        if image_count:
            capabilities = await self._capabilities.refresh_answer()
            check_answer_image_capability(
                image_count=image_count,
                capability=capabilities.answer,
            )
        links: list[LinkReference] = []
        pinned_link_attachments: list[AttachmentReference] = []
        pinned_link_bytes: list[bytes] = []
        for link in request.links:
            if not (link.mime_type or "").strip().casefold().startswith("image/"):
                links.append(link)
                continue
            data = await self.materialize_link_image(link.url)
            try:
                if data is None:
                    raise ValueError("image link did not materialize")
                mime_type, _data_uri = await asyncio.to_thread(
                    _verified_current_image_data_uri,
                    data,
                    max_pixels=self._settings.image_max_pixels,
                )
            except ValueError:
                links.append(replace(link, mime_type=None))
                continue
            pinned_link_attachments.append(
                AttachmentReference(
                    digest=artifact_digest(data),
                    filename=safe_source_filename(link.filename),
                    mime_type=mime_type,
                    ordinal=len(pinned_link_attachments),
                    byte_size=len(data),
                )
            )
            pinned_link_bytes.append(data)
        offset = len(pinned_link_attachments)
        attachments = [
            *pinned_link_attachments,
            *(
                replace(attachment, ordinal=offset + index)
                for index, attachment in enumerate(request.attachments)
            ),
        ]
        return (
            replace(
                request,
                links=tuple(links),
                attachments=tuple(attachments),
            ),
            [*pinned_link_bytes, *attachment_bytes],
        )

    async def resolve(
        self,
        resources: list[ResourceInput] | None,
        *,
        models: RequestModelContext,
        text_window_budget: TextWindowBudget,
        confirm_image_context: Callable[
            [RequestModelContext],
            Awaitable[tuple[RequestModelContext, AnswerImageCapability | None]],
        ],
        fetched_bytes_sink: FetchedBytesSink | None = None,
        research: bool | None = None,
    ) -> ResolvedAnswerResources:
        """Resolve resource capabilities, manifests, tools, and image transport."""
        if resources and not models.query.supports_tools:
            raise AnswerModelCapabilityError()
        declared_image_count = sum(
            1
            for resource in resources or ()
            if resource.loader is None
            and (resource.declared_mime or "").lower().startswith("image/")
        )
        image_capability: AnswerImageCapability | None = None
        if declared_image_count:
            models, image_capability = await confirm_image_context(models)
            check_answer_image_capability(
                image_count=declared_image_count,
                capability=image_capability,
            )
        (
            current_images,
            remaining_resources,
            current_image_resources,
        ) = await self.prepare_current_images(resources)
        if current_images and not declared_image_count:
            models, image_capability = await confirm_image_context(models)
        check_answer_image_capability(
            image_count=len(current_images),
            capability=image_capability,
        )

        web_search = self._models.web_search()
        registry, resource_tools = self.build_resource_context(
            remaining_resources,
            text_window_budget=text_window_budget,
            web_search=web_search,
            fetched_bytes_sink=fetched_bytes_sink,
            vlm_profile=models.vlm,
        )
        try:
            current_image_resource_ids = (
                tuple(registry.register(resource) for resource in current_image_resources)
                if registry is not None
                else ()
            )
            resource_manifest = registry.manifest() if registry is not None else ()
            inferred = web_search is not None or bool(resource_manifest)
            research = inferred if research is None else research
            image_budget: AnswerImageBudget | None = None
            query_images: list[dict[str, Any]] | None = current_images or None
            if research:
                image_budget = self._capabilities.answer_image_policy(models.query).new_budget()
                query_images = (
                    await self.budget_agent_images(
                        current_images,
                        image_budget,
                        current_image_resource_ids,
                    )
                    or None
                )
            return ResolvedAnswerResources(
                models=models,
                web_search=web_search,
                registry=registry,
                resource_tools=resource_tools,
                resource_manifest=resource_manifest,
                current_images=current_images,
                current_image_count=len(current_images),
                research=research,
                image_budget=image_budget,
                query_images=query_images,
            )
        except BaseException:
            if registry is not None:
                await registry.aclose()
            raise

    async def prepare_current_images(
        self,
        resources: list[ResourceInput] | None,
    ) -> tuple[list[dict[str, Any]], list[ResourceInput], list[ResourceInput]]:
        """Build verified current-image blocks while retaining attachments as resources."""
        if not resources:
            return [], [], []
        images: list[dict[str, Any]] = []
        remaining: list[ResourceInput] = []
        image_resources: list[ResourceInput] = []
        for resource in resources:
            data: bytes | None = None
            if resource.loader is not None:
                remaining.append(resource)
                continue
            if resource.content is not None:
                data = resource.content
            elif resource.url is not None and (resource.declared_mime or "").lower().startswith(
                "image/"
            ):
                data = await self.materialize_link_image(resource.url)
            if data is None:
                remaining.append(resource)
                continue
            try:
                mime, data_uri = await asyncio.to_thread(
                    _verified_current_image_data_uri,
                    data,
                    max_pixels=self._settings.image_max_pixels,
                )
            except ValueError:
                remaining.append(resource)
                continue
            images.append({"type": "image_url", "image_url": {"url": data_uri}})
            image_resource = ResourceInput(
                filename=resource.filename,
                content=data,
                declared_mime=mime,
            )
            remaining.append(image_resource)
            image_resources.append(image_resource)
        return images, remaining, image_resources

    async def materialize_link_image(self, url: str) -> bytes | None:
        """Fetch one current-image link under SSRF revalidation."""
        try:
            await avalidate_public_https_url(url)
            return await afetch_public_https_bytes(
                url,
                max_bytes=self._settings.image_max_bytes,
                timeout=120.0,
            )
        except Exception:
            logger.warning("Failed to materialize current image link", exc_info=True)
            return None

    def build_resource_context(
        self,
        resources: list[ResourceInput] | None,
        *,
        text_window_budget: TextWindowBudget,
        web_search: ExaSearch | None = None,
        fetched_bytes_sink: FetchedBytesSink | None = None,
        vlm_profile: ModelProfile,
    ) -> tuple[ResourceRegistry | None, list[AgentTool]]:
        """Register resources and compose their text and visual peer tools."""
        if not resources and web_search is None:
            return None, []
        registry = ResourceRegistry(
            max_attachments=self._settings.max_attachments,
            max_attachment_bytes=self._settings.max_attachment_bytes,
            max_total_attachment_bytes=self._settings.max_total_attachment_bytes,
            url_text_fallback=(web_search.contents_text if web_search is not None else None),
            fetched_bytes_sink=fetched_bytes_sink,
        )
        try:
            for resource in resources or []:
                registry.register(resource)
        except (ValueError, ResourceRegistryError) as exc:
            raise AnswerResourceAdmissionError() from exc

        vlm_policy = self._capabilities.vlm_image_policy(vlm_profile)
        visual_supported = vlm_profile.supports_images and vlm_policy.max_images > 0
        inspector = (
            ResourceInspector(
                registry,
                vlm_func=self._models.vlm_func(),
                image_policy=vlm_policy,
            )
            if visual_supported
            else None
        )
        tools = build_resource_tools(
            registry,
            text_window_budget=text_window_budget,
            inspector=inspector,
            visual_supported=visual_supported,
        )
        return registry, tools

    @staticmethod
    async def budget_agent_images(
        current_images: list[dict[str, Any]],
        budget: AnswerImageBudget,
        resource_ids: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        def build() -> list[dict[str, Any]]:
            blocks: list[dict[str, Any]] = []
            for index, image in enumerate(current_images, start=1):
                block = budget.add_user_image(image, label=f"query_image_{index}")
                if block is None:
                    raise CurrentImagePayloadError(
                        f"current image query_image_{index} could not fit the answer image budget"
                    )
                if index <= len(resource_ids):
                    blocks.append(
                        {
                            "type": "text",
                            "text": (
                                f"[current image {index} | resource: {resource_ids[index - 1]}]"
                            ),
                        }
                    )
                blocks.append(block)
            return blocks

        return await asyncio.to_thread(build)


class AnswerExecutor:
    """Execute durable Answer runs without composition or storage dependencies."""

    def __init__(
        self,
        *,
        store: ArtifactReader,
        pool: WorkspacePool,
        retrieve: RawRetrieval,
        models: AnswerModelRuntime,
        capabilities: AnswerCapabilityCoordinator,
        resources: AnswerResourceResolver,
        settings: AnswerExecutorSettings,
        telemetry: Telemetry,
        execution_environment: str = "disabled",
        workspace_root: str | None = None,
        working_dir: str = "./dlightrag_storage",
    ) -> None:
        self._store = store
        self._pool = pool
        self._retrieve_result = retrieve
        self._models = models
        self._capabilities = capabilities
        self._resources = resources
        self._settings = settings
        self._telemetry = telemetry
        self._execution_environment = execution_environment
        self._workspace_root_setting = workspace_root
        self._working_dir = working_dir

    async def execute(self, session: RunSession) -> Mapping[str, Any]:
        with model_call_scope((session.owner_id, session.run_id)):
            try:
                return await self._execute(session)
            except (
                asyncio.CancelledError,
                RunCancelledError,
                LeaseLostError,
                RunExecutionError,
            ):
                raise
            except Exception as exc:
                logger.warning("Answer run %s execution failed", session.run_id, exc_info=True)
                message = (
                    exc.public_message
                    if isinstance(exc, AnswerInputError | InvalidToolConfigurationError)
                    and exc.public_message
                    else "Answer run failed."
                )
                raise RunExecutionError(classify_answer_error(exc), message) from exc

    async def _ensure_resolved_mode(
        self, session: RunSession, request: AnswerRunInput
    ) -> tuple[str | None, str | None]:
        load = getattr(self._store, "load_routing", None)
        resolve = getattr(self._store, "resolve", None)
        if not inspect.iscoroutinefunction(load) or not inspect.iscoroutinefunction(resolve):
            return None, None
        record = await load(owner_id=session.owner_id, run_id=session.run_id)  # type: ignore[misc]
        if record is None:
            raise RunExecutionError("routing_failed", "Routing record is missing.")
        if record.resolved_mode:
            return str(record.resolved_mode), record.research_session_id
        try:
            decided = decide_resolved_mode(
                requested_mode=record.requested_mode,
                valid_modes=frozenset(record.valid_modes),
            )
        except ValueError as exc:
            raise RunExecutionError("routing_failed", "Answer mode routing failed.") from exc
        if decided is None:
            decided = await self._route_with_model(request, valid_modes=record.valid_modes)
        research_session_id = None
        if decided == "research" and record.requested_mode == "auto":
            research_session_id = SessionId.new().value
        written = await resolve(  # type: ignore[misc]
            owner_id=session.owner_id,
            run_id=session.run_id,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
            resolved_mode=decided,
            research_session_id=research_session_id,
        )
        return written or decided, research_session_id or record.research_session_id

    async def _route_with_model(
        self, request: AnswerRunInput, *, valid_modes: tuple[str, ...]
    ) -> str:
        model, _telemetry = self._models.new_highlight_model()
        router = AnswerModeRouter(model)
        resources = tuple(
            ModeResource(role=resource_role(filename=item.filename, mime_type=item.mime_type))
            for item in (*request.attachments, *request.history_attachments)
        )
        tools = ["search_knowledge_base"]
        if self._models.web_search() is not None:
            tools.append("search_web")
        try:
            return await router.choose(
                query=request.query,
                history=request.history,
                resources=resources,
                tool_categories=tools,
                has_images=any(item.role == "image" for item in resources),
                valid_modes=valid_modes,
            )
        except RoutingFailedError as exc:
            raise RunExecutionError("routing_failed", "Answer mode routing failed.") from exc

    async def _execute(self, session: RunSession) -> Mapping[str, Any]:
        request = AnswerRunInput.from_prepared_input(session.prepared_input)
        model_profiles = self.validate_pinned_model_profiles(request)
        await session.enter_phase("routing")
        resolved_mode, research_session_id = await self._ensure_resolved_mode(session, request)
        await session.enter_phase("planning")
        projected_history = PriorTurns([dict(message) for message in request.history])

        boundaries: JournalRunBoundaries | None = None
        fast_boundaries: FastRunBoundaries | None = None

        fetched_buffer: list[FetchedResourceBytes] = []

        run = await self.prepare_orchestrated_run(
            workspaces=list(request.workspaces),
            top_k=request.top_k,
            chunk_top_k=request.chunk_top_k,
            filters=MetadataFilter.model_validate(request.filters) if request.filters else None,
            resources=await self._answer_run_resources(request, owner_id=session.owner_id),
            fetched_bytes_sink=_buffered_fetched_bytes_sink(fetched_buffer),
            research=None if resolved_mode is None else resolved_mode == "research",
            pinned_image_descriptions=request.image_descriptions,
            projected_history=projected_history,
            model_profiles=model_profiles,
        )
        stream: AsyncIterator[str] | None = None
        try:
            journal = session.execution.session_store
            prepared_early: Any = None
            research = (
                run.orchestrator.uses_research_path
                if resolved_mode is None
                else resolved_mode == "research"
            )
            if research:
                from dlightrag.answer.execution_settings import validate_agent_execution

                root = validate_agent_execution(
                    execution_environment=self._execution_environment,
                    workspace_root=self._workspace_root_setting,
                    working_dir=self._working_dir,
                )
                if root is not None:
                    try:
                        bound = await bind_run_workspace(
                            workspace_root=root,
                            owner_id=session.owner_id,
                            run_id=session.run_id,
                            fencing_epoch=session.execution.fencing_epoch,
                            recorded_epoch=session.workspace_epoch,
                            store=session.execution.workspace_store,
                        )
                    except WorkspaceRecoveryFailed as exc:
                        raise RunExecutionError("workspace_recovery_failed", str(exc)) from exc
                    except WorkspaceIntegrityError as exc:
                        raise RunExecutionError("workspace_integrity_error", str(exc)) from exc
                    run.orchestrator.bind_workspace(bound)
            if research:
                session_id = SessionId(
                    request.session_id or research_session_id or SessionId.new().value
                )
                prepared_early = run.orchestrator.prepare_run(
                    request.query,
                    conversation_history=run.history,
                    query_images=run.query_images,
                    registry=run.registry,
                )
                snapshot = await journal.load(session_id)
                if snapshot.version == 0:
                    snapshot = await self._seed_session(journal, session_id, request, snapshot)
                else:
                    await run.orchestrator.recover_from_fold(prepared_early, snapshot)
                    await _adopt_durable_evidence(prepared_early, journal, session_id)
                boundaries = JournalRunBoundaries(
                    session=session,
                    journal=journal,
                    session_id=session_id,
                    tools_by_name={tool.name: tool for tool in prepared_early.tools},
                    ledger_state=lambda: prepared_early.evidence.ledger_state_json(),
                    fetched_buffer=fetched_buffer,
                    run_id=session.run_id,
                    initial_version=snapshot.version,
                )
                if snapshot.version > 0:
                    await boundaries.recover_pending_intents(snapshot)
            else:
                fast_boundaries = FastRunBoundaries(
                    session=session,
                    progress=session.execution.progress_store,
                    run_id=session.run_id,
                    plan={
                        "query": request.query,
                        "workspaces": list(request.workspaces),
                        "top_k": request.top_k,
                        "chunk_top_k": request.chunk_top_k,
                    },
                )
                await fast_boundaries.settle_planner()

            async with self._telemetry.observe(
                "answer_orchestration",
                as_type="chain",
                input={"query": request.query},
                metadata={
                    "run_id": session.run_id,
                    "research": research,
                    "workspaces": run.workspaces,
                    "history_turns": len(run.history or []),
                    "query_image_count": run.current_image_count,
                    "semantic_highlights": request.semantic_highlights,
                },
            ) as pipeline_trace:
                prepared = prepared_early
                limit = boundaries if boundaries is not None else fast_boundaries
                contexts, stream = await run.orchestrator.answer_stream(
                    request.query,
                    conversation_history=run.history,
                    run=prepared,
                    boundaries=limit,  # type: ignore[arg-type]
                )
                answer_parts: list[str] = []
                if stream is not None:
                    async for chunk in stream:
                        answer_parts.append(chunk)
                        await session.emit_token(chunk)
                await session.flush_tokens()
                answer_text = getattr(stream, "answer", "") or "".join(answer_parts)
                finalized = finalize_answer(answer_text, contexts)
                if request.semantic_highlights:
                    finalized.sources = await enrich_semantic_highlights(
                        finalized.sources,
                        answer_text=finalized.answer,
                        settings=self._settings.semantic_highlights,
                        model_factory=self._models.new_highlight_model,
                    )
                trace = dict(getattr(stream, "trace", None) or {})
                trace["query_image_description_count"] = len(run.image_descriptions)
                images = answer_images_from_sources(finalized.sources, contexts=contexts)
                pipeline_trace.update(
                    output=answer_trace_output(
                        finalized.answer,
                        finalized.sources,
                        contexts,
                        capture_sensitive_data=self._telemetry.capture_sensitive_data,
                    )
                )
                publications, primary_handle, artifact_descriptors, report_sources = (
                    _stage_publications(
                        staged=run.orchestrator.staged_artifacts(),
                        answer=finalized.answer,
                        contexts=contexts,
                        require_answer=getattr(prepared_early, "stop_reason", None) == "model_stop",
                    )
                )
                session.pending_publications = publications
                stored = store_answer_result(
                    answer=finalized.answer,
                    contexts=project_contexts_for_client(contexts),
                    sources=finalized.sources,
                    answer_images=images,
                    trace=trace,
                    image_descriptions=run.image_descriptions,
                    primary_report=primary_handle,
                    artifacts=artifact_descriptors,
                    report_sources=report_sources,
                )
                if fast_boundaries is not None:
                    await fast_boundaries.settle_retrieval(contexts)
                    await fast_boundaries.settle_final(
                        result=stored,
                        result_digest=canonical_json(stored),
                    )
                return stored
        finally:
            await _close_execution_resources(stream, run.registry)

    async def _seed_session(
        self,
        journal: AgentSessionStore[M3HostUpdate],
        session_id: SessionId,
        request: AnswerRunInput,
        snapshot: Any,
    ) -> Any:
        """Append pinned history, objective, and profile facts atomically (M3-D25)."""
        entries: list[SessionEntry] = [
            UserMessageEntry(
                entry_id=EntryId.new(),
                session_id=session_id,
                timestamp=_entry_timestamp(),
                content={"text": turn.get("content", "")},
            )
            for turn in request.history
        ]
        entries.append(
            ProfileFactEntry(
                entry_id=EntryId.new(),
                session_id=session_id,
                timestamp=_entry_timestamp(),
                key="objective",
                value=request.query,
            )
        )
        entries.append(
            ProfileFactEntry(
                entry_id=EntryId.new(),
                session_id=session_id,
                timestamp=_entry_timestamp(),
                key="profile_facts",
                value={
                    "workspaces": list(request.workspaces),
                    "context_policy_revision": request.context_policy_revision,
                    "model_catalog_revision": request.model_catalog_revision,
                },
            )
        )
        initial = ContextProjection(
            projection_id=ProjectionId.new(),
            first_retained_sequence=1,
            covered_through_sequence=0,
            summary=None,
            token_anchors=(
                TokenAnchor(through_sequence=0, measured_input_tokens=0, measured_output_tokens=0),
            ),
        )
        commit = await journal.append(
            session_id=session_id, expected_version=0, entries=entries, projection=initial
        )
        if not isinstance(commit, SessionCommit):
            raise RunExecutionError(
                "run_execution_failed", "Cannot seed the pinned research session journal."
            )
        return await journal.load(session_id)

    async def prepare_orchestrated_run(
        self,
        *,
        workspaces: list[str],
        top_k: int | None,
        chunk_top_k: int | None,
        filters: MetadataFilter | None,
        resources: list[ResourceInput] | None,
        fetched_bytes_sink: FetchedBytesSink | None = None,
        pinned_image_descriptions: tuple[str, ...],
        projected_history: PriorTurns,
        model_profiles: Mapping[ModelRole, ModelProfile],
        environment: object | None = None,
        research: bool | None = None,
    ) -> OrchestratorRun:
        history = projected_history
        models = self._capabilities.request_model_context(model_profiles)
        query_profile = models.query
        if not workspaces:
            raise ValueError("an Answer run requires at least one workspace")
        warmup = asyncio.create_task(self._pool.warm(workspaces))
        warmup.add_done_callback(_observe_workspace_warmup)
        text_window_budget = TextWindowBudget(CONTEXT_POLICY.hard_input_limit(query_profile))
        resolved = await self._resources.resolve(
            resources,
            models=models,
            text_window_budget=text_window_budget,
            confirm_image_context=self._capabilities.pinned_answer_context,
            fetched_bytes_sink=fetched_bytes_sink,
            research=research,
        )
        try:
            models = resolved.models
            query_profile = models.query
            image_descriptions = list(pinned_image_descriptions)

            async def retrieve_knowledge_base(search_query: str) -> RetrievalResult:
                return await self._retrieve_result(
                    search_query,
                    workspaces=workspaces,
                    conversation_history=history.messages,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    filters=filters,
                    query_images=resolved.current_images,
                    image_descriptions=image_descriptions,
                    preserve_query=True if resolved.research else None,
                    model_profile=models.extract,
                )

            model_func: Callable[..., Any] | None = None
            stream_model_func: Callable[..., AsyncIterator[str]] | None = None
            if resolved.research:
                tool_model = self._models.query_tool_model()
                model_func = tool_model
                stream_model_func = tool_model.stream_text

            orchestrator = AnswerOrchestrator(
                synthesizer=self._models.answer_synthesizer(query_profile),
                retrieve_knowledge_base=retrieve_knowledge_base,
                search_web=(
                    resolved.web_search.search if resolved.web_search is not None else None
                ),
                model_func=model_func,
                stream_model_func=stream_model_func,
                resource_tools=resolved.resource_tools,
                resource_manifest=resolved.resource_manifest,
                register_web_source=(
                    resolved.registry.register_discovered_link
                    if resolved.registry is not None and resolved.web_search is not None
                    else None
                ),
                image_budget=resolved.image_budget,
                text_window_budget=text_window_budget,
                model_profile=query_profile,
                context_policy=CONTEXT_POLICY,
                telemetry=self._telemetry,
                environment=environment,
                research_path=research,
                resource_reader=(
                    make_resource_reader(resolved.registry, text_window_budget)
                    if resolved.registry is not None
                    else None
                ),
            )
            return OrchestratorRun(
                orchestrator=orchestrator,
                image_descriptions=image_descriptions,
                query_images=resolved.query_images,
                history=history,
                current_image_count=resolved.current_image_count,
                workspaces=workspaces,
                registry=resolved.registry,
            )
        except BaseException:
            if resolved.registry is not None:
                await resolved.registry.aclose()
            raise

    async def _answer_run_resources(
        self,
        request: AnswerRunInput,
        *,
        owner_id: str,
    ) -> list[ResourceInput] | None:
        if not request.links and not request.attachments and not request.history_attachments:
            return None

        async def load(digest: str) -> bytes:
            content = await self._store.load_artifact(owner_id=owner_id, digest=digest)
            if content is None:
                raise RunExecutionError(
                    "run_execution_failed",
                    "Answer run attachment bytes no longer exist.",
                )
            return content

        def loader(digest: str) -> Callable[[], Awaitable[bytes]]:
            async def read() -> bytes:
                return await load(digest)

            return read

        resources = await build_current_answer_resources(
            links=request.links,
            attachments=request.attachments,
            attachment_loaders=[loader(attachment.digest) for attachment in request.attachments],
        )
        resources.extend(
            ResourceInput(
                filename=attachment.filename,
                declared_mime=attachment.mime_type,
                loader=loader(attachment.digest),
            )
            for attachment in request.history_attachments
        )
        return resources

    async def _load_corpus_image(self, workspace: str, chunk_id: str) -> str | None:
        try:
            runtime = await self._pool.acquire(workspace)
            asset = await runtime.aget_visual_asset(chunk_id, size="full")
        except Exception:
            logger.info(
                "Knowledge-base visual for '%s' no longer resolves; dropping the image block",
                safe_log_text(chunk_id),
            )
            return None
        content = getattr(asset, "data", None)
        return base64.b64encode(content).decode("ascii") if content else None

    @staticmethod
    def validate_pinned_model_profiles(
        request: AnswerRunInput,
    ) -> dict[ModelRole, ModelProfile]:
        if request.context_policy_revision != CONTEXT_POLICY_REVISION:
            raise IncompatibleActiveRunError(
                "answer run context policy revision does not match this binary; "
                "drain active runs before deployment"
            )
        pinned = {item.role: item for item in request.pinned_models}
        if len(request.pinned_models) != len(MODEL_ROLE_NAMES) or set(pinned) != set(
            MODEL_ROLE_NAMES
        ):
            raise IncompatibleActiveRunError(
                "answer run does not contain the complete pinned model role set"
            )
        return {role: pinned[role].profile for role in MODEL_ROLE_NAMES}


def _buffered_fetched_bytes_sink(
    buffer: list[FetchedResourceBytes],
) -> FetchedBytesSink:
    """Buffer fetched bytes until the turn's settlements make them durable."""

    async def persist(fetched: FetchedResourceBytes) -> None:
        buffer.append(fetched)

    return persist


class JournalRunBoundaries:
    """Journal each complete turn and settle every intent in source order.

    One append transaction commits the assistant entry, all valid intents, and
    deterministic validation-result entries (M3-D26); each effect then settles
    one at a time in assistant source order (M3-D12), with the turn's durable
    evidence/fetched-resource host updates attached to the final settlement.
    """

    def __init__(
        self,
        *,
        session: RunSession,
        journal: AgentSessionStore[M3HostUpdate],
        session_id: SessionId,
        tools_by_name: Mapping[str, AgentTool],
        ledger_state: Callable[[], str],
        fetched_buffer: list[FetchedResourceBytes],
        run_id: str,
        initial_version: int = 0,
    ) -> None:
        self._session = session
        self._journal = journal
        self._session_id = session_id
        self._tools_by_name = tools_by_name
        self._ledger_state = ledger_state
        self._fetched_buffer = fetched_buffer
        self._run_id = run_id
        self._version = initial_version

    async def recover_pending_intents(self, snapshot: Any) -> None:
        """Settle intents a crash left unsettled, per their pinned policy.

        Committed effects fold and never execute again; unsettled ``safe``
        effects replay only when tool name, replay policy, contract version,
        and schema digest all match, unsettled ``never`` effects settle
        interrupted, and a changed contract settles ``tool_contract_changed``
        (M3-D13, 3C recovery).
        """
        import json as _json

        settled_ids = {
            entry.intent_id
            for entry in snapshot.entries
            if isinstance(entry, EffectResultEntry) and entry.intent_id is not None
        }
        for entry in snapshot.entries:
            if not isinstance(entry, EffectIntentEntry):
                continue
            intent = entry.intent
            if intent.intent_id in settled_ids:
                continue
            tool = self._tools_by_name.get(intent.tool_name)
            contract_matches = (
                tool is not None
                and tool.replay_policy == intent.replay_policy
                and tool.contract_version == intent.contract_version
                and tool.input_schema_digest == intent.input_schema_digest
            )
            progress: Literal["live", "prelude"] = "prelude"
            if intent.replay_policy == "safe" and contract_matches:
                if tool is None:
                    raise RuntimeError("matched contract lost its tool")
                try:
                    arguments = tool.input_model.model_validate(_json.loads(intent.canonical_input))
                    result = await tool.execute(arguments)
                    outcome = "succeeded"
                    content = result.content
                    cached = result.cached
                except Exception as exc:
                    outcome = "succeeded"
                    content = f'Tool "{intent.tool_name}" failed: {exc}'
                    cached = False
                progress = "live"
            elif intent.replay_policy == "safe":
                outcome = "tool_contract_changed"
                content = f'Tool "{intent.tool_name}" contract changed; result discarded.'
                cached = False
            else:
                outcome = "interrupted"
                content = f'Tool "{intent.tool_name}" was interrupted before it settled.'
                cached = False
            await self._settle_intent_recovery(
                intent,
                outcome=outcome,
                content=content,
                cached=cached,
                progress=progress,
            )

    async def _settle_intent_recovery(
        self,
        intent: EffectIntent,
        *,
        outcome: str,
        content: str,
        cached: bool,
        progress: Literal["live", "prelude"],
    ) -> None:
        result_entry = EffectResultEntry(
            entry_id=EntryId.new(),
            session_id=self._session_id,
            timestamp=_entry_timestamp(),
            intent_id=intent.intent_id,
            result=ToolResultEntry(
                tool_name=intent.tool_name,
                call_id=intent.source_call_id or "",
                outcome=outcome,  # type: ignore[arg-type]
                content=content,
                cached=cached,
            ),
        )
        committed = await self._journal.settle_effect(
            session_id=self._session_id,
            expected_version=self._version,
            intent_id=intent.intent_id,
            settlement=EffectSettlement(
                outcome=outcome,  # type: ignore[arg-type]
                result=result_entry.result,
                host_update=EvidenceSettlementUpdate(),
            ),
            entries=[result_entry],
            progress=progress,
        )
        await self._handle_settlement(committed, intent)

    @property
    def version(self) -> int:
        return self._version

    async def enter_phase(self, phase: str) -> None:
        await self._session.enter_phase(phase)  # type: ignore[arg-type]

    async def check_cancelled(self) -> None:
        await self._session.check_cancelled()

    async def commit_turn(self, executed: ExecutedTurn, *, turn_number: int) -> None:
        assistant = _assistant_entry(executed, self._session_id)
        intents = tuple(
            EffectIntentEntry(
                entry_id=EntryId.new(),
                session_id=self._session_id,
                timestamp=_entry_timestamp(),
                intent=intent,
            )
            for intent in executed.intents
        )
        validation = tuple(
            EffectResultEntry(
                entry_id=EntryId.new(),
                session_id=self._session_id,
                timestamp=_entry_timestamp(),
                result=result,
            )
            for result in executed.validation_results
        )
        commit = await self._journal.append(
            session_id=self._session_id,
            expected_version=self._version,
            entries=(assistant, *intents, *validation),
        )
        if isinstance(commit, (VersionConflict, LeaseLost)):
            raise LeaseLostError
        self._version = commit.version

        intent_list = list(executed.intents)
        for position, intent in enumerate(intent_list):
            await self._settle_intent(
                executed,
                intent,
                turn_number=turn_number,
                is_last=position == len(intent_list) - 1,
            )

    async def _settle_intent(
        self,
        executed: ExecutedTurn,
        intent: EffectIntent,
        *,
        turn_number: int,
        is_last: bool,
    ) -> None:
        execution = next(
            (result for result in executed.results if result.call.id == intent.source_call_id),
            None,
        )
        if execution is None:
            outcome: str = "interrupted"
            content = f'Tool "{intent.tool_name}" was interrupted before it settled.'
            cached = False
        else:
            outcome = "succeeded"
            content = execution.result.content
            cached = execution.result.cached
        result_entry = EffectResultEntry(
            entry_id=EntryId.new(),
            session_id=self._session_id,
            timestamp=_entry_timestamp(),
            intent_id=intent.intent_id,
            result=ToolResultEntry(
                tool_name=intent.tool_name,
                call_id=intent.source_call_id or "",
                outcome="succeeded" if outcome == "succeeded" else "interrupted",
                content=content,
                details=None if execution is None else execution.result.details,
                cached=cached,
            ),
        )
        settlement = EffectSettlement(
            outcome=outcome,  # type: ignore[arg-type]
            result=result_entry.result,
            host_update=self._host_update(intent, is_last=is_last),
        )
        committed = await self._journal.settle_effect(
            session_id=self._session_id,
            expected_version=self._version,
            intent_id=intent.intent_id,
            settlement=settlement,
            entries=[result_entry],
        )
        await self._handle_settlement(committed, intent)

    def _host_update(self, intent: EffectIntent, *, is_last: bool) -> M3HostUpdate:
        if not is_last:
            return EvidenceSettlementUpdate()
        updates: list[M3HostUpdate] = []
        if self._ledger_state() != "{}":
            content = self._ledger_state().encode("utf-8")
            updates.append(
                EvidenceSettlementUpdate(
                    evidence=(
                        OpaqueEvidenceWrite(
                            session_id=self._session_id.value,
                            intent_id=intent.intent_id.value,
                            result_ordinal=0,
                            content_digest=blob_digest(content),
                            locator_digest=blob_digest(b"{}"),
                            content=content,
                            locator=b"{}",
                        ),
                    )
                )
            )
        for fetched in self._fetched_buffer:
            plan = plan_blob(fetched.content)
            updates.append(
                FetchedResourceSettlementUpdate(
                    resource=OpaqueFetchedResourceWrite(
                        resource_id=fetched.resource_id,
                        safe_name=fetched.filename,
                        media_type=fetched.mime_type,
                        capabilities={},
                        blob_digest=plan.digest,
                        source_locator_digest=blob_digest(fetched.url.encode("utf-8")),
                        source_locator=fetched.url.encode("utf-8"),
                        session_id=self._session_id.value,
                        intent_id=intent.intent_id.value,
                    ),
                    complete_blob=CompleteBlobDescriptor(
                        digest=plan.digest,
                        total_bytes=plan.total_bytes,
                        chunks=tuple(
                            plan.chunk(fetched.content, index) for index in range(plan.chunk_count)
                        ),
                    ),
                )
            )
        self._fetched_buffer.clear()
        if len(updates) == 1:
            return updates[0]
        if len(updates) > 1:
            raise RunExecutionError(
                "run_execution_failed",
                "A turn settlement cannot carry both evidence and fetched resources",
            )
        return EvidenceSettlementUpdate()

    async def _handle_settlement(self, committed: SettleCommit, intent: EffectIntent) -> None:
        if isinstance(committed, EffectCommit):
            self._version = committed.version
            return
        if isinstance(committed, (VersionConflict, LeaseLost)):
            raise LeaseLostError
        if isinstance(committed, EffectAlreadySettled):
            # Load and fold the committed settlement; never re-execute.
            snapshot = await self._journal.load(self._session_id)
            self._version = snapshot.version
            return
        if isinstance(committed, EffectContractChanged):
            settled = await self._journal.settle_effect(
                session_id=self._session_id,
                expected_version=self._version,
                intent_id=intent.intent_id,
                settlement=EffectSettlement(
                    outcome="tool_contract_changed",
                    result=ToolResultEntry(
                        tool_name=intent.tool_name,
                        call_id=intent.source_call_id or "",
                        outcome="tool_contract_changed",
                        content=f'Tool "{intent.tool_name}" contract changed; result discarded.',
                    ),
                    host_update=EvidenceSettlementUpdate(),
                ),
                entries=[
                    EffectResultEntry(
                        entry_id=EntryId.new(),
                        session_id=self._session_id,
                        timestamp=_entry_timestamp(),
                        intent_id=intent.intent_id,
                        result=ToolResultEntry(
                            tool_name=intent.tool_name,
                            call_id=intent.source_call_id or "",
                            outcome="tool_contract_changed",
                            content=f'Tool "{intent.tool_name}" contract changed; result discarded.',
                        ),
                    )
                ],
            )
            if isinstance(settled, EffectCommit):
                self._version = settled.version
                return
            raise LeaseLostError
        if isinstance(committed, EvidenceConflict):
            raise RunExecutionError(
                "evidence_settlement_conflict",
                "Evidence identity conflict during settlement.",
            )
        if isinstance(committed, EffectMissing):
            raise RunExecutionError(
                "run_execution_failed", "Journal lost a committed effect intent."
            )
        raise RunExecutionError("run_execution_failed", "Unknown settlement outcome.")


class FastRunBoundaries:
    """Durable three-stage Fast boundaries (planner, retrieval, final_generation)."""

    def __init__(
        self,
        *,
        session: RunSession,
        progress: RunProgressStore,
        run_id: str,
        plan: Mapping[str, Any],
    ) -> None:
        self._session = session
        self._progress = progress
        self._run_id = run_id
        self._plan = plan
        self._progress_version = 0

    async def enter_phase(self, phase: str) -> None:
        await self._session.enter_phase(phase)  # type: ignore[arg-type]

    async def check_cancelled(self) -> None:
        await self._session.check_cancelled()

    async def commit_turn(self, executed: ExecutedTurn, *, turn_number: int) -> None:
        raise AssertionError("Fast Answers never commit agent turns")

    async def settle_planner(self) -> None:
        stage_id = StageIntentId.deterministic(run_id=self._run_id, name="fast:planner:0")
        committed = await self._progress.settle_stage(
            expected_progress_version=self._progress_version,
            stage_intent_id=stage_id,
            stage_name="planner",
            state=dict(self._plan),
            evidence=(),
        )
        await self._observe(committed)

    async def settle_retrieval(self, contexts: Any) -> None:
        stage_id = StageIntentId.deterministic(run_id=self._run_id, name="fast:retrieval:1")
        committed = await self._progress.settle_stage(
            expected_progress_version=self._progress_version,
            stage_intent_id=stage_id,
            stage_name="retrieval",
            state={"contexts": _contexts_summary(contexts)},
            evidence=(),
        )
        await self._observe(committed)

    async def settle_final(self, *, result: Mapping[str, Any], result_digest: str) -> None:
        stage_id = StageIntentId.deterministic(run_id=self._run_id, name="fast:final_generation:2")
        terminal = getattr(self._progress, "settle_terminal", None)
        if terminal is not None:
            committed = await terminal(
                expected_progress_version=self._progress_version,
                stage_intent_id=stage_id,
                state={"result_digest": result_digest},
                result=result,
            )
            await self._observe(committed)
            return
        committed = await self._progress.settle_stage(
            expected_progress_version=self._progress_version,
            stage_intent_id=stage_id,
            stage_name="final_generation",
            state={"result_digest": result_digest},
            evidence=(),
        )
        await self._observe(committed)

    async def _observe(self, committed: Any) -> None:
        if isinstance(committed, StageCommit):
            self._progress_version = committed.progress_version
            return
        raise LeaseLostError


def _assistant_entry(executed: ExecutedTurn, session_id: SessionId) -> AssistantMessageEntry:
    assistant = executed.assistant
    return AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_entry_timestamp(),
        content=assistant.text,
        stop_reason=assistant.stop_reason,
        reasoning=assistant.reasoning,
        tool_calls=assistant.tool_calls,
        usage=assistant.usage_details,
        cost=assistant.cost_details,
        provider_state=assistant.provider_state,
    )


def _contexts_summary(contexts: Any) -> list[dict[str, Any]]:
    return [
        {
            "kind": kind,
            "rows": len(contexts.get(kind, []) or []),
        }
        for kind in ("chunks", "entities", "relationships")
    ]


async def _close_execution_resources(
    stream: AsyncIterator[str] | None,
    registry: ResourceRegistry | None,
) -> None:
    cancellation: asyncio.CancelledError | None = None
    try:
        await aclose_answer_stream(stream)
    except asyncio.CancelledError as exc:
        cancellation = defer_cancellation(cancellation, exc)
    except Exception:
        logger.warning("Failed to close Answer stream", exc_info=True)
    if registry is not None:
        try:
            await registry.aclose()
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        except Exception:
            logger.warning("Failed to close Answer resource registry", exc_info=True)
    if cancellation is not None:
        raise cancellation


def _verified_current_image_data_uri(data: bytes, *, max_pixels: int) -> tuple[str, str]:
    from dlightrag_ai.media import image_bytes_to_data_uri, verify_web_image_bytes

    mime = verify_web_image_bytes(data, max_pixels=max_pixels)
    return mime, image_bytes_to_data_uri(data, fallback_mime=mime)


def _context_count(contexts: RetrievalContexts, key: str) -> int:
    items = contexts.get(key, [])
    return len(items) if isinstance(items, list) else 0


def _stage_publications(
    *,
    staged: Sequence[Any],
    answer: str,
    contexts: RetrievalContexts,
    require_answer: bool = False,
) -> tuple[list[PendingPublication], str | None, list[dict[str, Any]], list[Any]]:
    has_report = any(item.kind == "primary_report" for item in staged)
    if require_answer and is_empty_answer(answer=answer, has_primary_report=has_report):
        raise RunExecutionError("empty_answer", "The run produced no answer.")
    publications: list[PendingPublication] = []
    descriptors: list[dict[str, Any]] = []
    primary_handle: str | None = None
    report_sources: list[Any] = []
    for item in staged:
        payload = item.path.read_bytes()
        if item.kind == "primary_report":
            cleaned = finalize_answer(payload.decode("utf-8"), contexts)
            payload = cleaned.answer.encode("utf-8")
            report_sources = list(cleaned.sources)
            primary_handle = "primary_report"
            resource_id = "primary_report"
        else:
            resource_id = f"artifact-{item.relative_path.replace('/', '-')}"
        publications.append(
            PendingPublication(
                resource_id=resource_id,
                reference_kind=item.kind,
                filename=item.relative_path,
                mime_type=item.media_type,
                content=payload,
            )
        )
        descriptors.append(
            {
                "resource_id": resource_id,
                "kind": item.kind,
                "filename": item.relative_path,
                "media_type": item.media_type,
                "size_bytes": len(payload),
            }
        )
    return publications, primary_handle, descriptors, report_sources


def answer_trace_output(
    answer: str | None,
    sources: Sequence[Any] | None,
    contexts: RetrievalContexts,
    *,
    capture_sensitive_data: bool = False,
) -> dict[str, Any]:
    """Shape what a pipeline observation reports for one Answer result."""
    output: dict[str, Any] = {
        "answer_len": len(answer or ""),
        "source_count": len(sources or []),
        "context_chunk_count": _context_count(contexts, "chunks"),
    }
    if capture_sensitive_data:
        output["answer"] = answer or ""
    return output


def _observe_workspace_warmup(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    if error := task.exception():
        logger.debug("Workspace warm-up failed", exc_info=error)


__all__ = [
    "AnswerExecutor",
    "AnswerExecutorSettings",
    "AnswerResourceResolver",
    "AnswerResourceSettings",
    "IncompatibleActiveRunError",
    "OrchestratorRun",
    "ResolvedAnswerResources",
    "answer_trace_output",
]


def _entry_timestamp() -> datetime:
    return datetime.now(UTC)


def _agent_turn_count_from_snapshot(snapshot: Any) -> int:
    """Count complete assistant turns from the folded journal."""
    return sum(
        1 for entry in snapshot.entries if entry.__class__.__name__ == "AssistantMessageEntry"
    )


async def _adopt_durable_evidence(prepared: Any, journal: Any, session_id: SessionId) -> None:
    """Adopt the latest durable evidence state into the live ledger (recovery)."""
    loader = getattr(journal, "load_evidence", None)
    if loader is None or prepared is None:
        return
    writes = await loader(session_id)
    if not writes:
        return
    import json as _json

    latest = writes[-1]
    prepared.evidence.adopt_ledger_state(_json.loads(latest.content.decode("utf-8")))
