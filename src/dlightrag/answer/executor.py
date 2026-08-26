# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer execution over durable Runtime sessions and RAG workspaces."""

import asyncio
import base64
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from dlightrag_memory import Memory, MemoryStore

from dlightrag.agent.environment import (
    ExecutionEnvironment,
    resolve_execution_adapter,
)
from dlightrag.agent.extensions import TrustedExtensions
from dlightrag.agent.session.effects import (
    EffectIntent,
    ToolResultEntry,
    canonical_json,
)
from dlightrag.agent.session.fold import PriorTurns, project_session_messages
from dlightrag.agent.session.ids import (
    AttemptId,
    IntentId,
    LaneId,
    OperationId,
    SessionId,
    StageIntentId,
)
from dlightrag.agent.session.operation import (
    OperationCancelled,
    OperationCompleted,
    OperationFailed,
    ToolBatchItem,
)
from dlightrag.agent.session.plan import AgentRunPlan
from dlightrag.agent.session.registers import PendingInput, RequestSnapshot
from dlightrag.agent.session.runtime import (
    AgentOperationCancelled,
    AgentSessionEvent,
    AgentSessionRuntime,
    FollowUpCommand,
    OperationEffectFailed,
    ProviderAttemptFailed,
    ProviderContextOverflow,
    RuntimeContext,
    SessionLeaseLostError,
    SteerCommand,
    ToolEffectResult,
)
from dlightrag.agent.session.store import (
    AgentSessionStore,
)
from dlightrag.agent.session.transactions import TransactionCommit
from dlightrag.agent.tools import (
    AgentTool,
    ToolEffects,
    ToolResult,
    ToolRuntime,
    fit_tool_result,
)
from dlightrag.ai.capacity import CONTEXT_POLICY, CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.ai.fingerprints import ModelFingerprint
from dlightrag.ai.messages import AssistantTurn
from dlightrag.ai.providers.base import is_provider_context_overflow
from dlightrag.ai.scheduler import model_call_scope
from dlightrag.ai.settings import MODEL_ROLE_NAMES, ModelRole
from dlightrag.ai.telemetry import Telemetry, safe_log_text
from dlightrag.answer.agent.orchestrator import AnswerOrchestrator
from dlightrag.answer.capabilities import AnswerCapabilityCoordinator, RequestModelContext
from dlightrag.answer.capability import AnswerImageCapability, check_answer_image_capability
from dlightrag.answer.citations.finalization import finalize_answer
from dlightrag.answer.citations.streaming import aclose_answer_stream
from dlightrag.answer.errors import (
    AnswerInputError,
    AnswerResourceAdmissionError,
    CurrentImagePayloadError,
    InvalidToolConfigurationError,
    classify_answer_error,
)
from dlightrag.answer.highlights import SemanticHighlightSettings, enrich_semantic_highlights
from dlightrag.answer.images import AnswerImageBudget
from dlightrag.answer.media import evidence_images_from_sources
from dlightrag.answer.memory import memory_owner_allowed, render_auto_recall
from dlightrag.answer.mode import ModeResource, ResolvedMode, resource_role
from dlightrag.answer.model_runtime import AnswerModelRuntime
from dlightrag.answer.publication import (
    PublicationLimits,
    PublicationPlan,
    is_empty_answer,
    validate_publication,
)
from dlightrag.answer.resources import ResourceInput, ResourceRegistry
from dlightrag.answer.resources.models import (
    ResourceManifestEntry,
    ResourceRegistryError,
    TextWindowBudget,
)
from dlightrag.answer.resources.registry import (
    FetchedBytesSink,
    FetchedResourceBytes,
    ResourceEffectOwner,
)
from dlightrag.answer.resources.visual import ResourceInspector
from dlightrag.answer.router import AnswerModeRouter, RoutingFailedError
from dlightrag.answer.routing import AnswerRoutingStore, decide_resolved_mode
from dlightrag.answer.runs.execution import (
    AnswerRunInput,
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    build_current_answer_resources,
)
from dlightrag.answer.runs.results import store_answer_result
from dlightrag.answer.session_host import FastSessionHost
from dlightrag.answer.sources import project_contexts_for_client
from dlightrag.answer.tools.memory import MemoryHost
from dlightrag.answer.tools.resources import build_resource_tools, make_resource_reader
from dlightrag.answer.tools.subagents import (
    ChildContextSnapshot,
    ChildOutcome,
    ChildRequest,
    SpawnAgentInput,
    SubagentHost,
)
from dlightrag.answer.tools.web import ExaSearch
from dlightrag.answer.workspace import (
    WorkspaceIntegrityError,
    WorkspaceRecoveryFailed,
    bind_run_workspace,
)
from dlightrag.rag.lifecycle import defer_cancellation
from dlightrag.rag.pool import WorkspacePool
from dlightrag.rag.retrieval import (
    MetadataFilter,
    RetrievalContexts,
    RetrievalResult,
)
from dlightrag.rag.sourcing.source_contract import safe_source_filename
from dlightrag.rag.sourcing.url import afetch_public_https_bytes, avalidate_public_https_url
from dlightrag.runtime import (
    ANSWER_RUN_LEASE_SECONDS,
    LeaseLostError,
    RunCancelledError,
    RunExecutionError,
    RunSession,
)
from dlightrag.runtime.blob_chunks import blob_digest, plan_blob
from dlightrag.runtime.progress import RunProgressStore, StageCommit
from dlightrag.runtime.records import PendingPublication, artifact_digest
from dlightrag.runtime.settlements import (
    CommittedSpillUpdate,
    CompleteBlobDescriptor,
    EffectHostUpdate,
    FetchedResourceSettlementUpdate,
    InventoryPathRecord,
    MemoryOperationSettlement,
    OpaqueEvidenceWrite,
    OpaqueFetchedResourceWrite,
    WorkspaceInventoryUpdate,
)

logger = logging.getLogger(__name__)

_CHILD_LEASE_HEARTBEAT_SECONDS = ANSWER_RUN_LEASE_SECONDS / 3


class ArtifactReader(Protocol):
    """Stream one owner-scoped blob by digest (executor store surface)."""

    def stream_artifact(
        self,
        *,
        owner_id: str,
        digest: str,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes]: ...
    async def list_run_artifacts(self, *, owner_id: str, run_id: str) -> tuple[Any, ...]: ...


class AnswerExecutionStore(ArtifactReader, AnswerRoutingStore, Protocol):
    """Executor store: artifacts plus the lease-fenced Routing Record."""


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
    publication: PublicationLimits = PublicationLimits()


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
        resolved_mode: ResolvedMode,
    ) -> ResolvedAnswerResources:
        """Resolve resource capabilities, manifests, tools, and image transport."""
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
            image_budget: AnswerImageBudget | None = None
            query_images: list[dict[str, Any]] | None = current_images or None
            if resolved_mode == "research":
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


async def _memory_recall_allowed(
    checker: Callable[..., Awaitable[bool]] | None, *, owner_id: str
) -> bool:
    """Whether answer injection may use this owner's memory.

    A missing checker means the composition has no settings store and memory
    stays enabled — the historical default.
    """
    return checker is None or await checker(owner_id=owner_id)


class AnswerExecutor:
    """Execute durable Answer runs without composition or storage dependencies."""

    def __init__(
        self,
        *,
        store: AnswerExecutionStore,
        pool: WorkspacePool,
        retrieve: RawRetrieval,
        models: AnswerModelRuntime,
        capabilities: AnswerCapabilityCoordinator,
        resources: AnswerResourceResolver,
        settings: AnswerExecutorSettings,
        telemetry: Telemetry,
        model_fingerprint_for_role: Callable[[ModelRole], ModelFingerprint],
        execution_environment: str = "disabled",
        workspace_root: str | None = None,
        working_dir: str = "./dlightrag_storage",
        memory_store: MemoryStore | None = None,
        memory_recall_enabled: Callable[..., Awaitable[bool]] | None = None,
        memory_capability_current: Callable[..., Awaitable[bool]] | None = None,
        external_tools: tuple[AgentTool, ...] = (),
        trusted_extensions: TrustedExtensions | None = None,
        skills_global_root: Path | None = None,
    ) -> None:
        self._store = store
        self._pool = pool
        self._retrieve_result = retrieve
        self._models = models
        self._capabilities = capabilities
        self._resources = resources
        self._settings = settings
        self._telemetry = telemetry
        self._model_fingerprint_for_role = model_fingerprint_for_role
        self._execution_environment = execution_environment
        self._workspace_root_setting = workspace_root
        self._working_dir = working_dir
        self._memory_store = memory_store
        self._memory = Memory(memory_store) if memory_store is not None else None
        self._memory_recall_enabled = memory_recall_enabled
        self._memory_capability_current = memory_capability_current
        self._external_tools = external_tools
        self._trusted_extensions = trusted_extensions or TrustedExtensions()
        self._skills_global_root = skills_global_root
        if execution_environment not in {"disabled", "trust", "sandbox"}:
            raise ValueError(f"unknown agent execution mode: {execution_environment}")
        mode = execution_environment
        extension_adapter = self._trusted_extensions.execution_adapter(mode)  # type: ignore[arg-type]
        self._execution_adapter = resolve_execution_adapter(
            mode,  # type: ignore[arg-type]
            trust=extension_adapter if mode == "trust" else None,
            sandbox=extension_adapter if mode == "sandbox" else None,
        )

    def acceptance_research_tools(self) -> tuple[AgentTool, ...]:
        """Return non-resource definitions execution may expose to Research.

        Acceptance combines these exact factories with request-specific search
        and resource tools. The execute closures are never invoked here.
        """
        from dlightrag.agent.environment import AccessScheduler
        from dlightrag.agent.environment.local import LocalExecutionEnvironment
        from dlightrag.agent.skills import SkillCatalog, SkillMetadata, load_skill_tool
        from dlightrag.agent.tools.files import path_tools, read_tool
        from dlightrag.agent.tools.registry import ToolRegistry
        from dlightrag.answer.tools.memory import forget_tool, recall_memory_tool, remember_tool
        from dlightrag.answer.tools.subagents import subagent_tools

        access = AccessScheduler()
        # Resource reads exist independently of local execution and use the
        # same ReadArgs schema as the runtime's registry-backed read tool.
        tools: list[AgentTool] = [read_tool(None, access), *self._external_tools]
        if self._execution_adapter is not None:
            tools.extend(
                tool
                for tool in path_tools(
                    LocalExecutionEnvironment(Path.cwd()),
                    scheduler=access,
                )
                if tool.name != "read"
            )
        tools.extend(subagent_tools(host=SubagentHost()))
        if self._memory_store is not None:
            host = MemoryHost()
            tools.extend(
                (remember_tool(host=host), forget_tool(host=host), recall_memory_tool(host=host))
            )
        if self._skills_global_root is not None or self._execution_adapter is not None:
            placeholder = SkillMetadata(
                name="__acceptance__",
                description="Schema-only acceptance placeholder.",
                root=Path.cwd(),
                source="global",
            )
            # Membership follows configured roots, not discovered contents, so
            # workspace changes cannot alter an accepted Plan.
            tools.append(load_skill_tool(SkillCatalog((placeholder,))))
        registry = ToolRegistry(tools)
        self._trusted_extensions.register_tools(registry)
        return registry.resolve()

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
    ) -> ResolvedMode:
        record = await self._store.load_routing(owner_id=session.owner_id, run_id=session.run_id)
        if record is None:
            raise RunExecutionError("routing_failed", "Routing record is missing.")
        if record.resolved_mode:
            return _require_resolved_mode(record.resolved_mode)
        try:
            decided = decide_resolved_mode(
                requested_mode=record.requested_mode,
                valid_modes=frozenset(record.valid_modes),
            )
        except ValueError as exc:
            raise RunExecutionError("routing_failed", "Answer mode routing failed.") from exc
        if decided is None:
            decided = _require_resolved_mode(
                await self._route_with_model(request, valid_modes=record.valid_modes)
            )
        written = await self._store.resolve(
            owner_id=session.owner_id,
            run_id=session.run_id,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
            resolved_mode=decided,
        )
        return _require_resolved_mode(written or decided)

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
        resolved_mode = await self._ensure_resolved_mode(session, request)
        await session.enter_phase("planning")
        agent_session_id = SessionId(request.agent_session_id)
        agent_lane_id = LaneId(request.agent_lane_id)
        journal = session.execution.session_store
        canonical_snapshot = await journal.load(agent_session_id)
        if canonical_snapshot.entries:
            history_lane_id = (
                agent_lane_id
                if any(lane.lane_id == agent_lane_id for lane in canonical_snapshot.tree.lanes)
                else LaneId(request.source_lane_id or LaneId.main().value)
            )
            selected_snapshot = replace(
                canonical_snapshot,
                selected_lane_id=history_lane_id,
            )
            projected_history = PriorTurns(
                project_session_messages(
                    canonical_snapshot.tree.ancestry(history_lane_id),
                    selected_snapshot.active_projection,
                )
                if resolved_mode == "fast"
                else []
            )
        else:
            projected_history = PriorTurns(
                [dict(message) for message in request.history],
                episodic_summary=request.episodic_summary,
            )

        fast_boundaries: FastRunBoundaries | None = None
        fast_session_host: FastSessionHost | None = None
        fast_reservation_active = False
        agent_runtime: AgentSessionRuntime[EffectHostUpdate] | None = None
        research_operation_id: OperationId | None = None
        research_plan: AgentRunPlan | None = None
        agent_operations: list[dict[str, Any]] = []

        fetched_buffer = FetchedResourceBuffer()

        run = await self.prepare_orchestrated_run(
            workspaces=list(request.workspaces),
            top_k=request.top_k,
            chunk_top_k=request.chunk_top_k,
            filters=MetadataFilter.model_validate(request.filters) if request.filters else None,
            resources=await self._answer_run_resources(request, owner_id=session.owner_id),
            fetched_bytes_sink=_buffered_fetched_bytes_sink(fetched_buffer),
            resolved_mode=resolved_mode,
            pinned_image_descriptions=request.image_descriptions,
            projected_history=projected_history,
            model_profiles=model_profiles,
        )
        auth_mode = str((session.prepared_input or {}).get("auth_mode") or "none")
        prepared_input = session.prepared_input or {}
        recall_allowed = resolved_mode == "research" and bool(
            prepared_input.get("profile_memory_enabled", True)
        )
        memory_epoch = int(prepared_input.get("profile_memory_epoch") or 0)
        memory_recall_record_count = 0
        memory_recall_chars = 0
        if self._memory is None or not memory_owner_allowed(auth_mode):
            recall_allowed = False
        elif recall_allowed and self._memory_capability_current is not None:
            recall_allowed = await self._memory_capability_current(
                owner_id=session.owner_id, epoch=memory_epoch
            )
        elif recall_allowed:
            recall_allowed = await _memory_recall_allowed(
                self._memory_recall_enabled, owner_id=session.owner_id
            )
        stream: AsyncIterator[str] | None = None
        try:
            await _ensure_selected_lane(
                journal,
                session_id=agent_session_id,
                lane_id=agent_lane_id,
                source_lane_id=(
                    LaneId(request.source_lane_id) if request.source_lane_id is not None else None
                ),
            )
            prepared_early: Any = None
            if resolved_mode == "research":
                from dlightrag.answer.execution_settings import validate_agent_execution

                root = validate_agent_execution(
                    execution_environment=self._execution_environment,
                    workspace_root=self._workspace_root_setting,
                    working_dir=self._working_dir,
                    sandbox_adapter=(
                        self._execution_adapter
                        if self._execution_environment == "sandbox"
                        else None
                    ),
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
                            execution_adapter=self._execution_adapter,
                        )
                    except WorkspaceRecoveryFailed as exc:
                        raise RunExecutionError("workspace_recovery_failed", str(exc)) from exc
                    except WorkspaceIntegrityError as exc:
                        raise RunExecutionError("workspace_integrity_error", str(exc)) from exc
                    run.orchestrator.bind_workspace(bound)
                session_id = agent_session_id
                store = self._store
                run.orchestrator.bind_memory(
                    owner_id=session.owner_id,
                    auth_mode=str((session.prepared_input or {}).get("auth_mode") or "none"),
                    run_id=session.run_id,
                    session_id=session_id.value,
                    store=self._memory_store,
                    enabled=recall_allowed,
                    epoch=memory_epoch,
                    capability_current=self._memory_capability_current,
                )
                persist_child_runtime = _fenced_child_writer(store, "upsert_child_session", session)
                claim_child = _fenced_child_writer(store, "claim_child_session", session)
                renew_child = _fenced_child_writer(store, "heartbeat_child_session", session)
                if persist_child_runtime is None or claim_child is None or renew_child is None:
                    raise RunExecutionError(
                        "run_execution_failed",
                        "Child Session persistence is unavailable.",
                    )
                run.orchestrator.bind_subagents(
                    parent_session_id=session_id,
                    run_id=session.run_id,
                    owner_id=session.owner_id,
                    persist=_fenced_child_writer(store, "upsert_child_session", session),
                    load_child=_async_store_method(store, "load_child_session"),
                    finish_child=_fenced_child_writer(store, "finish_child_session", session),
                    run_child=_bound_child_runner(
                        orchestrator=run.orchestrator,
                        journal=journal,
                        session=session,
                        fetched_buffer=fetched_buffer,
                        parent_session_id=session_id,
                        persist_child_runtime=persist_child_runtime,
                        claim_child=claim_child,
                        renew_child=renew_child,
                    ),
                    check_cancelled=session.check_cancelled,
                )
                # Resolve and compare every accepted execution pin before the
                # first Session mutation. The later post-recall comparison is
                # a second guard immediately before provider/tool effects.
                pin_probe = run.orchestrator.prepare_run(
                    request.query,
                    conversation_history=run.history,
                    query_images=run.query_images,
                    registry=run.registry,
                )
                self.validate_pinned_model_profiles(request)
                self.validate_pinned_agent_run_plan(request, pin_probe.tools)
                snapshot = await journal.load(session_id)
                is_new_session = snapshot.commit_sequence == 0
                memory_text = ""
                if self._memory is not None and recall_allowed:
                    recalled = await self._memory.recall(
                        owner_id=session.owner_id,
                        query=request.query,
                    )
                    memory_text = render_auto_recall(recalled.records)
                    memory_recall_record_count = len(recalled.records)
                    memory_recall_chars = recalled.content_chars
                run.orchestrator.bind_recall(memory_text)
                prepared_early = run.orchestrator.prepare_run(
                    request.query,
                    conversation_history=run.history,
                    query_images=run.query_images,
                    registry=run.registry,
                )
                self.validate_pinned_model_profiles(request)
                self.validate_pinned_agent_run_plan(request, prepared_early.tools)
                if not is_new_session:
                    await _adopt_durable_evidence(prepared_early, journal, session_id)
                plan = request.agent_run_plan
                if plan is None:
                    raise RunExecutionError(
                        "run_execution_failed",
                        "Research answer run is missing its accepted Agent Plan",
                    )
                research_plan = plan

                def validate_research_pins() -> None:
                    self.validate_pinned_model_profiles(request)
                    self.validate_pinned_agent_run_plan(request, prepared_early.tools)

                effects = ResearchRuntimeEffects(
                    orchestrator=run.orchestrator,
                    prepared=prepared_early,
                    session=session,
                    session_id=session_id,
                    fetched_buffer=fetched_buffer,
                    persist_child_intent=_fenced_child_writer(
                        store, "upsert_child_session", session
                    ),
                    validate_pins=validate_research_pins,
                )
                control_reader = _fenced_control_reader(store, session)
                control_ack = _fenced_control_ack(store, session)
                controls = (
                    AnswerRuntimeControls(reader=control_reader, acknowledge=control_ack)
                    if control_reader is not None and control_ack is not None
                    else None
                )
                agent_runtime = AgentSessionRuntime(
                    transactions=journal,
                    load=journal.load,
                    effects=effects,
                    tools=prepared_early.tools,
                    fencing_epoch=session.execution.fencing_epoch,
                    provider_attempt_limit=plan.provider_attempt_limit,
                    event_sink=_answer_runtime_event_sink(session),
                    controls=controls,
                )
                accepted = await agent_runtime.accept(
                    session_id=session_id,
                    lane_id=agent_lane_id,
                    idempotency_key=f"answer-run:{session.run_id}",
                    content=request.query,
                    plan=plan,
                )
                research_operation_id = accepted.operation_id
                await session.enter_phase("researching")
                while True:
                    before = await journal.load(session_id)
                    operation = await _drive_answer_operation(
                        agent_runtime,
                        session=session,
                        session_id=session_id,
                        operation_id=accepted.operation_id,
                    )
                    if not isinstance(operation.state, OperationCompleted):
                        raise RunExecutionError(
                            "run_execution_failed",
                            f"Research Agent operation ended as {operation.state.state_type}.",
                        )
                    snapshot = await journal.load(session_id)
                    operation_usage = (
                        _usage_from_snapshot_entries(
                            snapshot_entries=(
                                entry
                                for entry in snapshot.entries
                                if entry.sequence > _last_entry_sequence(before)
                            )
                        )
                        or {}
                    )
                    agent_operations.append(
                        {
                            "operation_id": accepted.operation_id.value,
                            "purpose": "research" if not agent_operations else "follow_up",
                            "status": "completed",
                            "usage": operation_usage,
                        }
                    )
                    next_input = _oldest_pending_input(snapshot, agent_lane_id)
                    command_ids: tuple[str, ...] = ()
                    if next_input is None and controls is not None:
                        commands = await controls.poll(operation.context)
                        if commands:
                            command = commands[0]
                            command_ids = (command.command_id,)
                            if isinstance(command, FollowUpCommand):
                                next_input = (
                                    command.idempotency_key,
                                    command.content,
                                )
                            else:
                                next_input = (command.command_id, command.content)
                    if next_input is None:
                        break
                    if len(agent_operations) >= 1 + plan.max_pending_follow_ups:
                        raise RunExecutionError(
                            "run_execution_failed",
                            "Research linked-operation bound was exhausted.",
                        )
                    validate_research_pins()
                    accepted = await agent_runtime.accept(
                        session_id=session_id,
                        lane_id=agent_lane_id,
                        idempotency_key=next_input[0],
                        content=next_input[1],
                        plan=plan,
                    )
                    research_operation_id = accepted.operation_id
                    if command_ids and controls is not None:
                        if not await controls.acknowledge(command_ids):
                            raise LeaseLostError
                run.orchestrator.adopt_runtime_snapshot(prepared_early, snapshot)
            else:
                fast_session_host = FastSessionHost(
                    transactions=journal,
                    load=journal.load,
                    fencing_epoch=session.execution.fencing_epoch,
                )
                fast_turn = await fast_session_host.accept(
                    session_id=agent_session_id,
                    lane_id=agent_lane_id,
                    reservation_id=session.run_id,
                    idempotency_key=request.idempotency_fingerprint,
                    content=request.query,
                )
                fast_reservation_active = True
                fast_boundaries = FastRunBoundaries(
                    session=session,
                    progress=session.execution.progress_store,
                    run_id=session.run_id,
                    initial_progress_version=(
                        session.durable_progress_version + (1 if fast_turn.created else 0)
                    ),
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
                    "resolved_mode": resolved_mode,
                    "workspaces": run.workspaces,
                    "history_turns": len(run.history or []),
                    "query_image_count": run.current_image_count,
                    "semantic_highlights": request.semantic_highlights,
                },
            ) as pipeline_trace:
                prepared = prepared_early
                if resolved_mode == "research":
                    if prepared is None:
                        raise RunExecutionError(
                            "run_execution_failed",
                            "Research Runtime lost its prepared Host state.",
                        )
                    await session.enter_phase("generating")
                    contexts, stream = run.orchestrator.runtime_answer_stream(prepared)
                else:
                    contexts, stream = await run.orchestrator.answer_stream(
                        request.query,
                        conversation_history=run.history,
                        boundaries=fast_boundaries,
                    )
                answer_parts: list[str] = []
                if stream is not None:
                    async for chunk in stream:
                        answer_parts.append(chunk)
                        await session.emit_token(chunk)
                await session.flush_tokens()
                answer_text = getattr(stream, "answer", "") or "".join(answer_parts)
                finalized = finalize_answer(answer_text, contexts)
                publication = _publication_plan(
                    run.orchestrator.artifact_root(),
                    answer=finalized.answer,
                    limits=self._settings.publication,
                )
                finalized.answer = publication.answer
                if (
                    publication.issues
                    and agent_runtime is not None
                    and research_plan is not None
                    and prepared_early is not None
                ):
                    correction = await agent_runtime.accept(
                        session_id=agent_session_id,
                        lane_id=agent_lane_id,
                        idempotency_key=f"publication-correction:{session.run_id}",
                        content=publication.correction_feedback(),
                        plan=research_plan,
                    )
                    before_correction = await journal.load(agent_session_id)
                    corrected = await _drive_answer_operation(
                        agent_runtime,
                        session=session,
                        session_id=agent_session_id,
                        operation_id=correction.operation_id,
                    )
                    if not isinstance(corrected.state, OperationCompleted):
                        raise RunExecutionError(
                            "run_execution_failed",
                            "Publication correction Agent operation did not complete.",
                        )
                    corrected_snapshot = await journal.load(agent_session_id)
                    correction_usage = (
                        _usage_from_snapshot_entries(
                            snapshot_entries=(
                                entry
                                for entry in corrected_snapshot.entries
                                if entry.sequence > _last_entry_sequence(before_correction)
                            )
                        )
                        or {}
                    )
                    correction_record = {
                        "operation_id": correction.operation_id.value,
                        "purpose": "publication_correction",
                        "status": "completed",
                        "usage": correction_usage,
                    }
                    agent_operations.append(correction_record)
                    research_operation_id = correction.operation_id
                    run.orchestrator.adopt_runtime_snapshot(
                        prepared_early,
                        corrected_snapshot,
                    )
                    contexts, stream = run.orchestrator.runtime_answer_stream(prepared_early)
                    await session.reset_output()
                    corrected_parts: list[str] = []
                    if stream is not None:
                        async for chunk in stream:
                            corrected_parts.append(chunk)
                            await session.emit_token(chunk)
                    await session.flush_tokens()
                    answer_text = getattr(stream, "answer", "") or "".join(corrected_parts)
                    finalized = finalize_answer(answer_text, contexts)
                    publication = _publication_plan(
                        run.orchestrator.artifact_root(),
                        answer=finalized.answer,
                        limits=self._settings.publication,
                    )
                    finalized.answer = publication.answer
                    correction_record["publication_outcome"] = publication.outcome
                if request.semantic_highlights:
                    finalized.sources = await enrich_semantic_highlights(
                        finalized.sources,
                        answer_text=finalized.answer,
                        settings=self._settings.semantic_highlights,
                        model_factory=self._models.new_highlight_model,
                    )
                trace = dict(getattr(stream, "trace", None) or {})
                if agent_runtime is not None and research_operation_id is not None:
                    root_usage: dict[str, int] = {}
                    for item in agent_operations:
                        for key, value in item["usage"].items():
                            root_usage[key] = root_usage.get(key, 0) + int(value)
                    child_usage = await _durable_child_usage(
                        self._store,
                        owner_id=session.owner_id,
                        run_id=session.run_id,
                    )
                    if not child_usage:
                        child_usage = {
                            str(key): int(value)
                            for key, value in (trace.get("child_usage") or {}).items()
                            if isinstance(value, int)
                        }
                    inclusive = dict(root_usage)
                    for key, value in child_usage.items():
                        inclusive[key] = inclusive.get(key, 0) + value
                    trace["usage"] = {
                        "usage_details": root_usage,
                        "child_usage_details": child_usage,
                        "inclusive_usage_details": inclusive,
                    }
                    trace["agent_operations"] = list(agent_operations)
                trace.setdefault("agent_operations", list(agent_operations))
                trace["query_image_description_count"] = len(run.image_descriptions)
                trace["memory_recall_record_count"] = memory_recall_record_count
                trace["memory_recall_chars"] = memory_recall_chars
                images = evidence_images_from_sources(finalized.sources, contexts=contexts)
                pipeline_trace.update(
                    output=answer_trace_output(
                        finalized.answer,
                        finalized.sources,
                        contexts,
                        capture_sensitive_data=self._telemetry.capture_sensitive_data,
                    )
                )
                publications, artifact_descriptors, report_sources = _stage_publications(
                    plan=publication,
                    answer=finalized.answer,
                    contexts=contexts,
                    require_answer=getattr(prepared_early, "stop_reason", None) == "model_stop",
                )
                session.pending_publications = publications
                stored = store_answer_result(
                    answer=finalized.answer,
                    contexts=project_contexts_for_client(contexts),
                    sources=finalized.sources,
                    evidence_images=images,
                    trace=trace,
                    image_descriptions=run.image_descriptions,
                    artifacts=artifact_descriptors,
                    artifact_outcome=publication.outcome,
                    report_sources=report_sources,
                )
                if fast_boundaries is not None:
                    await fast_boundaries.settle_retrieval(contexts)
                if fast_session_host is not None:
                    fast_commit = await fast_session_host.complete(
                        session_id=agent_session_id,
                        lane_id=agent_lane_id,
                        reservation_id=session.run_id,
                        content=finalized.answer,
                        usage=(
                            trace.get("usage") if isinstance(trace.get("usage"), Mapping) else None
                        ),
                    )
                    fast_reservation_active = False
                    if fast_boundaries is not None and fast_commit is not None:
                        fast_boundaries.observe_session_progress()
                if fast_boundaries is not None:
                    await fast_boundaries.settle_final(
                        result=stored,
                        result_digest=canonical_json(stored),
                    )
                return stored
        except BaseException:
            if fast_session_host is not None and fast_reservation_active:
                try:
                    await fast_session_host.fail(
                        session_id=agent_session_id,
                        lane_id=agent_lane_id,
                        reservation_id=session.run_id,
                    )
                except Exception:
                    logger.exception("Failed to clear Fast Host turn reservation")
            raise
        finally:
            await _close_execution_resources(stream, run.registry)

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
        environment: ExecutionEnvironment | None = None,
        resolved_mode: ResolvedMode,
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
            resolved_mode=resolved_mode,
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
                    preserve_query=True if resolved_mode == "research" else None,
                    model_profile=models.extract,
                )

            model_func: Callable[..., Any] | None = None
            stream_model_func: Callable[..., AsyncIterator[str]] | None = None
            if resolved_mode == "research":
                tool_model = self._models.query_tool_model()
                model_func = tool_model
                stream_model_func = tool_model.stream_text

            def resolve_child_model(
                role: str,
            ) -> tuple[Callable[..., Any], Callable[..., AsyncIterator[str]], ModelProfile]:
                if role not in {"query", "extract"}:
                    raise ValueError(f"unknown child model role: {role}")
                selected_role: ModelRole = role  # type: ignore[assignment]
                profile = models.query if role == "query" else models.extract
                selected = self._models.tool_model(selected_role)
                return selected, selected.stream_text, profile

            orchestrator = AnswerOrchestrator(
                synthesizer=self._models.answer_synthesizer(query_profile),
                retrieve_knowledge_base=retrieve_knowledge_base,
                search_web=(
                    resolved.web_search.search if resolved.web_search is not None else None
                ),
                model_func=model_func,
                stream_model_func=stream_model_func,
                resource_tools=[*resolved.resource_tools, *self._external_tools],
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
                resolved_mode=resolved_mode,
                subagent_host=SubagentHost() if resolved_mode == "research" else None,
                memory_host=(
                    MemoryHost()
                    if resolved_mode == "research" and self._memory_store is not None
                    else None
                ),
                resource_reader=(
                    make_resource_reader(resolved.registry, text_window_budget)
                    if resolved.registry is not None
                    else None
                ),
                child_model_resolver=resolve_child_model,
                trusted_extensions=self._trusted_extensions,
                skills_global_root=self._skills_global_root,
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
            pieces: list[bytes] = []
            async for piece in self._store.stream_artifact(owner_id=owner_id, digest=digest):
                pieces.append(piece)
            if not pieces:
                raise RunExecutionError(
                    "run_execution_failed",
                    "Answer run attachment bytes no longer exist.",
                )
            return b"".join(pieces)

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
    def validate_pinned_agent_run_plan(
        request: AnswerRunInput,
        tools: Sequence[AgentTool],
    ) -> None:
        """Reject execution when runtime tools differ from acceptance."""
        pinned = request.agent_run_plan
        if pinned is None:
            raise IncompatibleActiveRunError(
                "Research answer run is missing its accepted Agent Plan"
            )
        actual = AgentRunPlan.from_tools(
            tools,
            model_role="query",
            context_policy_revision=request.context_policy_revision,
        )
        if (
            actual.model_role != pinned.model_role
            or actual.context_policy_revision != pinned.context_policy_revision
            or actual.tools != pinned.tools
        ):
            raise IncompatibleActiveRunError(
                "answer run Agent Plan differs from its accepted tool contracts"
            )

    def validate_pinned_model_profiles(
        self,
        request: AnswerRunInput,
    ) -> dict[ModelRole, ModelProfile]:
        # Capacity is recalculated from the pinned model facts for each segment.
        # A global arithmetic revision is not a reason to strand an otherwise
        # replayable run.
        pinned = {item.role: item for item in request.pinned_models}
        if len(request.pinned_models) != len(MODEL_ROLE_NAMES) or set(pinned) != set(
            MODEL_ROLE_NAMES
        ):
            raise IncompatibleActiveRunError(
                "answer run does not contain the complete pinned model role set"
            )
        if request.context_policy_revision != CONTEXT_POLICY_REVISION:
            raise IncompatibleActiveRunError("answer run uses another context policy revision")
        if any(
            pinned[role].fingerprint != self._model_fingerprint_for_role(role)
            for role in MODEL_ROLE_NAMES
        ):
            raise IncompatibleActiveRunError(
                "answer run targets another model endpoint configuration"
            )
        return {role: pinned[role].profile for role in MODEL_ROLE_NAMES}


class FetchedResourceBuffer:
    """Fetched bytes partitioned by explicit Session and model tool call."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], list[FetchedResourceBytes]] = {}

    def append(
        self,
        fetched: FetchedResourceBytes,
        owner: ResourceEffectOwner | None,
    ) -> None:
        key = (owner.execution_scope, owner.intent_id.value) if owner is not None else ("", "")
        self._items.setdefault(key, []).append(fetched)

    def drain(self, *, scope: str, intent_id: IntentId) -> tuple[FetchedResourceBytes, ...]:
        fetched = [*self._items.pop((scope, intent_id.value), ())]
        # Acceptance-time fetches happen before a tool task has a scope. Bind
        # them to the first durable settlement rather than sharing a live list.
        fetched.extend(self._items.pop(("", ""), ()))
        return tuple(fetched)


def _buffered_fetched_bytes_sink(buffer: FetchedResourceBuffer) -> FetchedBytesSink:
    """Buffer fetched bytes until their explicit owning intent settles."""

    async def persist(
        fetched: FetchedResourceBytes,
        owner: ResourceEffectOwner | None,
    ) -> None:
        buffer.append(fetched, owner)

    return persist


def _memory_operation_settlement(
    details: Mapping[str, Any] | None,
) -> MemoryOperationSettlement | None:
    """Decode the product-owned Memory receipt envelope; never infer from text."""
    if not isinstance(details, Mapping):
        return None
    raw = details.get("memory_operation")
    if not isinstance(raw, Mapping):
        return None
    operation = str(raw.get("operation") or "")
    outcome = str(raw.get("outcome") or "")
    if operation not in {"remember", "forget", "undo"}:
        return None
    if outcome not in {"changed", "unchanged", "rejected", "conflict"}:
        return None
    memory_ids = raw.get("memory_ids")
    return MemoryOperationSettlement(
        operation=operation,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        change_id=str(raw["change_id"]) if raw.get("change_id") else None,
        memory_ids=(
            tuple(str(memory_id) for memory_id in memory_ids)
            if isinstance(memory_ids, list | tuple)
            else ()
        ),
        kind=str(raw["kind"]) if raw.get("kind") else None,
        body=str(raw.get("body") or ""),
        supersedes_id=str(raw["supersedes_id"]) if raw.get("supersedes_id") else None,
        target_change_id=(str(raw["target_change_id"]) if raw.get("target_change_id") else None),
    )


def _build_effect_host_update(
    *,
    session_id: SessionId,
    intent: EffectIntent,
    ledger_state: Callable[[], str],
    fetched_buffer: FetchedResourceBuffer,
    execution_scope: str,
    tool_effects: ToolEffects,
    details: Mapping[str, Any] | None = None,
) -> EffectHostUpdate:
    """Convert one product Tool result into its typed atomic HostDelta."""
    evidence: tuple[OpaqueEvidenceWrite, ...] = ()
    encoded_ledger = ledger_state()
    if encoded_ledger != "{}":
        content = encoded_ledger.encode("utf-8")
        evidence = (
            OpaqueEvidenceWrite(
                session_id=session_id.value,
                intent_id=intent.intent_id.value,
                result_ordinal=0,
                content_digest=blob_digest(content),
                locator_digest=blob_digest(b"{}"),
                content=content,
                locator=b"{}",
            ),
        )
    fetched_updates: list[FetchedResourceSettlementUpdate] = []
    for fetched in fetched_buffer.drain(
        scope=execution_scope,
        intent_id=intent.intent_id,
    ):
        plan = plan_blob(fetched.content)
        fetched_updates.append(
            FetchedResourceSettlementUpdate(
                resource=OpaqueFetchedResourceWrite(
                    resource_id=fetched.resource_id,
                    safe_name=fetched.filename,
                    media_type=fetched.mime_type,
                    capabilities={},
                    blob_digest=plan.digest,
                    source_locator_digest=blob_digest(fetched.url.encode("utf-8")),
                    source_locator=fetched.url.encode("utf-8"),
                    session_id=session_id.value,
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
    inventory = tool_effects.workspace_inventory

    def blob_descriptor(content: bytes) -> CompleteBlobDescriptor:
        plan = plan_blob(content)
        return CompleteBlobDescriptor(
            digest=plan.digest,
            total_bytes=plan.total_bytes,
            chunks=tuple(plan.chunk(content, index) for index in range(plan.chunk_count)),
        )

    attached_updates = [
        FetchedResourceSettlementUpdate(
            resource=OpaqueFetchedResourceWrite(
                resource_id=attached.resource_id,
                safe_name=attached.filename,
                media_type=attached.mime_type,
                capabilities={"tool_attachment": True},
                blob_digest=blob_digest(attached.content),
                source_locator_digest=blob_digest(attached.source_locator.encode("utf-8")),
                source_locator=attached.source_locator.encode("utf-8"),
                session_id=session_id.value,
                intent_id=intent.intent_id.value,
            ),
            complete_blob=blob_descriptor(attached.content),
        )
        for attached in tool_effects.attached_resources
    ]
    return EffectHostUpdate(
        evidence=evidence,
        fetched=(*fetched_updates, *attached_updates),
        committed_outputs=tuple(
            CommittedSpillUpdate(
                resource_id=output.resource_id,
                content_digest=output.content_digest,
                size_bytes=output.size_bytes,
                session_id=session_id.value,
                intent_id=intent.intent_id.value,
            )
            for output in tool_effects.committed_outputs
        ),
        workspace_inventory=(
            None
            if inventory is None
            else WorkspaceInventoryUpdate(
                upserts=tuple(
                    InventoryPathRecord(
                        relative_path=record.relative_path,
                        entry_type=record.entry_type,
                        size_bytes=record.size_bytes,
                        mode=record.mode,
                        content_digest=record.content_digest,
                    )
                    for record in inventory.upserts
                ),
                deletes=inventory.deletes,
                replace_all=inventory.replace_all,
            )
        ),
        memory_operation=_memory_operation_settlement(details),
    )


class AnswerRuntimeControls:
    """Translate ordered Answer control rows into typed Runtime controls."""

    def __init__(
        self,
        *,
        reader: Callable[[], Awaitable[tuple[Mapping[str, Any], ...]]],
        acknowledge: Callable[[tuple[int, ...]], Awaitable[bool]],
    ) -> None:
        self._reader = reader
        self._acknowledge = acknowledge
        self._sequences: dict[str, int] = {}

    async def poll(self, context: RuntimeContext) -> tuple[Any, ...]:
        del context
        commands: list[Any] = []
        for row in await self._reader():
            sequence = int(row.get("control_sequence") or 0)
            kind = str(row.get("kind") or "steer")
            content = str(row.get("content") or "")
            command_id = f"answer-control:{sequence}"
            self._sequences[command_id] = sequence
            if kind == "follow_up":
                commands.append(
                    FollowUpCommand(
                        command_id=command_id,
                        input_id=command_id,
                        idempotency_key=command_id,
                        content=content,
                    )
                )
            else:
                commands.append(SteerCommand(command_id=command_id, content=content))
        return tuple(commands)

    async def acknowledge(self, command_ids: tuple[str, ...]) -> bool:
        sequences = tuple(self._sequences[command_id] for command_id in command_ids)
        return await self._acknowledge(sequences)


class ResearchRuntimeEffects:
    """Answer Host adapters for the product-neutral AgentSessionRuntime effects."""

    def __init__(
        self,
        *,
        orchestrator: AnswerOrchestrator,
        prepared: Any,
        session: RunSession,
        session_id: SessionId,
        fetched_buffer: FetchedResourceBuffer,
        persist_child_intent: Callable[..., Awaitable[Any]] | None,
        validate_pins: Callable[[], None] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._prepared = prepared
        self._session = session
        self._session_id = session_id
        self._fetched_buffer = fetched_buffer
        self._persist_child_intent = persist_child_intent
        self._validate_pins = validate_pins
        self._tools = {tool.name: tool for tool in prepared.tools}

    async def assemble_request(self, context: RuntimeContext) -> RequestSnapshot | Any:
        await self._check_cancelled()
        self._check_pins()
        try:
            return await self._orchestrator.assemble_runtime_request(self._prepared, context)
        except AnswerInputError as exc:
            raise OperationEffectFailed("context_overflow", str(exc)) from exc

    async def call_provider(
        self,
        context: RuntimeContext,
        request: RequestSnapshot,
        attempt_id: AttemptId,
        emit_ephemeral: Any,
    ) -> AssistantTurn:
        await self._check_cancelled()
        self._check_pins()
        await emit_ephemeral(
            AgentSessionEvent(
                kind="model_start",
                session_id=context.session_id,
                lane_id=context.lane_id,
                operation_id=context.operation_id,
                commit_sequence=None,
                data={"attempt_id": attempt_id.value},
                ephemeral=True,
            )
        )
        try:
            assistant = await self._orchestrator.call_runtime_provider(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if is_provider_context_overflow(exc):
                raise ProviderContextOverflow from exc
            raise ProviderAttemptFailed(str(exc), retryable=True) from exc
        await emit_ephemeral(
            AgentSessionEvent(
                kind="model_end",
                session_id=context.session_id,
                lane_id=context.lane_id,
                operation_id=context.operation_id,
                commit_sequence=None,
                data={
                    "attempt_id": attempt_id.value,
                    "stop_reason": assistant.stop_reason,
                    "tool_calls": len(assistant.tool_calls),
                },
                ephemeral=True,
            )
        )
        return assistant

    async def execute_tool(
        self,
        context: RuntimeContext,
        item: ToolBatchItem,
        arguments: Mapping[str, Any],
        attempt_id: AttemptId,
        emit_ephemeral: Any,
    ) -> ToolEffectResult[EffectHostUpdate]:
        await self._check_cancelled()
        self._check_pins()
        self._orchestrator.bind_child_context(self._prepared, context)
        if item.intent_id is None:
            raise RuntimeError("executable Tool item lost its IntentId")
        tool = self._tools.get(item.tool_name)
        if tool is None:
            raise RuntimeError(f"Tool contract unavailable: {item.tool_name}")
        await self._bind_subagent_intent(item, arguments)
        validated = tool.input_model.model_validate(arguments)

        async def update(result: ToolResult) -> None:
            await emit_ephemeral(
                AgentSessionEvent(
                    kind="tool_update",
                    session_id=context.session_id,
                    lane_id=context.lane_id,
                    operation_id=context.operation_id,
                    commit_sequence=None,
                    data={
                        "tool_name": item.tool_name,
                        "call_id": item.call_id,
                        "source_position": item.source_index,
                        "text_chars": len(result.text_content),
                    },
                    ephemeral=True,
                )
            )

        await emit_ephemeral(
            AgentSessionEvent(
                kind="tool_start",
                session_id=context.session_id,
                lane_id=context.lane_id,
                operation_id=context.operation_id,
                commit_sequence=None,
                data={
                    "tool_name": item.tool_name,
                    "call_id": item.call_id,
                    "source_position": item.source_index,
                    "attempt_id": attempt_id.value,
                },
                ephemeral=True,
            )
        )
        runtime = ToolRuntime(
            call_id=item.call_id,
            tool_name=item.tool_name,
            intent_id=item.intent_id,
            execution_scope=self._session_id.value,
            _update_sink=update,
        )
        result = await tool.execute(validated, runtime)
        fitted = fit_tool_result(
            result,
            max_tokens=CONTEXT_POLICY.observation_reserve_tokens,
        )
        outcome = "failed" if fitted.is_error else "succeeded"
        durable = ToolResultEntry(
            tool_name=item.tool_name,
            call_id=item.call_id,
            outcome=outcome,
            parts=fitted.parts,
            details=fitted.details,
            cached=fitted.cached,
        )
        self._prepared.trace["tool_observations"].append(
            {
                "tool": item.tool_name,
                "call_id": item.call_id,
                "outcome": outcome,
                "cached": fitted.cached,
                "is_error": fitted.is_error,
                "content_chars": len(fitted.text_content),
            }
        )
        intent = EffectIntent(
            intent_id=item.intent_id,
            tool_name=item.tool_name,
            replay_policy=item.replay_policy,
            contract_version=item.contract_version,
            input_schema_digest=item.input_schema_digest,
            canonical_input=canonical_json(arguments),
            source_call_id=item.call_id,
        )
        delta = _build_effect_host_update(
            session_id=self._session_id,
            intent=intent,
            ledger_state=lambda: self._prepared.evidence.ledger_state_json(),
            fetched_buffer=self._fetched_buffer,
            execution_scope=self._session_id.value,
            tool_effects=fitted.effects,
            details=fitted.details,
        )
        return ToolEffectResult(durable, delta)

    async def compact(self, context: RuntimeContext, attempt: int) -> Any:
        return await self._orchestrator.compact_runtime_context(
            self._prepared,
            context,
            attempt,
        )

    async def _bind_subagent_intent(
        self,
        item: ToolBatchItem,
        arguments: Mapping[str, Any],
    ) -> None:
        if (
            self._persist_child_intent is None
            or item.tool_name != "spawn_agent"
            or item.intent_id is None
        ):
            return
        from dlightrag.answer.tools.subagents import child_session_id

        spawn = SpawnAgentInput.model_validate(arguments)
        for position, request in enumerate(spawn.children):
            child_id = child_session_id(
                run_id=self._session.run_id,
                parent_session_id=self._session_id,
                parent_intent_id=item.intent_id,
                position=position,
            )
            await self._persist_child_intent(
                owner_id=self._session.owner_id,
                run_id=self._session.run_id,
                child_session_id=child_id.value,
                parent_session_id=self._session_id.value,
                parent_call_id=item.call_id,
                parent_intent_id=item.intent_id.value,
                objective=request.objective,
                context_mode=request.context,
                model_role=request.model_role,
                tools=request.tools,
            )

    def _check_pins(self) -> None:
        if self._validate_pins is None:
            return
        try:
            self._validate_pins()
        except IncompatibleActiveRunError as exc:
            raise OperationEffectFailed("plan_unavailable", str(exc)) from exc

    async def _check_cancelled(self) -> None:
        try:
            await self._session.check_cancelled()
        except RunCancelledError as exc:
            raise AgentOperationCancelled(exc) from exc


def _answer_runtime_event_sink(
    session: RunSession,
) -> Callable[[AgentSessionEvent], Awaitable[None]]:
    async def publish(event: AgentSessionEvent) -> None:
        event_type = {
            "tool_start": "tool_start",
            "tool_update": "tool_progress",
            "tool_result_committed": "tool_end",
        }.get(event.kind)
        if event_type is None:
            return
        allowed = {
            "tool_name",
            "call_id",
            "source_position",
            "source_index",
            "outcome",
            "text_chars",
            "attempt_id",
        }
        payload = {
            key: value
            for key, value in event.data.items()
            if key in allowed and isinstance(value, (str, int, float, bool))
        }
        if event.commit_sequence is not None:
            payload["session_commit_sequence"] = event.commit_sequence
        await session.emit_tool_event(event_type, payload)

    return publish


async def _drive_answer_operation(
    runtime: AgentSessionRuntime[EffectHostUpdate],
    *,
    session: RunSession,
    session_id: SessionId,
    operation_id: OperationId,
) -> Any:
    try:
        return await runtime.drive(session_id=session_id, operation_id=operation_id)
    except (
        asyncio.CancelledError,
        RunCancelledError,
        AgentOperationCancelled,
    ) as exc:
        await runtime.cancel(session_id=session_id, operation_id=operation_id)
        await runtime.close(session_id=session_id, operation_id=operation_id)
        if isinstance(exc, AgentOperationCancelled):
            raise exc.reason from exc
        raise
    except SessionLeaseLostError as exc:
        raise LeaseLostError from exc


def _oldest_pending_input(snapshot: Any, lane_id: LaneId) -> tuple[str, Any] | None:
    for record in snapshot.registers:
        if isinstance(record.value, PendingInput) and record.value.lane_id == lane_id:
            if not record.value.items:
                return None
            item = record.value.items[0]
            return item.idempotency_key, item.content
    return None


def _last_entry_sequence(snapshot: Any) -> int:
    return max((entry.sequence for entry in snapshot.entries), default=0)


async def _ensure_selected_lane(
    journal: AgentSessionStore[EffectHostUpdate],
    *,
    session_id: SessionId,
    lane_id: LaneId,
    source_lane_id: LaneId | None,
) -> None:
    snapshot = await journal.load(session_id)
    try:
        snapshot.tree.lane(lane_id)
        return
    except KeyError:
        pass
    if source_lane_id is None:
        if snapshot.commit_sequence == 0 and lane_id == LaneId.main():
            return
        raise RunExecutionError("run_execution_failed", "Agent Lane mapping is missing.")
    try:
        outcome = await journal.fork_lane(
            session_id=session_id,
            source_lane_id=source_lane_id,
            lane_id=lane_id,
        )
    except KeyError as exc:
        raise RunExecutionError(
            "agent_session_conflict",
            "The Agent Session changed before its branch could be created.",
        ) from exc
    if not isinstance(outcome, TransactionCommit):
        raise LeaseLostError


def _bound_child_runner(
    *,
    orchestrator: AnswerOrchestrator,
    journal: AgentSessionStore[EffectHostUpdate],
    session: RunSession,
    fetched_buffer: FetchedResourceBuffer,
    parent_session_id: SessionId,
    persist_child_runtime: Callable[..., Awaitable[Any]],
    claim_child: Callable[..., Awaitable[Any]],
    renew_child: Callable[..., Awaitable[Any]],
) -> Callable[[SessionId, ChildRequest, str, ChildContextSnapshot], Awaitable[ChildOutcome]]:
    async def run_child(
        child_id: SessionId,
        request: ChildRequest,
        parent_call_id: str,
        context_snapshot: ChildContextSnapshot,
    ) -> ChildOutcome:
        return await run_child_session(
            orchestrator=orchestrator,
            journal=journal,
            session=session,
            fetched_buffer=fetched_buffer,
            child_id=child_id,
            request=request,
            parent_call_id=parent_call_id,
            parent_session_id=parent_session_id,
            context_snapshot=context_snapshot,
            persist_child_runtime=persist_child_runtime,
            claim_child=claim_child,
            renew_child=renew_child,
        )

    return run_child


async def run_child_session(
    *,
    orchestrator: AnswerOrchestrator,
    journal: AgentSessionStore[EffectHostUpdate],
    session: RunSession,
    fetched_buffer: FetchedResourceBuffer,
    child_id: SessionId,
    request: ChildRequest,
    parent_call_id: str,
    parent_session_id: SessionId,
    context_snapshot: ChildContextSnapshot,
    persist_child_runtime: Callable[..., Awaitable[Any]],
    claim_child: Callable[..., Awaitable[Any]],
    renew_child: Callable[..., Awaitable[Any]] | None = None,
) -> ChildOutcome:
    """Run or restore one Child through the same deep AgentSessionRuntime."""
    if context_snapshot.parent_session_id != parent_session_id:
        raise RunExecutionError(
            "run_execution_failed",
            "Child ContextSnapshot parent identity changed.",
        )
    prepared = orchestrator.prepare_child_session(
        request,
        context_snapshot=context_snapshot,
        child_session_id=child_id.value,
    )
    plan = AgentRunPlan.from_tools(
        prepared.tools,
        model_role=request.model_role,
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_identity={"role": request.model_role, "scope": "child"},
        model_profile=asdict(prepared.model_profile),
    )
    await persist_child_runtime(
        owner_id=session.owner_id,
        run_id=session.run_id,
        child_session_id=child_id.value,
        parent_session_id=parent_session_id.value,
        parent_call_id=parent_call_id,
        objective=request.objective,
        context_mode=request.context,
        model_role=request.model_role,
        tools=request.tools,
        depth=context_snapshot.depth + 1,
        context_snapshot=context_snapshot.canonical_payload(),
        plan=plan.canonical_payload(),
        budget={
            "provider_attempt_limit": plan.provider_attempt_limit,
            "compaction_attempt_limit": plan.compaction_attempt_limit,
            "model_profile": asdict(prepared.model_profile),
        },
        host_state={"inherits_parent_evidence": bool(context_snapshot.evidence_state)},
    )
    child_epoch = await claim_child(
        owner_id=session.owner_id,
        run_id=session.run_id,
        child_session_id=child_id.value,
    )
    if not isinstance(child_epoch, int):
        raise LeaseLostError
    child_journal = journal
    bind_child = getattr(journal, "for_child", None)
    if callable(bind_child):
        child_journal = cast(
            AgentSessionStore[EffectHostUpdate],
            bind_child(child_id, fencing_epoch=child_epoch),
        )
    effects = ResearchRuntimeEffects(
        orchestrator=orchestrator,
        prepared=prepared,
        session=session,
        session_id=child_id,
        fetched_buffer=fetched_buffer,
        persist_child_intent=None,
    )
    runtime = AgentSessionRuntime(
        transactions=child_journal,
        load=child_journal.load,
        effects=effects,
        tools=prepared.tools,
        fencing_epoch=child_epoch,
        provider_attempt_limit=plan.provider_attempt_limit,
        event_sink=_answer_runtime_event_sink(session),
    )
    accepted = await runtime.accept(
        session_id=child_id,
        lane_id=LaneId.main(),
        idempotency_key=f"child-session:{child_id.value}",
        content=request.objective,
        plan=plan,
    )
    try:
        operation = await _drive_child_with_lease_renewal(
            runtime,
            session_id=child_id,
            operation_id=accepted.operation_id,
            child_fencing_epoch=child_epoch,
            renew_child=renew_child,
        )
    except (
        asyncio.CancelledError,
        RunCancelledError,
        AgentOperationCancelled,
    ) as exc:
        await runtime.cancel(session_id=child_id, operation_id=accepted.operation_id)
        await runtime.close(session_id=child_id, operation_id=accepted.operation_id)
        if isinstance(exc, AgentOperationCancelled):
            raise exc.reason from exc
        raise
    except SessionLeaseLostError as exc:
        raise LeaseLostError from exc
    snapshot = await child_journal.load(child_id)
    orchestrator.adopt_runtime_snapshot(prepared, snapshot)
    await _adopt_durable_evidence(prepared, child_journal, child_id)
    if isinstance(operation.state, OperationCompleted):
        status: Literal["succeeded", "failed", "cancelled"] = "succeeded"
    elif isinstance(operation.state, OperationCancelled):
        status = "cancelled"
    elif isinstance(operation.state, OperationFailed):
        status = "failed"
    else:
        raise RunExecutionError(
            "run_execution_failed",
            "Child Agent Runtime returned a non-terminal operation.",
        )
    summary = _child_summary(prepared, status)
    return ChildOutcome(
        status=status,
        summary=summary,
        handles=tuple(prepared.evidence.citation_handles()),
        usage=_usage_from_snapshot(snapshot),
        delta=_delta_from_ledger(prepared.evidence),
        child_session_id=child_id.value,
        evidence_state=prepared.evidence.durable_state(),
    )


async def _drive_child_with_lease_renewal(
    runtime: AgentSessionRuntime[EffectHostUpdate],
    *,
    session_id: SessionId,
    operation_id: OperationId,
    child_fencing_epoch: int,
    renew_child: Callable[..., Awaitable[Any]] | None,
) -> Any:
    if renew_child is None:
        return await runtime.drive(session_id=session_id, operation_id=operation_id)

    async def renew_forever() -> None:
        while True:
            await asyncio.sleep(_CHILD_LEASE_HEARTBEAT_SECONDS)
            try:
                renewed = await renew_child(
                    child_session_id=session_id.value,
                    child_fencing_epoch=child_fencing_epoch,
                )
            except LeaseLostError:
                raise
            except Exception:
                logger.warning(
                    "Child Session %s lease heartbeat failed; retrying next cadence",
                    session_id.value,
                    exc_info=True,
                )
                continue
            if renewed is not True:
                raise LeaseLostError

    drive_task = asyncio.create_task(
        runtime.drive(session_id=session_id, operation_id=operation_id)
    )
    heartbeat_task = asyncio.create_task(renew_forever())
    try:
        done, _pending = await asyncio.wait(
            (drive_task, heartbeat_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if drive_task in done:
            return await drive_task
        drive_task.cancel()
        await asyncio.gather(drive_task, return_exceptions=True)
        exception = heartbeat_task.exception()
        if exception is not None:
            raise exception
        raise LeaseLostError
    finally:
        if not drive_task.done():
            drive_task.cancel()
        heartbeat_task.cancel()
        await asyncio.gather(drive_task, heartbeat_task, return_exceptions=True)


def _child_status(reason: str) -> tuple[Literal["succeeded", "failed", "cancelled"], str]:
    if reason == "cancelled":
        return "cancelled", "cancelled"
    return "succeeded", "completed"


def _child_summary(prepared: Any, status: str) -> str:
    text = prepared.last_turn.assistant.text if prepared.last_turn is not None else ""
    return text.strip() or f"Child session {status}."


def _usage_from_snapshot(snapshot: Any) -> dict[str, int] | None:
    return _usage_from_snapshot_entries(snapshot_entries=snapshot.entries)


def _usage_from_snapshot_entries(*, snapshot_entries: Any) -> dict[str, int] | None:
    total: dict[str, int] = {}
    for entry in snapshot_entries or ():
        usage = getattr(entry, "usage", None)
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                total[key] = total.get(key, 0) + value
    return total or None


def _delta_from_ledger(evidence: Any) -> Any:
    from dlightrag.answer.evidence import EvidenceDelta

    contexts = getattr(evidence, "contexts", {}) or {}
    return EvidenceDelta(
        new_chunks=len(contexts.get("chunks") or ()),
        new_entities=len(contexts.get("entities") or ()),
        new_relationships=len(contexts.get("relationships") or ()),
    )


class FastRunBoundaries:
    """Durable three-stage Fast boundaries (planner, retrieval, final_generation)."""

    def __init__(
        self,
        *,
        session: RunSession,
        progress: RunProgressStore,
        run_id: str,
        initial_progress_version: int,
        plan: Mapping[str, Any],
    ) -> None:
        self._session = session
        self._progress = progress
        self._run_id = run_id
        self._plan = plan
        self._progress_version = initial_progress_version

    def observe_session_progress(self) -> None:
        """Account for the canonical Fast Assistant Session commit."""
        self._progress_version += 1

    async def enter_phase(self, phase: str) -> None:
        await self._session.enter_phase(phase)  # type: ignore[arg-type]

    async def check_cancelled(self) -> None:
        await self._session.check_cancelled()

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


def _require_resolved_mode(value: str | None) -> ResolvedMode:
    if value == "fast" or value == "research":
        return value
    raise RunExecutionError("routing_failed", "Answer mode routing failed.")


def _async_store_method(store: object, name: str) -> Any | None:
    method = getattr(store, name, None)
    if inspect.iscoroutinefunction(method):
        return method
    return None


async def _durable_child_usage(
    store: object,
    *,
    owner_id: str,
    run_id: str,
) -> dict[str, int]:
    """Aggregate settled child usage so recovery matches uninterrupted execution."""
    method = _async_store_method(store, "list_child_sessions")
    if method is None:
        return {}
    rows = await method(owner_id=owner_id, run_id=run_id)
    aggregate: dict[str, int] = {}
    for row in rows or ():
        usage = row.get("usage") if isinstance(row, Mapping) else None
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                name = str(key)
                aggregate[name] = aggregate.get(name, 0) + value
    return aggregate


def _fenced_control_reader(
    store: object, session: RunSession
) -> Callable[[], Awaitable[tuple[Mapping[str, Any], ...]]] | None:
    method = _async_store_method(store, "load_pending_agent_controls")
    if method is None:
        return None

    async def read() -> tuple[Mapping[str, Any], ...]:
        controls = await method(
            owner_id=session.owner_id,
            run_id=session.run_id,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
        )
        if controls is None:
            raise LeaseLostError
        return tuple(controls)

    return read


def _fenced_control_ack(
    store: object, session: RunSession
) -> Callable[[tuple[int, ...]], Awaitable[bool]] | None:
    method = _async_store_method(store, "acknowledge_agent_controls")
    if method is None:
        return None

    async def acknowledge(sequences: tuple[int, ...]) -> bool:
        held = await method(
            owner_id=session.owner_id,
            run_id=session.run_id,
            control_sequences=sequences,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
        )
        return bool(held)

    return acknowledge


def _fenced_child_writer(store: object, name: str, session: RunSession) -> Any | None:
    method = _async_store_method(store, name)
    if method is None:
        return None

    async def write(**kwargs: Any) -> Any:
        held = await method(
            **kwargs,
            worker_id=session.worker_id,
            fencing_epoch=session.fencing_epoch,
        )
        if held is False:
            raise LeaseLostError
        return held

    return write


def _verified_current_image_data_uri(data: bytes, *, max_pixels: int) -> tuple[str, str]:
    from dlightrag.ai.media import image_bytes_to_data_uri, verify_web_image_bytes

    mime = verify_web_image_bytes(data, max_pixels=max_pixels)
    return mime, image_bytes_to_data_uri(data, fallback_mime=mime)


def _context_count(contexts: RetrievalContexts, key: str) -> int:
    items = contexts.get(key, [])
    return len(items) if isinstance(items, list) else 0


def _publication_plan(
    root: Path | None,
    *,
    answer: str,
    limits: PublicationLimits,
) -> PublicationPlan:
    if not isinstance(root, Path):
        return PublicationPlan(answer=answer)
    return validate_publication(root, answer=answer, limits=limits)


def _stage_publications(
    *,
    plan: PublicationPlan,
    answer: str,
    contexts: RetrievalContexts,
    require_answer: bool = False,
) -> tuple[list[PendingPublication], list[dict[str, Any]], list[Any]]:
    has_report = any(item.role == "primary_report" for item in plan.artifacts)
    if require_answer and is_empty_answer(answer=answer, has_primary_report=has_report):
        raise RunExecutionError("empty_answer", "The run produced no answer.")
    publications: list[PendingPublication] = []
    report_sources: list[Any] = []
    descriptors = [dict(item) for item in plan.descriptors]
    for item in plan.artifacts:
        payload = item.content
        if item.role == "primary_report" and item.media_type == "text/markdown":
            cleaned = finalize_answer(payload.decode("utf-8"), contexts)
            payload = cleaned.answer.encode("utf-8")
            report_sources = list(cleaned.sources)
            for descriptor in descriptors:
                if descriptor.get("resource_id") == item.resource_id:
                    descriptor["byte_size"] = len(payload)
                    descriptor["digest"] = artifact_digest(payload)
                    break
        publications.append(
            PendingPublication(
                resource_id=item.resource_id,
                reference_kind=item.kind,
                filename=item.filename,
                mime_type=item.media_type,
                content=payload,
            )
        )
    return publications, descriptors, report_sources


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
    "AnswerExecutionStore",
    "AnswerExecutor",
    "AnswerExecutorSettings",
    "AnswerResourceResolver",
    "AnswerResourceSettings",
    "IncompatibleActiveRunError",
    "OrchestratorRun",
    "ResolvedAnswerResources",
    "answer_trace_output",
    "run_child_session",
]


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
