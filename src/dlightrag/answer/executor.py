# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer execution over durable Runtime sessions and RAG workspaces."""

import asyncio
import base64
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from dlightrag_agent.tools import AgentTool
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
from dlightrag.answer.model_runtime import AnswerModelRuntime
from dlightrag.answer.resources import ResourceInput, ResourceRegistry
from dlightrag.answer.resources.models import (
    ResourceManifestEntry,
    ResourceRegistryError,
    TextWindowBudget,
)
from dlightrag.answer.resources.registry import FetchedBytesSink, FetchedResourceBytes
from dlightrag.answer.resources.visual import ResourceInspector
from dlightrag.answer.runs.checkpoints import (
    ArtifactReader,
    encode_checkpoint_state,
    restore_agent_state,
)
from dlightrag.answer.runs.execution import (
    AnswerRunInput,
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    SessionBoundaries,
    build_current_answer_resources,
)
from dlightrag.answer.runs.models import AgentRunState
from dlightrag.answer.runs.results import store_answer_result
from dlightrag.answer.sources import project_contexts_for_client
from dlightrag.answer.tools.resources import build_resource_tools
from dlightrag.answer.tools.web import ExaSearch
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.runtime import (
    CheckpointError,
    LeaseLostError,
    PendingArtifact,
    PendingArtifactReference,
    RunCancelledError,
    RunExecutionError,
    RunSession,
    artifact_digest,
)

logger = logging.getLogger(__name__)


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
    max_agent_turns: int
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


class IncompatibleAnswerRunError(RuntimeError):
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
            research = web_search is not None or bool(resource_manifest)
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
    """Execute durable Answer runs without a manager or storage implementation dependency."""

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
    ) -> None:
        self._store = store
        self._pool = pool
        self._retrieve_result = retrieve
        self._models = models
        self._capabilities = capabilities
        self._resources = resources
        self._settings = settings
        self._telemetry = telemetry

    async def execute(self, session: RunSession) -> Mapping[str, Any]:
        with model_call_scope((session.owner_id, session.run_id)):
            try:
                return await self._execute(session)
            except (
                asyncio.CancelledError,
                RunCancelledError,
                LeaseLostError,
                CheckpointError,
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

    async def _execute(self, session: RunSession) -> Mapping[str, Any]:
        request = AnswerRunInput.from_request(session.request)
        model_profiles = self.validate_pinned_model_profiles(request)
        await session.enter_phase("planning")
        projected_history = PriorTurns([dict(message) for message in request.history])
        run = await self.prepare_orchestrated_run(
            workspaces=list(request.workspaces),
            top_k=request.top_k,
            chunk_top_k=request.chunk_top_k,
            filters=MetadataFilter.model_validate(request.filters) if request.filters else None,
            resources=await self._answer_run_resources(request, owner_id=session.owner_id),
            fetched_bytes_sink=_fetched_bytes_sink(session),
            pinned_image_descriptions=request.image_descriptions,
            projected_history=projected_history,
            model_profiles=model_profiles,
        )
        stream: AsyncIterator[str] | None = None
        try:
            research = run.orchestrator.uses_research_path
            if session.checkpoint is not None and not research:
                raise CheckpointError(
                    "checkpoint_incompatible",
                    "Answer run checkpoint requires a research path this deployment no longer offers.",
                )
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
                prepared = (
                    run.orchestrator.prepare_run(
                        request.query,
                        conversation_history=run.history,
                        query_images=run.query_images,
                        registry=run.registry,
                    )
                    if research
                    else None
                )
                if session.checkpoint is not None:
                    if prepared is None:  # guarded above
                        raise AssertionError("checkpointed Answer run has no research state")
                    await restore_agent_state(
                        prepared.state,
                        {
                            "version": session.checkpoint.version,
                            "completed_turns": session.checkpoint.completed_turns,
                            "state": session.checkpoint.state,
                        },
                        owner_id=session.owner_id,
                        run_id=session.run_id,
                        store=self._store,
                        expected_completed_turns=session.completed_turns,
                        load_corpus_image=self._load_corpus_image,
                    )

                async def encode(state: AgentRunState) -> Mapping[str, Any]:
                    return await encode_checkpoint_state(
                        state,
                        owner_id=session.owner_id,
                        run_id=session.run_id,
                        store=self._store,
                    )

                boundaries = SessionBoundaries(session, encode=encode)
                contexts, stream = await run.orchestrator.answer_stream(
                    request.query,
                    conversation_history=run.history,
                    run=prepared,
                    boundaries=boundaries,
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
                return store_answer_result(
                    answer=finalized.answer,
                    contexts=project_contexts_for_client(contexts),
                    sources=finalized.sources,
                    answer_images=images,
                    trace=trace,
                    image_descriptions=run.image_descriptions,
                )
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
                max_agent_turns=self._settings.max_agent_turns,
                telemetry=self._telemetry,
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
                raise CheckpointError(
                    "checkpoint_corrupt",
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
            raise IncompatibleAnswerRunError(
                "answer run context policy revision does not match this binary; "
                "drain active runs before deployment"
            )
        pinned = {item.role: item for item in request.pinned_models}
        if len(request.pinned_models) != len(MODEL_ROLE_NAMES) or set(pinned) != set(
            MODEL_ROLE_NAMES
        ):
            raise IncompatibleAnswerRunError(
                "answer run does not contain the complete pinned model role set"
            )
        return {role: pinned[role].profile for role in MODEL_ROLE_NAMES}


def _fetched_bytes_sink(session: RunSession) -> FetchedBytesSink:
    async def persist(fetched: FetchedResourceBytes) -> None:
        artifact = PendingArtifact(content=fetched.content)
        await session.attach_artifacts(
            artifacts=[artifact],
            references=[
                PendingArtifactReference(
                    resource_id=fetched.resource_id,
                    reference_kind="fetched_resource",
                    ordinal=fetched.ordinal,
                    digest=artifact.digest,
                    filename=fetched.filename,
                    mime_type=fetched.mime_type,
                    transform_locator={"url": fetched.url},
                )
            ],
        )

    return persist


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
    "IncompatibleAnswerRunError",
    "OrchestratorRun",
    "ResolvedAnswerResources",
    "answer_trace_output",
]
