# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Private process composition for a started Application."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dlightrag.application.application import Application, _ApplicationComponents
from dlightrag.application.config.loading import get_config
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.opaque_cursor import CursorSecretBox
from dlightrag.application.skills import (
    SkillProtection,
    skill_read_layers,
    skill_roots,
    skills_bundle_factory,
)
from dlightrag.engine.agent.environment.confinement import ConfinementPolicy
from dlightrag.engine.agent.environment.execution import (
    confinement_state,
)
from dlightrag.engine.agent.environment.toolchain import SearchToolchain
from dlightrag.engine.ai.embedding import MultimodalEmbedder
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.telemetry import SPAN_TYPES, Telemetry
from dlightrag.engine.runtime.agentspace import SessionNotesLimits

if TYPE_CHECKING:
    from dlightrag_memory.postgres import PostgresMemoryStore

    from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
    from dlightrag.application.memory.service import MemoryService
    from dlightrag.application.memory.session_pass import SessionMemoryPassRunner
    from dlightrag.application.runs import RunService
    from dlightrag.engine.answer.agent_browser.mailbox import (
        AgentMailbox,
    )

logger = logging.getLogger(__name__)


def _operational_pool_factory() -> Any:
    """Lazily resolve DlightRAG's process-wide operational pool."""
    from dlightrag.adapters.postgres.core._pool import pg_pool

    return pg_pool.get


def _memory_embedder(
    config: DlightragConfig, *, scheduler: ModelScheduler, telemetry: Telemetry
) -> MultimodalEmbedder:
    """Build the Memory dense leg from DlightRAG's embedding endpoint."""
    from dlightrag.engine.ai.embedding import create_embedding_model

    return create_embedding_model(config.models.embedding, scheduler=scheduler, telemetry=telemetry)


class _GoalRoundCanceller:
    """The narrow runtime action a Web pause receives: cancel one owner-scoped Run.

    The Web service knows a live Run's ``continuation_kind`` from its own read; this adapter is
    the only thing it can do about one, so the conversation layer never receives the Run service.
    """

    def __init__(self, runs: RunService) -> None:
        self._runs = runs

    async def cancel(self, *, owner_id: str, run_id: str) -> object:
        return await self._runs.cancel(owner_id=owner_id, run_id=run_id)


def _actionable_error(exc: Exception) -> str:
    """Return fixed operator guidance without reflecting endpoints or secrets."""
    text = str(exc).lower()
    if "authentication" in text or "password" in text or "denied" in text:
        return "Dependency authentication failed; check deployment credentials."
    if "connection" in text or "timeout" in text or "timed out" in text:
        return "A configured dependency is unreachable; check deployment endpoints."
    return f"Dependency initialization failed ({type(exc).__name__})."


def _package_release() -> str:
    """Default the release to the running DlightRAG version.

    Read from installed metadata rather than the package root: importing
    ``dlightrag`` pulls the composition root and every transport with it.
    """
    try:
        from importlib.metadata import version

        return version("dlightrag")
    except Exception:
        return "unknown"


def _initialize_process(config: DlightragConfig) -> None:
    """Initialize tracing and logging, and bind the process-wide operational pool."""
    from dlightrag_observability.otlp import start_log_export, start_telemetry

    from dlightrag.adapters.postgres.core._pool import pg_pool
    from dlightrag.engine.ai.providers.base import provider_usage_buckets

    observability = config.observability
    release = observability.release or _package_release()
    # The application owns the vocabulary and the provider dialects; the package
    # owns how an observation is rendered and sent. A destination that is absent
    # or broken costs telemetry, never a boot.
    start_telemetry(
        span_types=SPAN_TYPES,
        usage_buckets=provider_usage_buckets,
        endpoint=observability.otlp_endpoint,
        headers=observability.otlp_headers,
        service_name=observability.service_name,
        environment=observability.environment,
        release=release,
        sample_rate=observability.sample_rate,
        timeout=observability.timeout,
    )
    # The same seam carries the process's own log records, so a trace and the
    # lines it logged meet in ClickHouse. stdout is untouched: the export is one
    # more handler, and no endpoint leaves it off with nothing else to configure.
    start_log_export(
        endpoint=observability.otlp_endpoint,
        headers=observability.otlp_headers,
        service_name=observability.service_name,
        environment=observability.environment,
        release=release,
        timeout=observability.timeout,
    )
    pg_pool.bind(config)


async def _close_process() -> None:
    """Close process-wide resources while always flushing both telemetry channels."""
    from dlightrag_observability.otlp import shutdown_log_export, shutdown_telemetry

    from dlightrag.adapters.postgres.core._pool import pg_pool

    try:
        await pg_pool.close()
    finally:
        shutdown_log_export()
        shutdown_telemetry()


def agent_confinement_policy(config: DlightragConfig) -> ConfinementPolicy:
    """Return what an Agent's processes may see beyond their own workspace.

    The corpus working directory and the project tree are the two trees an Agent may
    never see (ADR 0024), refused here rather than configured so a deployment cannot
    trade the property away. Declared layers belong to the capabilities that need
    them and arrive as those capabilities land.
    """
    policy = ConfinementPolicy(
        forbidden=(config.working_dir_path, Path.cwd()),
        per_owner=lambda owner_id: skill_read_layers(config, owner_id=owner_id),
    )
    for path, capability in skill_roots(config):
        policy.refuse(path, capability)
    return policy


def _agent_mailbox(config: DlightragConfig) -> AgentMailbox | None:
    """The Agent Mailbox this deployment configures, or None where it names no bucket."""
    from dlightrag.adapters.agent_mailbox import S3AgentMailbox

    mailbox = config.answer.agent.mailbox
    if not mailbox.enabled:
        return None
    # The configuration requires the rest of the settings whenever it names a bucket.
    return S3AgentMailbox(
        alias_domain=cast(str, mailbox.alias_domain),
        bucket=cast(str, mailbox.bucket),
        prefix=mailbox.prefix,
        endpoint=mailbox.endpoint,
        region=mailbox.region,
        access_key_id=cast(str, mailbox.access_key_id),
        secret_access_key=cast(str, mailbox.secret_access_key),
    )


def _session_memory_passes(
    config: DlightragConfig,
    *,
    run_store: PGRunStore,
    memory_store: PostgresMemoryStore,
    memory: MemoryService,
    scheduler: ModelScheduler,
    telemetry: Telemetry,
) -> SessionMemoryPassRunner | None:
    """The Session Memory Pass runner (ADR 0045), or None when the deployment turns it off.

    The pass calls the `query` role's model at `memory_consolidation.reasoning`.
    """
    settings = config.answer.agent.memory_consolidation
    if not settings.enabled:
        return None
    import uuid

    from dlightrag_memory import Memory

    from dlightrag.application.memory.session_pass import (
        SessionMemoryPass,
        SessionMemoryPassRunner,
        SessionPassSettings,
    )
    from dlightrag.application.settings.projections import model_settings_for_role
    from dlightrag.engine.ai.completion import CompletionModel

    session_pass = SessionMemoryPass(
        memory=Memory(memory_store),
        store=run_store,
        gate=memory,
        # The pass sets its own reasoning level, not the query role's, which serve other work;
        # the model profile still clamps a level it does not support at call time.
        model=CompletionModel(
            model_settings_for_role(config, "query").model_copy(
                update={"reasoning": settings.reasoning}
            ),
            scheduler=scheduler,
            telemetry=telemetry,
        ),
        settings=SessionPassSettings(
            max_batch_tokens=settings.max_batch_tokens,
            session_record_limit=settings.session_record_limit,
        ),
        holder=f"session-pass-{uuid.uuid4().hex}",
    )
    return SessionMemoryPassRunner(session_pass=session_pass, store=run_store)


def _compose(config: DlightragConfig) -> _ApplicationComponents:
    """Construct this process's collaborators from one resolved configuration."""
    from dlightrag_memory.postgres import PostgresMemoryStore
    from dlightrag_observability.otlp import OtlpTelemetry
    from PIL import Image

    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.adapters.mcp.personal_http import PersonalMcpClient
    from dlightrag.adapters.postgres.answer.agent_accounts import (
        PGAgentAccountSettingsStore,
        PGAgentAccountStore,
    )
    from dlightrag.adapters.postgres.answer.goals import PGAnswerGoalStore
    from dlightrag.adapters.postgres.answer.memory_settings import PGMemorySettingsStore
    from dlightrag.adapters.postgres.connections import PGConnectionsStore
    from dlightrag.adapters.postgres.corpus.corpus import (
        PGReadinessProbe,
        build_pg_corpus_backend,
    )
    from dlightrag.adapters.postgres.corpus.file_panel import PGFilePanelStore
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import PGMetadataIndex
    from dlightrag.adapters.postgres.corpus.pg_metadata_search import PGMetadataSearchStore
    from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry
    from dlightrag.adapters.postgres.corpus_mutations import build_corpus_mutations
    from dlightrag.adapters.postgres.model_catalogue import PGModelCatalogueStore
    from dlightrag.adapters.postgres.runtime.run_blob_store import PGRunBlobStore
    from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
    from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
    from dlightrag.application.access.control import (
        access_control_from_settings,
    )
    from dlightrag.application.agent_accounts import AgentAccountMaintenance, AgentAccounts
    from dlightrag.application.answer.goal_rounds import GoalRoundDriver
    from dlightrag.application.answer.queued_turns import QueuedTurnDriver
    from dlightrag.application.answer_runs.service import AnswerService
    from dlightrag.application.connections.service import Connections
    from dlightrag.application.corpus_admin.mutations import (
        CorpusMutationExecutor,
        CorpusStageReclaimer,
        validate_corpus_mutation_prepared_input,
    )
    from dlightrag.application.corpus_admin.service import CorpusAdmin
    from dlightrag.application.health import ApplicationHealth
    from dlightrag.application.live_browser.service import LiveBrowserService
    from dlightrag.application.memory.service import MemoryService
    from dlightrag.application.model_catalogue.service import ModelCatalogueAdmin
    from dlightrag.application.policies.service import (
        RuntimePolicies,
    )
    from dlightrag.application.privacy import Privacy
    from dlightrag.application.retrieval._answer_projection import (
        AnswerQueryImagePreparer,
        project_answer_retrieval,
    )
    from dlightrag.application.retrieval.execution import RetrievalExecutor
    from dlightrag.application.retrieval.service import RetrievalService
    from dlightrag.application.runs import RunService
    from dlightrag.application.settings.projections import (
        access_settings,
        agent_browser_settings,
        answer_capability_settings,
        answer_executor_settings,
        answer_model_runtime_settings,
        answer_resource_settings,
        corpus_admin_settings,
        live_preview_settings,
        model_profile_for_role,
        model_settings_for_role,
        rag_settings,
        rerank_scoring_model_settings,
        retrieval_settings,
    )
    from dlightrag.application.web_conversations.service import WebConversationService
    from dlightrag.engine.ai.catalog import (
        MODEL_CATALOGUE,
        catalogue_overlay_revision,
        parse_catalogue_overlay,
    )
    from dlightrag.engine.ai.fingerprints import (
        ModelInvocationFingerprint,
        model_invocation_fingerprint,
    )
    from dlightrag.engine.ai.media import MAX_DECODE_IMAGE_PIXELS
    from dlightrag.engine.ai.providers.base import provider_usage_buckets
    from dlightrag.engine.ai.scheduler import ModelScheduler
    from dlightrag.engine.ai.settings import (
        CHAT_MODEL_SELECTORS,
        MODEL_ROLE_NAMES,
        ChatModelSelector,
    )
    from dlightrag.engine.ai.telemetry import safe_log_text
    from dlightrag.engine.ai.vision import ModelImageCapabilities
    from dlightrag.engine.answer.agent_browser.accounts import (
        AgentAccountsBinding,
        has_mailbox,
    )
    from dlightrag.engine.answer.agent_browser.contracts import (
        AgentBrowserBinding,
    )
    from dlightrag.engine.answer.agentspace import agentspace_reclaimer
    from dlightrag.engine.answer.capabilities import (
        AnswerCapabilityCoordinator,
        AnswerCapabilityView,
    )
    from dlightrag.engine.answer.execution.executor import AnswerExecutor
    from dlightrag.engine.answer.execution.resources import AnswerResourceResolver
    from dlightrag.engine.answer.goal_service import AnswerGoalService
    from dlightrag.engine.answer.model_runtime import AnswerModelRuntime
    from dlightrag.engine.credential_cipher import KEYRING_FILE, deployment_cipher
    from dlightrag.engine.dependencies import DependencyComponent
    from dlightrag.engine.rag.corpus.downloads import SourceDownloadService
    from dlightrag.engine.rag.retrieval.federation import FederatedReranker
    from dlightrag.engine.rag.retrieval.language import ProfileBM25Languages
    from dlightrag.engine.rag.retrieval.rerank import build_rerank_func
    from dlightrag.engine.rag.retrieval.rerank_breaker import RerankBreaker
    from dlightrag.engine.rag.retrieval.runtime import RetrievalPlannerRuntime
    from dlightrag.engine.rag.workspace.pool import WorkspacePool
    from dlightrag.engine.rag.workspace.ports import CorpusSchemaError
    from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag
    from dlightrag.engine.runtime.contracts import RunKind
    from dlightrag.engine.runtime.coordinator import (
        RunAgentspaceReclaimer,
        RunCoordinator,
        RunExecutor,
    )
    from dlightrag.engine.runtime.errors import IncompatibleActiveRunError, RunExecutionError

    # Large document scans are DlightRAG product policy, not an AI package import side effect.
    Image.MAX_IMAGE_PIXELS = MAX_DECODE_IMAGE_PIXELS
    startup_catalogue = parse_catalogue_overlay(
        config.models.catalogue_data(),
        source="startup model catalogue",
        path="models.catalogue",
    )
    health = ApplicationHealth(readiness_probe=PGReadinessProbe())
    health.set_agent_shell_confinement(confinement_state(config.answer.agent.execution_environment))
    scheduler = ModelScheduler(max_concurrency=config.models.max_concurrency)
    # One process-wide handle: startup installs the provider it resolves, so the
    # components built here and the exporter that starts later are the same seam.
    telemetry = OtlpTelemetry(span_types=SPAN_TYPES, usage_buckets=provider_usage_buckets)
    # Every workspace and the federation pass rerank through one configured endpoint, so
    # they share one breaker: a down reranker is noticed once, not once per workspace.
    rerank_breaker = RerankBreaker()
    corpus_backend = build_pg_corpus_backend(config)

    # Image capability is role-specific but cached per resolved model config,
    # so roles that share one model share one probe.
    capabilities = AnswerCapabilityCoordinator(
        settings=answer_capability_settings(config),
        profile_for_role=lambda role: model_profile_for_role(config, role),
        model_settings_for_role=lambda role: model_settings_for_role(config, role),
        rerank_model_settings=lambda: rerank_scoring_model_settings(config),
        image_capabilities=ModelImageCapabilities(scheduler=scheduler),
        on_answer_capability=health.set_answer_image_capability,
    )

    def workspace_config(workspace_id: str) -> DlightragConfig:
        deployment = config.deployment.model_copy(update={"workspace": workspace_id})
        return config.model_copy(update={"deployment": deployment})

    async def build_workspace(workspace_id: str) -> WorkspaceRag:
        resolved = workspace_config(workspace_id)
        settings = rag_settings(resolved)
        backend = build_pg_corpus_backend(resolved)
        try:
            runtime = await WorkspaceRag.acreate(
                workspace_id=workspace_id,
                settings=settings,
                backend=backend,
                scheduler=scheduler,
                telemetry=telemetry,
                rerank_supports_vision=capabilities.rerank_supports_vision,
                rerank_breaker=rerank_breaker,
            )
        except CorpusSchemaError:
            raise
        except Exception as exc:
            raise RuntimeError(_actionable_error(exc)) from exc
        logger.info("Created WorkspaceRag for workspace '%s'", safe_log_text(workspace_id))
        return runtime

    default_workspace = config.deployment.workspace_id

    def workspace_unavailable(workspace_id: str, component: DependencyComponent) -> None:
        if workspace_id == default_workspace:
            health.mark_component_degraded(component)

    def workspace_available(workspace_id: str, recovered: DependencyComponent | None) -> None:
        if workspace_id == default_workspace:
            # A built workspace reached corpus storage, and whatever it waited for.
            health.mark_component_healthy("corpus_storage")
            if recovered is not None:
                health.mark_component_healthy(recovered)

    pool = WorkspacePool(
        build=build_workspace,
        on_workspace_unavailable=workspace_unavailable,
        on_workspace_available=workspace_available,
    )
    cursor_secrets = CursorSecretBox(
        (
            f"{config.storage.postgres.host}\0"
            f"{config.storage.postgres.database}\0"
            f"{config.storage.postgres.password}"
        ).encode()
    )

    source_download_settings = rag_settings(config)
    corpora = CorpusAdmin(
        settings=corpus_admin_settings(config),
        pool=pool,
        maintenance=corpus_backend.maintenance,
        file_panel=PGFilePanelStore(),
        metadata_search=PGMetadataSearchStore(),
        promotion_worker=corpus_backend.promotion,
        source_download_for=lambda workspace: SourceDownloadService(
            settings=source_download_settings,
            metadata_index=PGMetadataIndex(workspace=workspace),
            workspace_id=workspace,
        ),
        # Stable across workers sharing the operational database. The cursor
        # carries no authorization state and expires on credential rotation.
        file_panel_cursor_secret=cursor_secrets.derive("dlightrag-file-panel-cursor"),
        metadata_search_cursor_secret=cursor_secrets.derive("dlightrag-metadata-search-cursor"),
        workspace_catalog_cursor_secret=cursor_secrets.derive("dlightrag-workspace-catalog-cursor"),
    )
    access_control = access_control_from_settings(
        access_settings(config), creators=PGWorkspaceRegistry()
    )

    model_catalogue = ModelCatalogueAdmin(
        store=PGModelCatalogueStore(initial_revision=catalogue_overlay_revision(())),
        configured_models=lambda: tuple(
            config.models.chat.resolve(role) for role in MODEL_ROLE_NAMES
        ),
        on_publish=lambda _snapshot: capabilities.invalidate_model_catalogue(),
        read_only=config.is_reader,
    )

    models = AnswerModelRuntime(
        settings=answer_model_runtime_settings(config),
        scheduler=scheduler,
        telemetry=telemetry,
        answer_image_policy=capabilities.answer_image_policy,
        vlm_image_policy=capabilities.vlm_image_policy,
        vlm_profile=lambda: capabilities.model_profile("vlm"),
    )
    resources = AnswerResourceResolver(
        settings=answer_resource_settings(config),
        models=models,
        capabilities=capabilities,
    )
    schema_index = PGMetadataIndex(workspace=config.deployment.workspace_id)

    async def schema_lookup(workspaces: Sequence[str]) -> dict[str, Any]:
        return await schema_index.get_field_schema(workspaces=tuple(workspaces))

    def federated_reranker_factory() -> FederatedReranker | None:
        """Build the shared federation reranker lazily, on first flagged request.

        Deferred past startup so the capabilities vision probe has settled;
        built from the same product reranker settings the workspaces use.
        """
        resolved_rerank = rag_settings(config).rerank
        return build_rerank_func(
            resolved_rerank,
            scoring_settings=rerank_scoring_model_settings(config),
            scheduler=scheduler,
            supports_vision=capabilities.rerank_supports_vision,
            telemetry=telemetry,
            breaker=rerank_breaker,
        )

    model_selectors: dict[str, ChatModelSelector] = {name: name for name in CHAT_MODEL_SELECTORS}

    def fingerprint_for_role(role: str) -> ModelInvocationFingerprint:
        """The invocation fingerprint a pinned role resolves to; unknown roles refuse."""
        selector = model_selectors.get(role)
        if selector is None:
            raise ValueError(f"unknown pinned model role: {role}")
        return model_invocation_fingerprint(model_settings_for_role(config, selector))

    run_retention_seconds = config.runtime.run_retention_days * 24 * 3600

    retrieval = RetrievalService(
        pool=pool,
        planners=RetrievalPlannerRuntime(
            model_settings=model_settings_for_role(config, "extract"),
            default_profile=lambda: capabilities.model_profile("extract"),
            scheduler=scheduler,
            telemetry=telemetry,
        ),
        schema_lookup=schema_lookup,
        image_preparer=AnswerQueryImagePreparer(capabilities=capabilities, models=models),
        projector=project_answer_retrieval,
        settings=retrieval_settings(config),
        telemetry=telemetry,
        model_profile_for_role=lambda role: capabilities.model_profile(role),
        model_invocation_fingerprint_for_role=fingerprint_for_role,
        federated_reranker_factory=federated_reranker_factory,
    )

    run_blob_store = PGRunBlobStore()
    run_store = PGRunStore(
        query_max_nonterminal_runs=config.runtime.query.max_nonterminal_runs,
        corpus_mutation_max_nonterminal_runs=(config.runtime.corpus_mutation.max_nonterminal_runs),
        promotion_doc_threshold=config.corpus.promotion.doc_threshold,
        promotion_chunk_threshold=config.corpus.promotion.chunk_threshold,
    )
    # An administrator's settings are kept in the Run store's schema, which every process shares,
    # and each request reads them afresh.
    policies = RuntimePolicies(store=run_store)
    # An owner's Conversation Contribution lives in the same schema, one row per owner who chose.
    privacy = Privacy(store=run_store)
    memory_embedder = _memory_embedder(config, scheduler=scheduler, telemetry=telemetry)
    memory_store = PostgresMemoryStore(
        pool_factory=_operational_pool_factory(),
        embedder=memory_embedder,
        languages=ProfileBM25Languages(config.corpus.retrieval.bm25_profiles),
    )
    memory_settings = PGMemorySettingsStore()
    goal_service = AnswerGoalService(store=PGAnswerGoalStore())
    memory = MemoryService(
        memory_store,
        settings_store=memory_settings,
        # Turning memory back on restarts every Session, which the Run store keeps.
        sessions=run_store,
        superseded_retention_days=config.runtime.run_retention_days,
        # Stable across workers sharing the operational database. Cursors
        # carry no authorization state and expire on credential rotation.
        memory_list_cursor_secret=cursor_secrets.derive("dlightrag-memory-list-cursor"),
    )
    agent_config = config.answer.agent
    search_toolchain = SearchToolchain(
        fd=agent_config.fd_path,
        ripgrep=agent_config.ripgrep_path,
        cache_root=(
            Path(agent_config.search_tool_cache_root)
            if agent_config.search_tool_cache_root is not None
            else None
        ),
        auto_install=agent_config.search_tool_auto_install,
    )
    # One key ring seals Connection credentials and Agent Account passwords alike.
    cipher = deployment_cipher(config.working_dir_path / KEYRING_FILE, create=not config.is_reader)
    personal_mcp = PersonalMcpClient()
    connections = Connections(
        store=PGConnectionsStore(),
        mcp=personal_mcp,
        # The same bounded transport serves on-demand resource reads; it is a separate
        # port so a deployment without it advertises no resource tools.
        resources=personal_mcp,
        oauth=PersonalOAuthClient(),
        policy=config.answer.agent.connections,
        cipher=cipher,
    )

    agent_account_store = PGAgentAccountStore()
    browser_settings = agent_browser_settings(config)
    # Whether the deployment lets the Agent register is one answer: there is an Agent Browser to
    # register with, and the operator has not turned account registration off. The binding,
    # health, Settings and acceptance all take it from here.
    registration_allowed = (
        browser_settings is not None and config.answer.agent.browser.account_registration
    )
    agent_browser = None
    live_sessions = None
    if browser_settings is not None:
        # Playwright loads only where an Agent Browser is configured.
        from dlightrag.adapters.agent_browser.live import (
            LiveBrowserSessions,
        )
        from dlightrag.adapters.agent_browser.pool import (
            PooledBrowserProvider,
        )
        from dlightrag.adapters.postgres.runtime.browser_leases import PGAgentBrowserLeaseStore

        browser_leases = PGAgentBrowserLeaseStore()
        browser_provider = PooledBrowserProvider(
            endpoints=browser_settings.endpoints,
            headed_endpoints=browser_settings.headed_endpoints,
            egress_proxy=browser_settings.egress_proxy,
            chromium_sandbox=browser_settings.chromium_sandbox,
            connect_timeout_seconds=browser_settings.connect_timeout_seconds,
            leases=browser_leases,
        )
        agent_browser = AgentBrowserBinding(
            browser_provider,
            browser_settings,
            AgentAccountsBinding(
                agent_account_store,
                cipher,
                _agent_mailbox(config),
                registration_allowed=registration_allowed,
            ),
        )
        # The live preview of a leased page, watched and co-driven over the Web surface. It
        # leases nothing itself: it shows the page a Run already holds, and ends with it,
        # so the pool tells it when a lease goes away rather than leaving a session open.
        live_sessions = LiveBrowserSessions(
            provider=browser_provider,
            leases=browser_leases,
            settings=live_preview_settings(config),
        )
        browser_provider.on_retire(live_sessions.retired)

    accounts = None if agent_browser is None else agent_browser.accounts
    agent_accounts = AgentAccounts(
        directory=agent_account_store,
        settings_store=PGAgentAccountSettingsStore(),
        available=accounts is not None,
        registration_allowed=registration_allowed,
    )
    health.set_agent_browser(
        endpoints=len(config.answer.agent.browser.endpoints),
        sandbox=config.answer.agent.browser.chromium_sandbox,
        accounts=accounts is not None,
        registration_allowed=registration_allowed,
        mailbox=has_mailbox(accounts),
    )

    skills_factory = skills_bundle_factory(config, ensure_dirs=True)
    # One process-wide content fingerprint index: every reader-facing surface asks it
    # whether a passage is a protected Skill document's own text (ADR 0061).
    skill_protection = SkillProtection(skills_factory)

    answer_executor = AnswerExecutor(
        store=run_store,
        blob_store=run_blob_store,
        pool=pool,
        warm=retrieval.warm,
        retrieve=retrieval.retrieve_result,
        planning=retrieval,
        models=models,
        capabilities=capabilities,
        resources=resources,
        settings=answer_executor_settings(config),
        telemetry=telemetry,
        model_invocation_fingerprint_for_role=fingerprint_for_role,
        execution_environment=config.answer.agent.execution_environment,
        shell_confinement=agent_confinement_policy(config),
        workspace_root=config.answer.agent.workspace_root,
        session_notes_limits=SessionNotesLimits(
            max_count=config.answer.agent.session_notes.max_count,
            max_bytes=config.answer.agent.session_notes.max_bytes,
        ),
        search_toolchain=search_toolchain,
        working_dir=config.deployment.working_dir,
        memory_store=memory_store,
        goal_service=goal_service,
        memory_recall_enabled=memory.recall_enabled,
        memory_capability_current=memory.capability_current,
        connection_tool_resolver=connections.restore_research,
        connection_resource_operations=connections,
        skills_bundle_factory=skills_factory,
        # Lazy, so the index is built by the first reader projection (a conversation page, an
        # answer stream or a publication) rather than on every process start; the value
        # resolves through SkillProtection's own revalidation, so a Skill installed later is
        # covered without a restart. Publication asks the prose-only index: executable kit
        # strings are engine output on a case/page, not reader-facing Skill prose.
        protected_index=lambda: skill_protection.publication_index,
        browser=agent_browser,
        on_dependency_unavailable=health.mark_component_degraded,
        on_dependency_recovered=health.mark_component_healthy,
    )

    retrieval_executor = RetrievalExecutor(
        operation=retrieval,
        telemetry=telemetry,
        timeout_seconds=config.corpus.retrieval.timeout,
        model_invocation_fingerprint_for_role=fingerprint_for_role,
        on_dependency_unavailable=health.mark_component_degraded,
        on_dependency_recovered=health.mark_component_healthy,
    )
    executors: dict[RunKind, RunExecutor] = {
        "answer": answer_executor,
        "retrieval": retrieval_executor,
    }
    workspace_reclaimers: list[RunAgentspaceReclaimer] = []
    agent_reclaimer = agentspace_reclaimer(
        execution_environment=config.answer.agent.execution_environment,
        workspace_root=config.answer.agent.workspace_root,
    )
    if agent_reclaimer is not None:
        workspace_reclaimers.append(agent_reclaimer)
    if not config.is_reader:
        executors["corpus_mutation"] = CorpusMutationExecutor(
            pool=pool,
            maintenance=corpus_backend.maintenance,
            store=run_store,
            corpus_root=config.corpus_dir_path,
            workspace_exists=corpora.workspace_exists,
            on_dependency_unavailable=health.mark_component_degraded,
            on_dependency_recovered=health.mark_component_healthy,
        )
        workspace_reclaimers.append(CorpusStageReclaimer(config.corpus_dir_path))

    async def validate_active_runs() -> None:
        """Compose operation-owned durable-input checks at the process boundary."""
        async for requirement in run_store.iter_active_run_requirements():
            kind = requirement.get("run_kind")
            prepared = requirement.get("prepared_input")
            try:
                if not isinstance(prepared, Mapping):
                    raise ValueError("prepared_input must be an object")
                if kind == "answer":
                    answer_executor.validate_active_prepared_input(prepared)
                elif kind == "retrieval":
                    retrieval_executor.validate_active_prepared_input(prepared)
                elif kind == "corpus_mutation":
                    validate_corpus_mutation_prepared_input(prepared)
                else:
                    raise ValueError("unsupported active run kind")
            except IncompatibleActiveRunError:
                if kind == "answer":
                    # The Answer executor settles this Run without external effects;
                    # one obsolete accepted contract must not prevent new admission.
                    continue
                raise
            except (AttributeError, KeyError, TypeError, ValueError, RunExecutionError) as exc:
                raise IncompatibleActiveRunError(
                    f"active {kind or 'unknown'} runs use an incompatible durable input schema; "
                    "drain or owner-cancel them before deployment"
                ) from exc

    session_memory_passes = _session_memory_passes(
        config,
        run_store=run_store,
        memory_store=memory_store,
        memory=memory,
        scheduler=scheduler,
        telemetry=telemetry,
    )
    coordinator = RunCoordinator(
        store=run_store,
        executors=executors,
        query_worker_concurrency=config.runtime.query.worker_concurrency,
        corpus_mutation_worker_concurrency=(config.runtime.corpus_mutation.worker_concurrency),
        workspace_reclaimers=workspace_reclaimers,
    )
    if session_memory_passes is not None:
        coordinator.observe_settled(session_memory_passes.run_settled)
    retrieval.bind_runtime(store=run_store, coordinator=coordinator)

    corpus_mutations = build_corpus_mutations(
        config=config,
        store=run_store,
        workspace_exists=corpora.workspace_exists,
        coordinator=coordinator,
    )
    cancelled_observers: list[Any] = [corpus_mutations.discard_cancelled_run]

    async def _run_cancelled(view: Any) -> None:
        for observer in tuple(cancelled_observers):
            await observer(view)

    runs = RunService(
        store=run_store,
        scheduler=coordinator,
        on_cancelled=_run_cancelled,
    )

    async def _cancel_local(owner: str, run_id: str) -> None:
        coordinator.cancel_local(owner, run_id)

    cancellation_listener = run_store.build_cancellation_listener(
        worker_id=coordinator.worker_id,
        on_cancel=_cancel_local,
    )

    answers = AnswerService(
        store=run_store,
        blob_store=run_blob_store,
        coordinator=coordinator,
        retrieval=retrieval,
        capabilities=capabilities,
        capability_view=AnswerCapabilityView(capabilities),
        models=models,
        resources=resources,
        model_invocation_fingerprint_for_role=fingerprint_for_role,
        research_tool_declarations=answer_executor.research_tool_declarations,
        memory_capability=memory.execution_capability,
        agent_may_register=agent_accounts.may_register,
        policies=policies,
        bind_research=connections.bind_research,
        # Stable across workers sharing the operational database. Cursors
        # carry no authorization state and expire on credential rotation.
        child_roster_cursor_secret=cursor_secrets.derive("dlightrag-child-roster-cursor"),
        protected_skill_documents=skill_protection.documents,
        protected_text=skill_protection.mask,
        run_retention_seconds=run_retention_seconds,
    )
    # The live preview exists wherever the Web surface does; without an Agent Browser it
    # answers `not_configured` and still serves the screenshots a Run stored earlier.
    live_browser = LiveBrowserService(sessions=live_sessions, runs=answers)
    web_store = PGWebConversationStore(run_store=run_store)
    # The driver and the Web conversations need each other: a round becomes a conversation
    # turn, and a human resume asks the driver to start its round. The driver is created first
    # and receives the conversations through its binding, so neither constructor closes a cycle.
    goal_rounds = GoalRoundDriver(
        goals=goal_service,
        answers=answers,
        runs=runs,
    )
    web_conversations = WebConversationService(
        store=web_store,
        answers=answers,
        max_attachments=config.answer.generation.max_attachments,
        # Stable across workers sharing the operational database. Cursors
        # carry no authorization state and naturally expire on credential rotation.
        cursor_secret=cursor_secrets.derive("dlightrag-web-conversation-cursor"),
        goals=goal_service,
        goal_rounds=goal_rounds,
        goal_round_canceller=_GoalRoundCanceller(runs),
        workspace_heads=agent_reclaimer,
    )
    goal_rounds.bind_round_turns(web_conversations)
    # One settlement observer, ordered: the reader's queued messages are delivered first and
    # the goal round driver then decides whether a round may follow. Two observers would race
    # for the same lane; see ADR 0071.
    queued_turns = QueuedTurnDriver(
        turns=web_conversations,
        runs=runs,
        goal_rounds=goal_rounds,
    )
    # The goal driver requests its next round as a next-turn continuation; this is the one
    # delivery path that admits it, so a continued goal moves immediately while a human send
    # still keeps the Session first.
    goal_rounds.bind_pending_delivery(queued_turns.deliver_pending_goal_round)
    coordinator.observe_settled(queued_turns.run_settled)
    cancelled_observers.append(queued_turns.run_cancelled)
    MODEL_CATALOGUE.replace_startup(startup_catalogue)

    return _ApplicationComponents(
        connections=connections,
        agent_accounts=agent_accounts,
        # A writer re-seals Agent Account envelopes whatever its browser configuration, since
        # envelopes sealed earlier must still follow a rotation; a reader writes nothing.
        agent_account_maintenance=None
        if config.is_reader
        else AgentAccountMaintenance(store=agent_account_store, cipher=cipher),
        health=health,
        capabilities=capabilities,
        model_catalogue=model_catalogue,
        pool=pool,
        models=models,
        run_store=run_store,
        web_store=web_store,
        coordinator=coordinator,
        cancellation_listener=cancellation_listener,
        validate_active_runs=validate_active_runs,
        corpora=corpora,
        access_control=access_control,
        policies=policies,
        privacy=privacy,
        corpus_mutations=corpus_mutations,
        retrieval=retrieval,
        runs=runs,
        answers=answers,
        goals=goal_service,
        skill_protection=skill_protection,
        live_browser=live_browser,
        memory=memory,
        memory_store=memory_store,
        memory_embedder=memory_embedder,
        web_conversations=web_conversations,
        search_toolchain=search_toolchain,
        # Active goals are taken up automatically at startup; ``deployment.goal_recovery`` decides
        # whether this process is the one that scans (the MCP service sets it false).
        recover_goal_rounds=goal_rounds.resume_after_restart,
        initialize_process=_initialize_process,
        close_agent_execution=answer_executor.aclose,
        close_process=_close_process,
        **(
            {
                "start_session_memory_passes": session_memory_passes.start,
                "close_session_memory_passes": session_memory_passes.aclose,
            }
            if session_memory_passes is not None
            else {}
        ),
    )


async def create_application(
    config: DlightragConfig | None = None,
    *,
    web_enabled: bool = False,
) -> Application:
    """Compose, start, and return one Application."""
    resolved = config or get_config()
    application = Application(resolved, _compose(resolved), web_enabled=web_enabled)
    await application.astart()
    return application


__all__ = ["create_application"]
