# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup and shutdown contract of the local composition root."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import pytest

from dlightrag._compose import _memory_embedder
from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
from dlightrag.application import Application, ApplicationClosedError
from dlightrag.application import application as application_module
from dlightrag.application.access import AllowAllAccessControl
from dlightrag.application.answer_runs import AnswerService
from dlightrag.application.application import _ApplicationComponents
from dlightrag.application.config import DlightragConfig
from dlightrag.application.corpus_admin import CorpusAdmin
from dlightrag.application.errors import StorageSchemaError
from dlightrag.application.health import ApplicationHealth
from dlightrag.application.memory import MemoryService
from dlightrag.application.retrieval import PinnedRetrievalModel, RetrievalService
from dlightrag.application.retrieval.execution import validate_active_retrieval_input
from dlightrag.application.settings import model_settings_for_role
from dlightrag.application.web_conversations import (
    WebConversationSchemaError,
    WebConversationService,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import (
    ModelInvocationFingerprint,
    model_invocation_fingerprint,
)
from dlightrag.engine.ai.settings import CHAT_MODEL_SELECTORS
from dlightrag.engine.answer.capabilities import AnswerCapabilityCoordinator
from dlightrag.engine.answer.execution.input import (
    PinnedModelProfile,
    model_reasoning_settings,
    new_resource_identity,
    validate_active_answer_input,
)
from dlightrag.engine.answer.model_runtime import AnswerModelRuntime
from dlightrag.engine.dependencies import ParserUnavailableError, ProviderUnavailableError
from dlightrag.engine.rag.workspace.pool import WorkspaceUnavailableError
from dlightrag.engine.rag.workspace.ports import CorpusSchemaError
from dlightrag.engine.rag.workspace.workspaces import normalize_workspace
from dlightrag.engine.runtime.coordinator import RunCoordinator
from dlightrag.engine.runtime.errors import (
    IncompatibleActiveRunError,
    RunSchemaError,
)
from tests.config_helpers import mutate_config

_CLOSE_ORDER = [
    "close:corpora",
    "close:coordinator",
    "close:agent_execution",
    "close:listener",
    "close:web_conversations",
    "close:retrieval",
    "close:models",
    "close:pool",
    "close:memory_embedder",
]


@pytest.fixture(autouse=True)
async def _release_domain_pool():
    """Startup binds the process-wide domain pool; unbind it between tests."""
    yield
    await pg_pool.close()


class _Recorder:
    """One shared, ordered log of everything startup and shutdown did."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def add(self, event: str) -> None:
        self.events.append(event)

    def started(self) -> list[str]:
        return [event for event in self.events if not event.startswith("close:")]

    def closed(self) -> list[str]:
        return [event for event in self.events if event.startswith("close:")]


class _Collaborator:
    """A collaborator that records its own lifecycle and can fail on demand."""

    def __init__(self, recorder: _Recorder, name: str) -> None:
        self._recorder = recorder
        self._name = name
        self.close_error: BaseException | None = None

    def _record(self, event: str) -> None:
        self._recorder.add(f"{self._name}:{event}")

    async def aclose(self) -> None:
        self._recorder.add(f"close:{self._name}")
        if self.close_error is not None:
            raise self.close_error


class _Capabilities(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "capabilities")

    def resolve_profiles(self) -> None:
        self._record("resolve_profiles")

    async def probe_rerank(self) -> None:
        self._record("probe_rerank")

    async def probe_chat(self) -> None:
        # Overlaps the default workspace's warm-up, so where it lands among the
        # recorded events is not a contract; its own test pins the overlap.
        pass


class _SearchToolchain:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder
        self.error: Exception | None = None
        self.provenance = {
            "fd": {"path": "/bin/fd", "version": "10.5.0", "sha256": "a" * 64},
            "rg": {"path": "/bin/rg", "version": "15.2.0", "sha256": "b" * 64},
        }

    async def ensure(self) -> None:
        self._recorder.add("search_toolchain:ensure")
        if self.error is not None:
            raise self.error


class _Pool(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "pool")
        self.acquire_error: Exception | None = None

    async def acquire(self, workspace_id: str) -> object:
        self._record(f"acquire:{workspace_id}")
        if self.acquire_error is not None:
            raise self.acquire_error
        return object()


class _RunStore(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "run_store")
        self.initialize_error: Exception | None = None
        self.requirements: tuple[dict[str, Any], ...] = ()

    async def initialize(self, *, validate_only: bool = False) -> None:
        self._record(f"initialize:{validate_only}")
        if self.initialize_error is not None:
            raise self.initialize_error

    async def iter_active_run_requirements(self) -> AsyncIterator[dict[str, Any]]:
        self._record("iter_active_run_requirements")
        for requirement in self.requirements:
            yield requirement


class _WebStore(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "web_store")
        self.initialize_error: Exception | None = None

    async def initialize(self, *, validate_only: bool = False) -> None:
        self._record(f"initialize:{validate_only}")
        if self.initialize_error is not None:
            raise self.initialize_error


class _MemoryStore(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "memory_store")

    async def initialize(self) -> None:
        self._record("initialize")


class _CancellationListener:
    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder
        self.ready = asyncio.Event()
        self.ready_on_start = True

    async def start(self) -> None:
        self.recorder.add("listener:start")
        if self.ready_on_start:
            self.ready.set()

    def become_ready(self) -> None:
        self.recorder.add("listener:ready")
        self.ready.set()

    async def aclose(self) -> None:
        self.recorder.add("close:listener")


class _Coordinator(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "coordinator")
        self.is_started = False
        self.start_error: Exception | None = None

    async def start(self) -> None:
        self._record("start")
        if self.start_error is not None:
            raise self.start_error
        self.is_started = True

    async def aclose(self) -> None:
        self.is_started = False
        await super().aclose()


class _RaceExecution(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "agent_execution")
        self.closed = asyncio.Event()
        self.starts_after_close = 0

    async def start_process(self) -> None:
        if self.closed.is_set():
            self.starts_after_close += 1

    async def aclose(self) -> None:
        self._recorder.add("close:agent_execution")
        self.closed.set()
        await asyncio.sleep(0)


class _LiveProducerCoordinator(_Coordinator):
    def __init__(self, recorder: _Recorder, execution: _RaceExecution) -> None:
        super().__init__(recorder)
        self._execution = execution
        self._producer: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await super().start()
        self._producer = asyncio.create_task(self._produce_after_adapter_close())

    async def _produce_after_adapter_close(self) -> None:
        await self._execution.closed.wait()
        if self.is_started:
            await self._execution.start_process()

    async def wait_for_producer(self) -> None:
        assert self._producer is not None
        await self._producer


class _Corpora(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "corpora")
        self.initialize_error: Exception | None = None
        self.recovery_error: Exception | None = None

    async def initialize(self) -> None:
        self._record("initialize")
        if self.initialize_error is not None:
            raise self.initialize_error

    async def start_recovery(self) -> None:
        self._record("start_recovery")
        if self.recovery_error is not None:
            raise self.recovery_error

    def start_promotion_worker(self) -> None:
        self._record("start_promotion_worker")


class _Memory(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "memory")

    async def purge_expired(self) -> int:
        self._record("purge_expired")
        return 0


class _Retrieval(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "retrieval")

    def planner_for(self, model_profile: object | None = None) -> object:
        del model_profile
        self._record("planner_for")
        return object()


class _WebConversations(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "web_conversations")

    async def start_retention(self) -> None:
        self._record("start_retention")


class _Maintenance(_Collaborator):
    def __init__(self, recorder: _Recorder) -> None:
        super().__init__(recorder, "agent_account_maintenance")

    def start(self) -> None:
        self._record("start")


class _Parts:
    """The fakes behind one Application, addressable by the test that wired them."""

    def __init__(self) -> None:
        self.recorder = _Recorder()
        self.health = ApplicationHealth(readiness_probe=None)
        self.capabilities = _Capabilities(self.recorder)
        self.pool = _Pool(self.recorder)
        self.models = _Collaborator(self.recorder, "models")
        self.run_store = _RunStore(self.recorder)
        self.web_store = _WebStore(self.recorder)
        self.memory_store = _MemoryStore(self.recorder)
        self.memory_embedder = _Collaborator(self.recorder, "memory_embedder")
        self.agent_execution = _Collaborator(self.recorder, "agent_execution")
        self.coordinator = _Coordinator(self.recorder)
        self.cancellation_listener = _CancellationListener(self.recorder)
        self.corpora = _Corpora(self.recorder)
        self.retrieval = _Retrieval(self.recorder)
        self.runs = object()
        self.answers = object()
        self.memory = _Memory(self.recorder)
        self.agent_accounts = object()
        self.web_conversations = _WebConversations(self.recorder)
        self.search_toolchain: _SearchToolchain | None = None
        self.agent_account_maintenance: _Maintenance | None = None

    def application(
        self,
        config: DlightragConfig,
        *,
        web_enabled: bool = True,
    ) -> Application:
        def current_fingerprint(role: str) -> ModelInvocationFingerprint:
            return model_invocation_fingerprint(model_settings_for_role(config, cast(Any, role)))

        async def validate_active_runs() -> None:
            # Deliberately strict injected validator: Application propagates its failures.
            # Production composition's per-Answer isolation is exercised by the PG tests.
            async for requirement in self.run_store.iter_active_run_requirements():
                kind = requirement.get("run_kind")
                prepared = requirement.get("prepared_input")
                if not isinstance(prepared, Mapping):
                    raise IncompatibleActiveRunError("incompatible durable input schema")
                if kind == "answer":
                    validate_active_answer_input(
                        prepared,
                        model_invocation_fingerprint_for_role=cast(Any, current_fingerprint),
                        model_settings_for_role=config.models.chat.resolve,
                    )
                elif kind == "retrieval":
                    validate_active_retrieval_input(
                        prepared,
                        model_invocation_fingerprint_for_role=current_fingerprint,
                    )
                else:
                    raise IncompatibleActiveRunError("incompatible durable input schema")

        return Application(
            config,
            _ApplicationComponents(
                health=self.health,
                capabilities=cast(AnswerCapabilityCoordinator, self.capabilities),
                pool=cast(Any, self.pool),
                models=cast(AnswerModelRuntime, self.models),
                run_store=cast(PGRunStore, self.run_store),
                web_store=cast(PGWebConversationStore, self.web_store),
                coordinator=cast(RunCoordinator, self.coordinator),
                cancellation_listener=cast(Any, self.cancellation_listener),
                validate_active_runs=validate_active_runs,
                corpora=cast(CorpusAdmin, self.corpora),
                access_control=AllowAllAccessControl(),
                retrieval=cast(RetrievalService, self.retrieval),
                runs=cast(Any, self.runs),
                answers=cast(AnswerService, self.answers),
                memory=cast(MemoryService, self.memory),
                memory_store=cast(Any, self.memory_store),
                memory_embedder=cast(Any, self.memory_embedder),
                web_conversations=cast(WebConversationService, self.web_conversations),
                agent_accounts=cast(Any, self.agent_accounts),
                search_toolchain=cast(Any, self.search_toolchain),
                agent_account_maintenance=cast(Any, self.agent_account_maintenance),
                close_agent_execution=self.agent_execution.aclose,
            ),
            web_enabled=web_enabled,
        )


def _requirement(
    config: DlightragConfig, *, run_kind: str = "answer", **overrides: Any
) -> dict[str, Any]:
    """One active run pinned to exactly this deployment's policy and models."""
    prepared = {
        "query": "why",
        "workspaces": ["default"],
        "context_policy_revision": CONTEXT_POLICY_REVISION,
        "model_catalog_revision": current_model_catalog_revision(),
        "idempotency_fingerprint": "test-fingerprint",
        "resource_identity": new_resource_identity(),
        "pinned_models": [
            PinnedModelProfile(
                role=role,
                fingerprint=model_invocation_fingerprint(config.models.chat.resolve(role)),
                profile=ModelProfile(context_window_tokens=200_000, max_output_tokens=32_000),
                reasoning_settings=model_reasoning_settings(config.models.chat.resolve(role)),
            ).as_json()
            for role in CHAT_MODEL_SELECTORS
        ],
        **overrides,
    }
    return {"run_kind": run_kind, "prepared_input": prepared}


def _retrieval_requirement(
    config: DlightragConfig, *, with_images: bool = False, **overrides: Any
) -> dict[str, Any]:
    roles = ("extract", "vlm") if with_images else ("extract",)
    prepared = {
        "query": "why",
        "workspaces": ["default"],
        "top_k": 40,
        "chunk_top_k": 20,
        "federated_rerank": False,
        "bm25_query": None,
        "filters": None,
        "query_images": (
            [{"type": "image_url", "image_url": {"url": "data:x"}}] if with_images else []
        ),
        "context_policy_revision": CONTEXT_POLICY_REVISION,
        "model_catalog_revision": current_model_catalog_revision(),
        "idempotency_fingerprint": "test-fingerprint",
        "pinned_models": [
            PinnedRetrievalModel(
                role=role,
                fingerprint=model_invocation_fingerprint(model_settings_for_role(config, role)),
                profile=ModelProfile(context_window_tokens=200_000),
            ).as_json()
            for role in roles
        ],
        **overrides,
    }
    return {"run_kind": "retrieval", "prepared_input": prepared}


def test_memory_dense_leg_reuses_root_embedding_settings(
    test_config: DlightragConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    expected = object()

    def create(settings: Any, *, scheduler: Any, telemetry: Any) -> object:
        captured.update(settings=settings, scheduler=scheduler, telemetry=telemetry)
        return expected

    monkeypatch.setattr("dlightrag.engine.ai.embedding.create_embedding_model", create)
    scheduler = object()
    telemetry = object()

    assert (
        _memory_embedder(
            test_config,
            scheduler=cast(Any, scheduler),
            telemetry=cast(Any, telemetry),
        )
        is expected
    )
    assert captured == {
        "settings": test_config.models.embedding,
        "scheduler": scheduler,
        "telemetry": telemetry,
    }


async def test_application_exposes_only_typed_services_and_closes_in_dependency_order(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    application = parts.application(test_config)

    await application.astart()

    assert parts.recorder.started() == [
        "capabilities:resolve_profiles",
        "run_store:initialize:False",
        "web_store:initialize:False",
        "memory_store:initialize",
        "run_store:iter_active_run_requirements",
        "corpora:initialize",
        "retrieval:planner_for",
        "capabilities:probe_rerank",
        f"pool:acquire:{normalize_workspace(test_config.deployment.workspace)}",
        "corpora:start_promotion_worker",
        "listener:start",
        "coordinator:start",
        "web_conversations:start_retention",
        "memory:purge_expired",
    ]
    assert application.health.is_ready is True
    assert application.answers is parts.answers
    assert application.retrieval is parts.retrieval
    assert application.corpora is parts.corpora
    assert application.web_conversations is parts.web_conversations

    await application.aclose()

    assert parts.recorder.closed() == _CLOSE_ORDER
    assert application.health.is_closed is True


async def test_agent_account_maintenance_runs_from_the_stores_start_to_the_applications_close(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.agent_account_maintenance = _Maintenance(parts.recorder)
    application = parts.application(test_config)

    await application.astart()

    # It needs the table the run stores create, and runs before anything starts claiming Runs.
    started = parts.recorder.started()
    assert (
        started.index("agent_account_maintenance:start")
        == started.index("memory_store:initialize") + 1
    )
    assert started.index("agent_account_maintenance:start") < started.index("coordinator:start")

    await application.aclose()

    assert parts.recorder.closed()[0] == "close:agent_account_maintenance"


async def test_close_stops_live_run_producer_before_execution_adapter(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    execution = _RaceExecution(parts.recorder)
    coordinator = _LiveProducerCoordinator(parts.recorder, execution)
    parts.agent_execution = execution
    parts.coordinator = coordinator
    application = parts.application(test_config)
    await application.astart()

    await application.aclose()
    await coordinator.wait_for_producer()

    assert execution.starts_after_close == 0
    assert parts.recorder.closed().index("close:coordinator") < parts.recorder.closed().index(
        "close:agent_execution"
    )


async def test_search_toolchain_is_preflighted_before_readiness(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.search_toolchain = _SearchToolchain(parts.recorder)
    application = parts.application(test_config)

    await application.astart()

    assert parts.recorder.started()[0] == "search_toolchain:ensure"
    assert application.health.search_toolchain["fd"]["version"] == "10.5.0"
    await application.aclose()


async def test_search_toolchain_preflight_failure_closes_startup(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.search_toolchain = _SearchToolchain(parts.recorder)
    parts.search_toolchain.error = RuntimeError("fd unavailable")
    application = parts.application(test_config)

    with pytest.raises(RuntimeError, match="fd unavailable"):
        await application.astart()

    assert application.health.is_closed is True


async def test_a_reader_validates_the_durable_schema_it_does_not_own(
    test_config: DlightragConfig,
) -> None:
    mutate_config(test_config, "deployment.service_role", "reader")
    parts = _Parts()

    await parts.application(test_config).astart()

    started = parts.recorder.started()
    assert "run_store:initialize:True" in started
    assert "web_store:initialize:True" in started


async def test_a_non_web_process_does_not_start_conversation_retention(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    application = parts.application(test_config, web_enabled=False)

    await application.astart()

    assert "web_conversations:start_retention" not in parts.recorder.started()


@pytest.mark.parametrize(
    ("failure", "error", "expected"),
    [
        pytest.param("run_store", RunSchemaError("run schema missing"), None, id="run-schema"),
        pytest.param(
            "web_store", WebConversationSchemaError("web schema missing"), None, id="web-schema"
        ),
        pytest.param(
            "corpora",
            StorageSchemaError("corpus schema missing"),
            StorageSchemaError,
            id="corpus-schema",
        ),
        pytest.param(
            "workspace",
            CorpusSchemaError("workspace schema missing"),
            StorageSchemaError,
            id="workspace",
        ),
    ],
)
async def test_a_startup_schema_failure_closes_the_application(
    test_config: DlightragConfig, failure: str, error: Exception, expected: type | None
) -> None:
    parts = _Parts()
    match failure:
        case "run_store":
            parts.run_store.initialize_error = error
        case "web_store":
            parts.web_store.initialize_error = error
        case "corpora":
            parts.corpora.initialize_error = error
        case _:
            parts.pool.acquire_error = error
    application = parts.application(test_config)

    with pytest.raises(expected or type(error)):
        await application.astart()

    assert parts.recorder.closed() == _CLOSE_ORDER
    assert application.health.is_ready is False
    assert application.health.is_closed is True


async def test_cancelling_startup_closes_every_initialized_collaborator(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    application = parts.application(test_config)
    warm_started = asyncio.Event()
    chat_probe_cancelled = asyncio.Event()

    async def blocked_acquire(_workspace: str) -> Any:
        warm_started.set()
        await asyncio.Event().wait()

    async def pending_chat_probe() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            chat_probe_cancelled.set()
            raise

    parts.pool.acquire = blocked_acquire  # type: ignore[method-assign]
    parts.capabilities.probe_chat = pending_chat_probe  # type: ignore[method-assign]
    startup = asyncio.create_task(application.astart())
    await warm_started.wait()

    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup

    assert chat_probe_cancelled.is_set()
    assert parts.recorder.closed() == _CLOSE_ORDER
    assert application.health.is_closed is True


async def test_active_runs_pinned_to_this_deployment_start_normally(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.run_store.requirements = (
        _requirement(test_config),
        _retrieval_requirement(test_config),
        _retrieval_requirement(test_config, with_images=True),
    )
    application = parts.application(test_config)

    await application.astart()

    assert application.health.is_ready is True


async def test_irrelevant_retrieval_capability_drift_does_not_block_startup(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.run_store.requirements = (
        _retrieval_requirement(
            test_config,
            capability_facts={"rerank_supports_vision": "obsolete-probe-value"},
        ),
    )

    await parts.application(test_config).astart()

    assert parts.health.is_ready is True


@pytest.mark.parametrize(
    ("override", "detail"),
    [
        pytest.param({"context_policy_revision": "stale"}, "context policy", id="policy"),
        pytest.param({"model_catalog_revision": "stale"}, "model catalog", id="model-catalog"),
        pytest.param({"pinned_models": "not-an-array"}, "durable input schema", id="schema"),
        pytest.param({"pinned_models": []}, "durable input schema", id="roles"),
        # A Run mints its handles from its own identity; without it a resume could
        # not print the handles its history already names.
        pytest.param({"resource_identity": None}, "durable input schema", id="no-identity"),
        pytest.param(
            {"resource_identity": "A" * 64}, "durable input schema", id="malformed-identity"
        ),
        pytest.param({"resource_identity": "a" * 63}, "durable input schema", id="short-identity"),
    ],
)
async def test_an_injected_answer_validation_failure_closes_startup(
    test_config: DlightragConfig, override: dict[str, Any], detail: str
) -> None:
    parts = _Parts()
    parts.run_store.requirements = (_requirement(test_config, **override),)
    application = parts.application(test_config)

    with pytest.raises(IncompatibleActiveRunError, match=detail):
        await application.astart()

    assert parts.recorder.closed() == _CLOSE_ORDER


@pytest.mark.parametrize(
    ("override", "detail"),
    [
        pytest.param({"context_policy_revision": "stale"}, "context policy", id="policy"),
        pytest.param({"model_catalog_revision": "stale"}, "model catalog", id="model-catalog"),
        pytest.param({"pinned_models": "not-an-array"}, "durable input schema", id="schema"),
        pytest.param({"pinned_models": []}, "durable input schema", id="roles"),
    ],
)
async def test_an_incompatible_active_retrieval_fails_startup(
    test_config: DlightragConfig, override: dict[str, Any], detail: str
) -> None:
    parts = _Parts()
    parts.run_store.requirements = (_retrieval_requirement(test_config, **override),)

    with pytest.raises(IncompatibleActiveRunError, match=detail):
        await parts.application(test_config).astart()

    assert parts.recorder.closed() == _CLOSE_ORDER


async def test_an_active_retrieval_on_another_model_endpoint_fails_startup(
    test_config: DlightragConfig,
) -> None:
    requirement = _retrieval_requirement(test_config)
    prepared = requirement["prepared_input"]
    foreign = dict(prepared["pinned_models"][0])
    foreign["fingerprint"] = {**foreign["fingerprint"], "model": "some-other-model"}
    prepared["pinned_models"] = [foreign]
    parts = _Parts()
    parts.run_store.requirements = (requirement,)

    with pytest.raises(IncompatibleActiveRunError, match="another model invocation"):
        await parts.application(test_config).astart()


async def test_an_injected_answer_endpoint_validation_failure_closes_startup(
    test_config: DlightragConfig,
) -> None:
    requirement = _requirement(test_config)
    prepared = requirement["prepared_input"]
    foreign = dict(prepared["pinned_models"][0])
    foreign["fingerprint"] = {**foreign["fingerprint"], "model": "some-other-model"}
    prepared["pinned_models"] = [foreign, *prepared["pinned_models"][1:]]
    parts = _Parts()
    parts.run_store.requirements = (requirement,)

    with pytest.raises(IncompatibleActiveRunError, match="another model invocation"):
        await parts.application(test_config).astart()


async def test_a_failed_default_workspace_degrades_instead_of_closing(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.pool.acquire_error = WorkspaceUnavailableError("workspace unavailable")
    application = parts.application(test_config)

    await application.astart()

    assert application.health.is_degraded is True
    assert application.health.is_closed is False
    assert application.health.warnings == ("Corpus storage unavailable",)
    assert application.health.is_ready is True
    # A degraded process still owns runs: the coordinator and Web retention start.
    started = parts.recorder.started()
    assert "coordinator:start" in started
    assert "web_conversations:start_retention" in started


@pytest.mark.parametrize(
    ("error", "warning"),
    [
        pytest.param(ProviderUnavailableError(), "Model providers unavailable", id="providers"),
        pytest.param(ParserUnavailableError(), "Document parser unavailable", id="parser"),
    ],
)
async def test_a_default_workspace_down_with_another_dependency_degrades_naming_it(
    test_config: DlightragConfig, error: Exception, warning: str
) -> None:
    """Building a workspace can reach more than corpus storage: its image probe asks
    the model provider. A transient outage of any dependency degrades startup and
    names that dependency; corpus storage itself stays healthy."""
    parts = _Parts()
    parts.pool.acquire_error = error
    application = parts.application(test_config)

    await application.astart()

    assert application.health.is_closed is False
    assert application.health.warnings == (warning,)
    assert application.health.components["corpus_storage"]["status"] == "healthy"
    assert application.health.is_ready is True
    assert "coordinator:start" in parts.recorder.started()


async def test_a_default_workspace_that_fails_for_good_closes_startup(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.pool.acquire_error = RuntimeError("embedding model is misconfigured")
    application = parts.application(test_config)

    with pytest.raises(RuntimeError, match="misconfigured"):
        await application.astart()

    assert application.health.is_closed is True


@pytest.mark.parametrize("warm_up", ["settles", "fails"])
async def test_the_chat_probes_run_beside_the_default_workspace_warm_up(
    test_config: DlightragConfig, warm_up: str
) -> None:
    """A workspace is built with the rerank verdict while the chat probes are still
    running, startup waits for them, and a failed warm-up leaves none running."""
    parts = _Parts()
    rerank_probed = False
    chat_probe = "idle"
    chat_started = asyncio.Event()
    chat_released = asyncio.Event()
    seen: dict[str, object] = {}

    async def probe_rerank() -> None:
        nonlocal rerank_probed
        await asyncio.sleep(0)
        rerank_probed = True

    async def probe_chat() -> None:
        nonlocal chat_probe
        chat_probe = "running"
        chat_started.set()
        try:
            await chat_released.wait()
        except asyncio.CancelledError:
            chat_probe = "cancelled"
            raise
        chat_probe = "finished"

    async def acquire(_workspace: str) -> object:
        await chat_started.wait()
        seen.update(rerank_probed=rerank_probed, chat_probe=chat_probe)
        if warm_up == "fails":
            raise CorpusSchemaError("workspace schema missing")
        chat_released.set()
        return object()

    parts.capabilities.probe_rerank = probe_rerank  # type: ignore[method-assign]
    parts.capabilities.probe_chat = probe_chat  # type: ignore[method-assign]
    parts.pool.acquire = acquire  # type: ignore[method-assign]
    application = parts.application(test_config)
    tasks_before = asyncio.all_tasks()

    async with asyncio.timeout(5):
        if warm_up == "fails":
            with pytest.raises(StorageSchemaError):
                await application.astart()
        else:
            await application.astart()

    assert seen == {"rerank_probed": True, "chat_probe": "running"}
    if warm_up == "fails":
        assert chat_probe == "cancelled"
        assert asyncio.all_tasks() <= tasks_before
    else:
        assert chat_probe == "finished"
        assert application.health.components["providers"]["status"] == "healthy"
        await application.aclose()


@pytest.mark.parametrize("failing_probe", ["probe_rerank", "probe_chat"])
async def test_a_failed_capability_probe_degrades_providers_without_failing_startup(
    test_config: DlightragConfig, failing_probe: str
) -> None:
    parts = _Parts()

    async def failed() -> None:
        raise RuntimeError("probe failed")

    setattr(parts.capabilities, failing_probe, failed)
    application = parts.application(test_config)

    await application.astart()

    assert application.health.warnings == ("Model providers unavailable",)
    assert application.health.is_ready is True
    assert "coordinator:start" in parts.recorder.started()
    await application.aclose()


async def test_transient_startup_faults_warn_without_starting_the_run_coordinator(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.run_store.initialize_error = RuntimeError("database unavailable")
    parts.corpora.initialize_error = RuntimeError("registry unavailable")
    parts.corpora.recovery_error = RuntimeError("recovery unavailable")
    application = parts.application(test_config)

    await application.astart()

    assert application.health.is_degraded is True
    assert set(application.health.warnings) == {
        "Operational State unavailable",
        "Corpus storage unavailable",
    }
    started = parts.recorder.started()
    assert "run_store:iter_active_run_requirements" not in started
    assert "coordinator:start" not in started
    assert "web_conversations:start_retention" not in started


async def test_registry_failure_alone_degrades_liveness_not_readiness(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.corpora.initialize_error = RuntimeError("registry unavailable")
    application = parts.application(test_config)

    await application.astart()

    assert application.health.is_ready is True
    assert application.health.is_degraded is True
    assert application.health.warnings == ("Corpus storage unavailable",)
    assert parts.coordinator.is_started is True


async def test_a_coordinator_start_failure_degrades_the_application(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.coordinator.start_error = RuntimeError("scheduler unavailable")
    application = parts.application(test_config)

    await application.astart()

    assert application.health.is_degraded is True
    assert "Operational State unavailable" in application.health.warnings
    assert "Run coordinator unavailable" in application.health.warnings
    assert application.health.is_ready is False
    assert parts.coordinator.is_started is False


async def test_a_late_cancellation_listener_starts_the_coordinator_once_ready(
    test_config: DlightragConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup stops waiting for the listener; the coordinator does not stop with it."""
    monkeypatch.setattr(application_module, "_LISTENER_STARTUP_WAIT_SECONDS", 0.01)
    parts = _Parts()
    parts.cancellation_listener.ready_on_start = False
    application = parts.application(test_config)

    await application.astart()

    assert "coordinator:start" not in parts.recorder.started()
    assert application.health.is_ready is False
    assert application.health.warnings == (
        "Operational State unavailable",
        "Run cancellation listener unavailable",
    )

    parts.cancellation_listener.become_ready()
    async with asyncio.timeout(2):
        while not application.health.is_ready:
            await asyncio.sleep(0)

    # Claiming begins only behind the ready listener, then what startup skipped.
    assert parts.recorder.started()[-3:] == [
        "listener:ready",
        "coordinator:start",
        "web_conversations:start_retention",
    ]
    assert application.health.warnings == ()
    await application.aclose()


@pytest.mark.parametrize("ready_during_close", [False, True], ids=["never-ready", "ready-at-close"])
async def test_closing_while_the_listener_is_late_never_starts_the_coordinator(
    test_config: DlightragConfig, monkeypatch: pytest.MonkeyPatch, ready_during_close: bool
) -> None:
    monkeypatch.setattr(application_module, "_LISTENER_STARTUP_WAIT_SECONDS", 0.01)
    parts = _Parts()
    parts.cancellation_listener.ready_on_start = False
    application = parts.application(test_config)
    await application.astart()

    if ready_during_close:
        parts.cancellation_listener.become_ready()
    await application.aclose()

    assert "coordinator:start" not in parts.recorder.started()
    assert parts.recorder.closed() == _CLOSE_ORDER
    late_starts = [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == application_module._LATE_COORDINATOR_START
    ]
    assert late_starts == []


async def test_config_is_read_only_application_state(test_config: DlightragConfig) -> None:
    application = _Parts().application(test_config)

    assert application.config is test_config
    with pytest.raises(AttributeError):
        application.config = test_config  # type: ignore[misc]


async def test_a_closed_application_refuses_services_but_stays_diagnosable(
    test_config: DlightragConfig,
) -> None:
    application = _Parts().application(test_config)
    await application.astart()

    await application.aclose()

    for name in ("answers", "retrieval", "corpora"):
        with pytest.raises(ApplicationClosedError) as closed:
            getattr(application, name)
        assert str(closed.value) == "Application is shutting down"
    assert application.config is test_config
    assert application.health.is_closed is True


async def test_closing_twice_closes_every_collaborator_once(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    application = parts.application(test_config)
    await application.astart()

    await application.aclose()
    await application.aclose()

    assert parts.recorder.closed() == _CLOSE_ORDER


async def test_concurrent_close_callers_join_the_same_cleanup(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    application = parts.application(test_config)
    await application.astart()
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def blocking_corpora_close() -> None:
        parts.recorder.add("close:corpora")
        close_started.set()
        await release_close.wait()

    parts.corpora.aclose = blocking_corpora_close  # type: ignore[method-assign]
    first = asyncio.create_task(application.aclose())
    await close_started.wait()
    second = asyncio.create_task(application.aclose())
    await asyncio.sleep(0)

    assert second.done() is False

    release_close.set()
    await asyncio.gather(first, second)
    assert parts.recorder.closed() == _CLOSE_ORDER


async def test_an_ordinary_close_failure_never_aborts_later_cleanup(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.coordinator.close_error = RuntimeError("coordinator close failed")
    application = parts.application(test_config)
    await application.astart()

    await application.aclose()

    assert parts.recorder.closed() == _CLOSE_ORDER


async def test_close_defers_cancellation_until_every_collaborator_is_closed(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.corpora.close_error = asyncio.CancelledError()
    application = parts.application(test_config)
    await application.astart()

    with pytest.raises(asyncio.CancelledError):
        await application.aclose()

    assert parts.recorder.closed() == _CLOSE_ORDER
