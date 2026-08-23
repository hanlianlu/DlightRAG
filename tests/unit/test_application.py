# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup and shutdown contract of the local composition root."""

import asyncio
from typing import Any, cast

import pytest

from dlightrag.adapters.postgres._pool import pg_pool
from dlightrag.adapters.postgres.answer_runs import PGAnswerRunStore
from dlightrag.adapters.postgres.web_conversations import PGWebConversationStore
from dlightrag.ai.capacity import CONTEXT_POLICY_REVISION
from dlightrag.ai.fingerprints import ModelFingerprint, model_fingerprint
from dlightrag.ai.settings import MODEL_ROLE_NAMES
from dlightrag.answer.capabilities import AnswerCapabilityCoordinator
from dlightrag.answer.executor import IncompatibleActiveRunError
from dlightrag.answer.model_runtime import AnswerModelRuntime
from dlightrag.application import (
    Application,
    ApplicationClosedError,
    _ApplicationComponents,
    _memory_embedder,
)
from dlightrag.config import DlightragConfig
from dlightrag.health import ApplicationHealth
from dlightrag.model_settings import model_settings_for_role
from dlightrag.rag.ports import CorpusSchemaError
from dlightrag.rag.workspaces import normalize_workspace
from dlightrag.runtime import RunCoordinator, RunSchemaError
from dlightrag.services.answers import AnswerService
from dlightrag.services.corpora import CorpusAdmin
from dlightrag.services.errors import StorageSchemaError
from dlightrag.services.retrieval import RetrievalService
from dlightrag.web.conversation_models import WebConversationSchemaError
from dlightrag.web.conversations import WebConversationService
from tests.config_helpers import mutate_config

_CLOSE_ORDER = [
    "close:corpora",
    "close:coordinator",
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

    def validate_startup(self) -> None:
        self._record("validate_startup")

    async def probe_all(self) -> None:
        self._record("probe_all")


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

    async def list_active_run_requirements(self) -> tuple[dict[str, Any], ...]:
        self._record("list_active_run_requirements")
        return self.requirements


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

    async def start(self) -> None:
        self.recorder.add("listener:start")
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
        self.coordinator = _Coordinator(self.recorder)
        self.cancellation_listener = _CancellationListener(self.recorder)
        self.corpora = _Corpora(self.recorder)
        self.retrieval = _Retrieval(self.recorder)
        self.answers = object()
        self.web_conversations = _WebConversations(self.recorder)

    def application(
        self,
        config: DlightragConfig,
        *,
        web_enabled: bool = True,
    ) -> Application:
        return Application(
            config,
            _ApplicationComponents(
                health=self.health,
                capabilities=cast(AnswerCapabilityCoordinator, self.capabilities),
                pool=cast(Any, self.pool),
                models=cast(AnswerModelRuntime, self.models),
                run_store=cast(PGAnswerRunStore, self.run_store),
                web_store=cast(PGWebConversationStore, self.web_store),
                coordinator=cast(RunCoordinator, self.coordinator),
                cancellation_listener=cast(Any, self.cancellation_listener),
                corpora=cast(CorpusAdmin, self.corpora),
                retrieval=cast(RetrievalService, self.retrieval),
                answers=cast(AnswerService, self.answers),
                memory=cast(Any, self.answers),
                memory_store=cast(Any, self.memory_store),
                memory_embedder=cast(Any, self.memory_embedder),
                web_conversations=cast(WebConversationService, self.web_conversations),
            ),
            web_enabled=web_enabled,
        )


def _pinned(fingerprint: ModelFingerprint, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "fingerprint": {
            "provider": fingerprint.provider,
            "model": fingerprint.model,
            "endpoint_fingerprint": fingerprint.endpoint_fingerprint,
        },
        "profile": {
            "context_window_tokens": 200_000,
            "max_input_tokens": None,
            "max_output_tokens": 32_000,
            "supports_images": False,
            "supports_reasoning": False,
        },
    }


def _requirement(config: DlightragConfig, **overrides: Any) -> dict[str, Any]:
    """One active run pinned to exactly this deployment's policy and models."""
    return {
        "context_policy_revision": CONTEXT_POLICY_REVISION,
        "pinned_models": [
            _pinned(model_fingerprint(model_settings_for_role(config, role)), role)
            for role in MODEL_ROLE_NAMES
        ],
        **overrides,
    }


def test_memory_dense_leg_reuses_root_embedding_settings(
    test_config: DlightragConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    expected = object()

    def create(settings: Any, *, scheduler: Any, telemetry: Any) -> object:
        captured.update(settings=settings, scheduler=scheduler, telemetry=telemetry)
        return expected

    monkeypatch.setattr("dlightrag.ai.embedding.create_embedding_model", create)
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
        "capabilities:validate_startup",
        "run_store:initialize:False",
        "web_store:initialize:False",
        "memory_store:initialize",
        "run_store:list_active_run_requirements",
        "corpora:initialize",
        "retrieval:planner_for",
        "capabilities:probe_all",
        f"pool:acquire:{normalize_workspace(test_config.deployment.workspace)}",
        "corpora:start_recovery",
        "listener:start",
        "coordinator:start",
        "web_conversations:start_retention",
    ]
    assert application.health.is_ready is True
    assert application.answers is parts.answers
    assert application.retrieval is parts.retrieval
    assert application.corpora is parts.corpora
    assert application.web_conversations is parts.web_conversations

    await application.aclose()

    assert parts.recorder.closed() == _CLOSE_ORDER
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

    async def blocked_acquire(_workspace: str) -> Any:
        warm_started.set()
        await asyncio.Event().wait()

    parts.pool.acquire = blocked_acquire  # type: ignore[method-assign]
    startup = asyncio.create_task(application.astart())
    await warm_started.wait()

    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup

    assert parts.recorder.closed() == _CLOSE_ORDER
    assert application.health.is_closed is True


async def test_active_runs_pinned_to_this_deployment_start_normally(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.run_store.requirements = (_requirement(test_config),)
    application = parts.application(test_config)

    await application.astart()

    assert application.health.is_ready is True


@pytest.mark.parametrize(
    ("override", "detail"),
    [
        pytest.param({"context_policy_revision": "stale"}, "context policy", id="policy"),
        pytest.param({"pinned_models": "not-an-array"}, "durable input schema", id="schema"),
        pytest.param({"pinned_models": []}, "complete model role set", id="roles"),
    ],
)
async def test_an_incompatible_active_run_fails_startup_and_closes(
    test_config: DlightragConfig, override: dict[str, Any], detail: str
) -> None:
    parts = _Parts()
    parts.run_store.requirements = (_requirement(test_config, **override),)
    application = parts.application(test_config)

    with pytest.raises(IncompatibleActiveRunError, match=detail):
        await application.astart()

    assert parts.recorder.closed() == _CLOSE_ORDER


async def test_an_active_run_on_another_model_endpoint_fails_startup(
    test_config: DlightragConfig,
) -> None:
    requirement = _requirement(test_config)
    foreign = dict(requirement["pinned_models"][0])
    foreign["fingerprint"] = {**foreign["fingerprint"], "model": "some-other-model"}
    requirement["pinned_models"] = [foreign, *requirement["pinned_models"][1:]]
    parts = _Parts()
    parts.run_store.requirements = (requirement,)

    with pytest.raises(IncompatibleActiveRunError, match="another model endpoint"):
        await parts.application(test_config).astart()


async def test_a_failed_default_workspace_degrades_instead_of_closing(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.pool.acquire_error = RuntimeError("workspace unavailable")
    application = parts.application(test_config)

    await application.astart()

    assert application.health.is_degraded is True
    assert application.health.is_closed is False
    assert any("workspace unavailable" in warning for warning in application.health.warnings)
    # A degraded process still owns runs: the coordinator and Web retention start.
    started = parts.recorder.started()
    assert "coordinator:start" in started
    assert "web_conversations:start_retention" in started


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
        "Answer run store unavailable",
        "Answer runtime unavailable",
        "Workspace registry unavailable",
        "Ingest job recovery unavailable",
        "Web conversations unavailable",
    }
    started = parts.recorder.started()
    assert "run_store:list_active_run_requirements" not in started
    assert "coordinator:start" not in started
    assert "web_conversations:start_retention" not in started


async def test_registry_failure_alone_degrades_readiness(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.corpora.initialize_error = RuntimeError("registry unavailable")
    application = parts.application(test_config)

    await application.astart()

    assert application.health.is_ready is False
    assert application.health.is_degraded is True
    assert application.health.warnings == ("Workspace registry unavailable",)
    assert parts.coordinator.is_started is True


async def test_ingest_recovery_failure_alone_degrades_readiness(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.corpora.recovery_error = RuntimeError("recovery unavailable")
    application = parts.application(test_config)

    await application.astart()

    assert application.health.is_ready is False
    assert application.health.is_degraded is True
    assert application.health.warnings == ("Ingest job recovery unavailable",)
    assert parts.coordinator.is_started is True


async def test_a_coordinator_start_failure_degrades_the_application(
    test_config: DlightragConfig,
) -> None:
    parts = _Parts()
    parts.coordinator.start_error = RuntimeError("scheduler unavailable")
    application = parts.application(test_config)

    await application.astart()

    assert application.health.is_degraded is True
    assert "Answer runtime unavailable" in application.health.warnings
    assert parts.coordinator.is_started is False


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
        assert closed.value.detail == "Application is shutting down"
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
