# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures for dlightrag tests."""

import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid7

import pytest

from dlightrag.adapters.postgres.runtime.run_store import (
    PGRunStore,
)
from dlightrag.application.config import DlightragConfig, reset_config, set_config
from dlightrag.application.config import sections as config_sections
from dlightrag.engine.ai.settings import (
    EmbeddingSettings,
    ModelRoleSettings,
    ModelSettings,
)
from dlightrag.engine.answer.execution.connection_binding import RunConnectionBinding
from dlightrag.engine.answer.runs.envelope import accepted_input_envelope
from dlightrag.engine.answer.runs.routing import RoutingAcceptance
from dlightrag.engine.runtime.policy import DEFAULT_RUN_RETENTION_SECONDS
from dlightrag.engine.runtime.records import (
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    RunAccessScope,
    RunCreation,
    run_request_fingerprint,
)


class FingerprintingRunStore(PGRunStore):
    """Test adapter for low-level suites whose raw request is the public input.

    ``create_run`` and ``create_run_in`` build an Answer envelope from a raw
    request, or pass a ready one through, to ``accept_run`` and ``accept_run_in``.
    """

    async def create_run(
        self,
        *,
        envelope: PreparedRunEnvelope | None = None,
        run_id: str | None = None,
        owner_id: str | None = None,
        request: Mapping[str, Any] | None = None,
        prepared_input: Mapping[str, Any] | None = None,
        idempotency_fingerprint: str | None = None,
        idempotency_key: str | None = None,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
        connection_bindings: tuple[RunConnectionBinding, ...] = (),
    ) -> RunCreation:
        if envelope is not None:
            if run_id is None:
                raise ValueError("run_id is required with an envelope")
            return await super().accept_run(
                envelope=envelope,
                run_id=run_id,
                artifacts=artifacts,
                references=references,
                routing=routing,
                connection_bindings=connection_bindings,
            )
        if owner_id is None:
            raise ValueError("owner_id is required for raw test acceptance")

        from dlightrag.engine.agent.session.ids import SessionId

        if prepared_input is not None:
            prepared = {
                "agent_session_id": SessionId.new().value,
                "agent_lane_id": "main",
                **dict(prepared_input),
            }
            return await self._create_answer_run(
                owner_id=owner_id,
                prepared=prepared,
                fingerprint=idempotency_fingerprint or run_request_fingerprint(prepared_input),
                idempotency_key=idempotency_key,
                artifacts=artifacts,
                references=references,
                connection_bindings=connection_bindings,
            )
        request = request or {}
        prepared: dict[str, Any] = {
            "agent_session_id": SessionId.new().value,
            "agent_lane_id": "main",
            **dict(request),
        }
        return await self._create_answer_run(
            owner_id=owner_id,
            prepared=prepared,
            fingerprint=idempotency_fingerprint or run_request_fingerprint(request),
            idempotency_key=idempotency_key,
            artifacts=artifacts,
            references=references,
            connection_bindings=connection_bindings,
        )

    async def _create_answer_run(
        self,
        *,
        owner_id: str,
        prepared: Mapping[str, Any],
        fingerprint: str,
        idempotency_key: str | None,
        artifacts: Sequence[PendingArtifact],
        references: Sequence[PendingArtifactReference],
        connection_bindings: tuple[RunConnectionBinding, ...],
    ) -> RunCreation:
        run_id = str(uuid7())
        return await super().accept_run(
            envelope=_answer_envelope(
                owner_id=owner_id,
                prepared=prepared,
                fingerprint=fingerprint,
                submission_key=idempotency_key or run_id,
            ),
            run_id=run_id,
            artifacts=artifacts,
            references=references,
            connection_bindings=connection_bindings,
        )

    async def create_run_in(
        self,
        conn: Any,
        *,
        envelope: PreparedRunEnvelope | None = None,
        run_id: str | None = None,
        owner_id: str | None = None,
        request: Mapping[str, Any] | None = None,
        idempotency_fingerprint: str | None = None,
        idempotency_key: str | None = None,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
        connection_bindings: tuple[RunConnectionBinding, ...] = (),
    ) -> RunCreation:
        if envelope is not None:
            if run_id is None:
                raise ValueError("run_id is required with an envelope")
            return await super().accept_run_in(
                conn,
                envelope=envelope,
                run_id=run_id,
                artifacts=artifacts,
                references=references,
                routing=routing,
                connection_bindings=connection_bindings,
            )
        if owner_id is None:
            raise ValueError("owner_id is required for raw test acceptance")

        from dlightrag.engine.agent.session.ids import SessionId

        request = request or {}
        prepared: dict[str, Any] = {
            "agent_session_id": SessionId.new().value,
            "agent_lane_id": "main",
            **dict(request),
        }
        run_id = str(uuid7())
        return await super().accept_run_in(
            conn,
            envelope=_answer_envelope(
                owner_id=owner_id,
                prepared=prepared,
                fingerprint=idempotency_fingerprint or run_request_fingerprint(request),
                submission_key=idempotency_key or run_id,
            ),
            run_id=run_id,
            artifacts=artifacts,
            references=references,
            routing=routing,
            connection_bindings=connection_bindings,
        )


def _answer_envelope(
    *, owner_id: str, prepared: Mapping[str, Any], fingerprint: str, submission_key: str
) -> PreparedRunEnvelope:
    return PreparedRunEnvelope(
        run_kind="answer",
        lane="query",
        submitted_by=owner_id,
        access_scope=RunAccessScope(kind="owner", scope_id=owner_id),
        submission_key=submission_key,
        request_fingerprint=fingerprint,
        payload=prepared,
        accepted_input=accepted_input_envelope(prepared),
        retention_seconds=DEFAULT_RUN_RETENTION_SECONDS,
    )


# The operator's .env, config.yaml, shell settings, and home are deployment inputs,
# not product contracts: a test that reads them asserts whatever this checkout is
# tuned to. Tests that mean to exercise a YAML config or an environment set their
# own. The suite gates (RUN_E2E_PG18, RUN_LOAD_RUNTIME, E2E_*) live outside the
# DLIGHTRAG_ namespace, so hiding every DLIGHTRAG_* name leaves them visible.
# The config.yaml files present when the run starts: this checkout's and the
# invocation directory's.
_STARTUP_CONFIG_YAMLS = frozenset(
    (directory / "config.yaml").resolve()
    for directory in (Path(__file__).resolve().parents[1], Path.cwd())
)
# Bound before the isolation patches the name, otherwise the wrapper recurses.
_FIND_YAML_CONFIG = config_sections._find_yaml_config


def _yaml_config_ignoring_startup_files() -> Path | None:
    """Resolve config.yaml as production does, minus the files present at startup."""
    found = _FIND_YAML_CONFIG()
    if found is not None and found.resolve() in _STARTUP_CONFIG_YAMLS:
        return None
    return found


def _hide_operator_inputs(patch: pytest.MonkeyPatch) -> None:
    """Hide the checkout's .env and config.yaml and the shell's DLIGHTRAG_* names."""
    patch.setenv("PYTHON_DOTENV_DISABLED", "1")  # LightRAG load_dotenv()s .env on import
    patch.setitem(DlightragConfig.model_config, "env_file", None)
    patch.setattr(config_sections, "_find_yaml_config", _yaml_config_ignoring_startup_files)
    for key in list(os.environ):
        if key.upper().startswith("DLIGHTRAG_"):
            patch.delenv(key, raising=False)


def _playwright_browsers(home: Path) -> Path:
    """Playwright's default browser cache under the operator's real home."""
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "ms-playwright"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local") / "ms-playwright"
    return Path(os.environ.get("XDG_CACHE_HOME") or home / ".cache") / "ms-playwright"


# Applied once for the whole session, as this conftest loads and before any suite's
# conftest, session fixture, or module imports LightRAG or builds a config; undone
# in pytest_unconfigure. Each test re-applies it in case one wrote os.environ.
_SESSION = pytest.MonkeyPatch()
_SESSION_HOME = Path(tempfile.mkdtemp(prefix="dlightrag-test-home-")).resolve()
if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
    _SESSION.setenv("PLAYWRIGHT_BROWSERS_PATH", str(_playwright_browsers(Path.home())))
_SESSION.setenv("HOME", str(_SESSION_HOME))
_hide_operator_inputs(_SESSION)


def pytest_unconfigure(config: pytest.Config) -> None:  # noqa: ARG001
    _SESSION.undo()
    shutil.rmtree(_SESSION_HOME, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolated_from_operator_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the operator's .env, config.yaml and DLIGHTRAG_* settings out of every test."""
    _hide_operator_inputs(monkeypatch)


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture(autouse=True)
def _pg_pool_without_leftover_patches():
    """Drop what undoing a patch of the process pool's methods leaves on the instance.

    ``monkeypatch.setattr(pg_pool, "run", ...)`` undoes by setting the original
    bound method on the instance, which then shadows every later patch of the
    ``PGPool`` class in the same process: a test's outcome depended on which
    tests ran before it.
    """
    yield
    pool_module = sys.modules.get("dlightrag.adapters.postgres.core._pool")
    if pool_module is None:
        return
    pool = pool_module.pg_pool
    for name in [name for name in vars(pool) if callable(getattr(type(pool), name, None))]:
        delattr(pool, name)


@pytest.fixture
def tmp_working_dir(tmp_path: Path) -> Path:
    """Create a temporary working directory structure."""
    working_dir = tmp_path / "dlightrag_storage"
    (working_dir / "artifacts" / "local").mkdir(parents=True)
    return working_dir


@pytest.fixture
def test_config(tmp_working_dir: Path) -> DlightragConfig:
    """Create a test config with temporary paths.

    Also sets the global singleton so that code calling get_config()
    directly (e.g. /health endpoint) gets the test config.
    """
    cfg = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        # type: ignore[call-arg]
        deployment={"working_dir": str(tmp_working_dir)},
        models={
            "chat": ModelRoleSettings(
                default=ModelSettings(
                    model="z-ai/glm-5.3-flash",
                    base_url="https://openrouter.ai/api/v1",
                    api_key="test-key-for-unit-tests",
                )
            ),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="test-key-for-unit-tests",
                startup_probe=False,
            ),
        },
    )
    set_config(cfg)
    return cfg
