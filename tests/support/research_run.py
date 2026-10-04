# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One accepted Run, executed by a real ``AnswerExecutor`` until its model is first called.

``research_rig`` makes the collaborators an executor needs for a Run to reach its model, and the
model records the tools it is offered and ends the Run, so what a test observes is what a provider
would have been sent. An executor that composed other tools than the Run was accepted with is
refused before the model is reached, as in production, and so is one whose model fingerprints are
not the ones the Run was accepted with.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.answer.capabilities import RequestModelContext
from dlightrag.engine.answer.execution import AnswerExecutor
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.runtime.coordinator import RunSession
from dlightrag.engine.runtime.errors import RunExecutionError
from dlightrag.engine.runtime.progress import StageCommit
from tests.in_memory_session_repository import MemoryAgentSessionRepository
from tests.support.workspace_store import InMemoryWorkspaceStore


class ModelCalled(Exception):
    """Ends a driven Run at its first model call."""


@dataclass
class ResearchRig:
    """What an executor is built from to run a Run to its first model call, and what that call
    was offered."""

    collaborators: dict[str, Any]
    """Keyword arguments for the ``AnswerExecutor`` under test."""
    offered: list[dict[str, Any]] = field(default_factory=list)
    """The tool definitions each model call was offered, by name."""

    async def run(self, executor: AnswerExecutor, prepared_input: Mapping[str, Any]) -> Any:
        """Execute the Run accepted as ``prepared_input``, and return the session it ran under.

        The session holds the claim ``owner-1``, ``trusted-run``, ``trusted-worker`` and epoch 7,
        whatever ``prepared_input`` says.
        """
        session = MagicMock(
            owner_id="owner-1",
            run_id="trusted-run",
            worker_id="trusted-worker",
            fencing_epoch=7,
            prepared_input=dict(prepared_input),
            durable_progress_version=0,
            workspace_epoch=None,
            checkpoint=None,
        )
        for method in (
            "check_cancelled",
            "enter_phase",
            "emit_token",
            "flush_tokens",
            "reset_output",
        ):
            setattr(session, method, AsyncMock())
        progress = MagicMock()
        progress.load_stage = AsyncMock(return_value=None)
        progress.settle_stage = AsyncMock(
            return_value=StageCommit(
                progress_version=1, stage_intent_id=MagicMock(), evidence_count=0
            )
        )
        session.execution.session_repository = MemoryAgentSessionRepository[Any](fencing_epoch=11)
        session.execution.workspace_store = InMemoryWorkspaceStore()
        session.execution.progress_store = progress
        session.execution.fencing_epoch = 11

        with pytest.raises(RunExecutionError):
            await executor.execute(cast(RunSession, session))
        return session


def research_rig(
    *,
    tmp_path: Path,
    planning: Any,
    mode: str = "research",
    registry: ResourceRegistry | None = None,
) -> ResearchRig:
    """The collaborators for a Run routed to ``mode``, with ``registry`` for its Resources.

    Its models answer with ``ModelSettings(model="test")`` for every role, which a Run accepted
    with those settings pinned, and the executor is still given the fingerprints it pinned.
    """
    (tmp_path / "corpus").mkdir()
    profile = ModelProfile(context_window_tokens=1_000_000)
    models = RequestModelContext(extract=profile, query=profile, vlm=profile)

    store = MagicMock()
    store.load_routing = AsyncMock(return_value=MagicMock(resolved_mode=mode))
    store.list_child_sessions = AsyncMock(return_value=[])
    store.load_pending_agent_controls = AsyncMock(return_value=[])
    store.list_fetched_resources = AsyncMock(return_value=[])
    capabilities = MagicMock()
    capabilities.request_model_context = MagicMock(return_value=models)
    resources = MagicMock()
    resources.resolve = AsyncMock(
        return_value=SimpleNamespace(
            models=models,
            current_images=[],
            web_sources=None,
            registry=registry,
            resource_manifest=(),
            image_budget=None,
            query_images=None,
            current_image_count=0,
        )
    )
    rig = ResearchRig({})

    class _ResearchModel:
        async def __call__(self, **kwargs: Any) -> Any:
            rig.offered.append({tool.name: tool for tool in kwargs["tools"]})
            raise ModelCalled

        async def stream_text(self, **_kwargs: Any) -> Any:
            raise AssertionError("the first model call ends the Run")

    class _FastGeneration:
        async def generate_stream(self, *_args: Any, **kwargs: Any) -> Any:
            rig.offered.append({tool.name: tool for tool in kwargs.get("tools") or ()})
            raise ModelCalled

    runtime = MagicMock()
    runtime.model_settings = MagicMock(return_value=ModelSettings(model="test"))
    runtime.query_tool_model = MagicMock(return_value=_ResearchModel())
    runtime.answer_synthesizer = MagicMock(return_value=_FastGeneration())
    rig.collaborators = {
        "store": store,
        "capabilities": capabilities,
        "resources": resources,
        "models": runtime,
        "planning": planning,
        "workspace_root": str(tmp_path / "workspaces"),
        "working_dir": str(tmp_path / "corpus"),
    }
    return rig


__all__ = ["ModelCalled", "ResearchRig", "research_rig"]
