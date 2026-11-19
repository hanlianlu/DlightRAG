# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A model provider that records every request it is sent and answers from a script, under the real
``ToolModel`` and ``CompletionModel``. ``expected_call`` states what ADR 0041 defines over a request
from ``hashlib`` and ``json``, not from the product's helpers, so a stored step is checked against
the definition and not against the code that implements it.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import pytest

from dlightrag.engine.ai.completion import CompletionModel
from dlightrag.engine.ai.messages import AssistantTurn, ToolDefinition
from dlightrag.engine.ai.providers.base import (
    CompletionOutput,
    CompletionProvider,
    capture_served_model,
    capture_stream_usage,
)
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.ai.tool_model import ToolModel


def canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderRequest:
    model: str
    messages: list[dict[str, Any]]
    tools: list[ToolDefinition]
    tool_choice: str | None
    temperature: float | None
    max_tokens: int | None
    model_kwargs: dict[str, Any]
    response_format: dict[str, Any] | None


def expected_call(
    request: ProviderRequest,
    *,
    role: str,
    endpoint: str | None,
    plan_digest: str | None,
    reasoning: dict[str, str] | None,
    served_model: str | None = "served-model-1",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``call`` a step must state for the request a provider received, timings apart.
    ``options`` is what the options digest covers when that is less than the request carried."""
    carried = request.model_kwargs if options is None else options
    system = "\n\n".join(str(m["content"]) for m in request.messages if m["role"] == "system")
    tools = [asdict(tool) for tool in request.tools]
    stated: dict[str, Any] = {
        "model": {
            "role": role,
            "provider": "openai",
            "model": request.model,
            "endpoint_fingerprint": None if endpoint is None else sha256(endpoint),
            "api_family": "chat_completion",
            "served_model": served_model,
        },
        "params": {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "tool_choice": request.tool_choice,
            "response_format": (request.response_format or {}).get("type"),
            "reasoning": reasoning,
            "options_digest": sha256(canonical(carried)),
        },
        "prompt": {
            "system": sha256(system),
            "tools": sha256(canonical(tools)),
            "request": sha256(canonical({"messages": request.messages, "tools": tools})),
        },
    }
    return stated if plan_digest is None else {**stated, "plan_digest": plan_digest}


def untimed(call: Mapping[str, Any] | None) -> dict[str, Any]:
    """A step's ``call`` without its two timings, once each is checked to be whole milliseconds."""
    assert call is not None, "the step states no call"
    timings = {name: call[name] for name in ("queued_ms", "duration_ms")}
    assert all(type(ms) is int and ms >= 0 for ms in timings.values()), timings
    return {key: value for key, value in call.items() if key not in timings}


@dataclass
class RecordingProvider(CompletionProvider):
    """Answers tool turns from ``reply`` and streams ``text``, recording each request; ``fail`` may
    return an exception to refuse one, and ``delay`` is the seconds each takes."""

    reply: Callable[[ProviderRequest], AssistantTurn | Awaitable[AssistantTurn]] | None = None
    text: str | Callable[[ProviderRequest], str] = "## Goal\nKeep the answer."
    served_model: str | None = "served-model-1"
    fail: Callable[[ProviderRequest], Exception | None] | None = None
    delay: float = 0.0
    requests: list[ProviderRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        CompletionProvider.__init__(self)

    async def _receive(self, model: str, messages: list[dict[str, Any]], **sent: Any) -> Any:
        request = ProviderRequest(
            model,
            copy.deepcopy(messages),
            list(sent.get("tools", ())),
            sent.get("tool_choice"),
            sent.get("temperature"),
            sent.get("max_tokens"),
            copy.deepcopy(sent.get("model_kwargs") or {}),
            copy.deepcopy(sent.get("response_format")),
        )
        self.requests.append(request)
        await asyncio.sleep(self.delay)
        if (problem := self.fail(request) if self.fail else None) is not None:
            raise problem
        return request

    def _text(self, request: ProviderRequest) -> str:
        return self.text if isinstance(self.text, str) else self.text(request)

    async def complete(self, messages: list[dict[str, Any]], model: str, **sent: Any) -> Any:
        return CompletionOutput(self._text(await self._receive(model, messages, **sent)))

    async def stream(
        self, messages: list[dict[str, Any]], model: str, **sent: Any
    ) -> AsyncGenerator[str]:
        usage_holder = sent.pop("usage_holder", None)
        text = self._text(await self._receive(model, messages, **sent))
        for piece in (text[: len(text) // 2], text[len(text) // 2 :]):
            yield piece
        capture_stream_usage(usage_holder, {"prompt_tokens": 7, "completion_tokens": 3})
        capture_served_model(usage_holder, self.served_model)

    async def stream_tool_text(
        self, messages: list[dict[str, Any]], model: str, **sent: Any
    ) -> AsyncGenerator[str]:
        async for piece in self.stream(messages, model, **sent):
            yield piece

    async def complete_tool_turn(
        self, messages: list[dict[str, Any]], model: str, **sent: Any
    ) -> AssistantTurn:
        assert self.reply is not None, "this provider was given no script for tool turns"
        turn = self.reply(await self._receive(model, messages, **sent))
        return replace(
            await turn if inspect.isawaitable(turn) else turn, served_model=self.served_model
        )


def tool_model_over(
    monkeypatch: pytest.MonkeyPatch,
    provider: CompletionProvider,
    settings: ModelSettings,
    *,
    max_concurrency: int = 4,
) -> ToolModel:
    """A real ``ToolModel`` for ``settings`` whose provider is the scripted one."""
    monkeypatch.setattr("dlightrag.engine.ai.providers.get_provider", lambda *_a, **_k: provider)
    slots = ModelScheduler(max_concurrency=max_concurrency)
    return ToolModel(settings, scheduler=slots, telemetry=NOOP_TELEMETRY)


def completion_model_over(
    monkeypatch: pytest.MonkeyPatch, provider: CompletionProvider, settings: ModelSettings
) -> CompletionModel:
    """A real ``CompletionModel`` for ``settings`` whose provider is the scripted one."""
    monkeypatch.setattr("dlightrag.engine.ai.providers.get_provider", lambda *_a, **_k: provider)
    return CompletionModel(
        settings, scheduler=ModelScheduler(max_concurrency=1), telemetry=NOOP_TELEMETRY
    )
