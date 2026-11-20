# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A model provider that records every request it is sent and answers from a script.

It sits where a vendor's SDK sits, under the real ``ToolModel`` and ``CompletionModel``, so what a
test reads back is what a provider would have been sent: the model it was asked for, the sampling
and output limits, the provider options, and the exact messages and tools. ``ProviderRequest``
also states the digests ADR 0043 defines over that request, computed here from first principles
(``hashlib`` and ``json``, not the product's own helpers), so a test that compares them to what a
Session stored checks the definition and not the code that implements it.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import pytest

from dlightrag.engine.ai.messages import AssistantTurn, ToolChoice, ToolDefinition
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
    """Canonical JSON: sorted keys, no whitespace, characters as written."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderRequest:
    """One request, exactly as the provider received it."""

    model: str
    messages: list[dict[str, Any]]
    tools: list[ToolDefinition]
    tool_choice: str | None
    temperature: float | None
    max_tokens: int | None
    model_kwargs: dict[str, Any]
    response_format: dict[str, Any] | None = None

    @property
    def system_text(self) -> str:
        """The system messages' text alone, joined by a blank line."""
        return "\n\n".join(
            str(message["content"]) for message in self.messages if message["role"] == "system"
        )

    def digests(self) -> dict[str, str]:
        """The three digests of the request: system text, tools, and messages with tools."""
        tools = [asdict(tool) for tool in self.tools]
        return {
            "system": sha256(self.system_text),
            "tools": sha256(canonical(tools)),
            "request": sha256(canonical({"messages": self.messages, "tools": tools})),
        }


def expected_call(
    request: ProviderRequest,
    *,
    role: str,
    endpoint: str | None,
    plan_digest: str | None,
    reasoning: dict[str, str] | None,
    provider: str = "openai",
    api_family: str = "chat_completion",
    served_model: str | None = "served-model-1",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``call`` a step must state for the request a provider received.

    Everything but the role, the endpoint's URL and the Plan comes from the request itself, and
    the endpoint's fingerprint is the SHA-256 of its canonical URL, so the expectation shares no
    code with the product that writes the real one. The options digest covers the options the
    request carried, unless ``options`` says which of them a digest may cover: the credentials
    a request carried are not among them.
    """
    stated: dict[str, Any] = {
        "model": {
            "role": role,
            "provider": provider,
            "model": request.model,
            "endpoint_fingerprint": None if endpoint is None else sha256(endpoint),
            "api_family": api_family,
            "served_model": served_model,
        },
        "params": {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "tool_choice": request.tool_choice,
            "response_format": (request.response_format or {}).get("type"),
            "reasoning": reasoning,
            "options_digest": sha256(
                canonical(request.model_kwargs if options is None else options)
            ),
        },
        "prompt": request.digests(),
    }
    if plan_digest is not None:
        stated["plan_digest"] = plan_digest
    return stated


_TIMINGS = ("queued_ms", "duration_ms")


def untimed(call: Mapping[str, Any] | None) -> dict[str, Any]:
    """A step's ``call`` without its two timings, once each is checked to be whole milliseconds.

    The timings depend on the machine and the moment, so a comparison with ``expected_call``
    leaves them out; what they must be is asserted where a test controls the clock of its provider.
    """
    assert call is not None, "the step states no call"
    for name in _TIMINGS:
        value = call[name]
        assert type(value) is int and value >= 0, (name, value)
    return {key: value for key, value in call.items() if key not in _TIMINGS}


Reply = Callable[[ProviderRequest], AssistantTurn | Awaitable[AssistantTurn]]


@dataclass
class RecordingProvider(CompletionProvider):
    """Answers tool turns from ``reply`` and streams ``text``, recording each request.

    ``served_model`` is the model name its server would state; ``fail`` may raise for a request
    to script a rejection; ``delay`` is the seconds a request takes before it is answered or
    refused, the same for every request or chosen per request.
    """

    reply: Reply | None = None
    text: str | Callable[[ProviderRequest], str] = "## Goal\nKeep the answer."
    served_model: str | None = "served-model-1"
    fail: Callable[[ProviderRequest], Exception | None] | None = None
    delay: float | Callable[[ProviderRequest], float] = 0.0
    requests: list[ProviderRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        CompletionProvider.__init__(self)

    async def _receive(self, request: ProviderRequest) -> None:
        """Take a request: note it, spend the time the script gives it, then answer or refuse."""
        self.requests.append(request)
        seconds = self.delay(request) if callable(self.delay) else self.delay
        if seconds:
            await asyncio.sleep(seconds)
        problem = self.fail(request) if self.fail is not None else None
        if problem is not None:
            raise problem

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> CompletionOutput:
        await self._receive(
            ProviderRequest(
                model,
                copy.deepcopy(messages),
                [],
                None,
                temperature,
                max_tokens,
                copy.deepcopy(model_kwargs or {}),
                copy.deepcopy(response_format),
            )
        )
        return CompletionOutput(self.text if isinstance(self.text, str) else "")

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        usage_holder: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:
        request = ProviderRequest(
            model,
            copy.deepcopy(messages),
            [],
            None,
            temperature,
            max_tokens,
            copy.deepcopy(model_kwargs or {}),
            copy.deepcopy(response_format),
        )
        await self._receive(request)
        text = self.text(request) if callable(self.text) else self.text
        for piece in (text[: len(text) // 2], text[len(text) // 2 :]):
            yield piece
        capture_stream_usage(usage_holder, {"prompt_tokens": 7, "completion_tokens": 3})
        capture_served_model(usage_holder, self.served_model)

    async def stream_tool_text(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        usage_holder: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str]:
        async for piece in self.stream(
            messages,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            usage_holder=usage_holder,
        ):
            yield piece

    async def complete_tool_turn(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        tools: list[ToolDefinition],
        tool_choice: ToolChoice = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> AssistantTurn:
        assert self.reply is not None, "this provider was given no script for tool turns"
        request = ProviderRequest(
            model,
            copy.deepcopy(messages),
            list(tools),
            tool_choice,
            temperature,
            max_tokens,
            copy.deepcopy(model_kwargs or {}),
        )
        await self._receive(request)
        turn = self.reply(request)
        if inspect.isawaitable(turn):
            turn = await turn
        return AssistantTurn(
            text=turn.text,
            tool_calls=turn.tool_calls,
            stop_reason=turn.stop_reason,
            reasoning=turn.reasoning,
            usage_details=turn.usage_details,
            cost_details=turn.cost_details,
            provider_state=turn.provider_state,
            served_model=self.served_model,
        )


def tool_model_over(
    monkeypatch: pytest.MonkeyPatch,
    provider: CompletionProvider,
    settings: ModelSettings,
    *,
    max_concurrency: int = 4,
) -> ToolModel:
    """A real ``ToolModel`` for ``settings`` whose provider is the scripted one."""
    monkeypatch.setattr(
        "dlightrag.engine.ai.providers.get_provider", lambda *_args, **_kwargs: provider
    )
    return ToolModel(
        settings,
        scheduler=ModelScheduler(max_concurrency=max_concurrency),
        telemetry=NOOP_TELEMETRY,
    )
