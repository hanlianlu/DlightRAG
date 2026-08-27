# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Structured AnswerModeRouter for auto when both Fast and Research are valid."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from dlightrag.ai.structured import StructuredOutput
from dlightrag.ai.tokens import estimate_messages_tokens
from dlightrag.application.answer_runs.mode import ModeResource, ResolvedMode

_ROUTER_SYSTEM = (
    "Choose exactly one answer mode. "
    "fast: one-shot retrieve and generate. "
    "research: multi-step tools. "
    "Use only the allowed modes. "
    'Reply with JSON: {"mode": "fast"} or {"mode": "research"}.'
)


class _ModeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["fast", "research"]


ROUTER_STRUCTURED_OUTPUT = StructuredOutput(name="answer_mode", schema=_ModeDecision)


class RoutingFailedError(RuntimeError):
    """The router did not return a legal structured mode."""


class AnswerModeRouter:
    """One structured call that picks fast or research."""

    def __init__(self, llm: Callable[..., Awaitable[Any]]) -> None:
        self._llm = llm

    def history_input_measure(
        self,
        query: str,
        *,
        resources: Sequence[ModeResource] = (),
        valid_modes: Sequence[str] = ("fast", "research"),
    ) -> Callable[[list[dict[str, Any]]], int]:
        def measure(history: list[dict[str, Any]]) -> int:
            return estimate_messages_tokens(
                self._messages(
                    query,
                    history=history,
                    resources=resources,
                    valid_modes=valid_modes,
                    tool_categories=(),
                    has_images=False,
                )
            )

        return measure

    async def choose(
        self,
        *,
        query: str,
        history: Sequence[Mapping[str, Any]] = (),
        resources: Sequence[ModeResource] = (),
        tool_categories: Sequence[str] = (),
        has_images: bool = False,
        valid_modes: Sequence[str],
    ) -> ResolvedMode:
        raw = await self._llm(
            messages=self._messages(
                query,
                history=[dict(item) for item in history],
                resources=resources,
                valid_modes=valid_modes,
                tool_categories=tool_categories,
                has_images=has_images,
            ),
            structured_output=ROUTER_STRUCTURED_OUTPUT,
        )
        try:
            parsed = ROUTER_STRUCTURED_OUTPUT.parse(raw)
        except (ValueError, TypeError) as exc:
            raise RoutingFailedError("router output was not a valid mode") from exc
        mode = getattr(parsed, "mode", None)
        if mode not in {"fast", "research"} or mode not in valid_modes:
            raise RoutingFailedError(f"router chose {mode!r} outside {list(valid_modes)}")
        return mode

    def _messages(
        self,
        query: str,
        *,
        history: list[dict[str, Any]],
        resources: Sequence[ModeResource],
        valid_modes: Sequence[str],
        tool_categories: Sequence[str],
        has_images: bool,
    ) -> list[dict[str, Any]]:
        roles = ",".join(resource.role for resource in resources) or "none"
        tools = ",".join(tool_categories) or "none"
        allowed = ",".join(valid_modes)
        user = (
            f"query: {query}\n"
            f"resources: {roles}\n"
            f"images: {has_images}\n"
            f"tools: {tools}\n"
            f"allowed: {allowed}"
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": _ROUTER_SYSTEM}]
        messages.extend(history)
        messages.append({"role": "user", "content": user})
        return messages


__all__ = ["AnswerModeRouter", "ROUTER_STRUCTURED_OUTPUT", "RoutingFailedError"]
