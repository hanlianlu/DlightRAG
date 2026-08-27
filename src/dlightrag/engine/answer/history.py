# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Project one pinned history across every reachable model call."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.engine.agent.session.fold import PriorTurns
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ContextPolicy, ModelProfile
from dlightrag.engine.ai.tokens import truncate_to_estimated_tokens


class HistoryInputMeasure(Protocol):
    """Exact target serializer for recent messages plus a projected summary."""

    def __call__(
        self,
        messages: list[dict[str, Any]],
        projected_summary: str = "",
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class HistoryProjectionTarget:
    """One reachable model call and its exact history-aware input serializer."""

    name: str
    profile: ModelProfile
    measure_input: HistoryInputMeasure
    proactive_compaction: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("history projection target name must be non-empty")


class HistoryProjectionOverflowError(ValueError):
    """A reachable call's zero-history fixed envelope cannot be accepted."""

    def __init__(
        self,
        *,
        target: str,
        fixed_input_tokens: int,
        acceptance_limit_tokens: int,
    ) -> None:
        self.target = target
        self.fixed_input_tokens = fixed_input_tokens
        self.acceptance_limit_tokens = acceptance_limit_tokens
        super().__init__(
            f"{target} fixed input uses {fixed_input_tokens} tokens but its acceptance "
            f"limit is {acceptance_limit_tokens}"
        )


@dataclass(frozen=True, slots=True)
class _ResolvedTarget:
    target: HistoryProjectionTarget
    fixed_input_tokens: int
    allowance_tokens: int


def project_history(
    messages: Sequence[dict[str, Any]],
    *,
    targets: Sequence[HistoryProjectionTarget],
    context_policy: ContextPolicy = CONTEXT_POLICY,
) -> PriorTurns:
    """Keep the newest contiguous complete pairs accepted by every target."""
    resolved = tuple(_resolve_target(target, context_policy) for target in targets)
    pairs = _complete_pairs(messages)
    if not resolved:
        return PriorTurns()
    kept: list[dict[str, Any]] = []
    for pair in reversed(pairs):
        candidate = [*pair, *kept]
        if not all(_fits(candidate, "", target) for target in resolved):
            break
        kept = candidate
    omitted_pairs = pairs[: len(pairs) - len(kept) // 2]
    summary = _fit_episodic_summary(
        kept,
        omitted_pairs,
        resolved=resolved,
        max_tokens=context_policy.episodic_summary_tokens,
    )
    return PriorTurns(kept, episodic_summary=summary)


def _resolve_target(
    target: HistoryProjectionTarget,
    context_policy: ContextPolicy,
) -> _ResolvedTarget:
    hard_limit = context_policy.hard_input_limit(target.profile)
    acceptance_limit = (
        context_policy.compaction_trigger(target.profile)
        if target.proactive_compaction
        else hard_limit
    )
    fixed_input = target.measure_input([], "")
    if fixed_input > acceptance_limit:
        raise HistoryProjectionOverflowError(
            target=target.name,
            fixed_input_tokens=fixed_input,
            acceptance_limit_tokens=acceptance_limit,
        )
    allowance = max(
        0,
        min(
            context_policy.history_allowance_cap(target.profile),
            acceptance_limit - fixed_input,
        ),
    )
    return _ResolvedTarget(target, fixed_input, allowance)


def _fits(
    messages: list[dict[str, Any]],
    episodic_summary: str,
    resolved: _ResolvedTarget,
) -> bool:
    history_tokens = max(
        0,
        resolved.target.measure_input(messages, episodic_summary) - resolved.fixed_input_tokens,
    )
    return history_tokens <= resolved.allowance_tokens


def _fit_episodic_summary(
    kept: list[dict[str, Any]],
    omitted_pairs: Sequence[Sequence[dict[str, Any]]],
    *,
    resolved: Sequence[_ResolvedTarget],
    max_tokens: int,
) -> str:
    """Fit the extractive continuation after preserving the recent-pair set."""
    summary = _episodic_summary(omitted_pairs, max_tokens=max_tokens)
    if not summary or all(_fits(kept, summary, target) for target in resolved):
        return summary

    low = 0
    high = max_tokens
    fitted = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = truncate_to_estimated_tokens(summary, middle) if middle else ""
        if all(_fits(kept, candidate, target) for target in resolved):
            fitted = candidate
            low = middle + 1
        else:
            high = middle - 1
    return fitted


def _episodic_summary(
    pairs: Sequence[Sequence[dict[str, Any]]],
    *,
    max_tokens: int,
) -> str:
    if not pairs or max_tokens <= 0:
        return ""
    lines: list[str] = ["Earlier conversation (extractive continuation):"]
    for pair in pairs:
        for message in pair:
            role = str(message.get("role") or "message")
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            lines.append(f"{role}: {text}")
    return truncate_to_estimated_tokens("\n".join(lines), max_tokens)


def _complete_pairs(messages: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    pairs: list[list[dict[str, Any]]] = []
    index = 0
    while index + 1 < len(messages):
        user = messages[index]
        assistant = messages[index + 1]
        if user.get("role") == "user" and assistant.get("role") == "assistant":
            pairs.append([dict(user), dict(assistant)])
            index += 2
            continue
        index += 1
    return pairs


__all__ = [
    "HistoryInputMeasure",
    "HistoryProjectionOverflowError",
    "HistoryProjectionTarget",
    "project_history",
]
