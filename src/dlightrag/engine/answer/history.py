# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Project one pinned history across every reachable model call."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.engine.agent.session.fold import PriorTurns
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ContextPolicy, ModelProfile
from dlightrag.engine.ai.tokens import estimate_tokens, truncate_to_estimated_tokens


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
    require_full_dynamic_reserve: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("history projection target name must be non-empty")
        if self.require_full_dynamic_reserve and not self.proactive_compaction:
            raise ValueError("a full dynamic reserve requires proactive compaction")

    def acceptance_limit(self, context_policy: ContextPolicy = CONTEXT_POLICY) -> int:
        """The input this call is accepted and compacted against.

        A proactively compacting call stops at its compaction trigger, so the
        history it is handed leaves room for the output it still has to produce;
        any other call may use its model's whole hard input limit.
        """
        if self.proactive_compaction:
            return context_policy.compaction_trigger(
                self.profile,
                require_full_dynamic_reserve=self.require_full_dynamic_reserve,
            )
        return context_policy.hard_input_limit(self.profile)


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


class IncrementalHistoryProjector:
    """Bounded two-phase projection for a durable history adapter.

    Recent pairs are offered newest-first until the first exact target rejection.
    When that happens, omitted pairs are offered oldest-first; only the bounded
    extractive prefix is retained. This is the same policy as :func:`project_history`
    without materializing an arbitrary number of durable turns.
    """

    def __init__(self, *, targets: Sequence[HistoryProjectionTarget]) -> None:
        self._resolved = tuple(_resolve_target(target, CONTEXT_POLICY) for target in targets)
        self._max_summary_tokens = CONTEXT_POLICY.episodic_summary_tokens
        self._kept: list[dict[str, Any]] = []
        self._recent_complete = False
        self._summary = ""
        self._summary_saturated = False

    @property
    def accepts_history(self) -> bool:
        return bool(self._resolved)

    @property
    def needs_omitted_pairs(self) -> bool:
        return (
            bool(self._resolved)
            and self._recent_complete
            and self._max_summary_tokens > 0
            and not self._summary_saturated
        )

    def offer_newest_pair(
        self,
        user: Mapping[str, Any],
        assistant: Mapping[str, Any],
    ) -> bool:
        """Retain one next-older complete pair, or establish the recent cutoff."""
        if self._recent_complete or not self._resolved:
            return False
        pair = [dict(user), dict(assistant)]
        candidate = [*pair, *self._kept]
        if not all(_fits(candidate, "", target) for target in self._resolved):
            self._recent_complete = True
            return False
        self._kept = candidate
        return True

    def offer_oldest_omitted_pair(
        self,
        user: Mapping[str, Any],
        assistant: Mapping[str, Any],
    ) -> bool:
        """Append one omitted pair while it can still alter the bounded prefix."""
        if not self.needs_omitted_pairs:
            return False
        lines = []
        if not self._summary:
            lines.append("Earlier conversation (extractive continuation):")
        for message in (user, assistant):
            role = str(message.get("role") or "message")
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            lines.append(f"{role}: {text}")
        candidate = "\n".join(filter(None, (self._summary, *lines)))
        if estimate_tokens(candidate) > self._max_summary_tokens:
            self._summary = truncate_to_estimated_tokens(candidate, self._max_summary_tokens)
            self._summary_saturated = True
        else:
            self._summary = candidate.strip()
        return True

    def finish(self) -> PriorTurns:
        summary = _fit_summary_text(
            self._kept,
            self._summary,
            resolved=self._resolved,
            max_tokens=self._max_summary_tokens,
        )
        return PriorTurns(self._kept, episodic_summary=summary)


def project_history(
    messages: Sequence[dict[str, Any]],
    *,
    targets: Sequence[HistoryProjectionTarget],
) -> PriorTurns:
    """Keep the newest contiguous complete pairs accepted by every target."""
    projector = IncrementalHistoryProjector(targets=targets)
    pairs = _complete_pairs(messages)
    if not projector.accepts_history:
        return PriorTurns()
    retained_pairs = 0
    for pair in reversed(pairs):
        if not projector.offer_newest_pair(pair[0], pair[1]):
            break
        retained_pairs += 1
    if retained_pairs != len(pairs):
        for pair in pairs[: len(pairs) - retained_pairs]:
            if not projector.offer_oldest_omitted_pair(pair[0], pair[1]):
                break
    return projector.finish()


def _resolve_target(
    target: HistoryProjectionTarget,
    context_policy: ContextPolicy,
) -> _ResolvedTarget:
    acceptance_limit = target.acceptance_limit(context_policy)
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
            context_policy.history_allowance_cap(
                target.profile,
                require_full_dynamic_reserve=target.require_full_dynamic_reserve,
            ),
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


def _fit_summary_text(
    kept: list[dict[str, Any]],
    summary: str,
    *,
    resolved: Sequence[_ResolvedTarget],
    max_tokens: int,
) -> str:
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
    "IncrementalHistoryProjector",
    "HistoryProjectionOverflowError",
    "HistoryProjectionTarget",
    "project_history",
]
