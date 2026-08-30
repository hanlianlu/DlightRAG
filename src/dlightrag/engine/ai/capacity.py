# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immutable model capacity facts and unified request policy."""

from dataclasses import dataclass, field
from typing import Literal

from dlightrag.engine.ai.reasoning import ReasoningProfile

CONTEXT_POLICY_REVISION = "agent-v4-dynamic-context"
type ModelInputOverflowKind = Literal[
    "hard_input_limit_exceeded",
    "context_exhausted",
]


class ModelInputOverflowError(ValueError):
    """One model request exceeds the current policy or physical context."""

    def __init__(
        self,
        *,
        kind: ModelInputOverflowKind,
        input_tokens: int,
        input_limit_tokens: int,
    ) -> None:
        self.kind = kind
        self.input_tokens = input_tokens
        self.input_limit_tokens = input_limit_tokens
        super().__init__(
            f"model input uses {input_tokens} tokens but the {kind} limit is {input_limit_tokens}"
        )


class ModelCapabilityError(RuntimeError):
    """A resolved model profile cannot perform a required operation."""

    def __init__(self, *, role: str, capability: str) -> None:
        self.role = role
        self.capability = capability
        super().__init__(f"{role} model profile does not support {capability}")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Capacity and capability facts for one resolved model endpoint."""

    context_window_tokens: int
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    supports_images: bool = False
    reasoning: ReasoningProfile | None = None

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        if self.max_input_tokens is not None:
            if self.max_input_tokens <= 0:
                raise ValueError("max_input_tokens must be positive when provided")
            if self.max_input_tokens > self.context_window_tokens:
                raise ValueError("max_input_tokens cannot exceed the context window")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive when provided")


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """Explicit model input, output, and dynamic-context reserves.

    Provider limits remain physical facts. Absolute reserves express the
    product request instead of multiplying opaque percentages. Research may
    clamp the dynamic reserve on small profiles; Fast explicitly requests the
    full reserve and is rejected when the resolved profile cannot preserve it.
    """

    requested_output_reserve_tokens: int = 16_384
    dynamic_context_reserve_tokens: int = 40_000
    safety_reserve_tokens: int = 1_024
    retained_tail_tokens: int = 20_000
    episodic_summary_tokens: int = 8_000
    minimum_input_tokens: int = 1_024
    revision: str = field(default=CONTEXT_POLICY_REVISION, init=False)

    def __post_init__(self) -> None:
        for name in (
            "requested_output_reserve_tokens",
            "dynamic_context_reserve_tokens",
            "safety_reserve_tokens",
            "retained_tail_tokens",
            "episodic_summary_tokens",
            "minimum_input_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    def _reserved_output(self, profile: ModelProfile) -> int:
        if profile.max_output_tokens is None:
            return self.requested_output_reserve_tokens
        return min(profile.max_output_tokens, self.requested_output_reserve_tokens)

    def hard_input_limit(self, profile: ModelProfile) -> int:
        """Return provider input capacity after explicit output and safety reserves."""
        provider_limit = profile.max_input_tokens or profile.context_window_tokens
        context = profile.context_window_tokens
        floor = min(self.minimum_input_tokens, provider_limit, context)
        output = min(self._reserved_output(profile), max(0, context - floor))
        safety = min(
            self.safety_reserve_tokens,
            max(0, context - output - floor),
        )
        context_input_limit = max(1, context - output - safety)
        return min(provider_limit, context_input_limit)

    def compaction_trigger(
        self,
        profile: ModelProfile,
        *,
        require_full_dynamic_reserve: bool = False,
    ) -> int:
        """Return the proactive input ceiling before dynamic context is added.

        Fast passes ``require_full_dynamic_reserve=True``: a negative or tiny
        ceiling is intentional and makes its fixed envelope fail admission
        instead of silently shrinking the 40K product reserve.
        """
        hard_limit = self.hard_input_limit(profile)
        if require_full_dynamic_reserve:
            return hard_limit - self.dynamic_context_reserve_tokens
        floor = min(self.minimum_input_tokens, hard_limit)
        reserve = min(self.dynamic_context_reserve_tokens, max(0, hard_limit - floor))
        return hard_limit - reserve

    def history_allowance_cap(
        self,
        profile: ModelProfile,
        *,
        require_full_dynamic_reserve: bool = False,
    ) -> int:
        """Let each reachable call allocate history from its actual residual."""
        return self.compaction_trigger(
            profile,
            require_full_dynamic_reserve=require_full_dynamic_reserve,
        )

    def retained_tail_target(self, profile: ModelProfile) -> int:
        """Return the absolute recent Research exchange target retained verbatim."""
        return min(self.retained_tail_tokens, self.hard_input_limit(profile))

    def classify_input(
        self,
        profile: ModelProfile,
        *,
        input_tokens: int,
    ) -> ModelInputOverflowKind | None:
        """Classify an oversized request without changing model facts."""
        if input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")
        if input_tokens >= profile.context_window_tokens:
            return "context_exhausted"
        if input_tokens > self.hard_input_limit(profile):
            return "hard_input_limit_exceeded"
        return None

    def output_allowance(
        self,
        profile: ModelProfile,
        *,
        input_tokens: int,
    ) -> int | None:
        """Return the requested cap bounded by the physical context remainder."""
        overflow = self.classify_input(profile, input_tokens=input_tokens)
        if overflow is not None:
            limit = (
                profile.context_window_tokens - 1
                if overflow == "context_exhausted"
                else self.hard_input_limit(profile)
            )
            raise ModelInputOverflowError(
                kind=overflow,
                input_tokens=input_tokens,
                input_limit_tokens=limit,
            )
        physical_remainder = profile.context_window_tokens - input_tokens
        requested = (
            profile.max_output_tokens
            if profile.max_output_tokens is not None
            else self.requested_output_reserve_tokens
        )
        return min(requested, physical_remainder)


CONTEXT_POLICY = ContextPolicy()

__all__ = [
    "CONTEXT_POLICY",
    "CONTEXT_POLICY_REVISION",
    "ContextPolicy",
    "ModelCapabilityError",
    "ModelInputOverflowError",
    "ModelInputOverflowKind",
    "ModelProfile",
]
