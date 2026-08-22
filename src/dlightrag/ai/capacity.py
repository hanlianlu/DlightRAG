# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immutable model capacity facts and unified request policy."""

from dataclasses import dataclass, field
from typing import Literal

CONTEXT_POLICY_REVISION = "m1-v1"
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
    supports_tools: bool = False
    supports_reasoning: bool = False

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
    """Current revision of DlightRAG's model-input capacity arithmetic."""

    revision: str = field(default=CONTEXT_POLICY_REVISION, init=False)

    def hard_input_limit(self, profile: ModelProfile) -> int:
        """Return the hard request-input ceiling for one model profile."""
        physical_margin = profile.context_window_tokens * 85 // 100
        provider_limit = profile.max_input_tokens or profile.context_window_tokens
        return min(provider_limit, physical_margin)

    def compaction_trigger(self, profile: ModelProfile) -> int:
        """Return the proactive Research compaction threshold."""
        return self.hard_input_limit(profile) * 85 // 100

    def history_allowance_cap(self, profile: ModelProfile) -> int:
        """Return the maximum shared-history allowance before envelope fitting."""
        return self.hard_input_limit(profile) * 20 // 100

    def retained_tail_target(self, profile: ModelProfile) -> int:
        """Return the recent Research exchange target retained verbatim."""
        return self.hard_input_limit(profile) * 20 // 100

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
        """Return a required output cap, or None when the provider permits omission."""
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
        if profile.max_output_tokens is None:
            return None
        return min(
            profile.max_output_tokens,
            profile.context_window_tokens - input_tokens,
        )


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
