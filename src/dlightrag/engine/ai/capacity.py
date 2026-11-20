# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immutable model capacity facts and unified request policy."""

from dataclasses import dataclass, field
from typing import Literal

from dlightrag.engine.ai.reasoning import ReasoningProfile

CONTEXT_POLICY_REVISION = "agent-v6-absolute-reserves"

#: What a model must have for the whole 60K reserve to be affordable. Below it the
#: reserve shrinks with the profile rather than being charged flat: a constant a small
#: window cannot afford is not a reserve, it is a rewrite of that model's history budget.
_DYNAMIC_RESERVE_SHRINK_DIVISOR = 3

#: The flat reserve a profile was charged before the reserve became a floor. It survives as
#: the reserve's own floor, so no profile's guarantee is smaller than it was: shrinking a
#: constant is the one thing this change must not do.
_LEGACY_FLAT_RESERVE_TOKENS = 40_000

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
    """Capacity and capability facts for one resolved model endpoint.

    Two rules keep a turn from being cut off mid-thought.  A reasoning-capable
    profile must state ``max_output_tokens``: reasoning and the answer share one
    output allowance, so an unstated cap silently bounds both.  And context
    arithmetic gates input only — a turn's output cap comes from here, never
    from whatever room the compaction threshold has left over.
    """

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
        if self.reasoning is not None and self.max_output_tokens is None:
            raise ValueError(
                "max_output_tokens must be stated when reasoning is enabled, because "
                "reasoning and the answer share one output allowance"
            )


#: The label the runtime puts in front of a resource page.
_PAGE_LABEL_TOKENS = 128


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """Explicit model input, output, and dynamic-context reserves.

    Provider limits remain physical facts. Absolute reserves express the
    product request instead of multiplying opaque percentages. Research may
    clamp the dynamic reserve on small profiles; Fast explicitly requests the
    full reserve and is rejected when the resolved profile cannot preserve it.

    The dynamic reserve is a **floor**, not a flat charge: :meth:`dynamic_context_reserve`
    returns the configured floor where the profile can afford it, a third of the hard input
    limit otherwise, and never less than the flat reserve the product charged before this
    change. Small profiles therefore keep history instead of losing it to a constant, and no
    profile's guarantee shrinks. The reserve is still the guarantee that one Tool result fits
    under the proactive compaction trigger.

    ``tool_result_ceiling`` is that same guarantee, named for what it limits: the most one
    result may claim. It is not a share of the window on top of the reserve, because a batch of
    results larger than the reserve is a turn no compaction can shed.

    There is deliberately no estimator-safety margin. The removed
    ``safety_reserve_tokens`` was 1024 tokens against a character-heuristic
    estimator that undercounts recorded session content by a median of 8% and
    up to 46%, so it could never absorb the error it was named for. The boundary
    is covered instead by the provider's own rejection followed by compaction
    and a retry of the same turn; a rejection whose wording escapes the overflow
    matcher stays an explicit, diagnosable failure. Anchoring input measurement on
    the provider's reported usage is deliberately owned by request composition
    (:meth:`dlightrag.engine.answer.research.context.ContextAssembler.observe_provider_input`)
    rather than by this policy, which stays a pure function of model facts.
    """

    requested_output_reserve_tokens: int = 16_384
    dynamic_context_reserve_tokens: int = 60_000
    """The floor one Tool result is guaranteed and the reserve Fast admits against.

    It is not charged flat: :meth:`dynamic_context_reserve` shrinks it on profiles too
    small to afford it.
    """
    retained_tail_tokens: int = 20_000
    episodic_summary_tokens: int = 8_000
    minimum_input_tokens: int = 1_024
    revision: str = field(default=CONTEXT_POLICY_REVISION, init=False)

    def __post_init__(self) -> None:
        for name in (
            "requested_output_reserve_tokens",
            "dynamic_context_reserve_tokens",
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
        """Return provider input capacity after the explicit output reserve."""
        provider_limit = profile.max_input_tokens or profile.context_window_tokens
        context = profile.context_window_tokens
        floor = min(self.minimum_input_tokens, provider_limit, context)
        output = min(self._reserved_output(profile), max(0, context - floor))
        context_input_limit = max(1, context - output)
        return min(provider_limit, context_input_limit)

    def dynamic_context_reserve(self, profile: ModelProfile) -> int:
        """Return the reserve this profile can actually afford.

        The configured floor is charged whole while the profile has room for it, a third of the
        hard input limit otherwise, and never less than the flat reserve a profile used to be
        charged. The last clamp is a monotonicity guarantee rather than a tuning choice: the
        profile whose guarantee would otherwise shrink under this change — a 128K window, whose
        third is 37,205 against the 40,000 it paid before — keeps the larger of the two, so no
        profile's per-result capacity is a cut. The clamp is applied again where the ceiling is
        derived, which is what makes it an invariant rather than a convention.
        """
        hard_limit = self.hard_input_limit(profile)
        floor = min(self.minimum_input_tokens, hard_limit)
        previously_charged = min(_LEGACY_FLAT_RESERVE_TOKENS, max(0, hard_limit - floor))
        return max(
            0,
            min(
                self.dynamic_context_reserve_tokens,
                max(hard_limit // _DYNAMIC_RESERVE_SHRINK_DIVISOR, previously_charged),
            ),
        )

    def compaction_trigger(
        self,
        profile: ModelProfile,
        *,
        require_full_dynamic_reserve: bool = False,
    ) -> int:
        """Return the proactive input ceiling before dynamic context is added.

        Fast passes ``require_full_dynamic_reserve=True``: a negative or tiny
        ceiling is intentional and makes its fixed envelope fail admission
        instead of silently shrinking the product reserve.
        """
        hard_limit = self.hard_input_limit(profile)
        if require_full_dynamic_reserve:
            return hard_limit - self.dynamic_context_reserve_tokens
        floor = min(self.minimum_input_tokens, hard_limit)
        reserve = min(self.dynamic_context_reserve(profile), max(0, hard_limit - floor))
        return hard_limit - reserve

    def observation_capacity(self, profile: ModelProfile) -> int:
        """Return how many tokens one Tool result is guaranteed in a Research request.

        It is the room the compaction trigger leaves below the hard input limit. The
        runtime cuts a result at :meth:`tool_result_ceiling`, which is never smaller, and
        a resource read sizes its pages by this figure, so a page the reader returns is a
        page the runtime keeps whole.
        """
        return max(0, self.hard_input_limit(profile) - self.compaction_trigger(profile))

    def tool_result_ceiling(self, profile: ModelProfile) -> int:
        """Return the most one Tool result may add to a Research request.

        It is the guarantee :meth:`observation_capacity` states — the room the proactive
        compaction trigger leaves below the hard input limit — and not a share of the window
        on top of it. A larger ceiling was tried and retired: at three eighths of a 1M window
        one result could claim 368,856 tokens, so three of them in one exchange measured
        1,106,568 against a 923,616 trigger and a 983,616 limit, and compaction keeps whole
        exchanges, so the batch was not something a compaction could shed. Holding one result
        to the guarantee is what makes a batch of full-size results survivable: three of them
        cost 180,000, comfortably inside a trigger that then fires and bounds the turn. It is
        never more than the hard input limit less the request floor either, so a window too
        small to hold a request never admits a result into the room that floor keeps.
        """
        hard_limit = self.hard_input_limit(profile)
        request_floor = min(self.minimum_input_tokens, hard_limit)
        return max(0, min(self.observation_capacity(profile), hard_limit - request_floor))

    def read_window_tokens(self, profile: ModelProfile) -> int:
        """Return the longest page a resource read may return for this model.

        The capacity less the label the runtime puts before a page: a filename of up to
        128 characters, which a CJK name makes about a hundred tokens.
        """
        return max(1, self.observation_capacity(profile) - _PAGE_LABEL_TOKENS)

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
