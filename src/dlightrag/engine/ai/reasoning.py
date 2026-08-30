# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral reasoning levels, resolution, and request translation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

type ReasoningLevel = Literal[
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]

REASONING_LEVELS: tuple[ReasoningLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
REASONING_FORMATS = frozenset(
    {"openrouter", "openai", "deepseek", "anthropic_native", "gemini_native"}
)

# Raw provider fields owned by typed reasoning configuration. A caller may use
# them only when no semantic reasoning level is configured for that request.
PROVIDER_REASONING_KEYS = frozenset(
    {
        "reasoning",
        "reasoning_effort",
        "thinking",
        "thinking_config",
        "enable_thinking",
        "chat_template_args",
        "chat_template_kwargs",
        "output_config",
    }
)


class ReasoningConfigurationError(ValueError):
    """A semantic reasoning request cannot be honored by a model profile."""


@dataclass(frozen=True, slots=True)
class ReasoningLevels:
    """Explicit provider values for every DlightRAG reasoning level.

    ``None`` means unsupported. Every field is required so catalogue entries
    cannot accidentally inherit a capability when new model facts are edited.
    """

    off: str | None
    minimal: str | None
    low: str | None
    medium: str | None
    high: str | None
    xhigh: str | None
    max: str | None

    def __post_init__(self) -> None:
        for level in REASONING_LEVELS:
            value = getattr(self, level)
            if value is not None and (not value or value != value.strip()):
                raise ValueError(f"reasoning level {level} must be null or a canonical string")

    def value(self, level: ReasoningLevel) -> str | None:
        return getattr(self, level)

    def as_dict(self) -> dict[str, str | None]:
        return {level: self.value(level) for level in REASONING_LEVELS}


@dataclass(frozen=True, slots=True)
class ReasoningProfile:
    """One model endpoint's request dialect and supported semantic levels."""

    format: str
    levels: ReasoningLevels

    def __post_init__(self) -> None:
        if self.format not in REASONING_FORMATS:
            supported = ", ".join(sorted(REASONING_FORMATS))
            raise ValueError(f"reasoning format must be one of: {supported}")

    def as_dict(self) -> dict[str, Any]:
        return {"format": self.format, "levels": self.levels.as_dict()}


@dataclass(frozen=True, slots=True)
class ResolvedReasoning:
    """The requested level and deterministic effective model level."""

    requested: ReasoningLevel
    effective: ReasoningLevel
    profile: ReasoningProfile

    @property
    def provider_value(self) -> str:
        value = self.profile.levels.value(self.effective)
        if value is None:  # guarded by ``resolve_reasoning``
            raise RuntimeError("resolved reasoning level has no provider value")
        return value


def resolve_reasoning(
    profile: ReasoningProfile | None,
    requested: ReasoningLevel | None,
) -> ResolvedReasoning | None:
    """Resolve one semantic request against immutable endpoint facts.

    Explicit ``off`` is a hard requirement. Other unsupported levels clamp to
    the nearest supported non-off level, preferring the next higher level when
    distances tie, then walking downward.
    """
    if requested is None:
        return None
    if profile is None:
        raise ReasoningConfigurationError(
            f"reasoning={requested!r} requires a catalogued reasoning profile"
        )
    if requested == "off":
        if profile.levels.off is None:
            raise ReasoningConfigurationError("reasoning='off' cannot be honored by this model")
        return ResolvedReasoning(requested="off", effective="off", profile=profile)

    requested_index = REASONING_LEVELS.index(requested)
    # Find the nearest non-off level; ties clamp upward so a request never
    # silently receives less effort when equally close choices exist.
    for distance in range(len(REASONING_LEVELS)):
        higher = requested_index + distance
        if higher < len(REASONING_LEVELS):
            candidate = REASONING_LEVELS[higher]
            if candidate != "off" and profile.levels.value(candidate) is not None:
                return ResolvedReasoning(
                    requested=requested,
                    effective=candidate,
                    profile=profile,
                )
        lower = requested_index - distance
        if distance and lower > 0:
            candidate = REASONING_LEVELS[lower]
            if profile.levels.value(candidate) is not None:
                return ResolvedReasoning(
                    requested=requested,
                    effective=candidate,
                    profile=profile,
                )
    raise ReasoningConfigurationError(
        f"reasoning={requested!r} requires a supported non-off reasoning level"
    )


def cheapest_supported_reasoning(profile: ReasoningProfile | None) -> ReasoningLevel | None:
    """Return the cheapest controllable level for internal compaction policy."""
    if profile is None:
        return None
    for level in REASONING_LEVELS:
        if profile.levels.value(level) is not None:
            return level
    return None


def conflicting_reasoning_keys(values: Mapping[str, Any]) -> tuple[str, ...]:
    """Return non-empty raw fields owned by typed reasoning configuration."""

    def is_nonempty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str | Mapping | list | tuple):
            return bool(value)
        return True

    return tuple(
        sorted(
            key for key in PROVIDER_REASONING_KEYS.intersection(values) if is_nonempty(values[key])
        )
    )


def reasoning_request_kwargs(resolved: ResolvedReasoning | None) -> dict[str, Any]:
    """Translate a resolved semantic level into its provider request dialect.

    Catalogue validation admits only the formats handled here. Unknown endpoint
    controls remain available through raw model kwargs when typed reasoning is absent.
    """
    if resolved is None:
        return {}
    level = resolved.effective
    value = resolved.provider_value
    format_name = resolved.profile.format

    if format_name == "openrouter":
        if level == "off":
            return (
                {"reasoning": {"enabled": False}}
                if value == "disabled"
                else {"reasoning": {"effort": value}}
            )
        if value == "enabled":
            return {"reasoning": {"enabled": True}}
        return {"reasoning": {"effort": value}}

    if format_name == "openai":
        return {"reasoning_effort": value}

    if format_name == "deepseek":
        if level == "off":
            return {"thinking": {"type": "disabled"}}
        return {
            "thinking": {"type": "enabled"},
            "reasoning_effort": value,
        }

    if format_name == "anthropic_native":
        if level == "off":
            return {"thinking": {"type": "disabled"}}
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"thinking": {"effort": value}},
        }

    if format_name == "gemini_native":
        if level == "off":
            return {"thinking_config": {"thinking_budget": 0}}
        return {
            "thinking_config": {
                "include_thoughts": True,
                "thinking_level": value.upper(),
            }
        }

    return {}


def merge_reasoning_kwargs(
    raw: Mapping[str, Any],
    resolved: ResolvedReasoning | None,
) -> dict[str, Any]:
    """Merge typed reasoning into raw kwargs after enforcing single ownership."""
    merged = dict(raw)
    if resolved is None:
        return merged
    conflicts = conflicting_reasoning_keys(merged)
    if conflicts:
        raise ReasoningConfigurationError(
            "typed reasoning conflicts with raw model kwargs: " + ", ".join(conflicts)
        )
    merged.update(reasoning_request_kwargs(resolved))
    return merged


__all__ = [
    "PROVIDER_REASONING_KEYS",
    "REASONING_FORMATS",
    "REASONING_LEVELS",
    "ReasoningConfigurationError",
    "ReasoningLevel",
    "ReasoningLevels",
    "ReasoningProfile",
    "ResolvedReasoning",
    "cheapest_supported_reasoning",
    "conflicting_reasoning_keys",
    "merge_reasoning_kwargs",
    "reasoning_request_kwargs",
    "resolve_reasoning",
]
