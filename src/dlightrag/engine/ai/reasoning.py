# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral reasoning levels, resolution, and request translation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dlightrag.engine.ai.contracts import ApiFamily

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
REASONING_FORMATS = frozenset({"openrouter", "openai", "deepseek", "anthropic", "gemini"})
#: The thinking levels Gemini's Interactions API names, which a catalogue entry may map to.
#: It has none that turns thinking off, and a model accepts only the ones its documentation lists.
GEMINI_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high", "xhigh"})
#: The levels an uncatalogued Gemini model is asked for: the four every model accepts. ``xhigh``
#: belongs to the models that list it, and an older one answers it with a 400.
GEMINI_BEST_EFFORT_LEVELS = frozenset({"minimal", "low", "medium", "high"})

# Raw provider fields owned by typed reasoning configuration. A caller may use
# them only when no semantic reasoning level is configured for that request.
PROVIDER_REASONING_KEYS = frozenset(
    {
        "reasoning",
        "reasoning_effort",
        "thinking",
        "thinking_level",
        "thinking_summaries",
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
    best_effort: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.format not in REASONING_FORMATS:
            supported = ", ".join(sorted(REASONING_FORMATS))
            raise ValueError(f"reasoning format must be one of: {supported}")
        if self.format == "gemini" and (
            self.levels.off is not None
            or not {value for value in self.levels.as_dict().values() if value is not None}
            <= GEMINI_THINKING_LEVELS
        ):
            raise ValueError(
                "gemini reasoning maps levels to minimal, low, medium, high or xhigh, "
                "and off to null"
            )

    @classmethod
    def unverified(cls, *, format: str, levels: ReasoningLevels) -> ReasoningProfile:
        profile = cls(format=format, levels=levels)
        object.__setattr__(profile, "best_effort", True)
        return profile

    def as_dict(self) -> dict[str, Any]:
        return {"format": self.format, "levels": self.levels.as_dict()}


def best_effort_reasoning_profile(format_name: str) -> ReasoningProfile:
    """Build an unverified protocol mapping for one uncatalogued endpoint.

    An unknown Gemini model is offered the four thinking levels every model accepts, so higher
    requests clamp to ``high`` and ``off`` cannot be honored.
    """
    if format_name == "gemini":
        levels = {
            level: level if level in GEMINI_BEST_EFFORT_LEVELS else None
            for level in REASONING_LEVELS
        }
        return ReasoningProfile.unverified(format=format_name, levels=ReasoningLevels(**levels))
    off = "disabled" if format_name in {"openrouter", "deepseek", "anthropic"} else "none"
    return ReasoningProfile.unverified(
        format=format_name,
        levels=ReasoningLevels(
            off=off,
            minimal="minimal",
            low="low",
            medium="medium",
            high="high",
            xhigh="xhigh",
            max="max",
        ),
    )


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


def request_under_ceiling(
    profile: ReasoningProfile | None,
    requested: ReasoningLevel | None,
    ceiling: ReasoningLevel,
) -> ReasoningLevel | None:
    """The level to ask a model for so that it never runs above ``ceiling``.

    A request that already resolves at or below the ceiling is returned as it is, and so is one
    the profile cannot resolve, which fails where it always did. A request that would resolve
    above it, and a model left on its own level (``requested`` None) under a ceiling below the
    top of the ladder, are asked for the highest level the model offers at or below the ceiling
    instead. The ladder is walked downward, never rounded the way ``resolve_reasoning`` rounds
    a tie: a rounding that crossed the ceiling would be no ceiling at all.

    ``off`` is not a target: it is the explicit, hard request it is everywhere else. A model
    that offers nothing at or below the ceiling is asked for its lowest level, which is above
    it; the level this resolves to says so.
    """
    top = REASONING_LEVELS.index(ceiling)
    if profile is None or top == len(REASONING_LEVELS) - 1:
        return requested
    offered: list[ReasoningLevel] = [
        level for level in REASONING_LEVELS[1:] if profile.levels.value(level) is not None
    ]
    if not offered:
        return requested
    if requested is not None:
        try:
            resolved = resolve_reasoning(profile, requested)
        except ReasoningConfigurationError:
            return requested
        if resolved is not None and REASONING_LEVELS.index(resolved.effective) <= top:
            return requested
    within: list[ReasoningLevel] = [
        level for level in offered if REASONING_LEVELS.index(level) <= top
    ]
    return within[-1] if within else offered[0]


def cheapest_supported_reasoning(profile: ReasoningProfile | None) -> ReasoningLevel | None:
    """Return the cheapest verified level for internal compaction policy."""
    if profile is None or profile.best_effort:
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


def reasoning_request_kwargs(
    resolved: ResolvedReasoning | None,
    *,
    api_family: ApiFamily = "chat_completion",
) -> dict[str, Any]:
    """Translate a resolved semantic level into its family-native request dialect.

    Catalogue validation admits only the formats handled here. Unknown endpoint
    controls remain available through raw model kwargs when typed reasoning is absent.
    """
    if resolved is None:
        return {}
    level = resolved.effective
    value = resolved.provider_value
    format_name = resolved.profile.format

    if api_family == "interactions" and format_name != "gemini":
        raise ReasoningConfigurationError(
            f"reasoning format {format_name!r} cannot be represented by the Interactions API"
        )

    if api_family == "response":
        if format_name not in {"openai", "openrouter", "deepseek"}:
            raise ReasoningConfigurationError(
                f"reasoning format {format_name!r} cannot be represented by the Response API"
            )
        if level == "off":
            effort = "none"
        elif format_name == "openrouter" and value == "enabled":
            # An on/off-only endpoint has no effort to name. This path names reasoning
            # by effort, and OpenRouter documents enabled=true as medium effort.
            effort = "medium"
        else:
            effort = value
        return {"reasoning": {"effort": effort}}

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

    if format_name == "anthropic":
        if level == "off":
            return {"thinking": {"type": "disabled"}}
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": value},
        }

    if format_name == "gemini":
        # The level is a thinking_level as written (a gemini profile has no off), and a
        # summary is the only readable reasoning the Interactions API returns.
        return {"thinking_level": value, "thinking_summaries": "auto"}

    return {}


def merge_reasoning_kwargs(
    raw: Mapping[str, Any],
    resolved: ResolvedReasoning | None,
    *,
    api_family: ApiFamily = "chat_completion",
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
    merged.update(reasoning_request_kwargs(resolved, api_family=api_family))
    return merged


__all__ = [
    "GEMINI_BEST_EFFORT_LEVELS",
    "GEMINI_THINKING_LEVELS",
    "PROVIDER_REASONING_KEYS",
    "REASONING_FORMATS",
    "REASONING_LEVELS",
    "ReasoningConfigurationError",
    "ReasoningLevel",
    "ReasoningLevels",
    "ReasoningProfile",
    "ResolvedReasoning",
    "best_effort_reasoning_profile",
    "cheapest_supported_reasoning",
    "conflicting_reasoning_keys",
    "merge_reasoning_kwargs",
    "reasoning_request_kwargs",
    "request_under_ceiling",
    "resolve_reasoning",
]
