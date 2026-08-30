# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral reasoning resolution and kwargs ownership tests."""

import pytest
from pydantic import ValidationError

from dlightrag.engine.ai.reasoning import (
    ReasoningConfigurationError,
    ReasoningLevels,
    ReasoningProfile,
    cheapest_supported_reasoning,
    reasoning_request_kwargs,
    resolve_reasoning,
)
from dlightrag.engine.ai.settings import ModelRoleSettings, ModelSettings


def _profile(
    *,
    format: str = "openrouter",
    off: str | None = "disabled",
    minimal: str | None = None,
    low: str | None = "low",
    medium: str | None = None,
    high: str | None = "high",
    xhigh: str | None = None,
    max: str | None = "max",
) -> ReasoningProfile:
    return ReasoningProfile(
        format=format,
        levels=ReasoningLevels(
            off=off,
            minimal=minimal,
            low=low,
            medium=medium,
            high=high,
            xhigh=xhigh,
            max=max,
        ),
    )


def test_reasoning_profile_is_immutable_and_hashable() -> None:
    profile = _profile()

    assert hash(profile)
    with pytest.raises(AttributeError):
        profile.format = "openai"  # type: ignore[misc]


def test_off_is_a_hard_requirement() -> None:
    with pytest.raises(ReasoningConfigurationError, match="cannot be honored"):
        resolve_reasoning(_profile(off=None), "off")


def test_non_off_clamps_to_nearest_supported_level_and_breaks_ties_upward() -> None:
    profile = _profile(low="low", high="high", max=None)

    assert resolve_reasoning(profile, "medium").effective == "high"  # type: ignore[union-attr]
    assert resolve_reasoning(profile, "xhigh").effective == "high"  # type: ignore[union-attr]
    assert resolve_reasoning(profile, "minimal").effective == "low"  # type: ignore[union-attr]


def test_compaction_selects_cheapest_supported_control() -> None:
    assert cheapest_supported_reasoning(_profile()) == "off"
    assert cheapest_supported_reasoning(_profile(off=None, low=None, high="high")) == "high"
    assert cheapest_supported_reasoning(None) is None


@pytest.mark.parametrize(
    ("format", "level", "expected"),
    [
        ("openrouter", "off", {"reasoning": {"enabled": False}}),
        ("openrouter", "high", {"reasoning": {"effort": "high"}}),
        ("openai", "high", {"reasoning_effort": "high"}),
        (
            "deepseek",
            "high",
            {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        ),
        ("deepseek", "off", {"thinking": {"type": "disabled"}}),
        (
            "anthropic_native",
            "high",
            {
                "thinking": {"type": "adaptive"},
                "output_config": {"thinking": {"effort": "high"}},
            },
        ),
        ("anthropic_native", "off", {"thinking": {"type": "disabled"}}),
        (
            "gemini_native",
            "high",
            {"thinking_config": {"include_thoughts": True, "thinking_level": "HIGH"}},
        ),
        ("gemini_native", "off", {"thinking_config": {"thinking_budget": 0}}),
    ],
)
def test_resolved_reasoning_translates_only_by_catalogue_format(
    format: str,
    level: str,
    expected: dict[str, object],
) -> None:
    resolved = resolve_reasoning(_profile(format=format), level)  # type: ignore[arg-type]

    assert reasoning_request_kwargs(resolved) == expected


def test_openrouter_off_can_map_to_a_native_effort_value() -> None:
    resolved = resolve_reasoning(_profile(off="none"), "off")

    assert reasoning_request_kwargs(resolved) == {"reasoning": {"effort": "none"}}


def test_catalogue_rejects_an_unknown_reasoning_format() -> None:
    with pytest.raises(ValueError, match="format must be one of"):
        _profile(format="future-wire-shape")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_kwargs", {"reasoning": {"enabled": False}}),
        ("model_kwargs", {"thinking": {"type": "disabled"}}),
        ("model_kwargs", {"reasoning_effort": "low"}),
        ("agentic_model_kwargs", {"thinking_config": {"thinking_budget": 0}}),
        ("agentic_model_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
    ],
)
def test_typed_reasoning_rejects_provider_native_reasoning_ownership_conflicts(
    field: str,
    value: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="conflicts"):
        ModelSettings.model_validate({"model": "m", "reasoning": "off", field: value})


@pytest.mark.parametrize("empty_value", [None, "", {}, []])
def test_typed_reasoning_ignores_empty_raw_placeholders(empty_value: object) -> None:
    settings = ModelSettings(
        model="m",
        reasoning="low",
        model_kwargs={"reasoning": empty_value},
    )

    assert settings.reasoning == "low"


def test_raw_reasoning_kwargs_remain_an_explicit_escape_hatch_without_typed_reasoning() -> None:
    settings = ModelSettings(
        model="private-model",
        model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
        agentic_model_kwargs={"chat_template_kwargs": {"enable_thinking": True}},
    )

    assert settings.model_kwargs["chat_template_kwargs"]["enable_thinking"] is False
    assert settings.agentic_model_kwargs_copy()["chat_template_kwargs"]["enable_thinking"] is True


def test_partial_default_mapping_preserves_agentic_reasoning_inheritance() -> None:
    roles = ModelRoleSettings.model_validate({"default": {"model": "m", "reasoning": "high"}})

    assert roles.default.effective_agentic_reasoning == "high"
    assert "agentic_reasoning" not in roles.default.model_fields_set


def test_partial_default_mapping_preserves_explicit_null_agentic_reasoning() -> None:
    roles = ModelRoleSettings.model_validate(
        {
            "default": {
                "model": "m",
                "reasoning": "high",
                "agentic_reasoning": None,
            }
        }
    )

    assert roles.default.effective_agentic_reasoning is None
    assert "agentic_reasoning" in roles.default.model_fields_set


def test_explicit_null_agentic_reasoning_can_use_raw_agentic_escape_hatch() -> None:
    settings = ModelSettings(
        model="private-model",
        reasoning="low",
        agentic_reasoning=None,
        agentic_model_kwargs={"thinking": {"type": "enabled"}},
    )

    assert settings.effective_agentic_reasoning is None
