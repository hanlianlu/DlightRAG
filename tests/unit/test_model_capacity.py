# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for AI-owned model capacity resolution."""

from dataclasses import FrozenInstanceError

import pytest

from dlightrag.engine.ai.capacity import (
    CONTEXT_POLICY,
    ContextPolicy,
    ModelInputOverflowError,
    ModelProfile,
)
from dlightrag.engine.ai.catalog import (
    FALLBACK_MODEL_PROFILE,
    MODEL_CATALOGUE,
    parse_catalogue_entry,
    resolve_model_profile,
)
from dlightrag.engine.ai.fingerprints import ModelFingerprint, normalized_endpoint_fingerprint
from dlightrag.engine.ai.reasoning import (
    cheapest_supported_reasoning,
    reasoning_request_kwargs,
    resolve_reasoning,
)


def test_context_policy_applies_explicit_model_aware_reserves() -> None:
    profile = ModelProfile(
        context_window_tokens=200_000,
        max_input_tokens=180_000,
        max_output_tokens=16_000,
        supports_images=True,
    )
    policy = ContextPolicy()

    assert policy.revision == "agent-v4-dynamic-context"
    assert policy.dynamic_context_reserve_tokens == 40_000
    assert not hasattr(policy, "observation_reserve_tokens")
    assert policy.hard_input_limit(profile) == 180_000
    assert policy.compaction_trigger(profile) == 140_000
    assert policy.history_allowance_cap(profile) == 140_000
    assert policy.compaction_trigger(profile, require_full_dynamic_reserve=True) == 140_000
    assert policy.retained_tail_target(profile) == 20_000
    with pytest.raises(FrozenInstanceError):
        profile.context_window_tokens = 1  # type: ignore[misc]


def test_profile_resolution_prefers_runtime_complete_overlay_before_builtin() -> None:
    fingerprint = ModelFingerprint(
        provider="openai",
        model="xiaomi/mimo-v2.5",
        endpoint_fingerprint=normalized_endpoint_fingerprint("https://openrouter.ai/api/v1"),
    )
    overlay = parse_catalogue_entry(
        {
            "provider": "openai",
            "model": "xiaomi/mimo-v2.5",
            "base_url": "https://openrouter.ai/api/v1",
            "profile": {
                "context_window_tokens": 90_000,
                "max_input_tokens": None,
                "max_output_tokens": 10_000,
                "supports_images": False,
                "reasoning": None,
            },
        }
    )
    previous = MODEL_CATALOGUE.overlay
    try:
        MODEL_CATALOGUE.replace_overlay((overlay,))
        assert resolve_model_profile(fingerprint) == overlay.profile
    finally:
        MODEL_CATALOGUE.replace_overlay(previous)

    catalog_profile = resolve_model_profile(fingerprint)
    assert catalog_profile.context_window_tokens == 1_048_576
    assert catalog_profile.max_output_tokens == 131_072
    assert catalog_profile.supports_images is True
    assert catalog_profile.reasoning is not None


def test_unknown_model_resolves_to_the_fallback_profile() -> None:
    fingerprint = ModelFingerprint(
        provider="openai",
        model="private-model",
        endpoint_fingerprint="endpoint-hash",
    )

    resolved = resolve_model_profile(fingerprint)

    assert resolved.context_window_tokens == FALLBACK_MODEL_PROFILE.context_window_tokens
    assert resolved.max_input_tokens == FALLBACK_MODEL_PROFILE.max_input_tokens
    assert resolved.max_output_tokens == FALLBACK_MODEL_PROFILE.max_output_tokens
    assert resolved.supports_images is FALLBACK_MODEL_PROFILE.supports_images
    assert resolved.reasoning is not None
    assert resolved.reasoning.format == "openai"
    assert resolved.reasoning.best_effort is True
    assert cheapest_supported_reasoning(resolved.reasoning) is None
    assert not hasattr(resolved, "supports_tools")


@pytest.mark.parametrize(
    ("provider", "endpoint", "format_name", "expected"),
    [
        (
            "openai",
            "https://openrouter.ai/api/v1",
            "openrouter",
            {"reasoning": {"effort": "high"}},
        ),
        (
            "openai",
            "https://api.deepseek.com/v1",
            "deepseek",
            {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        ),
        ("openai", "https://api.example.test/v1", "openai", {"reasoning_effort": "high"}),
        (
            "anthropic",
            None,
            "anthropic",
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        ),
        (
            "gemini",
            None,
            "gemini",
            {"thinking_config": {"include_thoughts": True, "thinking_level": "HIGH"}},
        ),
    ],
)
def test_unknown_model_reasoning_uses_protocol_derived_best_effort_mapping(
    provider: str,
    endpoint: str | None,
    format_name: str,
    expected: dict[str, object],
) -> None:
    profile = resolve_model_profile(
        ModelFingerprint(
            provider=provider,
            model="private-model",
            endpoint_fingerprint=normalized_endpoint_fingerprint(endpoint),
        )
    )

    assert profile.reasoning is not None
    assert profile.reasoning.format == format_name
    assert profile.reasoning.best_effort is True
    assert reasoning_request_kwargs(resolve_reasoning(profile.reasoning, "high")) == expected


def test_fast_full_dynamic_reserve_is_not_clamped_on_small_profiles() -> None:
    profile = ModelProfile(context_window_tokens=30_000)
    policy = ContextPolicy(
        requested_output_reserve_tokens=0,
        dynamic_context_reserve_tokens=40_000,
        safety_reserve_tokens=0,
        minimum_input_tokens=1_024,
    )

    assert policy.compaction_trigger(profile) == 1_024
    assert policy.compaction_trigger(profile, require_full_dynamic_reserve=True) == -10_000


def test_policy_classifies_overflow_and_caps_required_output_to_physical_remainder() -> None:
    profile = ModelProfile(
        context_window_tokens=1_000,
        max_input_tokens=850,
        max_output_tokens=300,
    )
    policy = ContextPolicy(
        requested_output_reserve_tokens=0,
        dynamic_context_reserve_tokens=0,
        safety_reserve_tokens=0,
        minimum_input_tokens=0,
    )

    assert policy.output_allowance(profile, input_tokens=800) == 200
    with pytest.raises(ModelInputOverflowError) as hard_limit:
        policy.output_allowance(profile, input_tokens=851)
    assert hard_limit.value.kind == "hard_input_limit_exceeded"
    with pytest.raises(ModelInputOverflowError) as exhausted:
        policy.output_allowance(profile, input_tokens=1_000)
    assert exhausted.value.kind == "context_exhausted"

    uncapped = ModelProfile(context_window_tokens=1_000)
    assert CONTEXT_POLICY.output_allowance(uncapped, input_tokens=800) == 200


def test_unknown_output_profile_still_reserves_and_caps_requested_output() -> None:
    profile = ModelProfile(context_window_tokens=100_000, max_input_tokens=95_000)

    assert CONTEXT_POLICY.hard_input_limit(profile) == 82_592
    assert CONTEXT_POLICY.output_allowance(profile, input_tokens=80_000) == 16_384
