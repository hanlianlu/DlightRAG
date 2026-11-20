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
from dlightrag.engine.ai.fingerprints import (
    ModelEndpointFingerprint,
    normalized_endpoint_fingerprint,
)
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

    assert policy.revision == "agent-v6-absolute-reserves"
    # The configured 60K is the reserve a large window keeps whole; this profile's hard input
    # limit cannot afford it, so it keeps the larger of a third of itself and the flat 40,000
    # the product charged before this change — 54,666 here.
    assert policy.dynamic_context_reserve_tokens == 60_000
    assert policy.dynamic_context_reserve(profile) == 60_000
    assert policy.hard_input_limit(profile) == 180_000
    assert policy.compaction_trigger(profile) == 120_000
    assert policy.history_allowance_cap(profile) == 120_000
    # Fast takes the reserve whole, so its ceiling is the same either way on this profile.
    assert policy.compaction_trigger(profile, require_full_dynamic_reserve=True) == 120_000
    assert policy.observation_capacity(profile) == 60_000
    # The ceiling is the guarantee named for what it limits: one result may not claim a share
    # of the window on top of the reserve, because a batch larger than the reserve is a turn
    # no compaction can shed.
    assert policy.tool_result_ceiling(profile) == 60_000
    assert policy.retained_tail_target(profile) == 20_000
    with pytest.raises(FrozenInstanceError):
        profile.context_window_tokens = 1  # type: ignore[misc]


def test_the_dynamic_reserve_shrinks_with_the_profile_and_never_cuts_its_history() -> None:
    """The 60K is a floor, not a flat charge: a window keeps what it can afford.

    A window too small for 60,000 keeps the larger of a third of its hard input limit — the
    share that leaves history — and the flat 40,000 the product charged before this change, so
    shrinking the constant is never how the floor is paid for.
    """
    policy = ContextPolicy()

    def bounded(window: int) -> ModelProfile:
        # An output cap small enough that the window itself is the binding constraint.
        return ModelProfile(context_window_tokens=window, max_output_tokens=1024)

    for window in (8_000, 32_000, 128_000, 200_000, 500_000, 1_000_000):
        profile = bounded(window)
        hard = policy.hard_input_limit(profile)
        reserve = policy.dynamic_context_reserve(profile)
        ceiling = policy.tool_result_ceiling(profile)
        floor = min(policy.minimum_input_tokens, hard)
        previously_charged = min(40_000, max(0, hard - floor))
        assert reserve >= 0
        assert reserve <= policy.dynamic_context_reserve_tokens
        # No more than a third, never less than the guarantee the profile already had, and the
        # larger of the two decides when the profile cannot afford the configured floor.
        assert reserve == min(
            policy.dynamic_context_reserve_tokens, max(hard // 3, previously_charged)
        )
        assert reserve >= previously_charged
        assert ceiling <= hard
        assert ceiling >= policy.observation_capacity(profile)
        assert policy.observation_capacity(profile) == reserve
        assert policy.compaction_trigger(profile) == hard - reserve
        assert policy.compaction_trigger(profile) >= 0

    # Small windows keep proportionally less, large ones keep the whole floor. Either way the
    # region between 40K and 100K of hard input is where the old flat charge and the third
    # cross, and neither is ever below the other's guarantee.
    assert policy.dynamic_context_reserve(bounded(128_000)) >= 40_000
    assert policy.dynamic_context_reserve(bounded(1_000_000)) == 60_000
    # The ceiling is the guarantee, not a share of the window: three full-size results in one
    # exchange must still fit under the trigger that then bounds the turn.
    assert policy.tool_result_ceiling(bounded(1_000_000)) == policy.observation_capacity(
        bounded(1_000_000)
    )
    assert 3 * policy.tool_result_ceiling(bounded(1_000_000)) < policy.compaction_trigger(
        bounded(1_000_000)
    )


def test_the_reserve_never_shrinks_the_guarantee_a_profile_already_had() -> None:
    """The change raises capacity; it must not take any away.

    Shrinking a flat constant is the one thing a floor may not do, so for every window the
    effective reserve is at least the flat 40,000 (or the hard input limit less the request
    floor, for a window that cannot hold that much) that the product charged before this
    change. This is the regression that a third-only clamp would have introduced on mid-size
    profiles, where a third is 37,205 against 40,000.
    """
    policy = ContextPolicy()

    for window in (8_000, 17_408, 32_000, 64_000, 100_000, 128_000, 200_000, 500_000, 1_000_000):
        for output in (1_024, 16_000, None):
            profile = ModelProfile(context_window_tokens=window, max_output_tokens=output)
            hard = policy.hard_input_limit(profile)
            floor = min(policy.minimum_input_tokens, hard)
            previously_charged = min(40_000, max(0, hard - floor))
            assert policy.dynamic_context_reserve(profile) >= previously_charged


def test_profile_resolution_prefers_runtime_complete_overlay_before_builtin() -> None:
    fingerprint = ModelEndpointFingerprint(
        provider="openai",
        model="xiaomi/mimo-v2.6-pro",
        endpoint_fingerprint=normalized_endpoint_fingerprint("https://openrouter.ai/api/v1"),
    )
    overlay = parse_catalogue_entry(
        {
            "provider": "openai",
            "model": "xiaomi/mimo-v2.6-pro",
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
    fingerprint = ModelEndpointFingerprint(
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
        ("gemini", None, "gemini", {"thinking_level": "high", "thinking_summaries": "auto"}),
    ],
)
def test_unknown_model_reasoning_uses_protocol_derived_best_effort_mapping(
    provider: str,
    endpoint: str | None,
    format_name: str,
    expected: dict[str, object],
) -> None:
    profile = resolve_model_profile(
        ModelEndpointFingerprint(
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
    """Fast admits against the whole reserve without shrinking it to fit.

    A profile too small for the reserve compacts to its input floor and fails the Fast
    envelope loudly, rather than quietly taking a smaller reserve and calling it the same
    product request. This is the one place the reserve is charged flat: Research may shrink
    it, Fast may not.
    """
    profile = ModelProfile(context_window_tokens=30_000)
    policy = ContextPolicy(
        requested_output_reserve_tokens=0,
        dynamic_context_reserve_tokens=40_000,
        minimum_input_tokens=1_024,
    )

    # Research's proactive trigger keeps the request floor and takes the rest as reserve.
    assert policy.compaction_trigger(profile) == 1_024
    # Fast keeps the whole 40,000 reserve, so its ceiling is negative and admission fails.
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

    assert CONTEXT_POLICY.hard_input_limit(profile) == 83_616
    assert CONTEXT_POLICY.output_allowance(profile, input_tokens=80_000) == 16_384


def test_reasoning_profile_must_state_its_output_allowance() -> None:
    """Reasoning shares the output allowance, so an unstated cap bounds both."""
    reasoning = resolve_model_profile(
        ModelEndpointFingerprint(
            provider="openai",
            model="private-thinking-model",
            endpoint_fingerprint=normalized_endpoint_fingerprint("https://private.example/v1"),
        )
    ).reasoning
    assert reasoning is not None

    with pytest.raises(ValueError, match="max_output_tokens must be stated"):
        ModelProfile(context_window_tokens=200_000, reasoning=reasoning)

    # Stating the cap is the whole fix, and the stated value is then the cap.
    stated = ModelProfile(
        context_window_tokens=200_000,
        max_output_tokens=128_000,
        reasoning=reasoning,
    )
    assert CONTEXT_POLICY.output_allowance(stated, input_tokens=10_000) == 128_000


def test_output_allowance_ignores_the_compaction_gap() -> None:
    """Context arithmetic gates input; it never becomes a turn's output cap."""
    profile = ModelProfile(context_window_tokens=1_000_000, max_output_tokens=300_000)
    narrow = ContextPolicy(dynamic_context_reserve_tokens=40_000)
    wide = ContextPolicy(dynamic_context_reserve_tokens=400_000)

    assert narrow.output_allowance(profile, input_tokens=120_000) == 300_000
    assert wide.output_allowance(profile, input_tokens=120_000) == 300_000
    # The reserve moves the input ceiling, which is the only thing it owns.
    assert narrow.compaction_trigger(profile) > wide.compaction_trigger(profile)


def test_operator_overlay_cannot_declare_reasoning_without_an_output_cap() -> None:
    """A deployment overlay entry is rejected the same way a builtin would be."""
    with pytest.raises(RuntimeError, match=r"profile\.max_output_tokens"):
        parse_catalogue_entry(
            {
                "provider": "openai",
                "model": "private-thinking-model",
                "base_url": "https://private.example/v1",
                "profile": {
                    "context_window_tokens": 500_000,
                    "max_input_tokens": None,
                    "max_output_tokens": None,
                    "supports_images": True,
                    "reasoning": {
                        "format": "openai",
                        "levels": {
                            "off": "none",
                            "minimal": None,
                            "low": "low",
                            "medium": "medium",
                            "high": "high",
                            "xhigh": None,
                            "max": None,
                        },
                    },
                },
            }
        )


def test_input_ceiling_carries_no_estimator_safety_margin() -> None:
    """Pin the post-removal arithmetic so a safety margin cannot creep back.

    The deleted ``safety_reserve_tokens`` was 1024 tokens against an estimator
    that undercounts real content by a median of 8%, so it never absorbed the
    error it was named for. Its absence is load-bearing: these numbers are the
    input ceilings the compaction trigger and the output allowance derive from.
    """
    policy = ContextPolicy()

    # 1024-window edge: the floor keeps a usable ceiling instead of 1 token.
    assert (
        policy.hard_input_limit(ModelProfile(context_window_tokens=10_000, max_output_tokens=8_000))
        == 2_000
    )
    # Unknown output cap falls back to the absolute output reserve, nothing more.
    assert (
        policy.hard_input_limit(
            ModelProfile(context_window_tokens=100_000, max_input_tokens=95_000)
        )
        == 83_616
    )
    # A large window keeps only the 16k output reserve.
    assert (
        policy.hard_input_limit(
            ModelProfile(context_window_tokens=1_048_576, max_output_tokens=262_144)
        )
        == 1_032_192
    )
