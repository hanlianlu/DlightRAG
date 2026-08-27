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
    resolve_model_profile,
)
from dlightrag.engine.ai.fingerprints import ModelFingerprint, normalized_endpoint_fingerprint


def test_context_policy_applies_explicit_model_aware_reserves() -> None:
    profile = ModelProfile(
        context_window_tokens=200_000,
        max_input_tokens=180_000,
        max_output_tokens=16_000,
        supports_images=True,
        supports_reasoning=True,
    )
    policy = ContextPolicy()

    assert policy.revision == "agent-v3-reserves"
    assert policy.hard_input_limit(profile) == 180_000
    assert policy.compaction_trigger(profile) == 147_232
    assert policy.history_allowance_cap(profile) == 147_232
    assert policy.retained_tail_target(profile) == 20_000
    with pytest.raises(FrozenInstanceError):
        profile.context_window_tokens = 1  # type: ignore[misc]


def test_profile_resolution_prefers_the_complete_override_before_adapter_facts() -> None:
    fingerprint = ModelFingerprint(
        provider="openai",
        model="xiaomi/mimo-v2.5",
        endpoint_fingerprint=normalized_endpoint_fingerprint("https://openrouter.ai/api/v1"),
    )
    override = ModelProfile(
        context_window_tokens=90_000,
        max_output_tokens=10_000,
        supports_images=True,
        supports_reasoning=True,
    )
    adapter = ModelProfile(
        context_window_tokens=80_000,
        supports_images=False,
        supports_reasoning=False,
    )

    assert (
        resolve_model_profile(
            fingerprint,
            override=override,
            adapter_profile=adapter,
        )
        is override
    )
    assert resolve_model_profile(fingerprint, adapter_profile=adapter) is adapter
    catalog_profile = resolve_model_profile(fingerprint)
    assert catalog_profile.context_window_tokens == 1_050_000
    assert catalog_profile.max_output_tokens == 131_072
    assert catalog_profile.supports_images is True
    assert catalog_profile.supports_reasoning is True


def test_unknown_model_resolves_to_the_fallback_profile() -> None:
    fingerprint = ModelFingerprint(
        provider="openai",
        model="private-model",
        endpoint_fingerprint="endpoint-hash",
    )

    resolved = resolve_model_profile(fingerprint)

    assert resolved == FALLBACK_MODEL_PROFILE
    assert resolved.context_window_tokens == 1_048_576
    assert resolved.max_output_tokens == 262_144
    assert resolved.supports_images is True
    assert resolved.supports_reasoning is True
    assert not hasattr(resolved, "supports_tools")


def test_policy_classifies_overflow_and_caps_required_output_to_physical_remainder() -> None:
    profile = ModelProfile(
        context_window_tokens=1_000,
        max_input_tokens=850,
        max_output_tokens=300,
    )
    policy = ContextPolicy(
        requested_output_reserve_tokens=0,
        observation_reserve_tokens=0,
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
