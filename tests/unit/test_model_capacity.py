# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for AI-owned model capacity resolution."""

from dataclasses import FrozenInstanceError

import pytest

from dlightrag.ai.capacity import (
    CONTEXT_POLICY,
    ContextPolicy,
    ModelInputOverflowError,
    ModelProfile,
)
from dlightrag.ai.catalog import (
    UnknownModelProfileError,
    resolve_model_profile,
)
from dlightrag.ai.fingerprints import ModelFingerprint, normalized_endpoint_fingerprint


def test_context_policy_applies_hard_limit_and_compaction_formulas() -> None:
    profile = ModelProfile(
        context_window_tokens=200_000,
        max_input_tokens=180_000,
        max_output_tokens=16_000,
        supports_images=True,
        supports_tools=True,
        supports_reasoning=True,
    )
    policy = ContextPolicy()

    assert policy.revision == "m1-v1"
    assert policy.hard_input_limit(profile) == 170_000
    assert policy.compaction_trigger(profile) == 144_500
    assert policy.history_allowance_cap(profile) == 34_000
    assert policy.retained_tail_target(profile) == 34_000
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
        supports_tools=True,
        supports_reasoning=True,
    )
    adapter = ModelProfile(
        context_window_tokens=80_000,
        supports_images=False,
        supports_tools=True,
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
    assert catalog_profile.supports_tools is True
    assert catalog_profile.supports_reasoning is True


def test_unknown_model_requires_an_explicit_profile_override() -> None:
    fingerprint = ModelFingerprint(
        provider="openai",
        model="private-model",
        endpoint_fingerprint="endpoint-hash",
    )

    with pytest.raises(UnknownModelProfileError, match="capacity override") as caught:
        resolve_model_profile(fingerprint)

    assert caught.value.fingerprint == fingerprint


def test_policy_classifies_overflow_and_caps_required_output_to_physical_remainder() -> None:
    profile = ModelProfile(
        context_window_tokens=1_000,
        max_output_tokens=300,
    )

    assert CONTEXT_POLICY.output_allowance(profile, input_tokens=800) == 200
    with pytest.raises(ModelInputOverflowError) as hard_limit:
        CONTEXT_POLICY.output_allowance(profile, input_tokens=851)
    assert hard_limit.value.kind == "hard_input_limit_exceeded"
    with pytest.raises(ModelInputOverflowError) as exhausted:
        CONTEXT_POLICY.output_allowance(profile, input_tokens=1_000)
    assert exhausted.value.kind == "context_exhausted"

    uncapped = ModelProfile(context_window_tokens=1_000)
    assert CONTEXT_POLICY.output_allowance(uncapped, input_tokens=800) is None
