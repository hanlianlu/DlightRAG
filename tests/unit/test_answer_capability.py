# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for answer-model image capability derivation."""

import pytest

from dlightrag.config import DlightragConfig, EmbeddingConfig
from dlightrag.core.answer_capability import (
    AnswerImageCapability,
    CapabilityStatus,
    derive_effective_max_images,
)
from dlightrag.core.servicemanager import RAGServiceManager
from dlightrag.core.vision_probe import ImageProbeOutcome


def test_config_disabled_forces_zero() -> None:
    assert derive_effective_max_images("unsupported", 0, None) == 0


def test_supported_uses_configured_ceiling() -> None:
    assert derive_effective_max_images("supported", 6, None) == 6


def test_provider_max_caps_below_ceiling() -> None:
    assert derive_effective_max_images("supported", 6, 4) == 4


def test_provider_max_above_ceiling_is_clamped() -> None:
    assert derive_effective_max_images("supported", 6, 10) == 6


def test_unknown_and_unsupported_are_zero() -> None:
    assert derive_effective_max_images("unknown", 6, None) == 0
    assert derive_effective_max_images("unsupported", 6, None) == 0


def test_capability_snapshot_is_frozen() -> None:
    cap = AnswerImageCapability(
        status="supported",
        configured_ceiling=6,
        effective_max_images=6,
        provider="openai",
        base_url=None,
        model="gpt-4o",
        failure_kind=None,
    )
    assert cap.effective_max_images == 6


async def test_capability_probe_targets_query_role(monkeypatch) -> None:
    manager = RAGServiceManager.__new__(RAGServiceManager)
    manager._answer_image_capability = None
    manager._config = DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )
    probed: dict[str, object] = {}

    async def fake_probe(provider, *, model, ceiling, model_kwargs=None):
        probed["model"] = model
        probed["ceiling"] = ceiling
        return ImageProbeOutcome(status="supported")

    class _StubProvider:
        async def aclose(self) -> None:
            pass

    monkeypatch.setattr("dlightrag.models.providers.get_provider", lambda *a, **k: _StubProvider())
    monkeypatch.setattr("dlightrag.core.vision_probe.probe_image_capability", fake_probe)

    await manager._probe_answer_image_capability()

    cap = manager.answer_image_capability
    ceiling = int(manager._config.answer.max_images)
    query_cfg = manager._config.llm.roles.query or manager._config.llm.default
    assert isinstance(cap, AnswerImageCapability)
    assert cap.status == "supported"
    assert cap.effective_max_images == ceiling
    assert probed["model"] == query_cfg.model
    assert probed["ceiling"] == ceiling


def _reprobe_config() -> DlightragConfig:
    return DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )


def _capability(status: CapabilityStatus, effective: int) -> AnswerImageCapability:
    return AnswerImageCapability(
        status=status,
        configured_ceiling=8,
        effective_max_images=effective,
        provider="p",
        base_url=None,
        model="m",
        failure_kind=None,
    )


async def test_unknown_capability_lazily_reprobes_to_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RAGServiceManager(config=_reprobe_config())
    manager._answer_image_capability = _capability("unknown", 0)
    calls = 0

    async def fake_discover() -> AnswerImageCapability:
        nonlocal calls
        calls += 1
        return _capability("supported", 8)

    monkeypatch.setattr(manager, "_discover_answer_image_capability", fake_discover)

    await manager._maybe_reprobe_answer_image_capability()

    cap = manager.answer_image_capability
    assert calls == 1
    assert cap is not None and cap.status == "supported"
    assert cap.effective_max_images == 8


async def test_supported_capability_is_terminal_no_reprobe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RAGServiceManager(config=_reprobe_config())
    manager._answer_image_capability = _capability("supported", 8)
    calls = 0

    async def fake_discover() -> AnswerImageCapability:
        nonlocal calls
        calls += 1
        return _capability("unknown", 0)

    monkeypatch.setattr(manager, "_discover_answer_image_capability", fake_discover)

    await manager._maybe_reprobe_answer_image_capability()

    cap = manager.answer_image_capability
    assert calls == 0
    assert cap is not None and cap.status == "supported"


async def test_unsupported_capability_is_terminal_no_reprobe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RAGServiceManager(config=_reprobe_config())
    manager._answer_image_capability = _capability("unsupported", 0)
    calls = 0

    async def fake_discover() -> AnswerImageCapability:
        nonlocal calls
        calls += 1
        return _capability("supported", 8)

    monkeypatch.setattr(manager, "_discover_answer_image_capability", fake_discover)

    await manager._maybe_reprobe_answer_image_capability()

    cap = manager.answer_image_capability
    assert calls == 0
    assert cap is not None and cap.status == "unsupported"


async def test_reprobe_respects_cooldown_when_still_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RAGServiceManager(config=_reprobe_config())
    manager._answer_image_capability = _capability("unknown", 0)
    calls = 0

    async def fake_discover() -> AnswerImageCapability:
        nonlocal calls
        calls += 1
        return _capability("unknown", 0)

    monkeypatch.setattr(manager, "_discover_answer_image_capability", fake_discover)

    await manager._maybe_reprobe_answer_image_capability()  # re-probe #1
    await manager._maybe_reprobe_answer_image_capability()  # within cooldown -> skip

    assert calls == 1
