# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for answer-model image capability derivation."""

import dataclasses

import pytest

from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    ModelConfig,
)
from dlightrag.core.answer.capability import (
    AnswerImageCapability,
    CapabilityStatus,
    derive_effective_max_images,
)
from dlightrag.core.servicemanager import RAGServiceManager
from dlightrag.core.vision_probe import ImageProbeOutcome
from dlightrag.models.llm_roles import model_for_role


@pytest.mark.parametrize(
    ("status", "configured_ceiling", "expected"),
    [
        pytest.param("unsupported", 0, 0, id="unsupported_zero_ceiling_forces_zero"),
        pytest.param("supported", 6, 6, id="supported_uses_configured_ceiling"),
        pytest.param("unknown", 6, 0, id="unknown_is_zero"),
        pytest.param("unsupported", 6, 0, id="unsupported_is_zero"),
    ],
)
def test_derive_effective_max_images(
    status: CapabilityStatus, configured_ceiling: int, expected: int
) -> None:
    assert derive_effective_max_images(status, configured_ceiling) == expected


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

    with pytest.raises(dataclasses.FrozenInstanceError):
        cap.effective_max_images = 0  # type: ignore[misc]


async def test_capability_probe_targets_resolved_query_role_without_borrowing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RAGServiceManager.__new__(RAGServiceManager)
    manager._answer_image_capability = None
    manager._config = DlightragConfig(
        llm=LLMConfig(
            default=ModelConfig(model="default-model", api_key="default-key"),
            roles=LLMRolesConfig(
                query=ModelConfig(
                    model="local-query",
                    api_key=None,
                    base_url="http://host.docker.internal:8888/v1",
                )
            ),
        ),
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

    def fake_get_provider(*_args, **kwargs):
        probed["api_key"] = kwargs["api_key"]
        return _StubProvider()

    monkeypatch.setattr("dlightrag.models.providers.get_provider", fake_get_provider)
    monkeypatch.setattr("dlightrag.core.vision_probe.probe_image_capability", fake_probe)

    await manager._probe_answer_image_capability()

    cap = manager.answer_image_capability
    ceiling = int(manager._config.answer.max_images)
    query_cfg = model_for_role(manager._config, "query")
    assert isinstance(cap, AnswerImageCapability)
    assert cap.status == "supported"
    assert cap.effective_max_images == ceiling
    assert probed["model"] == query_cfg.model
    assert probed["api_key"] is None
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


@pytest.mark.parametrize(
    ("status", "effective"),
    [
        pytest.param("supported", 8, id="supported_is_terminal_no_reprobe"),
        pytest.param("unsupported", 0, id="unsupported_is_terminal_no_reprobe"),
    ],
)
async def test_terminal_status_is_terminal_no_reprobe(
    monkeypatch: pytest.MonkeyPatch,
    status: CapabilityStatus,
    effective: int,
) -> None:
    manager = RAGServiceManager(config=_reprobe_config())
    manager._answer_image_capability = _capability(status, effective)
    calls = 0

    async def fake_discover() -> AnswerImageCapability:
        nonlocal calls
        calls += 1
        return _capability("unknown" if status == "supported" else "supported", 0)

    monkeypatch.setattr(manager, "_discover_answer_image_capability", fake_discover)

    await manager._maybe_reprobe_answer_image_capability()

    cap = manager.answer_image_capability
    assert calls == 0
    assert cap is not None and cap.status == status


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
