# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for answer-model image capability derivation."""

import base64
import dataclasses
import io

import pytest
from PIL import Image

from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    ModelConfig,
)
from dlightrag.core.answer.capability import (
    AnswerImageCapability,
    derive_effective_max_images,
)
from dlightrag.core.servicemanager import RAGServiceManager
from dlightrag.core.vision_probe import ImageCapabilityStatus, ImageProbeOutcome
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
    status: ImageCapabilityStatus, configured_ceiling: int, expected: int
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
    config = DlightragConfig(
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
    manager = RAGServiceManager(config=config)
    probed: dict[str, object] = {}

    async def fake_probe(provider, *, model, model_kwargs=None):
        probed["model"] = model
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


def _reprobe_config() -> DlightragConfig:
    return DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )


def _capability(status: ImageCapabilityStatus, effective: int) -> AnswerImageCapability:
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


async def test_reprobe_updates_cached_synthesizer_image_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.core.answer.synthesizer import AnswerSynthesizer

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    contexts = {
        "chunks": [
            {
                "chunk_id": "visual-1",
                "file_path": "chart.png",
                "content": "chart",
                "image_data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        ],
        "entities": [],
        "relationships": [],
    }
    manager = RAGServiceManager(config=_reprobe_config())
    manager._answer_image_capability = _capability("unknown", 0)
    synthesizer = AnswerSynthesizer(image_policy=manager._answer_image_policy())
    manager._answer_synthesizer = synthesizer

    async def fake_discover() -> AnswerImageCapability:
        return _capability("supported", 8)

    monkeypatch.setattr(manager, "_discover_answer_image_capability", fake_discover)

    before = synthesizer._prepare_prompt_context("question", contexts)
    await manager._maybe_reprobe_answer_image_capability()
    after = synthesizer._prepare_prompt_context("question", contexts)

    assert before.trace["answer_context_images_sent"] == 0
    assert after.trace["answer_context_images_sent"] == 1


@pytest.mark.parametrize(
    ("status", "effective"),
    [
        pytest.param("supported", 8, id="supported_is_terminal_no_reprobe"),
        pytest.param("unsupported", 0, id="unsupported_is_terminal_no_reprobe"),
    ],
)
async def test_terminal_status_is_terminal_no_reprobe(
    monkeypatch: pytest.MonkeyPatch,
    status: ImageCapabilityStatus,
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
    # The cooldown lives in the shared probe cache, so a manager re-probe only
    # reaches a model call when that model's own cooldown has elapsed.
    probed = _probed_models(monkeypatch, "unknown")
    manager = RAGServiceManager(config=_reprobe_config())
    manager._answer_image_capability = _capability("unknown", 0)

    await manager._maybe_reprobe_answer_image_capability()
    await manager._maybe_reprobe_answer_image_capability()

    assert len(probed) == 1


# --- Role-specific capability resolved from each role's own model config ------


def _probed_models(monkeypatch: pytest.MonkeyPatch, *statuses: ImageCapabilityStatus) -> list[str]:
    """Record every probed model, replying with *statuses* in order (last repeats)."""
    probed: list[str] = []
    replies = list(statuses) or ["supported"]

    async def fake_probe(_provider, *, model, model_kwargs=None):
        probed.append(model)
        return ImageProbeOutcome(status=replies[min(len(probed) - 1, len(replies) - 1)])

    class _StubProvider:
        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        "dlightrag.models.providers.get_provider", lambda *_a, **_k: _StubProvider()
    )
    monkeypatch.setattr("dlightrag.core.vision_probe.probe_image_capability", fake_probe)
    return probed


async def test_identical_resolved_configurations_share_one_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.core.vision_probe import ModelImageCapabilities

    probed = _probed_models(monkeypatch, "supported")
    capabilities = ModelImageCapabilities()
    first = ModelConfig(model="shared", api_key="k", base_url="https://api.example/v1")
    second = ModelConfig(model="shared", api_key="k", base_url="https://api.example/v1")

    assert (await capabilities.resolve(first)).status == "supported"
    assert (await capabilities.resolve(second)).status == "supported"
    assert probed == ["shared"]


async def test_distinct_resolved_configurations_are_probed_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.core.vision_probe import ModelImageCapabilities

    probed = _probed_models(monkeypatch, "supported", "unsupported")
    capabilities = ModelImageCapabilities()

    answer = await capabilities.resolve(ModelConfig(model="answer-model", api_key="k"))
    vlm = await capabilities.resolve(ModelConfig(model="vlm-model", api_key="k"))

    assert probed == ["answer-model", "vlm-model"]
    assert (answer.status, vlm.status) == ("supported", "unsupported")


async def test_same_endpoint_with_a_different_key_is_not_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.core.vision_probe import ModelImageCapabilities

    probed = _probed_models(monkeypatch, "supported")
    capabilities = ModelImageCapabilities()

    await capabilities.resolve(ModelConfig(model="m", api_key="key-one"))
    await capabilities.resolve(ModelConfig(model="m", api_key="key-two"))

    assert probed == ["m", "m"]


async def test_only_unknown_reprobes_and_only_once_per_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.core.vision_probe import ModelImageCapabilities

    probed = _probed_models(monkeypatch, "unknown")
    capabilities = ModelImageCapabilities(reprobe_cooldown_seconds=3600.0)
    unknown = ModelConfig(model="flaky", api_key="k")
    terminal = ModelConfig(model="steady", api_key="k")

    await capabilities.resolve(unknown)
    await capabilities.resolve(unknown)  # inside the cooldown -> no second probe
    assert probed == ["flaky"]

    monkeypatch.setattr(capabilities, "_cooldown_seconds", 0.0)
    await capabilities.resolve(unknown)
    assert probed == ["flaky", "flaky"]

    probed.clear()
    monkeypatch.setattr(
        "dlightrag.core.vision_probe.probe_image_capability",
        _recording_probe(probed, "supported"),
    )
    await capabilities.resolve(terminal)
    await capabilities.resolve(terminal)
    assert probed == ["steady"]


def _recording_probe(sink: list[str], status: ImageCapabilityStatus):
    async def probe(_provider, *, model, model_kwargs=None):
        sink.append(model)
        return ImageProbeOutcome(status=status)

    return probe


async def test_a_slow_probe_does_not_spend_its_own_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable model must not be re-probed the moment its timeout returns."""
    import asyncio

    from dlightrag.core.vision_probe import ModelImageCapabilities

    probed: list[str] = []

    async def slow_probe(_provider, *, model, model_kwargs=None):
        probed.append(model)
        await asyncio.sleep(0.05)
        return ImageProbeOutcome(status="unknown", failure_kind="TimeoutError")

    class _StubProvider:
        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        "dlightrag.models.providers.get_provider", lambda *_a, **_k: _StubProvider()
    )
    monkeypatch.setattr("dlightrag.core.vision_probe.probe_image_capability", slow_probe)
    capabilities = ModelImageCapabilities(reprobe_cooldown_seconds=0.04)
    cfg = ModelConfig(model="unreachable", api_key="k")

    await capabilities.resolve(cfg)
    await capabilities.resolve(cfg)

    assert probed == ["unreachable"]


async def test_concurrent_resolution_of_one_configuration_probes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from dlightrag.core.vision_probe import ModelImageCapabilities

    probed = _probed_models(monkeypatch, "supported")
    capabilities = ModelImageCapabilities()
    cfg = ModelConfig(model="single-flight", api_key="k")

    await asyncio.gather(*(capabilities.resolve(cfg) for _ in range(4)))

    assert probed == ["single-flight"]


def _role_config(**roles: ModelConfig) -> DlightragConfig:
    return DlightragConfig(
        llm=LLMConfig(
            default=ModelConfig(model="default-model", api_key="default-key"),
            roles=LLMRolesConfig(**roles),
        ),
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            startup_probe=False,
        ),
    )


async def test_inspect_resource_follows_vlm_capability_not_answer_capability() -> None:
    from dlightrag.core.resources import ResourceInput

    manager = RAGServiceManager(config=_role_config())
    manager._answer_image_capability = _capability("unsupported", 0)
    manager._vlm_image_status = "supported"

    _registry, tools = manager._build_resource_context(
        [ResourceInput(filename="chart.png", content=b"\x89PNG", declared_mime="image/png")]
    )

    assert [tool.name for tool in tools] == ["read_resource", "inspect_resource"]


async def test_inspect_resource_is_withheld_when_only_the_answer_model_sees_images() -> None:
    from dlightrag.core.resources import ResourceInput

    manager = RAGServiceManager(config=_role_config())
    manager._answer_image_capability = _capability("supported", 8)
    manager._vlm_image_status = "unsupported"

    _registry, tools = manager._build_resource_context(
        [ResourceInput(filename="chart.png", content=b"\x89PNG", declared_mime="image/png")]
    )

    assert [tool.name for tool in tools] == ["read_resource"]


async def test_query_image_description_follows_vlm_capability() -> None:
    manager = RAGServiceManager(config=_role_config())
    manager._answer_image_capability = _capability("unsupported", 0)
    manager._vlm_image_status = "supported"
    manager._vlm_func = lambda **_kwargs: None

    describer = manager._query_image_describer()

    assert describer._max_images > 0
    assert describer._image_policy.max_images > 0


async def test_query_image_description_is_disabled_without_vlm_image_support() -> None:
    manager = RAGServiceManager(config=_role_config())
    manager._answer_image_capability = _capability("supported", 8)
    manager._vlm_image_status = "unknown"
    manager._vlm_func = lambda **_kwargs: None

    describer = manager._query_image_describer()

    assert describer._max_images == 0
    assert await describer.describe([{"type": "image_url", "image_url": {"url": "data:x"}}]) == []


async def test_zero_configured_ceiling_disables_answer_images_without_a_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed = _probed_models(monkeypatch, "supported")
    config = _role_config()
    config.answer.max_images = 0
    manager = RAGServiceManager(config=config)

    capability = await manager._discover_answer_image_capability()

    assert probed == []
    assert capability.status == "unsupported"
    assert capability.failure_kind == "config_disabled"
    assert capability.effective_max_images == 0


async def test_zero_configured_ceiling_settles_the_vlm_role_without_a_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No role has an image slot under a zero ceiling, so the probe buys nothing."""
    probed = _probed_models(monkeypatch, "supported")
    config = _role_config()
    config.answer.max_images = 0
    manager = RAGServiceManager(config=config)

    await manager._probe_vlm_image_capability()

    assert probed == []
    assert manager._vlm_image_status == "unsupported"


async def test_rerank_capability_is_probed_from_the_rerank_scoring_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.config import RerankConfig

    probed = _probed_models(monkeypatch, "unsupported")
    config = _role_config()
    config.rerank = RerankConfig(
        enabled=True,
        strategy="chat_llm_reranker",
        provider="openai",
        model="rerank-scorer",
        api_key="rerank-key",
    )
    manager = RAGServiceManager(config=config)

    await manager._probe_rerank_image_capability()

    assert probed == ["rerank-scorer"]
    assert manager._rerank_supports_vision is False
