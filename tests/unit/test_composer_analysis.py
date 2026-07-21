# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for isolated Composer multimodal sidecar analysis."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from lightrag import LightRAG
from lightrag.constants import (
    DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
    DEFAULT_MM_IMAGE_MIN_PIXEL,
)
from lightrag.multimodal_context import DEFAULT_SURROUNDING_MAX_TOKENS

from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    ModelConfig,
    ParserSidecarsConfig,
    VLMSidecarConfig,
)
from dlightrag.core.request.composer_analysis import (
    ComposerAnalysisOutcome,
    aanalyze_composer_sidecars,
    build_composer_analysis_signature,
    create_composer_analysis_proxy,
)
from dlightrag.models.composer import ComposerAnalysisSettings

_ANALYSIS_ENV_KEYS = (
    "VLM_MAX_IMAGE_BYTES",
    "VLM_MIN_IMAGE_PIXEL",
    "SURROUNDING_LEADING_MAX_TOKENS",
    "SURROUNDING_TRAILING_MAX_TOKENS",
    "MAX_EXTRACT_INPUT_TOKENS",
)


class _Tokenizer:
    def __init__(self, model_name: str = "test-tokenizer") -> None:
        self.model_name = model_name

    def encode(self, value: str) -> list[str]:
        return value.split()


def _config(*, enabled: bool = True) -> DlightragConfig:
    return DlightragConfig(
        llm=LLMConfig(
            default=ModelConfig(
                provider="openai",
                model="default-model",
                api_key="sk-default-secret",
            ),
            roles=LLMRolesConfig(
                vlm=ModelConfig(
                    provider="openai",
                    model="vision-model",
                    api_key="sk-vlm-secret",
                    base_url="https://user:pass@vision.example/v1?token=secret",
                    temperature=0.1,
                    model_kwargs={"top_p": 0.9, "api_token": "must-not-leak"},
                ),
                extract=ModelConfig(
                    provider="openai",
                    model="extract-model",
                    api_key="sk-extract-secret",
                    base_url="https://extract.example/v1#private",
                    temperature=0.2,
                    model_kwargs={"reasoning_effort": "none", "secret": "hidden"},
                ),
            ),
        ),
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-embed-secret",
            startup_probe=False,
        ),
        parser_sidecars=ParserSidecarsConfig(vlm=VLMSidecarConfig(enabled=enabled)),
    )


def _bundle(
    *,
    vlm: Any | None = None,
    extract: Any | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        vlm_func=vlm or AsyncMock(return_value="{}"),
        extract_func=extract or AsyncMock(return_value="{}"),
        vlm_identity={
            "provider": "openai",
            "model": "vision-model",
            "base_url": "https://vision.example/v1",
        },
        extract_identity={
            "provider": "openai",
            "model": "extract-model",
            "base_url": "https://extract.example/v1",
        },
    )


def _lightrag() -> SimpleNamespace:
    return SimpleNamespace(tokenizer=_Tokenizer())


def _unset_analysis_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ANALYSIS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _blocks(tmp_path: Path, *, kind: str | None = None, item: dict[str, Any] | None = None) -> Path:
    blocks_path = tmp_path / "document.blocks.jsonl"
    blocks_path.write_text('{"type":"meta"}\n', encoding="utf-8")
    if kind is not None:
        root = {"drawing": "drawings", "table": "tables", "equation": "equations"}[kind]
        suffix = {"drawing": "drawings", "table": "tables", "equation": "equations"}[kind]
        sidecar = blocks_path.with_name(f"document.{suffix}.json")
        sidecar.write_text(
            json.dumps({root: {f"{kind}-1": item or {}}}),
            encoding="utf-8",
        )
    return blocks_path


def _sidecar(blocks_path: Path, kind: str) -> Path:
    suffix = {"drawing": "drawings", "table": "tables", "equation": "equations"}[kind]
    return blocks_path.with_name(f"document.{suffix}.json")


def test_proxy_has_direct_and_global_llm_cache_none() -> None:
    bundle = _bundle()
    proxy = create_composer_analysis_proxy(
        lightrag=_lightrag(),
        model_bundle=bundle,
        config=_config(),
    )

    assert proxy.llm_response_cache is None
    global_config = proxy._build_global_config()
    assert global_config["llm_response_cache"] is None
    assert global_config["enable_llm_cache_for_entity_extract"] is False
    assert global_config["role_llm_funcs"] == {
        "vlm": bundle.vlm_func,
        "extract": bundle.extract_func,
    }


@pytest.mark.parametrize(
    "attribute",
    [
        "full_docs",
        "doc_status",
        "text_chunks",
        "entities_vdb",
        "relationships_vdb",
        "chunks_vdb",
        "chunk_entity_relation_graph",
        "pipeline_status",
    ],
)
def test_proxy_rejects_workspace_storage_attributes(attribute: str) -> None:
    proxy = create_composer_analysis_proxy(
        lightrag=_lightrag(),
        model_bundle=_bundle(),
        config=_config(),
    )

    with pytest.raises(AttributeError):
        getattr(proxy, attribute)


async def test_disabled_preflight_never_calls_lightrag_analyzer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analyze = AsyncMock()
    monkeypatch.setattr(LightRAG, "analyze_multimodal", analyze)
    blocks_path = _blocks(tmp_path)

    result = await aanalyze_composer_sidecars(
        lightrag=_lightrag(),
        model_bundle=_bundle(),
        config=_config(enabled=False),
        doc_id="doc-1",
        file_path="report.pdf",
        parsed_data={"blocks_path": str(blocks_path)},
        process_options="iteP",
    )

    assert result.outcome is ComposerAnalysisOutcome.INTENTIONALLY_DISABLED
    assert result.cacheable is True
    assert result.mm_chunk_count == 0
    analyze.assert_not_awaited()


async def test_explicit_process_options_are_always_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def analyze(proxy: Any, doc_id: str, file_path: str, parsed_data: dict, **kwargs: Any):
        parsed_data["multimodal_processed"] = True
        return parsed_data

    analyze_mock = AsyncMock(side_effect=analyze)
    monkeypatch.setattr(LightRAG, "analyze_multimodal", analyze_mock)
    blocks_path = _blocks(tmp_path)

    result = await aanalyze_composer_sidecars(
        lightrag=_lightrag(),
        model_bundle=_bundle(),
        config=_config(),
        doc_id="doc-1",
        file_path="report.pdf",
        parsed_data={"blocks_path": str(blocks_path)},
        process_options="iteP",
    )

    assert result.outcome is ComposerAnalysisOutcome.SUCCESS
    await_args = analyze_mock.await_args
    assert await_args is not None
    assert await_args.kwargs["process_options"] == "iteP"
    assert await_args.kwargs["pipeline_status"] is not None
    assert await_args.kwargs["pipeline_status_lock"] is not None


async def test_real_unbound_lightrag_analyzer_produces_success(
    tmp_path: Path,
) -> None:
    blocks_path = _blocks(
        tmp_path,
        kind="table",
        item={
            "content": "<table><tr><td>Revenue</td><td>42</td></tr></table>",
            "format": "html",
        },
    )
    extract = AsyncMock(
        return_value=json.dumps(
            {
                "name": "Revenue table",
                "description": "A table showing revenue of 42.",
            }
        )
    )

    result = await aanalyze_composer_sidecars(
        lightrag=_lightrag(),
        model_bundle=_bundle(extract=extract),
        config=_config(),
        doc_id="doc-real",
        file_path="report.pdf",
        parsed_data={"blocks_path": str(blocks_path)},
        process_options="tP",
    )

    assert result.outcome is ComposerAnalysisOutcome.SUCCESS
    assert result.cacheable is True
    assert result.mm_chunk_count == 1
    extract.assert_awaited_once()


@pytest.mark.parametrize("mark_processed", [False, True])
async def test_success_requires_multimodal_processed_and_terminal_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mark_processed: bool,
) -> None:
    blocks_path = _blocks(tmp_path, kind="drawing", item={})

    async def analyze(proxy: Any, doc_id: str, file_path: str, parsed_data: dict, **kwargs: Any):
        if mark_processed:
            parsed_data["multimodal_processed"] = True
        return parsed_data

    monkeypatch.setattr(LightRAG, "analyze_multimodal", analyze)

    result = await aanalyze_composer_sidecars(
        lightrag=_lightrag(),
        model_bundle=_bundle(),
        config=_config(),
        doc_id="doc-1",
        file_path="report.pdf",
        parsed_data={"blocks_path": str(blocks_path)},
        process_options="iP",
    )

    assert result.outcome is ComposerAnalysisOutcome.DEGRADED
    assert result.cacheable is False


async def test_soft_swallowed_upstream_error_becomes_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def analyze(proxy: Any, doc_id: str, file_path: str, parsed_data: dict, **kwargs: Any):
        return parsed_data

    monkeypatch.setattr(LightRAG, "analyze_multimodal", analyze)
    blocks_path = _blocks(tmp_path)

    result = await aanalyze_composer_sidecars(
        lightrag=_lightrag(),
        model_bundle=_bundle(),
        config=_config(),
        doc_id="doc-1",
        file_path="report.pdf",
        parsed_data={"blocks_path": str(blocks_path)},
        process_options="iP",
    )

    assert result.outcome is ComposerAnalysisOutcome.DEGRADED
    assert result.error_type == "MissingSuccessPostcondition"


@pytest.mark.parametrize("failed_status", ["failure", "cancelled"])
async def test_failure_items_are_demoted_to_skipped_before_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_status: str
) -> None:
    blocks_path = _blocks(
        tmp_path,
        kind="drawing",
        item={"llm_analyze_result": {"status": failed_status, "message": "bad model"}},
    )

    async def analyze(proxy: Any, doc_id: str, file_path: str, parsed_data: dict, **kwargs: Any):
        parsed_data["multimodal_processed"] = True
        return parsed_data

    monkeypatch.setattr(LightRAG, "analyze_multimodal", analyze)

    result = await aanalyze_composer_sidecars(
        lightrag=_lightrag(),
        model_bundle=_bundle(),
        config=_config(),
        doc_id="doc-1",
        file_path="report.pdf",
        parsed_data={"blocks_path": str(blocks_path)},
        process_options="iP",
    )

    payload = json.loads(_sidecar(blocks_path, "drawing").read_text(encoding="utf-8"))
    analysis = payload["drawings"]["drawing-1"]["llm_analyze_result"]
    assert analysis["status"] == "skipped"
    assert result.outcome is ComposerAnalysisOutcome.DEGRADED
    assert result.mm_chunk_count == 0
    assert result.cacheable is False


async def test_cancellation_sets_private_flag_waits_cleanup_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()
    captured_status: dict[str, Any] = {}

    async def analyze(
        proxy: Any,
        doc_id: str,
        file_path: str,
        parsed_data: dict,
        *,
        pipeline_status: dict[str, Any],
        pipeline_status_lock: asyncio.Lock,
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured_status.update(pipeline_status)
        captured_status["live"] = pipeline_status
        started.set()
        try:
            while not await proxy._cancellation_requested(pipeline_status, pipeline_status_lock):
                await asyncio.sleep(0)
            parsed_data["multimodal_processed"] = True
            return parsed_data
        finally:
            cleaned.set()

    monkeypatch.setattr(LightRAG, "analyze_multimodal", analyze)
    blocks_path = _blocks(tmp_path)
    task = asyncio.create_task(
        aanalyze_composer_sidecars(
            lightrag=_lightrag(),
            model_bundle=_bundle(),
            config=_config(),
            doc_id="doc-1",
            file_path="report.pdf",
            parsed_data={"blocks_path": str(blocks_path)},
            process_options="iP",
        )
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert captured_status["live"]["cancellation_requested"] is True
    assert cleaned.is_set()


async def test_repeated_cancellation_waits_for_child_cleanup_and_reraises_first_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleaned = asyncio.Event()
    captured_status: dict[str, Any] = {}

    async def analyze(
        proxy: Any,
        doc_id: str,
        file_path: str,
        parsed_data: dict,
        *,
        pipeline_status: dict[str, Any],
        pipeline_status_lock: asyncio.Lock,
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured_status["live"] = pipeline_status
        started.set()
        try:
            while not await proxy._cancellation_requested(pipeline_status, pipeline_status_lock):
                await asyncio.sleep(0)
            cancellation_seen.set()
            await allow_cleanup.wait()
            return parsed_data
        finally:
            cleaned.set()

    monkeypatch.setattr(LightRAG, "analyze_multimodal", analyze)
    blocks_path = _blocks(tmp_path)
    task = asyncio.create_task(
        aanalyze_composer_sidecars(
            lightrag=_lightrag(),
            model_bundle=_bundle(),
            config=_config(),
            doc_id="doc-1",
            file_path="report.pdf",
            parsed_data={"blocks_path": str(blocks_path)},
            process_options="iP",
        )
    )
    await started.wait()

    task.cancel("first cancellation")
    await cancellation_seen.wait()
    assert captured_status["live"]["cancellation_requested"] is True

    for message in ("second cancellation", "third cancellation"):
        task.cancel(message)
        await asyncio.sleep(0)
        assert not task.done()
        assert not cleaned.is_set()

    allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    assert exc_info.value.args == ("first cancellation",)
    assert cleaned.is_set()


@pytest.mark.parametrize(
    ("setting", "replacement"),
    [
        ("max_images", 7),
        ("image_max_bytes", 2_000_000),
        ("image_max_total_bytes", 20_000_000),
        ("image_max_px", 1400),
        ("image_min_px", 900),
        ("image_quality", 88),
        ("image_min_quality", 78),
    ],
)
def test_analysis_signature_changes_with_each_image_transport_setting(
    setting: str,
    replacement: int,
) -> None:
    config = _config()
    changed = config.model_copy(
        update={"answer": config.answer.model_copy(update={setting: replacement})}
    )

    baseline = build_composer_analysis_signature(
        lightrag=_lightrag(),
        model_bundle=_bundle(),
        config=config,
        process_options="iteP",
    )
    updated = build_composer_analysis_signature(
        lightrag=_lightrag(),
        model_bundle=_bundle(),
        config=changed,
        process_options="iteP",
    )

    assert updated != baseline


@pytest.mark.parametrize(
    ("env_name", "env_value", "payload_path", "expected"),
    [
        ("VLM_MAX_IMAGE_BYTES", "4000000", ("sidecar_limits", "max_image_bytes"), 4_000_000),
        ("VLM_MIN_IMAGE_PIXEL", "96", ("sidecar_limits", "min_image_pixel"), 96),
        (
            "SURROUNDING_LEADING_MAX_TOKENS",
            "321",
            ("sidecar_limits", "surrounding_leading_max_tokens"),
            321,
        ),
        (
            "SURROUNDING_TRAILING_MAX_TOKENS",
            "654",
            ("sidecar_limits", "surrounding_trailing_max_tokens"),
            654,
        ),
        ("MAX_EXTRACT_INPUT_TOKENS", "12345", ("max_extract_input_tokens",), 12_345),
    ],
)
def test_analysis_signature_tracks_each_effective_lightrag_env_override(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    payload_path: tuple[str, ...],
    expected: int,
) -> None:
    _unset_analysis_env(monkeypatch)
    baseline_config = _config()
    baseline = build_composer_analysis_signature(
        lightrag=_lightrag(),
        model_bundle=_bundle(),
        config=baseline_config,
        process_options="iteP",
    )

    monkeypatch.setenv(env_name, env_value)
    overridden_config = _config()
    assert os.environ[env_name] == env_value
    updated = build_composer_analysis_signature(
        lightrag=_lightrag(),
        model_bundle=_bundle(),
        config=overridden_config,
        process_options="iteP",
    )

    assert updated != baseline
    recorded: object = json.loads(updated)
    for key in payload_path:
        assert isinstance(recorded, dict)
        recorded = recorded[key]
    assert recorded == expected


def test_effective_analysis_settings_use_lightrag_defaults_when_env_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _unset_analysis_env(monkeypatch)
    config = _config()
    _unset_analysis_env(monkeypatch)

    settings = ComposerAnalysisSettings.resolve(config)
    payload = json.loads(
        build_composer_analysis_signature(
            lightrag=_lightrag(),
            model_bundle=_bundle(),
            config=config,
            process_options="iteP",
        )
    )

    assert settings.vlm_max_image_bytes == 5 * 1024 * 1024
    assert settings.vlm_min_image_pixel == DEFAULT_MM_IMAGE_MIN_PIXEL
    assert settings.surrounding_leading_max_tokens == DEFAULT_SURROUNDING_MAX_TOKENS
    assert settings.surrounding_trailing_max_tokens == DEFAULT_SURROUNDING_MAX_TOKENS
    assert settings.max_extract_input_tokens == DEFAULT_MAX_EXTRACT_INPUT_TOKENS
    assert payload["sidecar_limits"] == {
        "max_image_bytes": 5 * 1024 * 1024,
        "min_image_pixel": DEFAULT_MM_IMAGE_MIN_PIXEL,
        "surrounding_leading_max_tokens": DEFAULT_SURROUNDING_MAX_TOKENS,
        "surrounding_trailing_max_tokens": DEFAULT_SURROUNDING_MAX_TOKENS,
    }
    assert payload["max_extract_input_tokens"] == DEFAULT_MAX_EXTRACT_INPUT_TOKENS


@pytest.mark.parametrize(
    ("env_name", "env_value", "attribute", "payload_path", "expected"),
    [
        (
            "VLM_MAX_IMAGE_BYTES",
            "1",
            "vlm_max_image_bytes",
            ("sidecar_limits", "max_image_bytes"),
            256 * 1024,
        ),
        (
            "VLM_MIN_IMAGE_PIXEL",
            "0",
            "vlm_min_image_pixel",
            ("sidecar_limits", "min_image_pixel"),
            1,
        ),
        (
            "SURROUNDING_LEADING_MAX_TOKENS",
            "-1",
            "surrounding_leading_max_tokens",
            ("sidecar_limits", "surrounding_leading_max_tokens"),
            0,
        ),
        (
            "SURROUNDING_TRAILING_MAX_TOKENS",
            "-2",
            "surrounding_trailing_max_tokens",
            ("sidecar_limits", "surrounding_trailing_max_tokens"),
            0,
        ),
        (
            "MAX_EXTRACT_INPUT_TOKENS",
            "-3",
            "max_extract_input_tokens",
            ("max_extract_input_tokens",),
            -3,
        ),
    ],
)
def test_effective_analysis_settings_apply_only_upstream_clamps(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    attribute: str,
    payload_path: tuple[str, ...],
    expected: int,
) -> None:
    _unset_analysis_env(monkeypatch)
    config = _config()
    monkeypatch.setenv(env_name, env_value)

    settings = ComposerAnalysisSettings.resolve(config)
    recorded: object = json.loads(
        build_composer_analysis_signature(
            lightrag=_lightrag(),
            model_bundle=_bundle(),
            config=config,
            process_options="iteP",
        )
    )

    assert getattr(settings, attribute) == expected
    for key in payload_path:
        assert isinstance(recorded, dict)
        recorded = recorded[key]
    assert recorded == expected


@pytest.mark.parametrize("env_name", ["VLM_MAX_IMAGE_BYTES", "VLM_MIN_IMAGE_PIXEL"])
@pytest.mark.parametrize("env_value", ["", "not-an-int"])
def test_effective_vlm_limits_raise_on_invalid_env_like_lightrag(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
) -> None:
    _unset_analysis_env(monkeypatch)
    config = _config()
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(ValueError):
        ComposerAnalysisSettings.resolve(config)


@pytest.mark.parametrize(
    ("env_name", "attribute", "expected"),
    [
        (
            "SURROUNDING_LEADING_MAX_TOKENS",
            "surrounding_leading_max_tokens",
            DEFAULT_SURROUNDING_MAX_TOKENS,
        ),
        (
            "SURROUNDING_TRAILING_MAX_TOKENS",
            "surrounding_trailing_max_tokens",
            DEFAULT_SURROUNDING_MAX_TOKENS,
        ),
        (
            "MAX_EXTRACT_INPUT_TOKENS",
            "max_extract_input_tokens",
            DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
        ),
    ],
)
@pytest.mark.parametrize("env_value", ["", "not-an-int"])
def test_effective_token_limits_fall_back_on_invalid_env_like_lightrag(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    attribute: str,
    expected: int,
    env_value: str,
) -> None:
    _unset_analysis_env(monkeypatch)
    config = _config()
    monkeypatch.setenv(env_name, env_value)

    settings = ComposerAnalysisSettings.resolve(config)

    assert getattr(settings, attribute) == expected


def test_effective_analysis_settings_preserve_answer_fingerprint_and_tighten_vlm_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _unset_analysis_env(monkeypatch)
    config = _config()
    monkeypatch.setenv("VLM_MAX_IMAGE_BYTES", "300000")
    monkeypatch.setenv("VLM_MIN_IMAGE_PIXEL", "2048")

    settings = ComposerAnalysisSettings.resolve(config)

    assert settings.answer_image_transport.image_max_bytes == config.answer.image_max_bytes
    assert settings.answer_image_transport.image_min_px == config.answer.image_min_px
    assert settings.vlm_image_transport.image_max_bytes == 300_000
    assert settings.vlm_image_transport.image_min_px == 2_048
    assert vars(ComposerAnalysisSettings)["__dataclass_params__"].frozen is True


def test_analysis_signature_changes_with_tokenizer_model() -> None:
    config = _config()
    bundle = _bundle()

    first = build_composer_analysis_signature(
        lightrag=SimpleNamespace(tokenizer=_Tokenizer("tokenizer-model-a")),
        model_bundle=bundle,
        config=config,
        process_options="iteP",
    )
    second = build_composer_analysis_signature(
        lightrag=SimpleNamespace(tokenizer=_Tokenizer("tokenizer-model-b")),
        model_bundle=bundle,
        config=config,
        process_options="iteP",
    )

    assert second != first


def test_analysis_signature_uses_canonical_immutable_image_transport_settings() -> None:
    from dlightrag.models.composer import ComposerImageTransportSettings

    config = _config()
    settings = ComposerImageTransportSettings.from_config(config)
    expected = {
        "max_images": config.answer.max_images,
        "image_max_bytes": config.answer.image_max_bytes,
        "image_max_total_bytes": config.answer.image_max_total_bytes,
        "image_max_px": config.answer.image_max_px,
        "image_min_px": config.answer.image_min_px,
        "image_quality": config.answer.image_quality,
        "image_min_quality": config.answer.image_min_quality,
    }

    assert settings.fingerprint_payload() == expected
    assert vars(ComposerImageTransportSettings)["__dataclass_params__"].frozen is True

    signature = build_composer_analysis_signature(
        lightrag=_lightrag(),
        model_bundle=_bundle(),
        config=config,
        process_options="iteP",
    )
    assert json.loads(signature)["image_transport"] == expected


def test_analysis_signature_is_complete_deterministic_and_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "12345")
    config = _config()
    bundle = _bundle()

    first = build_composer_analysis_signature(
        lightrag=_lightrag(),
        model_bundle=bundle,
        config=config,
        process_options="iteP",
    )
    second = build_composer_analysis_signature(
        lightrag=_lightrag(),
        model_bundle=bundle,
        config=config,
        process_options="iteP",
    )

    assert first == second
    payload = json.loads(first)
    assert payload["contract_version"]
    assert payload["lightrag_version"]
    assert payload["process_options"] == "iteP"
    assert payload["tokenizer"].endswith("._Tokenizer")
    assert payload["tokenizer_model"] == "test-tokenizer"
    assert payload["language"] == config.extraction.language
    assert payload["max_extract_input_tokens"] == 12345
    assert payload["roles"]["vlm"]["temperature"] == 0.1
    assert payload["roles"]["vlm"]["model_options"] == {"top_p": 0.9}
    assert payload["roles"]["extract"]["model_options"] == {"reasoning_effort": "none"}
    assert payload["roles"]["vlm"]["base_url"] == "https://vision.example/v1"
    assert "secret" not in first.lower()
    assert "sk-" not in first.lower()
