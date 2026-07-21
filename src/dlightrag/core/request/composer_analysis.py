# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Cache-neutral LightRAG multimodal analysis for temporary Composer sidecars."""

import asyncio
import json
import logging
import math
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from dlightrag.config import DlightragConfig, ModelConfig
from dlightrag.models.composer import ComposerAnalysisSettings, normalized_endpoint_fingerprint
from dlightrag.models.llm_roles import model_for_role

COMPOSER_ANALYSIS_CONTRACT_VERSION = "3"
logger = logging.getLogger(__name__)


def _is_bool(value: Any) -> bool:
    return type(value) is bool


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _is_reasoning_effort(value: Any) -> bool:
    return isinstance(value, str) and value in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    }


def _is_thinking_type(value: Any) -> bool:
    return isinstance(value, str) and value in {"adaptive", "auto", "disabled", "enabled"}


_MODEL_OPTION_SCALAR_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "enable_thinking": _is_bool,
    "max_tokens": _is_int,
    "reasoning_effort": _is_reasoning_effort,
    "seed": _is_int,
    "top_k": _is_int,
    "top_p": _is_finite_number,
}
_MODEL_OPTION_NESTED_VALIDATORS: dict[str, dict[str, Callable[[Any], bool]]] = {
    "reasoning": {
        "effort": _is_reasoning_effort,
        "enabled": _is_bool,
        "exclude": _is_bool,
        "max_tokens": _is_int,
    },
    "thinking": {
        "budget_tokens": _is_int,
        "enabled": _is_bool,
        "include_thoughts": _is_bool,
        "thinking_budget": _is_int,
        "type": _is_thinking_type,
    },
}


class ComposerAnalysisOutcome(StrEnum):
    SUCCESS = "success"
    INTENTIONALLY_DISABLED = "intentionally_disabled"
    DEGRADED = "degraded"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ComposerAnalysisResult:
    outcome: ComposerAnalysisOutcome
    mm_chunk_count: int
    error_type: str | None = None
    mm_chunks: tuple[dict[str, Any], ...] = ()

    @property
    def cacheable(self) -> bool:
        return self.outcome in {
            ComposerAnalysisOutcome.SUCCESS,
            ComposerAnalysisOutcome.INTENTIONALLY_DISABLED,
        }


class ComposerLightRAGProxy:
    """Strict allowlist required by ``LightRAG.analyze_multimodal``."""

    __slots__ = ("_global_config", "llm_response_cache", "role_llm_funcs", "tokenizer")

    def __init__(
        self,
        *,
        tokenizer: Any,
        role_llm_funcs: dict[str, Any],
        global_config: dict[str, Any],
    ) -> None:
        self.tokenizer = tokenizer
        self.role_llm_funcs = role_llm_funcs
        self.llm_response_cache = None
        self._global_config = global_config

    def _build_global_config(self) -> dict[str, Any]:
        return {
            **self._global_config,
            "llm_response_cache": None,
            "enable_llm_cache_for_entity_extract": False,
            "role_llm_funcs": dict(self.role_llm_funcs),
            "llm_cache_identities": {
                key: dict(value)
                for key, value in self._global_config["llm_cache_identities"].items()
            },
        }

    async def _raise_if_cancelled(
        self,
        pipeline_status: dict[str, Any],
        pipeline_status_lock: asyncio.Lock,
    ) -> None:
        from lightrag.exceptions import PipelineCancelledException

        if await self._cancellation_requested(pipeline_status, pipeline_status_lock):
            raise PipelineCancelledException("User cancelled")

    async def _cancellation_requested(
        self,
        pipeline_status: dict[str, Any],
        pipeline_status_lock: asyncio.Lock,
    ) -> bool:
        async with pipeline_status_lock:
            return bool(pipeline_status.get("cancellation_requested", False))


def _safe_role_identity(identity: Any) -> dict[str, str | None]:
    value = identity if isinstance(identity, dict) else {}
    endpoint_fingerprint = value.get("endpoint_fingerprint")
    if not isinstance(endpoint_fingerprint, str):
        endpoint_fingerprint = normalized_endpoint_fingerprint(value.get("base_url"))
    return {
        "provider": str(value.get("provider") or ""),
        "model": str(value.get("model") or ""),
        "endpoint_fingerprint": endpoint_fingerprint,
    }


def create_composer_analysis_proxy(
    *,
    lightrag: Any,
    model_bundle: Any,
    config: DlightragConfig,
) -> ComposerLightRAGProxy:
    """Create a storage-free proxy over borrowed Composer role callables."""
    analysis_settings = ComposerAnalysisSettings.resolve(config)
    role_llm_funcs = {
        "vlm": model_bundle.vlm_func,
        "extract": model_bundle.extract_func,
    }
    identities = {
        "vlm": _safe_role_identity(model_bundle.vlm_identity),
        "extract": _safe_role_identity(model_bundle.extract_identity),
    }
    language = config.extraction.language
    global_config = {
        "addon_params": {"language": language},
        "_resolved_summary_language": language,
        "vlm_process_enable": analysis_settings.enabled,
        "llm_response_cache": None,
        "enable_llm_cache_for_entity_extract": False,
        "role_llm_funcs": role_llm_funcs,
        "llm_cache_identities": identities,
    }
    return ComposerLightRAGProxy(
        tokenizer=lightrag.tokenizer,
        role_llm_funcs=role_llm_funcs,
        global_config=global_config,
    )


def _role_signature(identity: Any, cfg: ModelConfig) -> dict[str, Any]:
    return {
        **_safe_role_identity(identity),
        "temperature": cfg.temperature,
        "model_options": _safe_model_options(cfg.model_kwargs),
    }


def _safe_model_options(options: Any) -> dict[str, Any]:
    if not isinstance(options, dict):
        return {}
    safe: dict[str, Any] = {}
    for key, validator in _MODEL_OPTION_SCALAR_VALIDATORS.items():
        value = options.get(key)
        if validator(value):
            safe[key] = value
    for key, validators in _MODEL_OPTION_NESTED_VALIDATORS.items():
        value = options.get(key)
        if not isinstance(value, dict):
            continue
        nested = {
            nested_key: value[nested_key]
            for nested_key, validator in validators.items()
            if nested_key in value and validator(value[nested_key])
        }
        if nested:
            safe[key] = nested
    return {key: safe[key] for key in sorted(safe)}


def _lightrag_version() -> str:
    try:
        return version("lightrag-hku")
    except PackageNotFoundError:
        import lightrag

        return str(getattr(lightrag, "__version__", "unknown"))


def build_composer_analysis_signature(
    *,
    lightrag: Any,
    model_bundle: Any,
    config: DlightragConfig,
    process_options: str,
) -> str:
    """Return the deterministic, non-secret Composer analysis cache signature."""
    analysis_settings = ComposerAnalysisSettings.resolve(config)
    tokenizer_type = type(lightrag.tokenizer)
    tokenizer_model = getattr(lightrag.tokenizer, "model_name", None)
    payload = {
        "contract_version": COMPOSER_ANALYSIS_CONTRACT_VERSION,
        "lightrag_version": _lightrag_version(),
        "process_options": process_options,
        "tokenizer": f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}",
        "tokenizer_model": str(tokenizer_model) if tokenizer_model is not None else None,
        "image_transport": analysis_settings.answer_image_transport.fingerprint_payload(),
        "language": config.extraction.language,
        "enabled": analysis_settings.enabled,
        "sidecar_limits": {
            "max_image_bytes": analysis_settings.vlm_max_image_bytes,
            "min_image_pixel": analysis_settings.vlm_min_image_pixel,
            "surrounding_leading_max_tokens": (analysis_settings.surrounding_leading_max_tokens),
            "surrounding_trailing_max_tokens": (analysis_settings.surrounding_trailing_max_tokens),
        },
        "max_extract_input_tokens": analysis_settings.max_extract_input_tokens,
        "roles": {
            "vlm": _role_signature(
                model_bundle.vlm_identity,
                model_for_role(config, "vlm"),
            ),
            "extract": _role_signature(
                model_bundle.extract_identity,
                model_for_role(config, "extract"),
            ),
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _enabled_sidecars(blocks_path: str, process_options: str) -> list[tuple[Path, str]]:
    from lightrag.parser.routing import parse_process_options

    opts = parse_process_options(process_options)
    block_file = Path(blocks_path)
    base = str(block_file)
    if base.endswith(".blocks.jsonl"):
        base = base[: -len(".blocks.jsonl")]
    return [
        (Path(base + suffix), root)
        for enabled, suffix, root in (
            (opts.images, ".drawings.json", "drawings"),
            (opts.tables, ".tables.json", "tables"),
            (opts.equations, ".equations.json", "equations"),
        )
        if enabled
    ]


def _terminal_sidecar_state(
    blocks_path: str,
    process_options: str,
) -> tuple[bool, bool]:
    all_terminal = True
    has_failure = False
    for sidecar_path, root in _enabled_sidecars(blocks_path, process_options):
        if not sidecar_path.exists():
            continue
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            return False, True
        items = payload.get(root, {})
        if not isinstance(items, dict):
            return False, True
        for item in items.values():
            if not isinstance(item, dict):
                all_terminal = False
                continue
            result = item.get("llm_analyze_result")
            status = result.get("status") if isinstance(result, dict) else None
            if status in {"failure", "cancelled"}:
                has_failure = True
            if status not in {"success", "skipped"}:
                all_terminal = False
    return all_terminal, has_failure


def _normalize_failed_sidecars(blocks_path: str, process_options: str) -> None:
    for sidecar_path, root in _enabled_sidecars(blocks_path, process_options):
        if not sidecar_path.exists():
            continue
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except OSError, UnicodeError, json.JSONDecodeError:
            logger.warning("Could not normalize Composer sidecar %s", sidecar_path)
            continue
        items = payload.get(root, {})
        if not isinstance(items, dict):
            continue
        changed = False
        for item in items.values():
            if not isinstance(item, dict):
                continue
            result = item.get("llm_analyze_result")
            if not isinstance(result, dict) or result.get("status") not in {
                "failure",
                "cancelled",
            }:
                continue
            item["llm_analyze_result"] = {
                **result,
                "status": "skipped",
                "message": str(result.get("message") or "analysis degraded"),
            }
            changed = True
        if changed:
            try:
                sidecar_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError:
                logger.warning("Could not write normalized Composer sidecar %s", sidecar_path)


async def _request_cancellation(
    pipeline_status: dict[str, Any],
    pipeline_status_lock: asyncio.Lock,
) -> None:
    async with pipeline_status_lock:
        pipeline_status["cancellation_requested"] = True


def _clear_pending_cancellation() -> None:
    task = asyncio.current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()


async def _cancel_and_wait_for_analyzer(
    analyzer_task: asyncio.Task[dict[str, Any]],
    pipeline_status: dict[str, Any],
    pipeline_status_lock: asyncio.Lock,
) -> None:
    await _request_cancellation(pipeline_status, pipeline_status_lock)
    try:
        await analyzer_task
    except Exception, asyncio.CancelledError:
        logger.debug("Composer analyzer child stopped during cancellation", exc_info=True)


async def aanalyze_composer_sidecars(
    *,
    lightrag: Any,
    model_bundle: Any,
    config: DlightragConfig,
    doc_id: str,
    file_path: str,
    parsed_data: dict[str, Any],
    process_options: str,
) -> ComposerAnalysisResult:
    """Run real LightRAG analysis over a strict cache-neutral proxy."""
    from lightrag import LightRAG
    from lightrag.parser.routing import parse_process_options

    opts = parse_process_options(process_options)
    analysis_settings = ComposerAnalysisSettings.resolve(config)
    if not analysis_settings.enabled or not (opts.images or opts.tables or opts.equations):
        return ComposerAnalysisResult(
            outcome=ComposerAnalysisOutcome.INTENTIONALLY_DISABLED,
            mm_chunk_count=0,
        )

    proxy = create_composer_analysis_proxy(
        lightrag=lightrag,
        model_bundle=model_bundle,
        config=config,
    )
    pipeline_status: dict[str, Any] = {
        "cancellation_requested": False,
        "latest_message": "",
        "history_messages": [],
    }
    pipeline_status_lock = asyncio.Lock()
    analyzer = cast(
        Callable[..., Coroutine[Any, Any, dict[str, Any]]],
        LightRAG.analyze_multimodal,
    )
    analyzer_task = asyncio.create_task(
        analyzer(
            proxy,
            doc_id,
            file_path,
            parsed_data,
            process_options=process_options,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
        )
    )

    error_type: str | None = None
    analyzed = parsed_data
    try:
        analyzed = await asyncio.shield(analyzer_task)
    except asyncio.CancelledError as first_cancellation:
        _clear_pending_cancellation()
        cleanup_task = asyncio.create_task(
            _cancel_and_wait_for_analyzer(
                analyzer_task,
                pipeline_status,
                pipeline_status_lock,
            )
        )
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                _clear_pending_cancellation()
        await cleanup_task
        raise first_cancellation
    except Exception as exc:
        error_type = type(exc).__name__

    blocks_path = str(parsed_data.get("blocks_path") or "")
    all_terminal, has_failure = _terminal_sidecar_state(blocks_path, process_options)
    processed = bool(analyzed.get("multimodal_processed"))
    outcome = ComposerAnalysisOutcome.SUCCESS
    if error_type is not None:
        outcome = ComposerAnalysisOutcome.DEGRADED
    elif not processed:
        outcome = ComposerAnalysisOutcome.DEGRADED
        error_type = "MissingSuccessPostcondition"
    elif has_failure:
        outcome = ComposerAnalysisOutcome.DEGRADED
        error_type = "SidecarAnalysisFailure"
    elif not all_terminal:
        outcome = ComposerAnalysisOutcome.DEGRADED
        error_type = "NonTerminalSidecarItem"

    if outcome is ComposerAnalysisOutcome.DEGRADED:
        _normalize_failed_sidecars(blocks_path, process_options)

    mm_chunks: tuple[dict[str, Any], ...] = ()
    if blocks_path:
        try:
            renderer = cast(
                Callable[..., list[dict[str, Any]]], LightRAG._build_mm_chunks_from_sidecars
            )
            mm_chunks = tuple(
                renderer(
                    proxy,
                    doc_id=doc_id,
                    file_path=file_path,
                    blocks_path=blocks_path,
                    base_order_index=0,
                    process_options=process_options,
                )
            )
        except Exception as exc:
            outcome = ComposerAnalysisOutcome.DEGRADED
            error_type = error_type or type(exc).__name__

    return ComposerAnalysisResult(
        outcome=outcome,
        mm_chunk_count=len(mm_chunks),
        error_type=error_type,
        mm_chunks=mm_chunks,
    )


__all__ = [
    "COMPOSER_ANALYSIS_CONTRACT_VERSION",
    "ComposerAnalysisOutcome",
    "ComposerAnalysisResult",
    "ComposerLightRAGProxy",
    "aanalyze_composer_sidecars",
    "build_composer_analysis_signature",
    "create_composer_analysis_proxy",
]
