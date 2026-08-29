# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Versioned model-capacity catalog and deterministic resolution."""

import hashlib
import json
import logging
import re
from importlib.resources import files
from types import MappingProxyType
from typing import Never, cast

from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.fingerprints import ModelFingerprint, normalized_endpoint_fingerprint

_logger = logging.getLogger(__name__)

_ROOT_KEYS = frozenset({"revision", "models"})
_MODEL_KEYS = frozenset({"provider", "model", "base_url", "profile"})
_PROFILE_KEYS = frozenset(
    {
        "context_window_tokens",
        "max_input_tokens",
        "max_output_tokens",
        "supports_images",
        "supports_reasoning",
    }
)
_REVISION_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


class _JSONObject:
    """Object pairs retained until duplicate-member validation has paths."""

    def __init__(self, pairs: list[tuple[str, object]]) -> None:
        self.pairs = pairs


class UnknownModelProfileError(ValueError):
    """Raised when no trusted capacity facts exist for a model endpoint."""

    def __init__(self, fingerprint: ModelFingerprint) -> None:
        self.fingerprint = fingerprint
        endpoint = fingerprint.endpoint_fingerprint or "default"
        super().__init__(
            "No trusted model profile for "
            f"provider={fingerprint.provider!r}, model={fingerprint.model!r}, "
            f"endpoint={endpoint[:12]!r}; configure an explicit per-model capacity override"
        )


def _reject_json_constant(_value: str) -> Never:
    raise ValueError("non-standard JSON constants are not allowed")


def _materialize_json(value: object, path: str) -> object:
    if isinstance(value, _JSONObject):
        materialized: dict[str, object] = {}
        for key, item in value.pairs:
            member_path = f"{path}.{key}"
            if key in materialized:
                raise RuntimeError(f"{member_path} is duplicated")
            materialized[key] = _materialize_json(item, member_path)
        return materialized
    if type(value) is list:
        return [
            _materialize_json(item, f"{path}[{index}]")
            for index, item in enumerate(cast(list[object], value))
        ]
    return value


def _decode_catalog_json(text: str) -> object:
    """Decode strict JSON without exposing source text in failures."""
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_JSONObject,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError, ValueError:
        raise RuntimeError("model catalog is not valid JSON") from None
    return _materialize_json(decoded, "root")


def _require_object(
    value: object,
    *,
    path: str,
    keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError(f"{path} must be an object")
    obj = cast(dict[str, object], value)
    missing = sorted(keys - obj.keys())
    if missing:
        raise RuntimeError(f"{path}.{missing[0]} is missing")
    unknown = sorted(obj.keys() - keys)
    if unknown:
        raise RuntimeError(f"{path}.{unknown[0]} is not allowed")
    return obj


def _canonical_identity(value: object, *, path: str, lowercase: bool) -> str:
    if type(value) is not str:
        raise RuntimeError(f"{path} must be a string")
    identity = cast(str, value)
    if not identity or identity != identity.strip():
        raise RuntimeError(f"{path} must be a non-empty canonical string")
    if lowercase and identity != identity.lower():
        raise RuntimeError(f"{path} must be lowercase")
    return identity


def _profile_integer(
    profile: dict[str, object],
    field: str,
    *,
    path: str,
    optional: bool,
) -> int | None:
    value = profile[field]
    if optional and value is None:
        return None
    if type(value) is not int:
        suffix = " or null" if optional else ""
        raise RuntimeError(f"{path}.{field} must be an integer{suffix}")
    return cast(int, value)


def _profile_boolean(profile: dict[str, object], field: str, *, path: str) -> bool:
    value = profile[field]
    if type(value) is not bool:
        raise RuntimeError(f"{path}.{field} must be a boolean")
    return cast(bool, value)


def _validated_profile(value: object, *, path: str) -> ModelProfile:
    facts = _require_object(value, path=path, keys=_PROFILE_KEYS)
    context_window_tokens = cast(
        int,
        _profile_integer(
            facts,
            "context_window_tokens",
            path=path,
            optional=False,
        ),
    )
    max_input_tokens = _profile_integer(
        facts,
        "max_input_tokens",
        path=path,
        optional=True,
    )
    max_output_tokens = _profile_integer(
        facts,
        "max_output_tokens",
        path=path,
        optional=True,
    )
    supports_images = _profile_boolean(facts, "supports_images", path=path)
    supports_reasoning = _profile_boolean(facts, "supports_reasoning", path=path)
    try:
        return ModelProfile(
            context_window_tokens=context_window_tokens,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            supports_images=supports_images,
            supports_reasoning=supports_reasoning,
        )
    except ValueError as exc:
        field = next(
            (
                candidate
                for candidate in (
                    "context_window_tokens",
                    "max_input_tokens",
                    "max_output_tokens",
                )
                if str(exc).startswith(candidate)
            ),
            "context_window_tokens",
        )
        raise RuntimeError(f"{path}.{field}: {exc}") from None


def _model_catalog_revision(models: list[object]) -> str:
    """Hash canonical parsed model content, excluding top-level revision metadata."""
    canonical = json.dumps(
        models,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _parse_catalog(
    text: str,
) -> tuple[str, MappingProxyType[ModelFingerprint, ModelProfile]]:
    """Parse and validate model-catalog JSON without I/O or fallback behavior."""
    root = _require_object(_decode_catalog_json(text), path="root", keys=_ROOT_KEYS)
    models_value = root["models"]
    if type(models_value) is not list:
        raise RuntimeError("root.models must be an array")
    models = cast(list[object], models_value)

    parsed: dict[ModelFingerprint, ModelProfile] = {}
    fingerprint_indices: dict[ModelFingerprint, int] = {}
    for index, value in enumerate(models):
        path = f"models[{index}]"
        item = _require_object(value, path=path, keys=_MODEL_KEYS)
        provider = _canonical_identity(item["provider"], path=f"{path}.provider", lowercase=True)
        model = _canonical_identity(item["model"], path=f"{path}.model", lowercase=False)

        base_url = item["base_url"]
        if base_url is None:
            endpoint_fingerprint = None
        else:
            if type(base_url) is not str or not base_url:
                raise RuntimeError(f"{path}.base_url must be null or a valid HTTP(S) URL")
            endpoint_fingerprint = normalized_endpoint_fingerprint(base_url)
            if endpoint_fingerprint is None:
                raise RuntimeError(f"{path}.base_url must be null or a valid HTTP(S) URL")

        profile = _validated_profile(item["profile"], path=f"{path}.profile")
        fingerprint = ModelFingerprint(
            provider=provider,
            model=model,
            endpoint_fingerprint=endpoint_fingerprint,
        )
        if fingerprint in parsed:
            first_index = fingerprint_indices[fingerprint]
            raise RuntimeError(
                f"models[{index}] duplicates the normalized fingerprint from models[{first_index}]"
            )
        parsed[fingerprint] = profile
        fingerprint_indices[fingerprint] = index

    revision_value = root["revision"]
    if type(revision_value) is not str or _REVISION_PATTERN.fullmatch(revision_value) is None:
        raise RuntimeError("root.revision has invalid form; expected sha256:<64 lowercase hex>")
    revision = cast(str, revision_value)
    expected_revision = _model_catalog_revision(models)
    if revision != expected_revision:
        raise RuntimeError("root.revision does not match canonical models content")
    return revision, MappingProxyType(parsed)


def _load_catalog() -> tuple[str, MappingProxyType[ModelFingerprint, ModelProfile]]:
    text = files("dlightrag.engine.ai").joinpath("model_catalog.json").read_text("utf-8")
    return _parse_catalog(text)


MODEL_CATALOG_REVISION, _MODEL_CATALOG = _load_catalog()

#: The unconditional capacity guess for endpoints the catalog does not know.
#: It is deliberately generous: the first real provider rejection is the
#: calibration signal, never a wasted probe call.
FALLBACK_MODEL_PROFILE = ModelProfile(
    context_window_tokens=1_048_576,
    max_input_tokens=None,
    max_output_tokens=262_144,
    supports_images=True,
    supports_reasoning=True,
)


def resolve_model_profile(
    fingerprint: ModelFingerprint,
    *,
    override: ModelProfile | None = None,
    adapter_profile: ModelProfile | None = None,
) -> ModelProfile:
    """Resolve override, trusted adapter facts, then the versioned catalog.

    Unknown endpoints resolve to the shared fallback profile instead of
    failing: uncatalogued models work out of the box, and any mistake in
    capacity facts surfaces as an ordinary provider rejection at first use.
    """
    if override is not None:
        return override
    if adapter_profile is not None:
        return adapter_profile
    profile = _MODEL_CATALOG.get(fingerprint)
    if profile is not None:
        return profile
    _logger.warning(
        "uncatalogued model resolved to the fallback profile: provider=%r model=%r endpoint=%r",
        fingerprint.provider,
        fingerprint.model,
        fingerprint.endpoint_fingerprint,
    )
    return FALLBACK_MODEL_PROFILE


__all__ = [
    "FALLBACK_MODEL_PROFILE",
    "MODEL_CATALOG_REVISION",
    "UnknownModelProfileError",
    "resolve_model_profile",
]
