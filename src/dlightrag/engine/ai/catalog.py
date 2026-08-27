# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Versioned model-capacity catalog and deterministic resolution."""

import json
import logging
from importlib.resources import files
from types import MappingProxyType
from typing import Any, cast

from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.fingerprints import ModelFingerprint, model_endpoint_fingerprint

_logger = logging.getLogger(__name__)


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


def _load_catalog() -> tuple[str, MappingProxyType]:
    payload = cast(
        dict[str, Any],
        json.loads(files("dlightrag.engine.ai").joinpath("model_catalog.json").read_text("utf-8")),
    )
    revision = str(payload.get("revision") or "")
    if not revision:
        raise RuntimeError("model catalog revision is missing")
    catalog: dict[ModelFingerprint, ModelProfile] = {}
    for item in cast(list[dict[str, Any]], payload.get("models") or []):
        facts = cast(dict[str, Any], item["profile"])
        fingerprint = model_endpoint_fingerprint(
            str(item["provider"]),
            str(item["model"]),
            str(item["base_url"]) if item.get("base_url") is not None else None,
        )
        if fingerprint in catalog:
            raise RuntimeError("model catalog contains a duplicate fingerprint")
        catalog[fingerprint] = ModelProfile(
            context_window_tokens=int(facts["context_window_tokens"]),
            max_input_tokens=(
                int(facts["max_input_tokens"])
                if facts.get("max_input_tokens") is not None
                else None
            ),
            max_output_tokens=(
                int(facts["max_output_tokens"])
                if facts.get("max_output_tokens") is not None
                else None
            ),
            supports_images=bool(facts.get("supports_images")),
            supports_reasoning=bool(facts.get("supports_reasoning")),
        )
    return revision, MappingProxyType(catalog)


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
