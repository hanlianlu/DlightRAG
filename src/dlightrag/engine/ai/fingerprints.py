# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Non-recoverable identities for AI endpoints and model invocations."""

import hashlib
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from dlightrag.engine.ai.contracts import ApiFamily
from dlightrag.engine.ai.settings import EmbeddingSettings, ModelSettings


@dataclass(frozen=True, slots=True)
class ModelEndpointFingerprint:
    """Safe endpoint identity used to resolve provider-independent model facts."""

    provider: str
    model: str
    endpoint_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class ModelInvocationFingerprint:
    """Exact chat-model wire identity used by replay and capability caches."""

    provider: str
    model: str
    endpoint_fingerprint: str | None
    api_family: ApiFamily

    @property
    def endpoint(self) -> ModelEndpointFingerprint:
        """Project the invocation down to catalogue endpoint identity."""
        return ModelEndpointFingerprint(
            provider=self.provider,
            model=self.model,
            endpoint_fingerprint=self.endpoint_fingerprint,
        )

    def as_json(self) -> dict[str, object]:
        """Encode the complete invocation identity for durable state or telemetry."""
        return {
            "provider": self.provider,
            "model": self.model,
            "endpoint_fingerprint": self.endpoint_fingerprint,
            "api_family": self.api_family,
        }

    @classmethod
    def from_json(cls, value: object) -> ModelInvocationFingerprint:
        """Decode a complete invocation identity without legacy family defaults."""
        if not isinstance(value, Mapping):
            raise ValueError("model invocation fingerprint must be an object")
        provider = str(value.get("provider") or "")
        model = str(value.get("model") or "")
        api_family = value.get("api_family")
        if not provider or not model:
            raise ValueError("model invocation fingerprint requires provider and model")
        if api_family not in {"chat_completion", "response"}:
            raise ValueError("model invocation fingerprint requires an explicit API family")
        endpoint = value.get("endpoint_fingerprint")
        return cls(
            provider=provider,
            model=model,
            endpoint_fingerprint=str(endpoint) if endpoint is not None else None,
            api_family=cast(ApiFamily, api_family),
        )


def normalized_endpoint_fingerprint(value: object) -> str | None:
    """Hash a canonical HTTP endpoint without retaining routing data."""
    if not value:
        return None
    try:
        parsed = urlsplit(str(value))
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            return None
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if not hostname:
            return None
        port = parsed.port
        if port == {"http": 80, "https": 443}[scheme]:
            port = None
        authority = f"[{hostname}]" if ":" in hostname else hostname
        if port is not None:
            authority = f"{authority}:{port}"
        path = posixpath.normpath(parsed.path or "/")
        if not path.startswith("/"):
            path = f"/{path}"
        canonical = urlunsplit((scheme, authority, path, "", ""))
    except ValueError:
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_endpoint_fingerprint(
    provider: str,
    model: str,
    base_url: str | None,
) -> ModelEndpointFingerprint:
    """Build one safe identity from canonical endpoint facts."""
    return ModelEndpointFingerprint(
        provider=provider,
        model=model,
        endpoint_fingerprint=normalized_endpoint_fingerprint(base_url),
    )


def model_invocation_fingerprint(settings: ModelSettings) -> ModelInvocationFingerprint:
    """Project resolved chat settings into their exact provider wire identity."""
    endpoint = model_endpoint_fingerprint(settings.provider, settings.model, settings.base_url)
    return ModelInvocationFingerprint(
        provider=endpoint.provider,
        model=endpoint.model,
        endpoint_fingerprint=endpoint.endpoint_fingerprint,
        api_family=settings.api_family,
    )


def embedding_endpoint_fingerprint(settings: EmbeddingSettings) -> ModelEndpointFingerprint:
    """Project embedding settings into their endpoint identity."""
    return model_endpoint_fingerprint(settings.provider, settings.model, settings.base_url)


__all__ = [
    "ModelEndpointFingerprint",
    "ModelInvocationFingerprint",
    "embedding_endpoint_fingerprint",
    "model_endpoint_fingerprint",
    "model_invocation_fingerprint",
    "normalized_endpoint_fingerprint",
]
