# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Non-recoverable identities for resolved AI model endpoints."""

import hashlib
import posixpath
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from dlightrag.ai.settings import EmbeddingSettings, ModelSettings


@dataclass(frozen=True, slots=True)
class ModelFingerprint:
    """Safe model identity suitable for logs, traces, and persisted facts."""

    provider: str
    model: str
    endpoint_fingerprint: str | None


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


def model_fingerprint(settings: ModelSettings | EmbeddingSettings) -> ModelFingerprint:
    """Project resolved settings into a safe provider/model identity."""
    return ModelFingerprint(
        provider=settings.provider,
        model=settings.model,
        endpoint_fingerprint=normalized_endpoint_fingerprint(settings.base_url),
    )


__all__ = ["ModelFingerprint", "model_fingerprint", "normalized_endpoint_fingerprint"]
