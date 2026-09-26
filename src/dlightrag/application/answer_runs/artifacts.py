# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Which of an Answer's resources count as its Published Artifacts.

A resource is published only when the Run's stored result lists it among its
artifacts. Input uploads and fetched resources share the resource id space but
are never artifacts, so every transport asks this one rule before serving bytes.
"""

from collections.abc import Mapping
from typing import Any


def artifact_descriptor(
    result: Mapping[str, Any] | None,
    resource_id: str,
) -> Mapping[str, Any] | None:
    """Return the stored result's descriptor for one artifact, whatever its status."""
    for item in (result or {}).get("artifacts") or ():
        if isinstance(item, Mapping) and item.get("resource_id") == resource_id:
            return item
    return None


def published_artifact_descriptor(
    result: Mapping[str, Any] | None,
    resource_id: str,
) -> Mapping[str, Any] | None:
    """Return the descriptor only while the artifact is available to read."""
    descriptor = artifact_descriptor(result, resource_id)
    if descriptor is None or descriptor.get("status") != "available":
        return None
    return descriptor


__all__ = ["artifact_descriptor", "published_artifact_descriptor"]
