# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared ingest execution policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def should_wait_for_ingest(
    *,
    source_type: str,
    path: str | None = None,
    prefix: str | None = None,
    wait: bool | None = None,
) -> bool:
    """Return whether a transport should wait for ingest completion."""
    if wait is not None:
        return wait
    return not is_batch_ingest(source_type=source_type, path=path, prefix=prefix)


def is_batch_ingest(
    *,
    source_type: str,
    path: str | None = None,
    prefix: str | None = None,
) -> bool:
    """Return whether an ingest request is naturally batch-shaped."""
    if prefix is not None:
        return True
    return source_type == "local" and bool(path) and Path(path).is_dir()


def is_ingest_job_row(value: Any) -> bool:
    """Return whether a payload is an ingest job row rather than an ingest result."""
    return isinstance(value, dict) and isinstance(value.get("job_id"), str)


__all__ = ["is_batch_ingest", "is_ingest_job_row", "should_wait_for_ingest"]
