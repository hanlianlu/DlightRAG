# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Helpers for app-scoped runtime state."""

from __future__ import annotations

from typing import Any

from dlightrag.config import DlightragConfig, get_config


def request_config(request: Any) -> DlightragConfig:
    """Return the config attached to a FastAPI request/app, falling back to the singleton."""
    state = getattr(getattr(request, "app", None), "state", None)
    if state is None:
        state = getattr(request, "state", None)
    cfg = getattr(state, "config", None)
    if isinstance(cfg, DlightragConfig):
        return cfg
    manager = getattr(state, "manager", None)
    cfg = getattr(manager, "config", None)
    if isinstance(cfg, DlightragConfig):
        return cfg
    return get_config()


__all__ = ["request_config"]
