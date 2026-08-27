# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DlightRAG's in-process Application facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from importlib.metadata import version as _version

    __version__ = _version("dlightrag")
except Exception:
    __version__ = "0.0.0"

if TYPE_CHECKING:
    from .application.application import Application
    from .application.config import DlightragConfig

__all__ = ["Application", "DlightragConfig", "create_application", "__version__"]


def __getattr__(name: str) -> Any:
    if name == "Application":
        from .application.application import Application

        return Application
    if name == "DlightragConfig":
        from .application.config import DlightragConfig

        return DlightragConfig
    raise AttributeError(name)


async def create_application(
    config: DlightragConfig | None = None,
    *,
    web_enabled: bool = False,
) -> Application:
    """Compose, start, and return one Application."""
    from ._compose import create_application as _create_application

    return await _create_application(config, web_enabled=web_enabled)
