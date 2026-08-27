# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lazy Application facade.

Importing a configuration submodule must not compose process adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .application import Application, ApplicationClosedError

__all__ = ["Application", "ApplicationClosedError"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .application import Application, ApplicationClosedError

        return {
            "Application": Application,
            "ApplicationClosedError": ApplicationClosedError,
        }[name]
    raise AttributeError(name)
