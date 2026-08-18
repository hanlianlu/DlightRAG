# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DlightRAG's durable product entry points."""

try:
    from importlib.metadata import version as _version

    __version__ = _version("dlightrag")
except Exception:
    __version__ = "0.0.0"
__maintainer__ = "HanlianLyu"
__credits__ = ["hllyu"]

from dlightrag.application import Application
from dlightrag.config import DlightragConfig

__all__ = [
    "Application",
    "DlightragConfig",
    "__version__",
]
