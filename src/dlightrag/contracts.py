# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency-free scalar contracts shared across application layers."""

from typing import Literal

type ServiceRole = Literal["writer", "reader"]

__all__ = [
    "ServiceRole",
]
