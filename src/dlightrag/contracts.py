# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency-free scalar contracts shared across application layers."""

from typing import Literal

type AsymmetricMode = Literal["auto", "require", "disable"]
type ChatProvider = Literal["openai", "anthropic", "gemini"]
type InputModality = Literal["auto", "text", "multimodal"]
type MetadataPolicy = Literal["validate", "reject_unknown", "store_only"]
type ResolvedInputModality = Literal["text", "multimodal"]
type ServiceRole = Literal["writer", "reader"]
type VisualAssetSize = Literal["full", "thumb"]

__all__ = [
    "AsymmetricMode",
    "ChatProvider",
    "InputModality",
    "MetadataPolicy",
    "ResolvedInputModality",
    "ServiceRole",
    "VisualAssetSize",
]
