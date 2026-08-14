# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency-free scalar contracts for model and provider behavior."""

from typing import Literal

type AsymmetricMode = Literal["auto", "require", "disable"]
type ChatProvider = Literal["openai", "anthropic", "gemini"]
type InputModality = Literal["auto", "text", "multimodal"]
type ResolvedInputModality = Literal["text", "multimodal"]

__all__ = ["AsymmetricMode", "ChatProvider", "InputModality", "ResolvedInputModality"]
