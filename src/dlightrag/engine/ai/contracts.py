# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Dependency-free scalar contracts for model and provider behavior."""

from typing import Literal

type ApiFamily = Literal["chat_completion", "response"]
type ChatProvider = Literal["openai", "anthropic", "gemini"]
type InputModality = Literal["auto", "text", "multimodal"]
type ResolvedInputModality = Literal["text", "multimodal"]

__all__ = ["ApiFamily", "ChatProvider", "InputModality", "ResolvedInputModality"]
