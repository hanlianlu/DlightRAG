# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical role accessors for DlightRAG and LightRAG LLM calls."""

from __future__ import annotations

from typing import Literal

from dlightrag.config import DlightragConfig, ModelConfig

RoleName = Literal["extract", "keyword", "query", "vlm"]
LIGHTRAG_ROLE_NAMES: tuple[RoleName, ...] = ("extract", "keyword", "query", "vlm")


def model_for_role(config: DlightragConfig, role: RoleName) -> ModelConfig:
    """Return the configured role model or the default LLM model."""
    role_cfg = getattr(config.llm.roles, role)
    return role_cfg or config.llm.default
