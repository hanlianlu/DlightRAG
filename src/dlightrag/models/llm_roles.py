# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical role accessors for DlightRAG and LightRAG LLM calls."""

from typing import Literal

from dlightrag.config import DlightragConfig, ModelConfig

RoleName = Literal["extract", "keyword", "query", "vlm"]
LIGHTRAG_ROLE_NAMES: tuple[RoleName, ...] = ("extract", "keyword", "query", "vlm")


def model_for_role(config: DlightragConfig, role: RoleName) -> ModelConfig:
    """Return a complete role configuration, otherwise the complete default."""
    role_cfg = getattr(config.llm.roles, role)
    return (
        role_cfg
        if role_cfg is not None and "api_key" in role_cfg.model_fields_set
        else config.llm.default
    )
