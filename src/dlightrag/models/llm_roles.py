# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical role accessors for DlightRAG and LightRAG LLM calls."""

from typing import Literal

from dlightrag.config import DlightragConfig, ModelConfig, RerankConfig

RoleName = Literal["extract", "keyword", "query", "vlm"]
LIGHTRAG_ROLE_NAMES: tuple[RoleName, ...] = ("extract", "keyword", "query", "vlm")


def has_complete_api_key_setting(config: ModelConfig | RerankConfig) -> bool:
    """Return whether an override explicitly selects keyed or keyless auth."""
    if "api_key" not in config.model_fields_set:
        return False
    return config.api_key is None or bool(config.api_key.strip())


def model_for_role(config: DlightragConfig, role: RoleName) -> ModelConfig:
    """Return a complete role configuration, otherwise the complete default."""
    role_cfg = getattr(config.llm.roles, role)
    return (
        role_cfg
        if role_cfg is not None and has_complete_api_key_setting(role_cfg)
        else config.llm.default
    )
