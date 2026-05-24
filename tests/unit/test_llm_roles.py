# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LightRAG-aligned LLM role configuration."""

from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    ModelConfig,
)
from dlightrag.models.llm_roles import LIGHTRAG_ROLE_NAMES, model_for_role


def _cfg() -> DlightragConfig:
    return DlightragConfig(
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="sk-test",
            dim=1024,
            startup_probe=False,
        ),
        llm=LLMConfig(
            default=ModelConfig(provider="openai", model="gpt-4.1"),
            roles=LLMRolesConfig(keyword=ModelConfig(provider="openai", model="gpt-4.1-mini")),
        ),
    )


def test_lightrag_role_names_are_canonical() -> None:
    assert LIGHTRAG_ROLE_NAMES == ("extract", "keyword", "query", "vlm")


def test_model_for_role_uses_override_then_default() -> None:
    cfg = _cfg()

    assert model_for_role(cfg, "keyword").model == "gpt-4.1-mini"
    assert model_for_role(cfg, "query").model == "gpt-4.1"
