# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid YAML + .env configuration loading."""

from __future__ import annotations

import os

import pytest
import yaml

from dlightrag.config import DlightragConfig, _find_yaml_config


@pytest.fixture
def yaml_config_dir(tmp_path, monkeypatch):
    """Create a config.yaml in a temp dir and cd into it."""
    config_data = {
        "chat": {
            "provider": "openai",
            "model": "qwen3:8b",
            "base_url": "http://localhost:11434/v1",
        },
        "embedding": {"model": "text-embedding-3-large", "dim": 512},
        "rerank": {"enabled": False},
        "rag_mode": "unified",
        "top_k": 100,
        "kg_entity_types": ["Person", "Company"],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_dlightrag_env(monkeypatch):
    """Remove any DLIGHTRAG_* env vars that could interfere with tests."""
    for key in list(os.environ):
        if key.startswith("DLIGHTRAG_"):
            monkeypatch.delenv(key, raising=False)


class TestFindYamlConfig:
    def test_finds_yaml_in_cwd(self, yaml_config_dir):
        result = _find_yaml_config()
        assert result is not None
        assert result.name == "config.yaml"

    def test_returns_none_when_no_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _find_yaml_config() is None


class TestYamlConfigLoading:
    def test_loads_from_yaml(self, yaml_config_dir):
        config = DlightragConfig()
        assert config.chat.model == "qwen3:8b"
        assert config.chat.base_url == "http://localhost:11434/v1"
        assert config.embedding.dim == 512
        assert config.rerank.enabled is False
        assert config.rag_mode == "unified"
        assert config.top_k == 100
        assert config.kg_entity_types == ["Person", "Company"]

    def test_env_overrides_yaml(self, yaml_config_dir, monkeypatch):
        monkeypatch.setenv("DLIGHTRAG_CHAT__MODEL", "gpt-4.1")
        monkeypatch.setenv("DLIGHTRAG_TOP_K", "200")

        config = DlightragConfig()
        assert config.chat.model == "gpt-4.1"  # env override
        assert config.chat.base_url == "http://localhost:11434/v1"  # from yaml
        assert config.top_k == 200  # env override

    def test_constructor_overrides_yaml(self, yaml_config_dir):
        config = DlightragConfig(top_k=300)
        assert config.top_k == 300  # constructor override
        assert config.chat.model == "qwen3:8b"  # from yaml


class TestBackwardCompat:
    def test_works_without_yaml(self, tmp_path, monkeypatch):
        """Pure .env users should not be affected."""
        monkeypatch.chdir(tmp_path)
        config = DlightragConfig(
            embedding={"model": "text-embedding-3-large", "api_key": "test"},
            chat={"model": "gpt-4.1-mini", "api_key": "test"},
        )
        assert config.chat.model == "gpt-4.1-mini"
        assert config.embedding.model == "text-embedding-3-large"

    def test_env_vars_still_work_without_yaml(self, tmp_path, monkeypatch):
        """DLIGHTRAG_* env vars should work exactly as before."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DLIGHTRAG_CHAT__MODEL", "gpt-4.1")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__MODEL", "text-embedding-3-large")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__API_KEY", "test")
        monkeypatch.setenv("DLIGHTRAG_RAG_MODE", "unified")

        config = DlightragConfig()
        assert config.chat.model == "gpt-4.1"
        assert config.rag_mode == "unified"
