# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid YAML + .env configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from dlightrag.config import DlightragConfig, _find_yaml_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def yaml_config_dir(tmp_path, monkeypatch):
    """Create a config.yaml in a temp dir and cd into it."""
    config_data = {
        "llm": {
            "default": {
                "provider": "openai",
                "model": "gemma4:26b-a4b-it-q8_0",
                "base_url": "http://localhost:11434/v1",
            }
        },
        "embedding": {
            "provider": "openai_compatible",
            "model": "text-embedding-3-large",
            "dim": 512,
            "startup_probe": False,
        },
        "rerank": {"enabled": False},
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
        assert config.llm.default.model == "gemma4:26b-a4b-it-q8_0"
        assert config.llm.default.base_url == "http://localhost:11434/v1"
        assert config.embedding.dim == 512
        assert config.rerank.enabled is False
        assert config.top_k == 100
        assert config.kg_entity_types == ["Person", "Company"]

    def test_env_overrides_yaml(self, yaml_config_dir, monkeypatch):
        monkeypatch.setenv("DLIGHTRAG_LLM__DEFAULT__MODEL", "gpt-4.1")
        monkeypatch.setenv("DLIGHTRAG_TOP_K", "200")

        config = DlightragConfig()
        assert config.llm.default.model == "gpt-4.1"  # env override
        assert config.llm.default.base_url == "http://localhost:11434/v1"  # from yaml
        assert config.top_k == 200  # env override

    def test_constructor_overrides_yaml(self, yaml_config_dir):
        config = DlightragConfig(top_k=300)
        assert config.top_k == 300  # constructor override
        assert config.llm.default.model == "gemma4:26b-a4b-it-q8_0"  # from yaml

    def test_repo_config_metadata_schema_uses_current_field_keys(self):
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        metadata = config["metadata"]

        assert set(metadata) == {"allow_ad_hoc_json", "default_ingest_policy", "fields"}
        for field_spec in metadata["fields"].values():
            assert set(field_spec) <= {"type", "normalizer", "filter_ops"}
            assert "filter" not in field_spec
            assert "normalize" not in field_spec
            assert "indexed" not in field_spec

    def test_repo_config_matches_concurrency_defaults(self):
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

        assert config["max_async"] == 8
        assert config["embedding_func_max_async"] == 16
        assert config["max_parallel_insert"] == 3
        assert config["postgres_lightrag_pool_max_size"] == 16
        assert config["postgres_pool_min_size"] == 2
        assert config["postgres_pool_max_size"] == 10
        assert (
            DlightragConfig.model_fields["max_parallel_insert"].default
            == config["max_parallel_insert"]
        )
        assert DlightragConfig.model_fields["max_async"].default == config["max_async"]
        assert (
            DlightragConfig.model_fields["embedding_func_max_async"].default
            == config["embedding_func_max_async"]
        )
        assert (
            DlightragConfig.model_fields["postgres_lightrag_pool_max_size"].default
            == config["postgres_lightrag_pool_max_size"]
        )
        assert (
            DlightragConfig.model_fields["postgres_pool_min_size"].default
            == config["postgres_pool_min_size"]
        )
        assert (
            DlightragConfig.model_fields["postgres_pool_max_size"].default
            == config["postgres_pool_max_size"]
        )


class TestConfigSources:
    def test_works_without_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = DlightragConfig(
            embedding={
                "provider": "openai_compatible",
                "model": "text-embedding-3-large",
                "api_key": "test",
                "startup_probe": False,
            },
            llm={"default": {"model": "gpt-4.1-mini", "api_key": "test"}},
        )
        assert config.llm.default.model == "gpt-4.1-mini"
        assert config.embedding.model == "text-embedding-3-large"

    def test_env_vars_still_work_without_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DLIGHTRAG_LLM__DEFAULT__MODEL", "gpt-4.1")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__PROVIDER", "openai_compatible")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__MODEL", "text-embedding-3-large")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__API_KEY", "test")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__STARTUP_PROBE", "false")

        config = DlightragConfig()
        assert config.llm.default.model == "gpt-4.1"
