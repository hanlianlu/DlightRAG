# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid YAML + .env configuration loading."""

import os
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from dlightrag.config import DlightragConfig, _find_yaml_config

ROOT = Path(__file__).resolve().parents[2]


def _settings_config(**kwargs: Any) -> DlightragConfig:
    return cast(Any, DlightragConfig)(**kwargs)


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
    @pytest.mark.usefixtures("yaml_config_dir")
    def test_finds_yaml_in_cwd(self):
        result = _find_yaml_config()
        assert result is not None
        assert result.name == "config.yaml"

    def test_returns_none_when_no_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _find_yaml_config() is None


@pytest.mark.usefixtures("yaml_config_dir")
class TestYamlConfigLoading:
    def test_loads_from_yaml(self):
        config = _settings_config()
        assert config.llm.default.model == "gemma4:26b-a4b-it-q8_0"
        assert config.llm.default.base_url == "http://localhost:11434/v1"
        assert config.embedding.dim == 512
        assert config.rerank.enabled is False
        assert config.top_k == 100
        assert config.kg_entity_types == ["Person", "Company"]

    def test_env_overrides_yaml(self, monkeypatch):
        monkeypatch.setenv("DLIGHTRAG_LLM__DEFAULT__MODEL", "gpt-5.4-mini")
        monkeypatch.setenv("DLIGHTRAG_TOP_K", "200")

        config = _settings_config()
        assert config.llm.default.model == "gpt-5.4-mini"  # env override
        assert config.llm.default.base_url == "http://localhost:11434/v1"  # from yaml
        assert config.top_k == 200  # env override

    def test_constructor_overrides_yaml(self):
        config = _settings_config(top_k=300)
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

    def test_repo_config_declares_embedding_input_modality(self) -> None:
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

        assert config["embedding"]["input_modality"] == "auto"

    def test_repo_config_documents_web_query_image_admission_policy(self) -> None:
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

        assert config["query_images"]["max_current_images"] == 3
        assert config["query_images"]["max_upload_bytes"] == 15 * 1024 * 1024

        documentation = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
        assert "max_upload_bytes: 15728640" in documentation


class TestConfigSources:
    def test_works_without_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _settings_config(
            embedding={
                "provider": "openai_compatible",
                "model": "text-embedding-3-large",
                "api_key": "test",
                "startup_probe": False,
            },
            llm={"default": {"model": "gpt-5.4-mini", "api_key": "test"}},
        )
        assert config.llm.default.model == "gpt-5.4-mini"
        assert config.embedding.model == "text-embedding-3-large"

    def test_env_vars_still_work_without_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DLIGHTRAG_LLM__DEFAULT__MODEL", "gpt-5.4-mini")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__PROVIDER", "openai_compatible")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__MODEL", "text-embedding-3-large")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__API_KEY", "test")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__STARTUP_PROBE", "false")

        config = _settings_config()
        assert config.llm.default.model == "gpt-5.4-mini"
