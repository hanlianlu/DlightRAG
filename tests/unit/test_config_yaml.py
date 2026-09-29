# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""YAML precedence for the canonical nested configuration."""

import operator
from collections.abc import Iterator
from functools import reduce
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from ruamel.yaml.constructor import DuplicateKeyError

from dlightrag.application.config import DlightragConfig, LaneRuntimeConfig, _find_yaml_config
from dlightrag.application.config import sections as config_sections

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_nested_yaml_loads_and_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text(
        """
deployment:
  workspace: yaml-space
storage:
  postgres:
    host: yaml-db
models:
  embedding:
    dim: 768
corpus:
  retrieval:
    top_k: 41
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    monkeypatch.setenv("DLIGHTRAG_CORPUS__RETRIEVAL__TOP_K", "43")

    config = DlightragConfig()

    assert config.deployment.workspace == "yaml-space"
    assert config.storage.postgres.host == "yaml-db"
    assert config.models.embedding.dim == 768
    assert config.corpus.retrieval.top_k == 43


def test_yaml_1_2_keeps_plain_reasoning_levels_as_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text(
        """
deployment:
  workspace: no
models:
  chat:
    default:
      model: reasoning-model
      reasoning: off
      agentic_reasoning: max
  rerank:
    enabled: false
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)

    config = DlightragConfig()

    assert config.deployment.workspace == "no"
    assert config.models.chat.default.reasoning == "off"
    assert config.models.chat.default.agentic_reasoning == "max"
    assert config.models.rerank.enabled is False


def test_yaml_1_1_directive_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text(
        "%YAML 1.1\n---\ndeployment:\n  workspace: test\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)

    with pytest.raises(ValueError, match="must use YAML 1.2"):
        DlightragConfig()


def test_yaml_1_2_rejects_duplicate_mapping_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text(
        "deployment:\n  workspace: first\n  workspace: second\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)

    with pytest.raises(DuplicateKeyError, match="duplicate key"):
        DlightragConfig()


def test_incomplete_yaml_role_falls_back_but_explicit_null_is_keyless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text(
        """
models:
  chat:
    default:
      model: default-model
    roles:
      query:
        model: incomplete-query
      keyword:
        model: local-keyword
        api_key: null
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)

    roles = DlightragConfig().models.chat

    assert roles.resolve("query").model == "default-model"
    assert roles.resolve("keyword").model == "local-keyword"


def test_constructor_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text(
        "deployment:\n  workspace: yaml-space\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)

    config = DlightragConfig(deployment={"workspace": "constructor-space"})  # type: ignore[arg-type]

    assert config.deployment.workspace == "constructor-space"


def test_nested_runtime_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    monkeypatch.setenv("DLIGHTRAG_RUNTIME__QUERY__WORKER_CONCURRENCY", "7")

    runtime = DlightragConfig().runtime

    assert runtime.query.worker_concurrency == 7
    assert runtime.query.max_nonterminal_runs == 30_000


def test_nested_runtime_yaml_retains_the_other_lane_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text(
        "runtime:\n  corpus_mutation:\n    max_nonterminal_runs: 57\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)

    runtime = DlightragConfig().runtime

    assert runtime.corpus_mutation.worker_concurrency == 2
    assert runtime.corpus_mutation.max_nonterminal_runs == 57


def test_runtime_lanes_share_one_required_config_type_with_lane_defaults() -> None:
    runtime = DlightragConfig(_env_file=None).runtime

    assert type(runtime.query) is LaneRuntimeConfig
    assert type(runtime.corpus_mutation) is LaneRuntimeConfig
    assert set(LaneRuntimeConfig.model_fields) == {
        "worker_concurrency",
        "max_nonterminal_runs",
    }
    assert all(field.is_required() for field in LaneRuntimeConfig.model_fields.values())
    assert runtime.query == LaneRuntimeConfig(
        worker_concurrency=16,
        max_nonterminal_runs=30_000,
    )
    assert runtime.corpus_mutation == LaneRuntimeConfig(
        worker_concurrency=2,
        max_nonterminal_runs=1_000,
    )


def test_flat_runtime_environment_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    monkeypatch.setenv("DLIGHTRAG_ANSWER_WORKER_CONCURRENCY", "7")

    with pytest.raises(ValueError, match="Unknown DlightRAG environment variables"):
        DlightragConfig()


def test_yaml_discovery_and_no_yaml_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _find_yaml_config() is None
    config = DlightragConfig(_env_file=None)
    assert config.deployment.workspace == "default"

    path = tmp_path / "config.yaml"
    path.write_text("deployment:\n  workspace: found\n", encoding="utf-8")
    assert _find_yaml_config() == Path("config.yaml")
    assert DlightragConfig(_env_file=None).deployment.workspace == "found"


def _yaml_leaves(
    node: object, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], object]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _yaml_leaves(value, (*path, key))
    else:
        yield path, node


def test_shipped_config_yaml_validates_and_applies_every_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shipped = _REPO_ROOT / "config.yaml"
    monkeypatch.setattr(config_sections, "_find_yaml_config", lambda: shipped)
    loaded = DlightragConfig().model_dump(mode="json")

    for path, value in _yaml_leaves(YAML(typ="safe").load(shipped.read_text(encoding="utf-8"))):
        if path == ("deployment", "working_dir"):
            value = str(Path(str(value)).resolve())
        assert reduce(operator.getitem, path, loaded) == value, path


def test_unit_runs_ignore_the_checkout_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unit isolation hides this checkout's .env and config.yaml."""
    monkeypatch.chdir(tmp_path)
    isolated = DlightragConfig().model_dump()
    monkeypatch.chdir(_REPO_ROOT)
    assert DlightragConfig().model_dump() == isolated


def test_only_the_named_suite_gates_stay_visible_to_tests() -> None:
    """A gate the settings refuse would break every run; a client name must stay hidden."""
    from tests.conftest import _SUITE_GATE_PREFIXES, _SUITE_GATES, _is_suite_gate

    assert all(config_sections._is_auxiliary_env_name(name) for name in _SUITE_GATES)
    assert all(
        config_sections._is_auxiliary_env_name(f"{prefix}HOST") for prefix in _SUITE_GATE_PREFIXES
    )
    assert _is_suite_gate("dlightrag_run_e2e_pg18")
    assert _is_suite_gate("DLIGHTRAG_E2E_POSTGRES_HOST")
    for hidden in ("DLIGHTRAG_API_TOKEN", "DLIGHTRAG_API_URL", "DLIGHTRAG_DEPLOYMENT__WORKSPACE"):
        assert not _is_suite_gate(hidden)


def test_shipped_config_and_env_example_use_canonical_sections() -> None:
    root = _REPO_ROOT
    config_text = (root / "config.yaml").read_text(encoding="utf-8")
    env_text = (root / ".env.example").read_text(encoding="utf-8")

    assert "models:\n" in config_text
    assert "  embedding:\n" in config_text
    assert "    input_modality: auto\n" in config_text
    assert config_text.count("model: deepseek-flash\n") == 4
    assert "model: deepseek-v4.1-flash\n" not in config_text
    assert config_text.count("reasoning: off\n") == 4
    assert 'reasoning: "off"\n' not in config_text
    assert "DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEY" in env_text
    assert "DLIGHTRAG_ANSWER__WEB_SOURCES__TAVILY__API_KEY" in env_text
    assert "DLIGHTRAG_ANSWER__WEB_SEARCH__API_KEY" not in env_text
    assert "DLIGHTRAG_WEB_SEARCH__API_KEY" not in env_text


def test_old_yaml_root_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text("postgres_host: old-db\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)

    with pytest.raises(Exception, match="Extra inputs"):
        DlightragConfig()
