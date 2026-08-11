# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit-test fixtures — isolate from the operator's .env and config.yaml.

Both are deployment inputs, not product contracts: a unit test that reads them
asserts whatever this checkout happens to be tuned to, so retuning config.yaml
breaks CI. Tests that mean to exercise a YAML config build their own file.
"""

import os
from pathlib import Path

import pytest

from dlightrag import config as config_module
from dlightrag.config import DlightragConfig

_REPO_CONFIG_YAML = Path(__file__).resolve().parents[2] / "config.yaml"
# Bound before the fixture patches the name, otherwise the wrapper recurses.
_FIND_YAML_CONFIG = config_module._find_yaml_config


def _yaml_config_ignoring_repo_file() -> Path | None:
    """Resolve config.yaml as production does, minus this checkout's own file."""
    found = _FIND_YAML_CONFIG()
    if found is not None and found.resolve() == _REPO_CONFIG_YAML:
        return None
    return found


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent .env and the repo's config.yaml from polluting unit tests."""
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    monkeypatch.setattr(config_module, "_find_yaml_config", _yaml_config_ignoring_repo_file)
    for key in list(os.environ):
        if key.startswith("DLIGHTRAG_"):
            monkeypatch.delenv(key, raising=False)
