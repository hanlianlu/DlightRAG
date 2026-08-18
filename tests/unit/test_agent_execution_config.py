# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup validation for optional trusted local execution."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dlightrag.application import validate_agent_execution
from dlightrag.config import AgentExecutionConfig, DlightragConfig


def test_disabled_ignores_workspace_root(tmp_path: Path) -> None:
    config = DlightragConfig(
        working_dir=str(tmp_path / "corpus"),
        agent=AgentExecutionConfig(
            execution_environment="disabled", workspace_root=str(tmp_path / "ws")
        ),
    )
    assert validate_agent_execution(config) is None


def test_local_trusted_requires_absolute_root(tmp_path: Path) -> None:
    config = DlightragConfig(
        working_dir=str(tmp_path / "corpus"),
        agent=AgentExecutionConfig(execution_environment="local_trusted", workspace_root=None),
    )
    with pytest.raises(ValueError, match="absolute path"):
        validate_agent_execution(config)


def test_workspace_root_must_not_overlap_working_dir(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    config = DlightragConfig(
        working_dir=str(corpus),
        agent=AgentExecutionConfig(
            execution_environment="local_trusted", workspace_root=str(corpus / "nested")
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_agent_execution(config)


def test_unknown_agent_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DlightragConfig.model_validate(
            {"agent": {"execution_environment": "disabled", "sandbox": True}}
        )
