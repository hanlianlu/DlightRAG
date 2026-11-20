# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the read-only Agent Workspace audit command."""

import sys
from pathlib import Path

import pytest


def test_workspace_audit_rejects_a_relative_root() -> None:
    from dlightrag.adapters.postgres.workspace_audit import build_parser, validate_args

    args = build_parser().parse_args(["--root", "relative/path"])

    with pytest.raises(SystemExit, match="--root must be an absolute path"):
        validate_args(args)


def test_workspace_audit_rejects_a_negative_sample() -> None:
    from dlightrag.adapters.postgres.workspace_audit import build_parser, validate_args

    args = build_parser().parse_args(["--sample", "-1"])

    with pytest.raises(SystemExit, match="--sample must be >= 0"):
        validate_args(args)


def test_workspace_audit_main_says_nothing_was_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator's command must be read-only and must say so."""
    from dlightrag.adapters.postgres import workspace_audit
    from dlightrag.engine.answer.agentspace import RunAgentspaceAudit

    async def audit(**_kwargs: object) -> RunAgentspaceAudit:
        return RunAgentspaceAudit(roots=3, orphans=("ab/one", "cd/two"), unreadable=0)

    monkeypatch.setattr(workspace_audit, "audit_run_agentspaces", audit)
    monkeypatch.setattr(workspace_audit, "resolve_agentspace_root", lambda **_kwargs: tmp_path)
    monkeypatch.setattr(workspace_audit, "get_config", lambda: _config())
    monkeypatch.setattr(workspace_audit, "set_config", lambda _config: None)
    monkeypatch.setattr(workspace_audit, "load_config", lambda _path: _config())
    monkeypatch.setattr(sys, "argv", ["dlightrag-workspace-audit"])

    workspace_audit.main()

    printed = capsys.readouterr().out
    assert "3 run root(s)" in printed
    assert "2 without a Run row" in printed
    assert "Nothing was deleted." in printed
    assert "ab/one" in printed


def _config() -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        observability=SimpleNamespace(log_level="INFO"),
        answer=SimpleNamespace(
            agent=SimpleNamespace(execution_environment="disabled", workspace_root=None)
        ),
    )
