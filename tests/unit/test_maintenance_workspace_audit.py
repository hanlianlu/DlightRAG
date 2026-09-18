# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the read-only Agent Workspace audit command."""

import sys
from pathlib import Path

import pytest


def test_workspace_audit_parser_defaults() -> None:
    from dlightrag.adapters.postgres.workspace_audit import build_parser

    args = build_parser().parse_args([])

    assert args.env_file is None
    assert args.root is None
    assert args.sample == 20


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


def test_pyproject_exposes_workspace_audit_console_script() -> None:
    import tomllib

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["dlightrag-workspace-audit"] == (
        "dlightrag.adapters.postgres.workspace_audit:main"
    )


def test_workspace_audit_main_says_nothing_was_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator's command must be read-only and must say so."""
    from dlightrag.adapters.postgres import workspace_audit
    from dlightrag.engine.answer.workspace import RunWorkspaceAudit

    async def audit(**_kwargs: object) -> RunWorkspaceAudit:
        return RunWorkspaceAudit(roots=3, orphans=("ab/one", "cd/two"), unreadable=0)

    monkeypatch.setattr(workspace_audit, "audit_run_workspaces", audit)
    monkeypatch.setattr(workspace_audit, "resolve_workspace_root", lambda **_kwargs: tmp_path)
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
