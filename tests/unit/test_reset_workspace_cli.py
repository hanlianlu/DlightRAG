# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the Corpus Workspace reset command."""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dlightrag.application.access import AccessDeniedError, AccessGate, AllowAllAccessControl

_reset_path = Path(__file__).resolve().parents[2] / "scripts" / "reset_workspace.py"
_spec = importlib.util.spec_from_file_location("reset_workspace_cli", _reset_path)
assert _spec is not None and _spec.loader is not None
_reset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reset)


def _allow_all_gate() -> AccessGate:
    return AccessGate(AllowAllAccessControl(), None)


def test_print_workspace_result_uses_current_reset_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _reset._print_workspace_result(
        "demo",
        {
            "lightrag_storages_dropped": 4,
            "domain_stores_dropped": ["metadata_index"],
            "pending_tasks_cancelled": 2,
            "ingest_jobs_cancelled": 1,
            "orphan_tables_cleaned": 3,
            "local_files_removed": 5,
            "errors": [],
        },
    )

    output = capsys.readouterr().out
    assert "LightRAG storages dropped: 4" in output
    assert "DlightRAG stores cleared: metadata_index" in output
    assert "Pending tasks cancelled: 2" in output
    assert "Ingest jobs cancelled: 1" in output
    assert "Orphan tables cleaned: 3" in output
    assert "Local files removed: 5" in output


def test_reset_scope_options_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _reset.build_parser().parse_args(["--all", "--workspace", "demo"])


def test_help_defines_all_as_authorized_corpus_workspaces(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        _reset.build_parser().parse_args(["--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "every authorized Corpus Workspace" in help_text
    assert "development" not in help_text.lower()
    assert "deployment" not in help_text.lower()


async def test_all_expands_catalog_before_reset() -> None:
    corpora = AsyncMock()
    corpora.alist_workspace_records.return_value = [
        {"workspace": "default"},
        {"workspace": "finance"},
    ]
    corpora.reset.return_value = {"workspaces": {}, "total_errors": 0}

    await _reset._reset_with_corpora(
        corpora,
        _allow_all_gate(),
        workspace=None,
        reset_all=True,
        default_workspace="default",
        keep_files=True,
        dry_run=True,
    )

    corpora.reset.assert_awaited_once_with(
        workspace_ids=("default", "finance"),
        keep_files=True,
        dry_run=True,
    )


async def test_explicit_absent_workspace_remains_eligible_for_orphan_cleanup() -> None:
    corpora = AsyncMock()

    await _reset._reset_with_corpora(
        corpora,
        _allow_all_gate(),
        workspace="Archived Reports",
        reset_all=False,
        default_workspace="default",
        keep_files=False,
        dry_run=False,
    )

    corpora.alist_workspace_records.assert_not_awaited()
    corpora.reset.assert_awaited_once_with(
        workspace_ids=("archived_reports",),
        keep_files=False,
        dry_run=False,
    )


@pytest.mark.parametrize("workspace", ["", "   ", "*"])
async def test_explicit_scope_rejects_policy_or_empty_selector(workspace: str) -> None:
    corpora = AsyncMock()

    with pytest.raises(ValueError, match="concrete workspace"):
        await _reset._reset_with_corpora(
            corpora,
            _allow_all_gate(),
            workspace=workspace,
            reset_all=False,
            default_workspace="default",
            keep_files=False,
            dry_run=False,
        )

    corpora.reset.assert_not_awaited()


async def test_explicit_scope_must_pass_the_injected_access_gate() -> None:
    access_control = AsyncMock()
    access_control.check.side_effect = AccessDeniedError("denied")
    corpora = AsyncMock()

    with pytest.raises(AccessDeniedError, match="denied"):
        await _reset._reset_with_corpora(
            corpora,
            AccessGate(access_control, None),
            workspace="finance",
            reset_all=False,
            default_workspace="default",
            keep_files=False,
            dry_run=False,
        )

    corpora.reset.assert_not_awaited()


def test_verbose_enables_debug_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(**kwargs: Any) -> dict[str, Any]:
        return {"workspaces": {}, "total_errors": 0}

    def fake_basic_config(*, level: int) -> None:
        captured["level"] = level

    monkeypatch.setattr(_reset, "_run", fake_run)
    monkeypatch.setattr(_reset.logging, "basicConfig", fake_basic_config)
    monkeypatch.setattr(sys, "argv", ["reset_workspace.py", "--dry-run", "--verbose"])

    assert _reset.main() == 0
    assert captured["level"] == logging.DEBUG
