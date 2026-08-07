# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the destructive reset CLI wrapper."""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

_reset_path = Path(__file__).resolve().parents[2] / "scripts" / "reset.py"
_spec = importlib.util.spec_from_file_location("reset_cli", _reset_path)
assert _spec is not None and _spec.loader is not None
_reset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reset)


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
    parser = _reset.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--all", "--workspace", "demo"])


def test_verbose_enables_debug_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(**kwargs: Any) -> dict[str, Any]:
        return {"workspaces": {}, "total_errors": 0}

    def fake_basic_config(*, level: int) -> None:
        captured["level"] = level

    monkeypatch.setattr(_reset, "_run", fake_run)
    monkeypatch.setattr(_reset.logging, "basicConfig", fake_basic_config)
    monkeypatch.setattr(sys, "argv", ["reset.py", "--dry-run", "--verbose"])

    assert _reset.main() == 0
    assert captured["level"] == logging.DEBUG
