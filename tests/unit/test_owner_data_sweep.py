# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The package sweep: what it removes, what it must never remove, and what it says."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dlightrag.adapters.postgres import owner_data
from dlightrag.adapters.postgres.owner_data import record_sweep_receipts
from dlightrag.adapters.postgres.owner_data_sweep import SweptPackage, sweep_expired_exports

NOW = datetime(2026, 5, 1, tzinfo=UTC)


def _package(root: Path, name: str, *, expires_at: str, operation: str = "export") -> Path:
    package = root / name
    package.mkdir(parents=True)
    (package / "manifest.json").write_text(
        json.dumps(
            {"operation": operation, "expires_at": expires_at, "owner_id": "o", "request_id": name}
        )
    )
    return package


def test_a_package_past_its_deadline_goes_and_a_fresh_one_stays(tmp_path: Path) -> None:
    stale = _package(tmp_path, "stale", expires_at="2026-04-30T00:00:00+00:00")
    fresh = _package(tmp_path, "fresh", expires_at="2026-05-02T00:00:00+00:00")
    swept = sweep_expired_exports(tmp_path, now=NOW)
    assert [item.request_id for item in swept] == ["stale"]
    assert swept[0].removed and not stale.exists() and fresh.is_dir()


def test_directories_this_command_did_not_write_are_never_touched(tmp_path: Path) -> None:
    """A delivery root may be shared; a sweep is not a licence to delete a stranger's data."""
    stranger = tmp_path / "operator-notes"
    stranger.mkdir()
    (stranger / "important.txt").write_text("do not delete me")
    other_operation = _package(
        tmp_path, "erase-package", expires_at="2020-01-01T00:00:00+00:00", operation="erase"
    )
    no_manifest = tmp_path / "half-written"
    no_manifest.mkdir()
    assert sweep_expired_exports(tmp_path, now=NOW) == []
    assert (
        (stranger / "important.txt").is_file() and other_operation.is_dir() and no_manifest.is_dir()
    )


def test_a_malformed_deadline_is_reported_and_a_zoneless_one_is_read_as_utc(
    tmp_path: Path, caplog
) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{ not json")
    _package(tmp_path, "naive", expires_at="2026-04-30T00:00:00")  # no zone: still expired
    with caplog.at_level(logging.WARNING):
        swept = sweep_expired_exports(tmp_path, now=NOW)
    assert [item.request_id for item in swept] == ["naive"] and broken.is_dir()
    assert any("states no readable deadline" in record.message for record in caplog.records)


def test_a_dry_run_names_what_would_go_and_removes_nothing(tmp_path: Path) -> None:
    stale = _package(tmp_path, "stale", expires_at="2026-04-01T00:00:00+00:00")
    swept = sweep_expired_exports(tmp_path, now=NOW, dry_run=True)
    assert [item.request_id for item in swept] == ["stale"]
    assert swept[0].removed is False and stale.is_dir()


class FakeLedger:
    """The request ledger as the sweep uses it: rows to add."""

    def __init__(self) -> None:
        self.rows: list[tuple[Any, ...]] = []

    async def execute(self, query: str, *args: Any) -> str:
        if query.strip().startswith("INSERT INTO dlightrag_owner_data_requests"):
            self.rows.append(args)
        return "INSERT 0 1"


def _swept(tmp_path: Path, name: str, *, removed: bool) -> SweptPackage:
    return SweptPackage(
        path=tmp_path / name,
        request_id=name,
        owner_id="owner-1",
        expires_at=NOW,
        removed=removed,
    )


def _stub_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict[str, Any]:
    """Run `main()` the way the operator's timer does, with the database kept out of it."""
    seen: dict[str, Any] = {"calls": []}

    async def ledger(swept: list[SweptPackage], *, executor: str, config: Any) -> list[str]:
        seen["calls"].append((list(swept), executor))
        return [f"receipt-{index}" for index in range(len(swept))]

    monkeypatch.setattr(owner_data, "_run_sweep_ledger", ledger)
    monkeypatch.setattr(
        owner_data,
        "get_config",
        lambda: SimpleNamespace(observability=SimpleNamespace(log_level="info")),
    )
    monkeypatch.setattr(owner_data, "set_config", lambda config: None)
    monkeypatch.setattr(sys, "argv", ["dlightrag-owner-data", *argv])
    return seen


def test_a_sweep_needs_no_owner_and_receipts_every_package_it_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep is owner-agnostic, and dispatch is where that can regress: the timer
    passes no --owner, so resolving one before the branch would refuse the run."""
    moment = datetime.now(UTC)  # the CLI sweeps on the real clock, not on a passed one
    expired = _package(tmp_path, "stale", expires_at=(moment - timedelta(days=1)).isoformat())
    fresh = _package(tmp_path, "fresh", expires_at=(moment + timedelta(days=1)).isoformat())
    seen = _stub_cli(monkeypatch, ["sweep-expired", "--out", str(tmp_path)])

    owner_data.main()  # no --owner, and no SystemExit

    assert not expired.exists() and fresh.is_dir()
    ((swept, executor),) = seen["calls"]
    assert [item.request_id for item in swept] == ["stale"] and executor == "operator"


def test_a_dry_run_removes_nothing_and_writes_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A removal that was not performed was not promised, so the ledger stays silent."""
    stale = _package(tmp_path, "stale", expires_at="2026-04-01T00:00:00+00:00")
    seen = _stub_cli(monkeypatch, ["sweep-expired", "--out", str(tmp_path), "--dry-run"])

    owner_data.main()

    assert stale.is_dir() and seen["calls"] == []


def test_a_package_that_outlives_its_expiry_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A removal that quietly failed is exactly the cleartext copy this bounds."""
    stale = _package(tmp_path, "stale", expires_at="2026-04-01T00:00:00+00:00")
    seen = _stub_cli(monkeypatch, ["sweep-expired", "--out", str(tmp_path)])
    monkeypatch.setattr(
        "dlightrag.adapters.postgres.owner_data_sweep.shutil.rmtree",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(SystemExit) as failure:
        owner_data.main()

    assert failure.value.code not in (0, None)
    assert "outlived their expiry" in str(failure.value.code)
    assert stale.is_dir()
    # What it hands the ledger says the removal failed rather than claiming success.
    ((swept, _executor),) = seen["calls"]
    assert [item.removed for item in swept] == [False]


def test_a_sweep_whose_ledger_cannot_be_written_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt is part of the job: a removal nobody can evidence is not done."""
    _package(tmp_path, "stale", expires_at="2026-04-01T00:00:00+00:00")

    async def broken(*args: Any, **kwargs: Any) -> list[str]:
        raise OSError("the ledger is unreachable")

    monkeypatch.setattr(
        owner_data,
        "get_config",
        lambda: SimpleNamespace(observability=SimpleNamespace(log_level="info")),
    )
    monkeypatch.setattr(owner_data, "set_config", lambda config: None)
    monkeypatch.setattr(owner_data, "_run_sweep_ledger", broken)
    monkeypatch.setattr(
        sys, "argv", ["dlightrag-owner-data", "sweep-expired", "--out", str(tmp_path)]
    )

    with pytest.raises(SystemExit) as failure:
        owner_data.main()
    assert "ledger could not be written" in str(failure.value.code)


def test_each_package_the_sweep_judged_gets_one_ledger_row(tmp_path: Path) -> None:
    """A package that went and one that would not are both facts about a person's data."""
    ledger = FakeLedger()
    ids = asyncio.run(
        record_sweep_receipts(
            ledger,
            executor="timer",
            swept=[
                _swept(tmp_path, "gone", removed=True),
                _swept(tmp_path, "stuck", removed=False),
            ],
        )
    )
    assert len(ids) == 2 and len(set(ids)) == 2
    gone, stuck = ledger.rows
    assert (gone[2], gone[3], gone[6]) == ("sweep", "timer", "completed")
    assert (stuck[2], stuck[6]) == ("sweep", "failed")
    assert gone[1] == "owner-1"  # the owner comes from the package's own manifest
    assert json.loads(gone[7]) == {
        "package_request_id": "gone",
        "expired_at": NOW.isoformat(),
        "removed": True,
    }
    assert json.loads(gone[8]) == []
    assert json.loads(stuck[8]) == [str(tmp_path / "stuck")]


def test_the_help_states_what_the_verb_does() -> None:
    from dlightrag.adapters.postgres.owner_data import build_parser

    help_text = build_parser().format_help()
    assert "sweep-expired" in help_text


def test_a_root_that_is_not_there_sweeps_nothing_rather_than_failing(tmp_path: Path) -> None:
    assert sweep_expired_exports(tmp_path / "missing", now=NOW) == []
