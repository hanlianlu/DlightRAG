# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public contracts and dependency boundary of the durable runtime."""

import ast
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_STATUS_VALUES = ("queued", "running", "succeeded", "failed", "cancelled")
_PHASE_VALUES = ("planning", "searching", "researching", "generating")
_RUNTIME_RECORD_NAMES = frozenset(
    {
        "AnswerRunEventType",
        "AnswerRunPhase",
        "AnswerRunRecord",
        "AnswerRunEvent",
        "AnswerRunStatus",
        "ArtifactAttachOutcome",
        "CancellationOutcome",
        "CheckpointCommit",
        "ClaimedRun",
        "IdempotencyKeyConflict",
        "LeaseRenewal",
        "PendingArtifact",
        "PendingArtifactReference",
        "RunArtifactReference",
        "RunCheckpoint",
        "RunCreation",
        "RunDeletion",
        "ShutdownOutcome",
        "SweepOutcome",
        "TerminalOutcome",
        "answer_run_request_fingerprint",
        "artifact_digest",
        "canonical_run_request_json",
    }
)


def test_runtime_contracts_import_without_storage_or_answer() -> None:
    script = """
import sys
from dlightrag.runtime import (
    AnswerRunEvent,
    AnswerRunPhase,
    AnswerRunRecord,
    AnswerRunStatus,
    AnswerRunStore,
    CheckpointError,
    PendingArtifact,
    RunCoordinator,
    RunCheckpoint,
    RunExecutionError,
    RunExecutor,
    answer_run_request_fingerprint,
    artifact_digest,
)

assert AnswerRunStatus is not None
assert AnswerRunPhase is not None
assert AnswerRunRecord is not None
assert AnswerRunEvent is not None
assert RunCheckpoint is not None
assert PendingArtifact is not None
assert AnswerRunStore is not None
assert CheckpointError is not None
assert RunCoordinator is not None
assert RunExecutionError is not None
assert RunExecutor is not None
assert answer_run_request_fingerprint({"query": "q"})
assert artifact_digest(b"bytes")
for forbidden in (
    "asyncpg",
    "dlightrag.storage",
    "dlightrag.answer",
    "dlightrag.api",
    "dlightrag.mcp",
    "dlightrag.web",
):
    assert not any(name == forbidden or name.startswith(forbidden + ".") for name in sys.modules)
"""

    subprocess.run([sys.executable, "-I", "-c", script], check=True)


def test_run_status_and_phase_literals_have_one_runtime_owner() -> None:
    owners: dict[tuple[str, ...], list[Path]] = {
        _STATUS_VALUES: [],
        _PHASE_VALUES: [],
    }
    for path in (_ROOT / "src/dlightrag").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "Literal"
                and isinstance(node.slice, ast.Tuple)
            ):
                continue
            values = tuple(
                item.value
                for item in node.slice.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
            if values in owners:
                owners[values].append(path)

    expected = [_ROOT / "src/dlightrag/runtime/contracts.py"]
    assert owners[_STATUS_VALUES] == expected
    assert owners[_PHASE_VALUES] == expected


def test_postgres_adapter_does_not_publish_or_supply_runtime_records() -> None:
    adapter_path = _ROOT / "src/dlightrag/adapters/postgres/answer_runs.py"
    adapter_tree = ast.parse(adapter_path.read_text(encoding="utf-8"), filename=str(adapter_path))
    public_names = {
        item.value
        for node in adapter_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        and isinstance(node.value, ast.List)
        for item in node.value.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    assert public_names.isdisjoint(_RUNTIME_RECORD_NAMES)

    stale_imports: list[tuple[Path, str]] = []
    for path in (_ROOT / "src/dlightrag").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module == "dlightrag.adapters.postgres.answer_runs"
            ):
                continue
            stale_imports.extend(
                (path, alias.name) for alias in node.names if alias.name in _RUNTIME_RECORD_NAMES
            )
    assert stale_imports == []
