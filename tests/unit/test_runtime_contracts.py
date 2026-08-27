# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public contracts and dependency boundary of the durable runtime."""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

from dlightrag.engine.runtime.contracts import ANSWER_RUN_PHASES

_ROOT = Path(__file__).resolve().parents[2]
_STATUS_VALUES = ("queued", "running", "succeeded", "failed", "cancelled")
_PHASE_VALUES = ANSWER_RUN_PHASES
_RUNTIME_RECORD_NAMES = frozenset(
    {
        "AnswerRunEventType",
        "AnswerRunPhase",
        "AnswerRunRecord",
        "AnswerRunEvent",
        "AnswerRunStatus",
        "CancellationOutcome",
        "ClaimedRun",
        "IdempotencyKeyConflict",
        "LeaseRenewal",
        "PendingArtifact",
        "PendingArtifactReference",
        "RunArtifactReference",
        "RunCreation",
        "RunDeletion",
        "RunExecutionContext",
        "ShutdownOutcome",
        "SweepOutcome",
        "TerminalOutcome",
        "answer_run_request_fingerprint",
        "artifact_digest",
        "canonical_run_request_json",
    }
)


def test_sdk_and_runtime_import_without_composition_or_transports() -> None:
    script = """
import sys
import dlightrag.sdk.client
import dlightrag.engine.runtime

forbidden = (
    "asyncpg",
    "fastapi",
    "lightrag",
    "PIL",
    "dlightrag.adapters.postgres",
    "dlightrag.engine.answer",
    "dlightrag.api",
    "dlightrag.mcp",
    "dlightrag.web",
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
if loaded:
    raise SystemExit("unexpected eager imports: " + ", ".join(loaded))
"""
    env = os.environ.copy()
    source = str(_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))

    subprocess.run(  # noqa: S603 - fixed interpreter and inline test program
        [sys.executable, "-c", script],
        cwd=_ROOT,
        env=env,
        check=True,
    )


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

    expected = [_ROOT / "src/dlightrag/engine/runtime/contracts.py"]
    assert owners[_STATUS_VALUES] == expected
    assert owners[_PHASE_VALUES] == expected


def test_frontend_phase_union_matches_runtime() -> None:
    text = (_ROOT / "frontend/ui/chat_feature.ts").read_text(encoding="utf-8")
    match = re.search(r"type AnswerPhase = ([^;]+);", text)
    assert match is not None
    values = tuple(part.strip().strip("'\"") for part in match.group(1).split("|"))
    assert values == _PHASE_VALUES


def test_postgres_adapter_does_not_publish_or_supply_runtime_records() -> None:
    adapter_path = _ROOT / "src/dlightrag/adapters/postgres/answer/answer_runs.py"
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
                and node.module == "dlightrag.adapters.postgres.answer.answer_runs"
            ):
                continue
            stale_imports.extend(
                (path, alias.name) for alias in node.names if alias.name in _RUNTIME_RECORD_NAMES
            )
    assert stale_imports == []
