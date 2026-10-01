# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public contracts and dependency boundary of the durable RunRuntime."""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, get_args

import pytest

from dlightrag.adapters.postgres.runtime.run_store import RUN_MIGRATIONS
from dlightrag.engine.runtime.contracts import RunKind, RunLane, RunStatus

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("vocabulary", "constraint"),
    [
        (RunStatus, "dlightrag_runs_status_check"),
        (RunKind, "dlightrag_runs_kind_check"),
        (RunLane, "dlightrag_runs_lane_check"),
    ],
)
def test_run_vocabulary_is_what_the_run_table_accepts(vocabulary: Any, constraint: str) -> None:
    schema = "\n".join(
        statement for migration in RUN_MIGRATIONS for statement in migration.statements
    )
    checks = re.findall(rf"CONSTRAINT {constraint}\s+CHECK \(\w+ IN \(([^)]*)\)\)", schema)

    assert checks, constraint
    # A later migration may redefine the constraint; its last definition is the live one.
    assert set(get_args(vocabulary)) == set(re.findall(r"'([^']*)'", checks[-1]))


def test_sdk_and_runtime_import_without_composition_or_transports() -> None:
    script = """
import sys
import dlightrag.adapters.http.client.client
import dlightrag.engine.runtime
forbidden = (
    "asyncpg", "fastapi", "lightrag", "PIL", "dlightrag.adapters.postgres",
    "dlightrag.engine.answer", "dlightrag.adapters.http.rest",
    "dlightrag.adapters.http.server", "dlightrag.adapters.mcp",
    "dlightrag.adapters.http.browser",
)
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
if loaded:
    raise SystemExit("unexpected eager imports: " + ", ".join(loaded))
"""
    env = os.environ.copy()
    source = str(_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        env=env,
        check=True,
    )


@pytest.mark.parametrize(
    "first_module",
    (
        "dlightrag.adapters.postgres.answer.session_repository",
        "dlightrag.adapters.postgres.runtime",
    ),
)
def test_postgres_runtime_and_session_repository_import_in_either_order(
    first_module: str,
) -> None:
    script = f"""
import importlib
first = importlib.import_module({first_module!r})
runtime = importlib.import_module("dlightrag.adapters.postgres.runtime")
sessions = importlib.import_module("dlightrag.adapters.postgres.answer.session_repository")
assert runtime.PGRunStore.__module__ == "dlightrag.adapters.postgres.runtime.run_store"
assert runtime.PGRunBlobStore.__module__ == "dlightrag.adapters.postgres.runtime.run_blob_store"
assert sessions.PGAgentSessionRepository.__module__ == "dlightrag.adapters.postgres.answer.session_repository"
"""
    env = os.environ.copy()
    source = str(_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))
    subprocess.run([sys.executable, "-c", script], cwd=_ROOT, env=env, check=True)


def test_runtime_has_no_answer_specific_public_symbols() -> None:
    import dlightrag.engine.runtime as runtime

    assert not any(name.startswith("Answer") for name in runtime.__all__)


def test_postgres_adapter_does_not_publish_runtime_records() -> None:
    adapter_path = _ROOT / "src/dlightrag/adapters/postgres/runtime/run_store.py"
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
    runtime_records = {"RunRecord", "RunEvent", "RunStatus", "RunKind", "RunLane"}
    assert public_names.isdisjoint(runtime_records)
