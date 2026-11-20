# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public contracts and dependency boundary of the durable RunRuntime."""

import ast
import re
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
    (definition,) = re.findall(rf"CONSTRAINT {constraint}\s+CHECK \(\w+ IN \(([^)]*)\)\)", schema)

    assert set(get_args(vocabulary)) == set(re.findall(r"'([^']*)'", definition))


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
