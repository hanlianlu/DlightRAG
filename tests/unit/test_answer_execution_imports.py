# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Fresh-process contracts for the answer execution facade."""

import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_IMPORT_PATHS = [
    str(_ROOT / "src"),
    str(_ROOT / "packages/memory/src"),
    sysconfig.get_path("purelib"),
]


def _run_import_probe(script: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            f"import sys; sys.path[:0] = {_IMPORT_PATHS!r}\n" + script,
        ],
        cwd=tmp_path,
        env={"HOME": str(tmp_path), "PYTHON_DOTENV_DISABLED": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)


@pytest.mark.parametrize(
    "module", ["execution.connection_binding", "execution.input", "resources.models"]
)
def test_answer_contract_import_does_not_load_executor_or_converters(
    module: str, tmp_path: Path
) -> None:
    _run_import_probe(
        f"""
import importlib
importlib.import_module("dlightrag.engine.answer.{module}")
if {module!r} == "execution.connection_binding":
    assert "dlightrag.engine.answer.resources" not in sys.modules
forbidden = (
    "dlightrag.engine.answer.execution.executor",
    "dlightrag.engine.answer.resources.registry",
    "dlightrag.engine.answer.resources.converters",
    "markitdown", "magika", "onnxruntime",
)
loaded = sorted(name for name in sys.modules
                if any(name == prefix or name.startswith(prefix + ".")
                       for prefix in forbidden))
assert not loaded, loaded
""",
        tmp_path,
    )


def test_resources_facade_exports_resolve_actual_definitions(tmp_path: Path) -> None:
    _run_import_probe(
        """
import dlightrag.engine.answer.resources as resources
assert "dlightrag.engine.answer.resources.registry" not in sys.modules
expected_models = {
    "EXTRACTION_TEXT", "ResourceAdmissionError", "ResourceCursorError",
    "ResourceDecodeError", "ResourceInput", "ResourceManifestEntry",
    "ResourceNotFoundError", "ResourceReadResult", "ResourceRegistryError",
    "TextWindowLocator", "VisualHandle",
}
expected_registry = {"ResourceRegistry", "UrlTextFallback"}
assert set(resources.__all__) == expected_models | expected_registry
from dlightrag.engine.answer.resources import models
for name in expected_models:
    assert getattr(resources, name) is getattr(models, name), name
assert "dlightrag.engine.answer.resources.registry" not in sys.modules
assert "onnxruntime" not in sys.modules
from dlightrag.engine.answer.resources import ResourceRegistry
from dlightrag.engine.answer.resources import registry
assert ResourceRegistry is registry.ResourceRegistry
for name in expected_registry:
    assert getattr(resources, name) is getattr(registry, name), name
try:
    resources.not_an_export
except AttributeError:
    pass
else:
    raise AssertionError("Unknown facade attribute did not raise AttributeError")
""",
        tmp_path,
    )


def test_execution_facade_exports_resolve_actual_definitions(tmp_path: Path) -> None:
    _run_import_probe(
        """
import dlightrag.engine.answer.execution as execution
assert "dlightrag.engine.answer.execution.executor" not in sys.modules
expected = {
    "AnswerExecutionStore", "AnswerExecutor", "AnswerExecutorSettings",
    "AnswerResourceResolver", "AnswerResourceSettings", "OrchestratorRun",
    "ResolvedAnswerResources", "answer_trace_output", "research_history_input_measure",
}
assert set(execution.__all__) == expected
from dlightrag.engine.answer.execution import AnswerExecutor
from dlightrag.engine.answer.execution import executor, acceptance
assert AnswerExecutor is executor.AnswerExecutor
for name in expected:
    source = acceptance if name == "research_history_input_measure" else executor
    assert getattr(execution, name) is getattr(source, name), name
try:
    execution.not_an_export
except AttributeError:
    pass
else:
    raise AssertionError("Unknown facade attribute did not raise AttributeError")
""",
        tmp_path,
    )
