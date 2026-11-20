# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a module may and may not have loaded after its import, read in a fresh interpreter.

``sys.modules`` of the test process is polluted by every other test, so each probe is a table
row: a script run in an isolated interpreter whose only paths are the source trees and the
installed dependencies, exiting non-zero when its own assertion fails.
"""

import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_IMPORT_PATHS = [
    str(_ROOT / "src"),
    *(str(source) for source in sorted((_ROOT / "packages").glob("*/src"))),
    sysconfig.get_path("purelib"),
]

_TRANSPORTS_COMPOSE_NO_APPLICATION = """
import importlib
import dlightrag

calls = 0
def forbidden_create_application(*args, **kwargs):
    global calls
    calls += 1
    raise AssertionError("transport import composed an Application")

dlightrag.create_application = forbidden_create_application

for name in (
    "dlightrag.adapters.http.server",
    "dlightrag.adapters.http.rest.routes.rag",
    "dlightrag.adapters.http.rest.routes.runs",
    "dlightrag.adapters.http.browser.auth",
    "dlightrag.adapters.http.browser.routes.chat",
    "dlightrag.adapters.mcp.server",
    "dlightrag.adapters.mcp.tools.retrieval",
    "dlightrag.adapters.mcp.tools.answer_runs",
):
    importlib.import_module(name)
assert calls == 0
"""

_SDK_AND_RUNTIME_LOAD_NO_COMPOSITION = """
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


def _postgres_runtime_and_sessions(first_module: str) -> str:
    return f"""
import importlib
first = importlib.import_module({first_module!r})
runs = importlib.import_module("dlightrag.adapters.postgres.runtime.run_store")
blobs = importlib.import_module("dlightrag.adapters.postgres.runtime.run_blob_store")
sessions = importlib.import_module("dlightrag.adapters.postgres.answer.session_repository")
assert runs.PGRunStore.__module__ == "dlightrag.adapters.postgres.runtime.run_store"
assert blobs.PGRunBlobStore.__module__ == "dlightrag.adapters.postgres.runtime.run_blob_store"
assert sessions.PGAgentSessionRepository.__module__ == "dlightrag.adapters.postgres.answer.session_repository"
"""


def _answer_contract_loads_no_executor(module: str) -> str:
    return f"""
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
"""


# The server answers without MarkItDown's detector and ONNX runtime loaded.
_API_ENTRY_LOADS_NO_CONVERSION_STACK = """
import importlib
importlib.import_module("dlightrag.adapters.http.server")
loaded = [name for name in ("magika", "onnxruntime") if name in sys.modules]
assert not loaded, f"the API entry loaded {loaded}"
"""

_PROBES = {
    "transports-compose-no-application": _TRANSPORTS_COMPOSE_NO_APPLICATION,
    "sdk-and-runtime-load-no-composition": _SDK_AND_RUNTIME_LOAD_NO_COMPOSITION,
    "postgres-runtime-then-sessions": _postgres_runtime_and_sessions(
        "dlightrag.adapters.postgres.runtime.run_store"
    ),
    "postgres-sessions-then-runtime": _postgres_runtime_and_sessions(
        "dlightrag.adapters.postgres.answer.session_repository"
    ),
    "connection-binding-loads-no-executor": _answer_contract_loads_no_executor(
        "execution.connection_binding"
    ),
    "execution-input-loads-no-executor": _answer_contract_loads_no_executor("execution.input"),
    "resource-models-load-no-executor": _answer_contract_loads_no_executor("resources.models"),
    "api-entry-loads-no-conversion-stack": _API_ENTRY_LOADS_NO_CONVERSION_STACK,
}


@pytest.mark.parametrize("script", _PROBES.values(), ids=_PROBES.keys())
def test_an_import_boundary_holds_in_a_fresh_interpreter(script: str, tmp_path: Path) -> None:
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
        timeout=240,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
