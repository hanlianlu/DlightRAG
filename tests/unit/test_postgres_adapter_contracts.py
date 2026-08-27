# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ownership contracts for the root PostgreSQL adapters."""

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOTS = (_ROOT / "src", _ROOT / "packages")
_POSTGRES_ADAPTER = _ROOT / "src/dlightrag/adapters/postgres"
_MEMORY_PG_ADAPTER = _ROOT / "packages/memory/src/dlightrag_memory/_storage"
_RAG_CORE = _ROOT / "src/dlightrag/rag"
_RAW_SQL_RE = re.compile(
    r"\b(?:SELECT\b.{0,500}\bFROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|"
    r"CREATE\s+(?:TABLE|INDEX)|ALTER\s+TABLE|DROP\s+INDEX)\b",
    re.IGNORECASE | re.DOTALL,
)
_POSTGRES_IDENTIFIERS = (
    "LIGHTRAG_DOC_CHUNKS",
    "dlightrag_doc_metadata",
    "dlightrag_ingest_jobs",
    "dlightrag_bm25_language",
)
_WEB_RECORD_NAMES = {
    "AnswerTurnCreation",
    "ConversationSnapshot",
    "ConversationSubmissionConflict",
    "LinkedTurn",
}


def _python_files() -> list[Path]:
    return [path for root in _SOURCE_ROOTS for path in root.rglob("*.py")]


def test_asyncpg_is_private_to_the_postgres_adapter() -> None:
    offenders: list[Path] = []
    for path in _python_files():
        if path.is_relative_to(_POSTGRES_ADAPTER) or path.is_relative_to(_MEMORY_PG_ADAPTER):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            (isinstance(node, ast.Import) and any(alias.name == "asyncpg" for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "asyncpg")
            for node in ast.walk(tree)
        ):
            offenders.append(path.relative_to(_ROOT))

    assert offenders == []


def test_owner_modules_do_not_import_the_postgres_adapter() -> None:
    owner_roots = (
        _ROOT / "src/dlightrag/application",
        _ROOT / "src/dlightrag/engine/runtime",
        _ROOT / "src/dlightrag/rag",
    )
    owner_files = [path for root in owner_roots for path in root.rglob("*.py")]
    owner_files.append(_ROOT / "src/dlightrag/api/routes/status.py")

    offenders: list[Path] = []
    for path in owner_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("dlightrag.adapters.postgres")
            )
            or (
                isinstance(node, ast.Import)
                and any(
                    alias.name.startswith("dlightrag.adapters.postgres") for alias in node.names
                )
            )
            for node in ast.walk(tree)
        ):
            offenders.append(path.relative_to(_ROOT))

    assert offenders == []


def test_rag_core_contains_no_postgres_schema_or_raw_sql() -> None:
    offenders: list[Path] = []
    for path in _RAG_CORE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings: set[int] = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not body or not isinstance(body, list):
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))
        literals = (
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        )
        if any(
            _RAW_SQL_RE.search(literal)
            or any(identifier in literal for identifier in _POSTGRES_IDENTIFIERS)
            for literal in literals
        ):
            offenders.append(path.relative_to(_ROOT))

    assert offenders == []


def test_callers_take_web_records_from_the_application_owner() -> None:
    offenders: list[tuple[Path, str]] = []
    for path in _python_files() + list((_ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module == "dlightrag.web.conversation_models"
            ):
                continue
            offenders.extend(
                (path.relative_to(_ROOT), alias.name)
                for alias in node.names
                if alias.name in _WEB_RECORD_NAMES
            )

    assert offenders == []
