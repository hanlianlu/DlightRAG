# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ownership contracts for the root PostgreSQL adapters."""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOTS = (_ROOT / "src", _ROOT / "packages")
_POSTGRES_ADAPTER = _ROOT / "src/dlightrag/adapters/postgres"
_WEB_RECORD_NAMES = {
    "AnswerTurnCreation",
    "ConversationSnapshot",
    "ConversationSubmissionConflict",
    "LinkedTurn",
}


def _python_files() -> list[Path]:
    return [path for root in _SOURCE_ROOTS for path in root.rglob("*.py")]


def test_legacy_storage_package_is_removed() -> None:
    assert not (_ROOT / "src/dlightrag/storage").exists()


def test_asyncpg_is_private_to_the_postgres_adapter() -> None:
    offenders: list[Path] = []
    for path in _python_files():
        if path.is_relative_to(_POSTGRES_ADAPTER):
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
        _ROOT / "packages/rag-core/src/dlightrag_rag",
        _ROOT / "src/dlightrag/runtime",
    )
    owner_files = [path for root in owner_roots for path in root.rglob("*.py")]
    owner_files.extend(
        (
            _ROOT / "src/dlightrag/application.py",
            _ROOT / "src/dlightrag/api/routes/status.py",
        )
    )

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


def test_callers_take_web_records_from_the_web_owner() -> None:
    offenders: list[tuple[Path, str]] = []
    for path in _python_files() + list((_ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module == "dlightrag.adapters.postgres.web_conversations"
            ):
                continue
            offenders.extend(
                (path.relative_to(_ROOT), alias.name)
                for alias in node.names
                if alias.name in _WEB_RECORD_NAMES
            )

    assert offenders == []
