# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ownership contracts for the root PostgreSQL adapters.

Which modules may import asyncpg or the PostgreSQL adapter are import-linter contracts in
pyproject.toml; what is left here reads the shape of the source those contracts cannot.
"""

import ast
import re
from pathlib import Path

from tests.support.source_tree import REPO_ROOT, python_files, source_text, syntax_tree

_ROOT = REPO_ROOT
_SOURCE_ROOTS = (_ROOT / "src", _ROOT / "packages")
_RAG_CORE = _ROOT / "src/dlightrag/engine/rag"
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
_WEB_MODELS_MODULE = "dlightrag.adapters.http.browser.conversation_models"
_WEB_RECORD_NAMES = {
    "AnswerTurnCreation",
    "ConversationHead",
    "ConversationHistoryPage",
    "SubmissionSeed",
    "ConversationSubmissionConflict",
    "LinkedTurn",
}


def test_rag_core_contains_no_postgres_schema_or_raw_sql() -> None:
    offenders: list[Path] = []
    assert _RAG_CORE.is_dir()
    for path in python_files(_RAG_CORE):
        tree = syntax_tree(path)
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
    for path in python_files(*_SOURCE_ROOTS, _ROOT / "tests"):
        if _WEB_MODELS_MODULE not in source_text(path):
            continue
        for node in ast.walk(syntax_tree(path)):
            if not (isinstance(node, ast.ImportFrom) and node.module == _WEB_MODELS_MODULE):
                continue
            offenders.extend(
                (path.relative_to(_ROOT), alias.name)
                for alias in node.names
                if alias.name in _WEB_RECORD_NAMES
            )

    assert offenders == []
