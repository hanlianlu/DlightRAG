# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for trusted PostgreSQL identifier guards."""

from __future__ import annotations

import pytest

from dlightrag.storage.sql_identifiers import pg_identifier, pg_qualified_identifier


def test_pg_identifier_accepts_simple_identifiers() -> None:
    assert pg_identifier("LIGHTRAG_DOC_CHUNKS") == "LIGHTRAG_DOC_CHUNKS"
    assert pg_identifier("_workspace_1") == "_workspace_1"


def test_pg_qualified_identifier_accepts_schema_qualified_identifiers() -> None:
    assert pg_qualified_identifier("public.LIGHTRAG_DOC_CHUNKS") == (
        "public.LIGHTRAG_DOC_CHUNKS"
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "",
        "1table",
        "public.",
        "public.table;drop",
        'public."quoted"',
    ],
)
def test_pg_qualified_identifier_rejects_unsafe_identifiers(identifier: str) -> None:
    with pytest.raises(ValueError, match="Unsafe PostgreSQL identifier"):
        pg_qualified_identifier(identifier)
