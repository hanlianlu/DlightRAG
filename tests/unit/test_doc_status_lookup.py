# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for indexed PostgreSQL deletion identity lookups."""

from typing import Any

from dlightrag.adapters.postgres.corpus.doc_status_lookup import PGDocStatusLookup
from tests.support.pg_fakes import FakePool


async def test_resolve_deletion_matches_empty_input_skips_database() -> None:
    class _Connection:
        async def fetch(self, *_args: Any) -> list[Any]:
            raise AssertionError("database should not be queried")

    lookup = PGDocStatusLookup(workspace="default", pool=FakePool(_Connection()))

    assert await lookup.resolve_deletion_matches(file_paths=(), doc_ids=()) == ()
