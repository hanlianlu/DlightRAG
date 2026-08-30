# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for indexed PostgreSQL deletion identity lookups."""

from typing import Any

from dlightrag.adapters.postgres.corpus.doc_status_lookup import PGDocStatusLookup


class _Acquire:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def __aenter__(self) -> Any:
        return self._connection

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _Pool:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def acquire(self) -> _Acquire:
        return _Acquire(self._connection)


async def test_resolve_deletion_matches_uses_only_id_and_exact_path_queries() -> None:
    class _Connection:
        def __init__(self) -> None:
            self.fetches: list[tuple[Any, ...]] = []

        async def fetch(self, *args: Any) -> list[dict[str, str]]:
            self.fetches.append(args)
            if "id = ANY" in args[0]:
                return [{"id": "doc-1", "file_path": "/tmp/report.pdf"}]
            return [
                {"id": "doc-1", "file_path": "/tmp/report.pdf"},
                {"id": "dup-1", "file_path": "/tmp/report.pdf"},
            ]

    connection = _Connection()
    lookup = PGDocStatusLookup(workspace="default", pool=_Pool(connection))

    matches = await lookup.resolve_deletion_matches(
        file_paths=("/tmp/report.pdf", "report.pdf"),
        doc_ids=("doc-1",),
    )

    assert [(match.doc_id, match.file_path) for match in matches] == [
        ("doc-1", "/tmp/report.pdf"),
        ("dup-1", "/tmp/report.pdf"),
    ]
    assert len(connection.fetches) == 2
    assert connection.fetches[0][1:] == ("default", ["doc-1"])
    assert connection.fetches[1][1:] == (
        "default",
        ["/tmp/report.pdf", "report.pdf"],
    )
    assert all("get_docs_by_status" not in str(call) for call in connection.fetches)


async def test_resolve_deletion_matches_empty_input_skips_database() -> None:
    class _Connection:
        async def fetch(self, *_args: Any) -> list[Any]:
            raise AssertionError("database should not be queried")

    lookup = PGDocStatusLookup(workspace="default", pool=_Pool(_Connection()))

    assert await lookup.resolve_deletion_matches(file_paths=(), doc_ids=()) == ()
