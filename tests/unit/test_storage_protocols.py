# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for storage protocol definitions."""

from __future__ import annotations

from typing import Any

import pytest

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.storage.protocols import MetadataIndexProtocol

# ---------------------------------------------------------------------------
# Concrete stubs that satisfy each protocol
# ---------------------------------------------------------------------------


class _MetadataIndexStub:
    """Minimal concrete class implementing MetadataIndexProtocol."""

    async def initialize(self) -> None: ...
    async def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None: ...
    async def get(self, doc_id: str) -> dict[str, Any] | None: ...
    async def get_many(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]: ...
    async def query(self, filters: MetadataFilter) -> list[str]: ...
    async def delete(self, doc_id: str) -> None: ...
    async def clear(self) -> None: ...
    async def find_by_filename(self, name: str) -> list[str]: ...
    async def find_by_file_path(self, file_path: str) -> list[str]: ...
    async def get_field_schema(self) -> dict[str, Any]: ...


# Deliberately incomplete stubs — missing required methods
class _IncompleteMetadataIndex:
    async def initialize(self) -> None: ...


# ---------------------------------------------------------------------------
# TestMetadataIndexProtocol
# ---------------------------------------------------------------------------


class TestMetadataIndexProtocol:
    """Verify MetadataIndexProtocol is runtime_checkable and defines the right API."""

    def test_runtime_checkable(self) -> None:
        """Protocol is decorated with @runtime_checkable."""
        stub = _MetadataIndexStub()
        assert isinstance(stub, MetadataIndexProtocol)

    def test_incomplete_impl_fails_isinstance(self) -> None:
        """An incomplete implementation should NOT satisfy the protocol."""
        stub = _IncompleteMetadataIndex()
        assert not isinstance(stub, MetadataIndexProtocol)

    @pytest.mark.parametrize(
        "method",
        [
            "initialize",
            "upsert",
            "get",
            "get_many",
            "query",
            "delete",
            "clear",
            "find_by_filename",
            "find_by_file_path",
            "get_field_schema",
        ],
    )
    def test_required_methods_exist(self, method: str) -> None:
        """Each required method is present on the protocol class."""
        assert hasattr(MetadataIndexProtocol, method)
        assert callable(getattr(MetadataIndexProtocol, method))
