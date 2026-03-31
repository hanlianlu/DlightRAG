# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for storage protocol definitions (MetadataIndexProtocol, HashIndexProtocol)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.storage.protocols import HashIndexProtocol, MetadataIndexProtocol

# ---------------------------------------------------------------------------
# Concrete stubs that satisfy each protocol
# ---------------------------------------------------------------------------


class _MetadataIndexStub:
    """Minimal concrete class implementing MetadataIndexProtocol."""

    async def initialize(self) -> None: ...
    async def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None: ...
    async def get(self, doc_id: str) -> dict[str, Any] | None: ...
    async def query(self, filters: MetadataFilter) -> list[str]: ...
    async def delete(self, doc_id: str) -> None: ...
    async def clear(self) -> None: ...
    async def find_by_filename(self, name: str) -> list[str]: ...
    async def get_field_schema(self) -> dict[str, Any]: ...


class _HashIndexStub:
    """Minimal concrete class implementing HashIndexProtocol."""

    async def check_exists(self, content_hash: str) -> tuple[bool, str | None]: ...
    async def register(self, content_hash: str, doc_id: str, file_path: str) -> None: ...
    async def remove(self, content_hash: str) -> bool: ...
    async def should_skip_file(
        self, file_path: Path, replace: bool
    ) -> tuple[bool, str | None, str | None]: ...
    async def clear(self) -> None: ...
    async def list_all(self) -> list[dict[str, Any]]: ...
    def invalidate(self) -> None: ...
    async def find_by_name(self, filename: str) -> tuple[str | None, str | None, str | None]: ...
    async def find_by_path(self, file_path: str) -> tuple[str | None, str | None, str | None]: ...

    @staticmethod
    def generate_doc_id_from_path(file_path: Path) -> str:
        return file_path.stem


# Deliberately incomplete stubs — missing required methods
class _IncompleteMetadataIndex:
    async def initialize(self) -> None: ...


class _IncompleteHashIndex:
    async def check_exists(self, content_hash: str) -> tuple[bool, str | None]: ...


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
            "query",
            "delete",
            "clear",
            "find_by_filename",
            "get_field_schema",
        ],
    )
    def test_required_methods_exist(self, method: str) -> None:
        """Each required method is present on the protocol class."""
        assert hasattr(MetadataIndexProtocol, method)
        assert callable(getattr(MetadataIndexProtocol, method))


# ---------------------------------------------------------------------------
# TestHashIndexProtocol
# ---------------------------------------------------------------------------


class TestHashIndexProtocol:
    """Verify HashIndexProtocol is runtime_checkable and defines the right API."""

    def test_runtime_checkable(self) -> None:
        """Protocol is decorated with @runtime_checkable."""
        stub = _HashIndexStub()
        assert isinstance(stub, HashIndexProtocol)

    def test_incomplete_impl_fails_isinstance(self) -> None:
        """An incomplete implementation should NOT satisfy the protocol."""
        stub = _IncompleteHashIndex()
        assert not isinstance(stub, HashIndexProtocol)

    @pytest.mark.parametrize(
        "method",
        [
            "check_exists",
            "register",
            "remove",
            "should_skip_file",
            "clear",
            "list_all",
            "invalidate",
            "find_by_name",
            "find_by_path",
            "generate_doc_id_from_path",
        ],
    )
    def test_required_methods_exist(self, method: str) -> None:
        """Each required method is present on the protocol class."""
        assert hasattr(HashIndexProtocol, method)
        assert callable(getattr(HashIndexProtocol, method))

    def test_generate_doc_id_is_static(self) -> None:
        """generate_doc_id_from_path should be callable as a static method."""
        # If it's a proper static method on a conforming class, it works without self
        result = _HashIndexStub.generate_doc_id_from_path(Path("/tmp/report.pdf"))
        assert result == "report"
