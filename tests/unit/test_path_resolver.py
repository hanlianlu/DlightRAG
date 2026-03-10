"""Tests for PathResolver — verifies all source type dispatch paths."""
from __future__ import annotations

from dlightrag.core.retrieval.path_resolver import PathResolver


class TestResolve:
    """Each test covers a distinct dispatch branch in resolve()."""

    def test_local_path_resolved_to_endpoint(self) -> None:
        """Core behavior: absolute local path → /api/files/ with relative path."""
        resolver = PathResolver(working_dir="/data/rag_storage")
        assert resolver.resolve("/data/rag_storage/sources/local/f.pdf") == "/api/files/sources/local/f.pdf"

    def test_file_scheme_treated_as_local(self) -> None:
        """Backward compat: file:// stripped, then treated as local."""
        resolver = PathResolver(working_dir="/data/rag_storage")
        assert resolver.resolve("file://sources/local/f.pdf") == "/api/files/sources/local/f.pdf"

    def test_azure_scheme_wrapped_into_endpoint(self) -> None:
        """Remote scheme preserved inside endpoint URL for 302 dispatch."""
        resolver = PathResolver(working_dir="/data/rag_storage")
        assert resolver.resolve("azure://mycontainer/doc.pdf") == "/api/files/azure://mycontainer/doc.pdf"

    def test_snowflake_scheme_wrapped_into_endpoint(self) -> None:
        resolver = PathResolver(working_dir="/data/rag_storage")
        assert resolver.resolve("snowflake://my_table") == "/api/files/snowflake://my_table"

    def test_fallback_marker_when_no_working_dir(self) -> None:
        """Without working_dir, falls back to sources/artifacts marker."""
        resolver = PathResolver(working_dir=None)
        assert resolver.resolve("/random/prefix/sources/local/f.pdf") == "/api/files/sources/local/f.pdf"

    def test_no_match_uses_raw_path(self) -> None:
        """Unresolvable path passed through as-is."""
        resolver = PathResolver(working_dir="/other/dir")
        assert resolver.resolve("/random/path/file.pdf") == "/api/files//random/path/file.pdf"
