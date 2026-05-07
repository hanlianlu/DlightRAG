"""Tests for PathResolver — verifies all source type dispatch paths."""

from __future__ import annotations

from dlightrag.core.retrieval.path_resolver import PathResolver


class TestResolve:
    """Each test covers a distinct dispatch branch in resolve()."""

    def test_local_path_resolved_to_endpoint(self) -> None:
        """Core behavior: absolute local path → /api/files/ with relative path."""
        resolver = PathResolver(working_dir="/data/rag_storage")
        assert (
            resolver.resolve("/data/rag_storage/files/f.pdf")
            == "/api/files/files/f.pdf"
        )

    def test_file_scheme_is_not_rewritten(self) -> None:
        """file:// is no longer a supported source URL scheme."""
        resolver = PathResolver(working_dir="/data/rag_storage")
        assert resolver.resolve("file://sources/local/f.pdf") == "/api/files/file://sources/local/f.pdf"

    def test_azure_scheme_wrapped_into_endpoint(self) -> None:
        """Remote scheme preserved inside endpoint URL for 302 dispatch."""
        resolver = PathResolver(working_dir="/data/rag_storage")
        assert (
            resolver.resolve("azure://mycontainer/doc.pdf")
            == "/api/files/azure://mycontainer/doc.pdf"
        )

    def test_s3_scheme_wrapped_into_endpoint(self) -> None:
        resolver = PathResolver(working_dir="/data/rag_storage")
        assert resolver.resolve("s3://my-bucket/doc.pdf") == "/api/files/s3://my-bucket/doc.pdf"

    def test_fallback_marker_when_no_working_dir(self) -> None:
        """Without working_dir, falls back to the artifacts marker."""
        resolver = PathResolver(working_dir=None)
        assert (
            resolver.resolve("/random/prefix/artifacts/page.png")
            == "/api/files/artifacts/page.png"
        )

    def test_sources_marker_is_not_special(self) -> None:
        """A directory named sources is treated as a normal path component."""
        resolver = PathResolver(working_dir=None)
        assert (
            resolver.resolve("/random/prefix/sources/local/f.pdf")
            == "/api/files//random/prefix/sources/local/f.pdf"
        )

    def test_no_match_uses_raw_path(self) -> None:
        """Unresolvable path passed through as-is."""
        resolver = PathResolver(working_dir="/other/dir")
        assert resolver.resolve("/random/path/file.pdf") == "/api/files//random/path/file.pdf"
