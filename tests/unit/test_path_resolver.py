"""Tests for PathResolver — verifies all source type dispatch paths."""

from __future__ import annotations

from dlightrag.core.retrieval.path_resolver import PathResolver


class TestResolve:
    """Each test covers a distinct dispatch branch in resolve()."""

    def test_local_path_resolved_to_endpoint(self) -> None:
        """Core behavior: absolute local path → /api/files/ with relative path."""
        resolver = PathResolver(working_dir="/data/rag_storage")
        assert resolver.resolve("/data/rag_storage/files/f.pdf") == "/api/files/files/f.pdf"

    def test_file_scheme_is_not_rewritten(self) -> None:
        """file:// is no longer a supported source URL scheme."""
        resolver = PathResolver(working_dir="/data/rag_storage")
        assert (
            resolver.resolve("file://sources/local/f.pdf")
            == "/api/files/file://sources/local/f.pdf"
        )

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
            resolver.resolve("/random/prefix/artifacts/page.png") == "/api/files/artifacts/page.png"
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


class TestInputDir:
    """Tests for the input_dir + workspace parameters added in Task 5."""

    def test_resolve_local_path_under_input_dir(self) -> None:
        """Local path under input_dir is stripped to relative."""
        resolver = PathResolver(input_dir="/data/input")
        result = resolver.resolve("/data/input/default/report.pdf")
        assert result == "/api/files/default/report.pdf"

    def test_resolve_local_path_outside_input_dir(self) -> None:
        """Local path outside input_dir falls through to raw path."""
        resolver = PathResolver(input_dir="/data/input")
        result = resolver.resolve("/tmp/report.pdf")
        assert result == "/api/files//tmp/report.pdf"

    def test_resolve_azure_path(self) -> None:
        """Azure scheme wraps as-is."""
        resolver = PathResolver(input_dir="/data/input")
        result = resolver.resolve("azure://mycontainer/doc.pdf")
        assert result == "/api/files/azure://mycontainer/doc.pdf"

    def test_resolve_s3_path(self) -> None:
        """S3 scheme wraps as-is."""
        resolver = PathResolver(input_dir="/data/input")
        result = resolver.resolve("s3://mybucket/key/doc.pdf")
        assert result == "/api/files/s3://mybucket/key/doc.pdf"

    def test_resolve_workspace_scoped(self) -> None:
        """Workspace is baked into the relative path under input_dir."""
        resolver = PathResolver(input_dir="/data/input", workspace="ws-a")
        result = resolver.resolve("/data/input/ws-a/subdir/doc.pdf")
        assert result == "/api/files/ws-a/subdir/doc.pdf"

    def test_input_dir_takes_priority_over_working_dir(self) -> None:
        """When both are set, input_dir is tried first."""
        resolver = PathResolver(
            working_dir="/data/rag_storage",
            input_dir="/data/input",
        )
        result = resolver.resolve("/data/input/default/doc.pdf")
        assert result == "/api/files/default/doc.pdf"

    def test_working_dir_fallback_when_input_dir_does_not_match(self) -> None:
        """When input_dir does not match, working_dir is used."""
        resolver = PathResolver(
            working_dir="/data/rag_storage",
            input_dir="/data/input",
        )
        result = resolver.resolve("/data/rag_storage/files/f.pdf")
        assert result == "/api/files/files/f.pdf"
