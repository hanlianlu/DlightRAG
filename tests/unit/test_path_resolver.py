"""Tests for PathResolver — verifies current source URL normalization."""

from __future__ import annotations

import inspect

from dlightrag.core.retrieval.path_resolver import PathResolver


def test_constructor_does_not_expose_working_dir_legacy_generation() -> None:
    """URL generation should only use current input_dir/workspace roots."""
    assert "working_dir" not in inspect.signature(PathResolver).parameters


class TestResolve:
    """Each test covers a distinct dispatch branch in resolve()."""

    def test_file_scheme_is_not_rewritten(self) -> None:
        """file:// is no longer a supported source URL scheme."""
        resolver = PathResolver()
        assert (
            resolver.resolve("file://sources/local/f.pdf")
            == "/api/files/file://sources/local/f.pdf"
        )

    def test_azure_scheme_wrapped_into_endpoint(self) -> None:
        """Remote scheme preserved inside endpoint URL for 302 dispatch."""
        resolver = PathResolver()
        assert (
            resolver.resolve("azure://mycontainer/doc.pdf")
            == "/api/files/azure://mycontainer/doc.pdf"
        )

    def test_s3_scheme_wrapped_into_endpoint(self) -> None:
        resolver = PathResolver()
        assert resolver.resolve("s3://my-bucket/doc.pdf") == "/api/files/s3://my-bucket/doc.pdf"

    def test_artifact_marker_paths_are_preserved_for_serving(self) -> None:
        """Parser artifact paths keep the artifact-relative suffix for /api/files."""
        resolver = PathResolver()
        assert (
            resolver.resolve("/random/prefix/artifacts/page.png") == "/api/files/artifacts/page.png"
        )

    def test_sources_marker_is_not_special(self) -> None:
        """A directory named sources is treated as a normal path component."""
        resolver = PathResolver()
        assert (
            resolver.resolve("/random/prefix/sources/local/f.pdf")
            == "/api/files//random/prefix/sources/local/f.pdf"
        )

    def test_no_match_uses_raw_path(self) -> None:
        """Unresolvable path passed through as-is."""
        resolver = PathResolver()
        assert resolver.resolve("/random/path/file.pdf") == "/api/files//random/path/file.pdf"


class TestInputDir:
    """Tests for current input_dir + workspace URL generation."""

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

    def test_resolve_workspace_bare_filename(self) -> None:
        """LightRAG basename-only file paths are scoped by workspace."""
        resolver = PathResolver(input_dir="/data/input", workspace="ws-a")
        result = resolver.resolve("report.pdf")
        assert result == "/api/files/ws-a/report.pdf"
