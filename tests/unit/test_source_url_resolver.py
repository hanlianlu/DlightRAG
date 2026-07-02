"""Tests for source URL resolution from stored retrieval paths."""

from __future__ import annotations

import inspect

from dlightrag.core.retrieval.source_url_resolver import SourceUrlResolver


def test_constructor_does_not_expose_working_dir_generation() -> None:
    """URL generation should only use current input_dir/workspace roots."""
    assert "working_dir" not in inspect.signature(SourceUrlResolver).parameters


class TestResolve:
    """Each test covers a distinct dispatch branch in resolve()."""

    def test_file_scheme_is_not_projected(self) -> None:
        """file:// is no longer a supported source URL scheme."""
        resolver = SourceUrlResolver()
        assert resolver.resolve("file://sources/local/f.pdf") is None

    def test_azure_scheme_wrapped_into_endpoint(self) -> None:
        """Remote scheme preserved inside endpoint URL for 302 dispatch."""
        resolver = SourceUrlResolver()
        assert (
            resolver.resolve("azure://mycontainer/doc.pdf")
            == "/api/files/azure://mycontainer/doc.pdf"
        )

    def test_s3_scheme_wrapped_into_endpoint(self) -> None:
        resolver = SourceUrlResolver()
        assert resolver.resolve("s3://my-bucket/doc.pdf") == "/api/files/s3://my-bucket/doc.pdf"

    def test_https_scheme_wrapped_into_endpoint(self) -> None:
        resolver = SourceUrlResolver()
        assert (
            resolver.resolve("https://api.bynder.com/docs/getting-started")
            == "/api/files/https://api.bynder.com/docs/getting-started"
        )

    def test_https_source_query_is_encoded_into_endpoint_path(self) -> None:
        resolver = SourceUrlResolver()
        assert resolver.resolve("https://cdn.example.com/report.pdf?sig=x&download=1") == (
            "/api/files/https://cdn.example.com/report.pdf%3Fsig%3Dx%26download%3D1"
        )

    def test_absolute_artifact_marker_paths_are_not_projected(self) -> None:
        """Only input_dir-scoped local files are projected into /api/files URLs."""
        resolver = SourceUrlResolver()
        assert resolver.resolve("/random/prefix/artifacts/page.png") is None

    def test_sources_marker_is_not_special(self) -> None:
        """A directory named sources is treated as a normal path component."""
        resolver = SourceUrlResolver()
        assert resolver.resolve("/random/prefix/sources/local/f.pdf") is None

    def test_no_match_returns_none(self) -> None:
        """Unresolvable absolute local paths are not exposed."""
        resolver = SourceUrlResolver()
        assert resolver.resolve("/random/path/file.pdf") is None

    def test_relative_path_is_projected(self) -> None:
        resolver = SourceUrlResolver()
        assert resolver.resolve("default/docs/report.pdf") == "/api/files/default/docs/report.pdf"

    def test_windows_relative_path_is_projected_as_posix(self) -> None:
        resolver = SourceUrlResolver()
        assert resolver.resolve(r"default\docs\report.pdf") == "/api/files/default/docs/report.pdf"

    def test_relative_traversal_is_not_projected(self) -> None:
        resolver = SourceUrlResolver()
        assert resolver.resolve("../secrets.txt") is None

    def test_windows_absolute_path_is_not_projected(self) -> None:
        resolver = SourceUrlResolver()
        assert resolver.resolve(r"C:\Users\me\report.pdf") is None


class TestInputDir:
    """Tests for current input_dir + workspace URL generation."""

    def test_resolve_local_path_under_input_dir(self) -> None:
        """Local path under input_dir is stripped to relative."""
        resolver = SourceUrlResolver(input_dir="/data/input")
        result = resolver.resolve("/data/input/default/report.pdf")
        assert result == "/api/files/default/report.pdf"

    def test_resolve_local_path_outside_input_dir(self) -> None:
        """Local path outside input_dir is not exposed."""
        resolver = SourceUrlResolver(input_dir="/data/input")
        result = resolver.resolve("/tmp/report.pdf")
        assert result is None

    def test_resolve_azure_path(self) -> None:
        """Azure scheme wraps as-is."""
        resolver = SourceUrlResolver(input_dir="/data/input")
        result = resolver.resolve("azure://mycontainer/doc.pdf")
        assert result == "/api/files/azure://mycontainer/doc.pdf"

    def test_resolve_s3_path(self) -> None:
        """S3 scheme wraps as-is."""
        resolver = SourceUrlResolver(input_dir="/data/input")
        result = resolver.resolve("s3://mybucket/key/doc.pdf")
        assert result == "/api/files/s3://mybucket/key/doc.pdf"

    def test_resolve_https_path(self) -> None:
        """URL sources wrap as-is."""
        resolver = SourceUrlResolver(input_dir="/data/input")
        result = resolver.resolve("https://api.bynder.com/docs/getting-started")
        assert result == "/api/files/https://api.bynder.com/docs/getting-started"

    def test_resolve_workspace_scoped(self) -> None:
        """Workspace is baked into the relative path under input_dir."""
        resolver = SourceUrlResolver(input_dir="/data/input", workspace="ws-a")
        result = resolver.resolve("/data/input/ws-a/subdir/doc.pdf")
        assert result == "/api/files/ws-a/subdir/doc.pdf"

    def test_resolve_workspace_bare_filename(self) -> None:
        """LightRAG basename-only file paths are scoped by workspace."""
        resolver = SourceUrlResolver(input_dir="/data/input", workspace="ws-a")
        result = resolver.resolve("report.pdf")
        assert result == "/api/files/ws-a/report.pdf"

    def test_input_dir_prefix_lookalike_is_not_treated_as_child(self) -> None:
        """Only real descendants of input_dir are stripped to endpoint-relative paths."""
        resolver = SourceUrlResolver(input_dir="/data/input")
        result = resolver.resolve("/data/input-shadow/default/report.pdf")
        assert result is None
