from __future__ import annotations

from pathlib import Path

import pytest

from dlightrag.sourcing.source_contract import (
    implicit_https_download_uri,
    local_source_uri,
    validate_download_uri,
    validate_source_uri,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("s3://bucket/docs/report.pdf", "s3://bucket/docs/report.pdf"),
        ("s3://bucket/docs/Q2%20report.pdf", "s3://bucket/docs/Q2%20report.pdf"),
        ("azure://container/docs/report.pdf", "azure://container/docs/report.pdf"),
        ("https://cdn.example.com/docs/report.pdf", "https://cdn.example.com/docs/report.pdf"),
        (
            "https://cdn.example.com/docs/Q2%20report.pdf",
            "https://cdn.example.com/docs/Q2%20report.pdf",
        ),
        ("HTTPS://cdn.example.com/docs/report.pdf", "https://cdn.example.com/docs/report.pdf"),
        ("S3://bucket/docs/report.pdf", "s3://bucket/docs/report.pdf"),
        ("AZURE://container/docs/report.pdf", "azure://container/docs/report.pdf"),
    ],
)
def test_validate_download_uri_accepts_durable_supported_locators(
    value: str, expected: str
) -> None:
    assert validate_download_uri(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "bynder://asset/report.pdf",
        "file:///tmp/report.pdf",
        "http://example.com/report.pdf",
        "https://user:secret@example.com/report.pdf",
        "https://example.com/download?id=1",
        "https://example.com/report.pdf#page=2",
        "s3://bucket",
        "azure://container",
        "s3://bucket/../secret.pdf",
        "s3://bucket/%2e%2e/secret.pdf",
        "azure://container/folder/%2E%2E/secret.pdf",
        "s3://user:secret@bucket/report.pdf",
    ],
)
def test_validate_download_uri_rejects_non_durable_or_unsupported_locators(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="durable download_uri"):
        validate_download_uri(value)


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com:not-a-port/report.pdf",
        "https://example.com:70000/report.pdf",
        "https://example.com/report file.pdf",
        " https://example.com/report.pdf",
        "https://example.com/report.pdf\n",
        "https://example.com/report\x7f.pdf",
        "s3://:443/report.pdf",
        "s3://bucket:/report.pdf",
        "s3://bucket:443/report.pdf",
        "s3://user%3Asecret%40bucket/report.pdf",
        "azure://@container/report.pdf",
        "azure://container name/report.pdf",
        "azure://container%20name/report.pdf",
    ],
)
def test_validate_download_uri_rejects_malformed_uri_syntax(value: str) -> None:
    with pytest.raises(ValueError, match="durable download_uri"):
        validate_download_uri(value)


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/report%ZZ.pdf",
        "s3://bucket/report%.pdf",
        "azure://container/report%2.pdf",
        r"https://example.com/report\file.pdf",
        "https://example.com/report<draft>.pdf",
        'https://example.com/report"draft.pdf',
        "https://example.com/report{draft}.pdf",
        "s3://bucket/report|draft.pdf",
        "s3://bucket/report^draft.pdf",
        "azure://container/report`draft.pdf",
        "https://example.com/résumé.pdf",
    ],
)
def test_validate_download_uri_rejects_invalid_lexical_form(value: str) -> None:
    with pytest.raises(ValueError, match="durable download_uri"):
        validate_download_uri(value)


def test_implicit_https_download_uri_never_persists_query_or_fragment() -> None:
    assert implicit_https_download_uri("https://example.com/report.pdf") == (
        "https://example.com/report.pdf"
    )
    assert implicit_https_download_uri("https://example.com/report.pdf?sig=secret") is None
    assert implicit_https_download_uri("https://example.com/report.pdf#page=2") is None


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/report.pdf?",
        "https://example.com/report.pdf#",
        "s3://bucket/report.pdf?",
        "azure://container/report.pdf#",
    ],
)
def test_validate_download_uri_rejects_empty_query_or_fragment_delimiters(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="durable download_uri"):
        validate_download_uri(value)


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/report.pdf?",
        "https://example.com/report.pdf#",
    ],
)
def test_implicit_https_download_uri_rejects_empty_query_or_fragment_delimiters(
    value: str,
) -> None:
    assert implicit_https_download_uri(value) is None


@pytest.mark.parametrize("value", ["", "\x00", "cms://asset/\x00secret"])
def test_validate_source_uri_rejects_empty_or_nul_values(value: str) -> None:
    with pytest.raises(ValueError, match="source_uri is invalid"):
        validate_source_uri(value)


def test_validate_source_uri_allows_connector_specific_identity() -> None:
    assert validate_source_uri("bynder://asset/1") == "bynder://asset/1"


def test_local_source_uri_is_workspace_relative_and_path_safe() -> None:
    assert local_source_uri("research", Path("reports/Q2 results.md")) == (
        "local://research/reports/Q2%20results.md"
    )


@pytest.mark.parametrize(
    "value",
    [
        r"C:\Users\alice\secret.pdf",
        r"\\server\share\secret.pdf",
    ],
)
def test_local_source_uri_rejects_windows_absolute_paths(value: str) -> None:
    with pytest.raises(ValueError, match="local source identity is invalid"):
        local_source_uri("research", value)
