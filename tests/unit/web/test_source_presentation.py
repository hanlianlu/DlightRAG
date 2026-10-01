# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A cited source offers only the links its caller may follow, as safe markup."""

import pytest

from dlightrag.adapters.http.browser.presentation import build_answer_presentation
from dlightrag.adapters.http.browser.safe_html import sanitize_html_fragment
from dlightrag.engine.answer.citations.contracts import SourceReferencePayload


def _presentation_source(*, source_uri: str, download_url: str | None = None):
    source = SourceReferencePayload(
        id="1",
        title=None,
        source_uri=source_uri,
        download_url=download_url,
        chunks=[],
    )
    return build_answer_presentation(
        answer="Cited [1].",
        sources=[source],
        evidence_images=[],
    ).sources[0]


def test_presentation_preserves_authorized_download_without_nesting_markup() -> None:
    source = _presentation_source(
        source_uri="local://default/notes.md",
        download_url="/web/api/files/raw/doc-notes?workspace=default",
    )
    assert source.download_url == "/web/api/files/raw/doc-notes?workspace=default"
    assert source.title == "Source"


def test_presentation_hides_download_without_caller_permission() -> None:
    source = _presentation_source(source_uri="local://default/notes.md")

    assert source.download_url is None


@pytest.mark.parametrize(
    "source_uri",
    [
        "https://exa.ai/library/weather/gothenburg-sweden?latitude=57.7052&longitude=11.9737",
        "http://www.sgas.ruc.edu.cn/xwgg/yjyxw/f1a3ff59a5894391b7b0db77951c08b4.htm",
    ],
)
def test_presentation_projects_public_web_provenance(source_uri: str) -> None:
    source = _presentation_source(source_uri=source_uri)

    assert source.source_url == source_uri


def test_presentation_rejects_non_public_provenance() -> None:
    for value in (
        "local://default/report.pdf",
        "https://127.0.0.1/private",
        "res-opaque",
    ):
        assert _presentation_source(source_uri=value).source_url is None


def test_source_anchor_allowlist_rejects_unsafe_attributes_and_targets() -> None:
    html = sanitize_html_fragment(
        '<a href="/web/api/files/raw/doc-notes" aria-label="Download source" '
        'onclick="alert(1)" style="display:none" target="_self">Download</a>'
    )

    assert 'aria-label="Download source"' in html
    assert "onclick" not in html
    assert "style=" not in html
    assert "target=" not in html
