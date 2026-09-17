# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A conversation addresses stored answer images on its own origin."""

import pytest

from dlightrag.adapters.http.browser.presentation import build_answer_presentation
from dlightrag.adapters.http.browser.run_resources import (
    image_rewrites,
    rewrite_image_sources,
    run_resource_url,
)

RUN_ID = "019893f4-0000-7000-8000-000000000001"
STORED = "https://public6.wolframalpha.com/files/GIF_thz351wxmi.gif"


def test_a_resource_address_quotes_the_ids_it_carries() -> None:
    assert run_resource_url(RUN_ID, "res-abc") == f"/web/api/runs/{RUN_ID}/resources/res-abc"
    assert (
        run_resource_url(RUN_ID, "attachment-occurrence:019893f4-0000-7000-8000-0000000000ff:1")
        == f"/web/api/runs/{RUN_ID}/resources/attachment-occurrence%3A019893f4-0000-7000-8000-0000000000ff%3A1"
    )


def test_each_stored_source_url_is_addressed_by_its_own_identity() -> None:
    rewrites = image_rewrites(
        RUN_ID,
        {
            "HTTPS://Public6.WolframAlpha.com/files/GIF_thz351wxmi.gif#frag": "res-1",
            "": "res-ignored",
        },
    )

    assert rewrites == {STORED: f"/web/api/runs/{RUN_ID}/resources/res-1"}


def test_only_stored_images_are_rewritten_and_links_are_left_alone() -> None:
    rewrites = image_rewrites(RUN_ID, {STORED: "res-1"})
    html = (
        f'<p><img src="{STORED}" alt="图">'
        '<img src="https://cdn.example.com/other.png" alt="未存">'
        '<img src="data:image/png;base64,AAAA" alt="内联">'
        f'<a href="{STORED}">来源</a></p>'
    )

    rewritten = rewrite_image_sources(html, rewrites)

    assert f'src="/web/api/runs/{RUN_ID}/resources/res-1"' in rewritten
    assert 'src="https://cdn.example.com/other.png"' in rewritten
    assert 'src="data:image/png;base64,AAAA"' in rewritten
    assert f'<a href="{STORED}">来源</a>' in rewritten


def test_an_answer_without_stored_images_renders_exactly_what_it_wrote() -> None:
    html = f'<p><img src="{STORED}"></p>'

    assert rewrite_image_sources(html, {}) == html
    assert rewrite_image_sources("<p>no images</p>", image_rewrites(RUN_ID, {STORED: "res-1"})) == (
        "<p>no images</p>"
    )


def test_the_presentation_applies_the_rewrites_to_rendered_answers() -> None:
    presentation = build_answer_presentation(
        answer=f"同一公式重画：\n\n![图]({STORED})\n",
        sources=[],
        evidence_images=[],
        image_rewrites=image_rewrites(RUN_ID, {STORED: "res-1"}),
    )

    html = presentation.parts[0].html or ""
    assert f'src="/web/api/runs/{RUN_ID}/resources/res-1"' in html
    assert STORED not in html


@pytest.mark.parametrize("url", ["not a url", "javascript:alert(1)", "/web/api/runs/x"])
def test_a_non_public_url_is_never_mapped(url: str) -> None:
    assert rewrite_image_sources(f'<img src="{url}">', image_rewrites(RUN_ID, {url: "res-1"})) == (
        f'<img src="{url}">'
    )
