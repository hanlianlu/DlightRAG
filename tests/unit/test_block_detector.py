# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a page that refuses an automated browser is, and what it is not.

The markers are read from real shapes: Ctrip's own answer, a Cloudflare interstitial, a
sliding-captcha widget, and a page that merely discusses one of these checks. The last is
the point of the suite: a named marker in running text is not a refusal.
"""

from __future__ import annotations

import pytest

from dlightrag.engine.answer.agent_browser.block import (
    detect_block,
    refused_on_navigation,
)
from dlightrag.engine.answer.agent_browser.contracts import (
    refusal_failure,
)

WHALEGUARD = b"""<!DOCTYPE html><html><head><title>whaleguard block</title></head>
<body><div class="tip">\xe8\xaf\xb7\xe5\xbc\x80\xe5\x90\xaf JavaScript</div></body></html>"""

SLIDER = """<html><head><title>安全验证</title></head>
<body><div id="nc_1_wrapper" class="nc-container"><span>滑动验证</span></div>
</body></html>"""

CLOUDFLARE = b"""<html><head><title>Just a moment...</title></head>
<body><h1>Checking your browser before accessing example.com</h1></body></html>"""

ACCESS_DENIED = b"""<html><body><h2>Access Denied</h2>
<p>You don't have permission to access this resource.</p></body></html>"""

#: A long article *about* a bot filter: the same words, in ordinary text.
ARTICLE = b"""<html><head><title>How Ctrip's bot filter works</title></head><body>
<h1>How Ctrip's bot filter works</h1>
<p>The filter answers a guard page to a bot, and the page says Access Denied in its body.
A slider captcha is what its API returns, and "verify you are human" is the English form.
"Checking your browser" is what Cloudflare serves while it decides. None of this is served
to a person.</p></body></html>"""


@pytest.mark.parametrize(
    ("html", "reason", "marker"),
    [
        pytest.param(WHALEGUARD, "blocked", "whaleguard", id="ctrip-whaleguard-block"),
        pytest.param(SLIDER, "verified", "安全验证", id="slider-widget"),
        pytest.param(CLOUDFLARE, "blocked", "just a moment", id="cloudflare"),
        pytest.param(ACCESS_DENIED, "blocked", "access denied", id="access-denied-heading"),
    ],
)
def test_a_refusal_is_named_with_the_marker_that_named_it(
    html: bytes | str, reason: str, marker: str
) -> None:
    block = refused_on_navigation(430, html.encode() if isinstance(html, str) else html)
    assert block is not None
    assert block.marker == marker
    assert block.status == 430
    assert block.reason == ("verification_required" if reason == "verified" else reason)


def test_a_page_that_merely_mentions_the_markers_is_not_a_refusal() -> None:
    # The article carries every phrase marker in its body, and none of them is the
    # document's own verdict on the caller. Only a title, a heading, a preformatted block,
    # or a widget the page itself marks as a check counts, so a page that merely discusses
    # these checks is served, whatever status it answers.
    for status in (200, 403, 430):
        assert refused_on_navigation(status, ARTICLE) is None


def test_a_phrase_marker_outside_a_verdict_element_is_not_a_refusal() -> None:
    html = b"<html><head><title>Report</title></head><body><p>Access Denied</p></body></html>"
    assert detect_block(status=200, html=html) is None


def test_a_vendor_token_in_running_text_is_still_a_refusal() -> None:
    # ``whaleguard`` is no page's prose, so a 4xx page that carries it is read as a refusal
    # wherever it says it: a page that is not a refusal and answers 4xx is an ordinary
    # status failure, which is the failure it would otherwise have been.
    html = b"<html><head><title>Report</title></head><body><p>whaleguard</p></body></html>"
    block = detect_block(status=430, html=html)
    assert block is not None and block.marker == "whaleguard"
    assert refused_on_navigation(200, html) is None


def test_the_refusal_ctrip_answers_with_is_named() -> None:
    # Verbatim from the live host: 432, no title, no heading, no class — 168 bytes.
    ctrip = (
        b'<html><head><meta name="color-scheme" content="light dark"></head><body>'
        b'<pre style="word-wrap: break-word; white-space: pre-wrap;">whaleguard block </pre>'
        b"</body></html>"
    )
    block = refused_on_navigation(432, ctrip)

    assert block is not None
    assert (block.reason, block.marker, block.status) == ("blocked", "whaleguard", 432)


def test_a_page_that_answers_200_is_served_whatever_it_mentions() -> None:
    # A block detector that fired on a 200 would refuse the page whose text discusses one.
    assert refused_on_navigation(200, WHALEGUARD) is None
    assert detect_block(status=200, html=WHALEGUARD) is not None


def test_a_verification_check_is_named_apart_from_a_plain_refusal() -> None:
    checking = b"<html><head><title>Verify you are human</title></head></html>"
    block = refused_on_navigation(403, checking)
    assert block is not None and block.reason == "verification_required"
    # The two say different things: one may be read on a headed browser, the other stops.
    assert "human-verification" in refusal_failure(block).public_message
    assert "never solve" in refusal_failure(block).public_message


def test_the_failure_carries_the_evidence_the_verdict_used() -> None:
    block = refused_on_navigation(430, WHALEGUARD)
    assert block is not None
    failure = refusal_failure(block)
    assert failure.reason == "blocked"
    assert '"whaleguard"' in failure.public_message
    assert "refused for being automated" in failure.public_message


@pytest.mark.parametrize(
    "html",
    [
        # Cloudflare's real interstitial marks its widget with ``id="challenge-…"``: the
        # page that refused stockanalysis.com reads with a 403 crashed the detector.
        pytest.param(
            b"<html><head><title>Just a moment...</title></head><body>"
            b'<div id="challenge-running">Checking your browser</div></body></html>',
            id="cloudflare-challenge-widget",
        ),
        # An empty title is a verdict element with no text.
        pytest.param(
            b"<html><head><title></title></head><body><p>Forbidden</p></body></html>",
            id="empty-title",
        ),
    ],
)
def test_a_4xx_page_with_a_widget_or_an_empty_verdict_is_read_not_crashed(html: bytes) -> None:
    # Each element pattern names its own group; reading one pattern's match by the other's
    # group raised ``IndexError: no such group`` on every 4xx read of such a page.
    refused_on_navigation(403, html)


def test_a_page_with_no_status_is_read_but_not_called_a_refusal() -> None:
    # A navigation the browser saw no response for has no refusal to name.
    assert refused_on_navigation(None, WHALEGUARD) is None
