# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The ``browser`` tool driving a real Chromium through a real Playwright run-server.

What stands in for the public Web is a loopback proxy serving ``http://*.example`` pages, as
the egress proxy would carry it, so a test observes the traffic the browser sends and what the
tool answers. No database is needed: the Resources the tool admits go to a recording sink.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from dlightrag.adapters.agent_browser import PooledBrowserProvider
from dlightrag.engine.agent.environment import AccessScheduler
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.tool_content import tool_content_attachments
from dlightrag.engine.agent.tools import AgentTool, ToolResult
from dlightrag.engine.agent.tools.contracts import CommittedOutput
from dlightrag.engine.agent.tools.files import (
    BashArgs,
    ImagePreparer,
    ResourceReader,
    ResourceReadRequest,
)
from dlightrag.engine.answer.agent_browser import BrowserHolder, RunAgentBrowser
from dlightrag.engine.answer.resources.registry import (
    FetchedResourceBytes,
    ResourceEffectOwner,
    ResourceRegistry,
)
from dlightrag.engine.answer.tools.browser import BrowserToolHost, browser_tool
from dlightrag.engine.answer.tools.resources import make_resource_reader
from dlightrag.engine.answer.workspace import spill_receipt, write_spill_file
from tests.support.agent_browser import (
    FakeLeases,
    RunServer,
    Served,
    WebProxy,
    browser_settings,
    run_server,
    web_proxy,
)
from tests.support.dns import public_dns
from tests.support.path_tools import path_tools
from tests.support.resources import preparer
from tests.tool_helpers import recording_tool_runtime

pytestmark = pytest.mark.asyncio

HOLDER = BrowserHolder("owner", "11111111-1111-1111-1111-111111111111", "worker", 1)
SHOP = "http://shop.example"
DOWNLOAD_DIRECTORIES = "dlightrag-browser-download-*"

HOME = """<html><head><title>Shop</title></head><body><h1>Shop</h1>
<form action="/results" method="get"><input aria-label="Search" name="q">
<button type="submit">Go</button></form>
<a href="/upload">Upload page</a> <a href="/ready">Ready page</a></body></html>"""


def results(query: str, page: int = 1) -> str:
    more = f"<a href='/results?q={query}&page=2'>Next page</a>" if page == 1 else ""
    return f"""<html><head><title>Results {page}</title></head><body><h1>Results</h1>
<p>Results for {query} page {page}</p>{more}
<a href="/details" target="_blank">Open details</a>
<button onclick="if (confirm('Sure?')) document.body.append(' deleted')">Delete</button>
<a href="/data.csv">Download CSV</a><button id="blob">Make CSV</button>
<script>document.getElementById('blob').onclick = () => {{
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob(['x,y\\n3,4\\n'], {{type: 'text/csv'}}));
  link.download = 'made.csv'; link.click(); }};</script></body></html>"""


UPLOAD = """<html><head><title>Upload</title></head><body>
<input type="file" aria-label="Attachment" id="f" multiple><p id="out">nothing chosen</p>
<script>document.getElementById('f').onchange = (e) => {
  document.getElementById('out').textContent =
    'chosen: ' + Array.from(e.target.files).map(f => f.name).join(','); };</script></body></html>"""
READY = (
    "<html><head><title>Ready</title></head><body><p>Waiting</p><script>"
    "setTimeout(() => { const p = document.createElement('p'); p.textContent = 'Ready now';"
    " document.body.append(p); }, 600);</script></body></html>"
)
# Each level of nesting is a list item, so the page is deeper than any snapshot depth below it.
DEEP = (
    "<html><head><title>Deep</title></head><body>"
    + "<ul><li>" * 12
    + "<button onclick=\"document.title = 'pressed'\">Deep button</button>"
    + "</li></ul>" * 12
    + "</body></html>"
)
HUGE = (
    "<html><head><title>Huge</title></head><body><ul>"
    + "".join(f"<li>item {number:05d} of the long list</li>" for number in range(3000))
    + "</ul></body></html>"
)
# One button opens a popup that closes itself, another opens one after its call has returned.
POPUPS = """<html><head><title>Popups</title></head><body>
<button onclick="window.open('/flash')">Open flash</button>
<button onclick="setTimeout(() => window.open('/late'), 600)">Open later</button></body></html>"""
# The page this link opens loads an image whose request is held, so it never goes quiet.
BUSY = '<html><head><title>Busy</title></head><body><a href="/slow">Open slow</a></body></html>'
COOKIE = (
    "<html><head><title>Cookie</title></head><body><script>"
    "document.body.append('cookie=' + document.cookie); document.cookie = 'who=' + location.hash;"
    "</script></body></html>"
)


def page(title: str, body: str) -> Served:
    return Served(f"<html><head><title>{title}</title></head><body>{body}</body></html>")


def csv(name: str, content: bytes) -> Served:
    return Served(
        content,
        headers={
            "content-type": "text/csv",
            "content-disposition": f"attachment; filename={name}",
        },
    )


PAGES = {
    f"{SHOP}/": Served(HOME),
    f"{SHOP}/results?q=lamp": Served(results("lamp")),
    f"{SHOP}/results?q=lamp&page=2": Served(results("lamp", 2)),
    f"{SHOP}/details": page("Details", "<button onclick='window.close()'>Close</button>"),
    f"{SHOP}/data.csv": csv("data.csv", b"a,b\n1,2\n"),
    f"{SHOP}/big.csv": csv("big.csv", b"x" * 10_000),
    f"{SHOP}/upload": Served(UPLOAD),
    f"{SHOP}/ready": Served(READY),
    f"{SHOP}/deep": Served(DEEP),
    f"{SHOP}/huge": Served(HUGE),
    f"{SHOP}/cookie": Served(COOKIE),
    f"{SHOP}/popups": Served(POPUPS),
    f"{SHOP}/flash": page(
        "Flash", "<p>Flash popup</p><script>setTimeout(() => window.close(), 300)</script>"
    ),
    f"{SHOP}/late": page("Late", "<p>Late popup</p>"),
    f"{SHOP}/busy": Served(BUSY),
    f"{SHOP}/slow": page("Slow", "<p>Slow page</p><img src='/held' alt='held'>"),
    f"{SHOP}/signed?token=abc": page("Signed", "<p>Signed page</p>"),
    "http://alpha.example/": page("Alpha", "<p>Alpha stock</p>"),
    "http://beta.example/": page("Beta", "<p>Beta stock</p>"),
}


@pytest.fixture(autouse=True)
def _hosts_resolve_public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)


@dataclass
class Browsing:
    """A Run's ``browser`` tool over a real browser, and everything it leaves behind."""

    tool: AgentTool
    registry: ResourceRegistry
    server: RunServer
    proxy: WebProxy
    leases: FakeLeases
    admitted: list[tuple[FetchedResourceBytes, ResourceEffectOwner | None]]
    subjects: list[str] = field(default_factory=list)

    async def call(self, scope: str = "parent", **arguments: Any) -> ToolResult:
        updates: list[ToolResult] = []
        runtime = recording_tool_runtime(updates, tool_name="browser", execution_scope=scope)
        result = await self.tool.execute(self.tool.input_model.model_validate(arguments), runtime)
        self.subjects = [update.subject for update in updates if update.subject]
        return result

    async def ref(self, text: str, scope: str = "parent") -> str:
        """The ref of the first element of the current page whose snapshot line holds ``text``."""
        found = await self.call(scope, action="find", query=text)
        return ref_of(found.text_content, text)


def ref_of(snapshot: str, text: str) -> str:
    """The ref on the first line of ``snapshot`` that holds ``text``."""
    for line in snapshot.splitlines():
        if text in line and (marker := re.search(r"\[ref=([^\]]+)\]", line)):
            return marker.group(1)
    raise AssertionError(f"no ref for {text!r} in:\n{snapshot}")


@asynccontextmanager
async def browsing(
    pages: dict[str, Served] | None = None,
    *,
    workspace: Path | LocalExecutionEnvironment | None = None,
    scheduler: AccessScheduler | None = None,
    spill: Path | None = None,
    images: int = 3,
    proxy_url: str | None = None,
    **bounds: Any,
) -> AsyncIterator[Browsing]:
    """A Run's browser tool over a run-server whose browser reaches ``pages`` through a proxy."""
    admitted: list[tuple[FetchedResourceBytes, ResourceEffectOwner | None]] = []

    async def sink(fetched: FetchedResourceBytes, owner: ResourceEffectOwner | None) -> None:
        admitted.append((fetched, owner))

    async def keep(text: str) -> CommittedOutput:
        assert spill is not None
        resource_id = f"spill_{uuid.uuid4().hex}"
        write_spill_file(spill, resource_id, text)
        return spill_receipt(resource_id, text)

    prepare: ImagePreparer = preparer(images)
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(web_proxy(PAGES if pages is None else pages))
        leases = FakeLeases()
        provider = PooledBrowserProvider(
            endpoints=(server.endpoint,),
            egress_proxy=proxy_url or proxy.url,
            chromium_sandbox=True,
            connect_timeout_seconds=20,
            leases=leases,
        )
        stack.push_async_callback(provider.aclose)
        settings = browser_settings(**{"navigation": 5, "settle": 0.3, "action": 3, **bounds})
        run = RunAgentBrowser(provider, HOLDER, settings)
        stack.push_async_callback(run.aclose)
        registry = await stack.enter_async_context(ResourceRegistry(fetched_bytes_sink=sink))
        host = BrowserToolHost(run, registry, make_resource_reader(registry, 4000))
        tool = browser_tool(
            host,
            environment=(
                LocalExecutionEnvironment(workspace) if isinstance(workspace, Path) else workspace
            ),
            scheduler=scheduler or AccessScheduler(),
            spill=keep if spill is not None else None,
            image_preparer=prepare,
        )
        yield Browsing(tool, registry, server, proxy, leases, admitted)


async def until(condition: Callable[[], object], seconds: float = 10) -> None:
    """Wait for something the browser does on its own, such as a request it sends."""
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.05)


def download_directories() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob(DOWNLOAD_DIRECTORIES))


async def test_a_search_form_is_driven_by_refs() -> None:
    async with browsing() as web:
        opened = await web.call(action="navigate", url=f"{SHOP}/")
        assert web.subjects == ["shop.example/"]
        assert opened.text_content.startswith(f"[browser: navigate | page: {SHOP}/ | title: Shop]")
        search = ref_of(opened.text_content, 'textbox "Search"')

        results_page = await web.call(action="type", ref=search, text="lamp", submit=True)

        assert web.subjects == [f"{search} · shop.example/"]
        assert f"page: {SHOP}/results?q=lamp | title: Results 1]" in results_page.text_content
        assert "Results for lamp page 1" in results_page.text_content
        assert not results_page.is_error
        # The browser asked the egress proxy for both pages, and nothing reached them directly.
        assert [request.target for request in web.proxy.requests] == [
            f"{SHOP}/",
            f"{SHOP}/results?q=lamp",
        ]


async def test_find_returns_refs_that_act() -> None:
    async with browsing() as web:
        await web.call(action="navigate", url=f"{SHOP}/results?q=lamp")
        found = await web.call(action="find", query="next")

        assert 'link "Next page"' in found.text_content
        assert '1 element(s) match "next":' in found.text_content
        assert web.subjects == ["next"]
        moved = await web.call(action="click", ref=ref_of(found.text_content, "Next page"))
        assert "Results for lamp page 2" in moved.text_content
        nothing = await web.call(action="find", query="nonexistent")
        assert nothing.text_content.splitlines()[-1] == 'No element matches "nonexistent".'


async def test_a_ref_below_the_depth_limit_still_acts() -> None:
    async with browsing(depth=4) as web:
        shown = await web.call(action="navigate", url=f"{SHOP}/deep")
        assert "Deep button" not in shown.text_content
        found = await web.call(action="find", query="Deep button")
        deep = ref_of(found.text_content, "Deep button")

        # A snapshot taken after the find still leaves the ref resolvable, though it is hidden.
        assert "Deep button" not in (await web.call(action="snapshot")).text_content
        clicked = await web.call(action="click", ref=deep)

        assert "title: pressed" in clicked.text_content


async def test_a_new_tab_becomes_the_active_page_until_it_closes() -> None:
    async with browsing() as web:
        await web.call(action="navigate", url=f"{SHOP}/results?q=lamp")
        opened = await web.call(action="click", ref=await web.ref("Open details"))

        assert f"page: {SHOP}/details | title: Details]" in opened.text_content
        assert "A new tab opened and is now the active page." in opened.text_content
        assert 'button "Close"' in opened.text_content

        closed = await web.call(action="click", ref=ref_of(opened.text_content, "Close"))

        assert f"page: {SHOP}/results?q=lamp" in closed.text_content
        assert "The active page closed; the previous page is active again." in closed.text_content


async def test_a_popup_that_closes_itself_while_the_call_waits_returns_to_the_page() -> None:
    async with browsing(settle=2) as web:
        await web.call(action="navigate", url=f"{SHOP}/popups")

        flashed = await web.call(action="click", ref=await web.ref("Open flash"))

        assert not flashed.is_error
        assert f"page: {SHOP}/popups | title: Popups]" in flashed.text_content
        assert "The active page closed; the previous page is active again." in flashed.text_content


async def test_a_popup_that_opens_after_its_call_returned_is_where_the_next_call_acts() -> None:
    async with browsing() as web:
        await web.call(action="navigate", url=f"{SHOP}/popups")

        clicked = await web.call(action="click", ref=await web.ref("Open later"))
        assert f"page: {SHOP}/popups" in clicked.text_content
        assert "new tab" not in clicked.text_content
        await until(lambda: web.proxy.fetched(f"{SHOP}/late"))
        await asyncio.sleep(0.2)

        moved = await web.call(action="navigate", url=f"{SHOP}/details")

        # The navigation happened in the popup, so its call shows where it went.
        assert f"page: {SHOP}/details | title: Details]" in moved.text_content
        assert "A new tab opened and is now the active page." in moved.text_content
        back = await web.call(action="back")
        assert f"page: {SHOP}/late | title: Late]" in back.text_content
        assert "new tab" not in back.text_content


async def test_a_confirm_is_accepted_and_reported() -> None:
    async with browsing() as web:
        await web.call(action="navigate", url=f"{SHOP}/results?q=lamp")

        deleted = await web.call(action="click", ref=await web.ref("Delete"))

        assert 'The page showed a confirm dialog: "Sure?" (accepted).' in deleted.text_content
        assert "deleted" in deleted.text_content


async def test_a_page_that_answers_an_error_status_is_shown_with_it() -> None:
    async with browsing() as web:
        missing = await web.call(action="navigate", url=f"{SHOP}/nothing-here")

        assert not missing.is_error
        assert missing.text_content.splitlines()[0].endswith("| HTTP 404]")


async def test_downloads_are_admitted_with_the_call_that_made_them() -> None:
    before = download_directories()
    async with browsing() as web:
        await web.call(action="navigate", url=f"{SHOP}/results?q=lamp")

        linked = await web.call(action="click", ref=await web.ref("Download CSV"))
        made = await web.call(action="click", ref=await web.ref("Make CSV"))
        direct = await web.call("fresh", action="navigate", url=f"{SHOP}/data.csv")

        linked_note = re.search(
            r"Downloaded data\.csv \(text/csv, 8 bytes\) as (res-\w+) \(browser_download\); "
            r"read\(resource_id='\1'\) reads it\.",
            linked.text_content,
        )
        assert linked_note is not None
        (link_row, made_row, direct_row) = (fetched for fetched, _ in web.admitted)
        assert (link_row.resource_id, link_row.url) == (linked_note.group(1), f"{SHOP}/data.csv")
        assert (link_row.acquisition, link_row.admission_origin) == ("browser_download", "agent")
        # A script-generated file has no public URL of its own, so the handle is its locator.
        assert (made_row.filename, made_row.url) == ("made.csv", made_row.resource_id)
        assert f"as {made_row.resource_id} (browser_download)" in made.text_content
        assert direct_row.url == f"{SHOP}/data.csv"
        assert f"as {direct_row.resource_id} (browser_download)" in direct.text_content
        # A URL that answers with a file loads no page: a session that has none stays blank.
        assert "page: about:blank" in direct.text_content
        # What a download holds is read like any Resource.
        text = await read(make_resource_reader(web.registry, 1000), link_row.resource_id)
        assert "| a | b |" in text and "| 1 | 2 |" in text
        assert len({row.resource_id for row, _ in web.admitted}) == 3
    assert download_directories() == before


async def test_a_download_over_the_limit_is_refused_and_leaves_nothing_behind() -> None:
    before = download_directories()
    async with browsing(download_bytes=1024) as web:
        await web.call(action="navigate", url=f"{SHOP}/")

        refused = await web.call(action="navigate", url=f"{SHOP}/big.csv")

        assert web.admitted == []
        assert (
            "The download big.csv was not admitted: it exceeds 1024 bytes." in refused.text_content
        )
        assert not refused.is_error
    assert download_directories() == before


async def test_a_download_this_process_cannot_save_is_refused_and_the_page_goes_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async with browsing() as web:
        await web.call(action="navigate", url=f"{SHOP}/results?q=lamp")
        link = await web.ref("Download CSV")

        # A temporary directory that is gone stands for a disk that cannot take the file.
        with monkeypatch.context() as patched:
            patched.setattr(tempfile, "tempdir", str(tmp_path / "gone"))
            refused = await web.call(action="click", ref=link)

        assert not refused.is_error and web.admitted == []
        assert "The download data.csv was not admitted: it could not be downloaded." in (
            refused.text_content
        )
        assert "Results for lamp page 1" in refused.text_content
        again = await web.call(action="click", ref=link)
        assert "Downloaded data.csv (text/csv, 8 bytes)" in again.text_content
        assert len(web.admitted) == 1


async def test_a_capture_admits_a_new_citable_web_resource_each_time() -> None:
    async with browsing() as web:
        await web.call(action="navigate", url=f"{SHOP}/results?q=lamp")

        first = await web.call(action="capture")
        second = await web.call(action="capture")

        header = re.search(r"\[resource: (res-\w+) \|", first.text_content)
        assert header is not None
        resource_id = header.group(1)
        assert f"Captured this page as Web Resource {resource_id} (browser_capture)." in (
            first.text_content
        )
        assert "Results for lamp page 1" in first.text_content
        (source,) = first.effects.evidence_sources
        assert (source.resource_id, source.source_type, source.source_uri) == (
            resource_id,
            "web_search",
            f"{SHOP}/results?q=lamp",
        )
        assert dict(source.attributes) == {
            "resource_kind": "web",
            "admission_origin": "agent",
            "acquisition": "browser_capture",
        }
        # The capture settles with its conversion view, restored without a browser.
        assert {effect.resource_kind for effect in first.effects.attached_resources} == {
            "conversion_snapshot"
        }
        assert second.effects.evidence_sources[0].resource_id != resource_id
        assert [fetched.url for fetched, _ in web.admitted] == [f"{SHOP}/results?q=lamp"] * 2

        await web.call(action="navigate", url=f"{SHOP}/signed?token=abc")
        signed = await web.call(action="capture")
        (signed_source,) = signed.effects.evidence_sources
        assert signed_source.source_uri == signed_source.resource_id
        assert signed_source.source_type == "web_attachment"


async def test_navigate_refuses_non_public_targets_before_any_browser_is_leased() -> None:
    async with browsing() as web:
        for url in (
            "http://127.0.0.1:8100/",
            "http://user:secret@shop.example/",
            "file:///etc/passwd",
            "http://localhost/",
        ):
            refused = await web.call(action="navigate", url=url)

            assert refused.is_error
            assert refused.text_content.startswith("browser navigate refused this URL:")
            assert "secret" not in refused.text_content

        assert (web.leases.claimed, web.proxy.requests) == ([], [])


async def test_actions_name_stale_refs_and_missing_pages() -> None:
    async with browsing() as web:
        without_page = await web.call("child", action="snapshot")
        assert without_page.is_error and "No page is open in this Agent Session" in (
            without_page.text_content
        )
        assert web.leases.claimed == []

        await web.call(action="navigate", url=f"{SHOP}/")
        stale = await web.call(action="click", ref="e9999")

        assert stale.is_error and "No element on the current page has ref e9999" in (
            stale.text_content
        )
        # A session that never navigated is still without a page beside one that has one.
        assert (await web.call("child", action="snapshot")).is_error
        assert len(web.leases.claimed) == 1


async def test_a_huge_snapshot_is_kept_in_the_workspace_and_its_head_shown(
    tmp_path: Path,
) -> None:
    async with browsing(workspace=tmp_path / "work", spill=tmp_path / "spill") as web:
        shown = await web.call(action="navigate", url=f"{SHOP}/huge")

        assert "browser snapshot exceeded 51200 UTF-8 bytes or 2000 lines" in shown.text_content
        assert "item 00000 of the long list" in shown.text_content
        assert "item 02999" not in shown.text_content
        assert shown.protected_text.startswith("Full output: read(resource_id='spill_")
        assert shown.text_content.endswith(shown.protected_text)
        (receipt,) = shown.effects.committed_outputs
        assert receipt.resource_id in shown.protected_text
        kept = (tmp_path / "spill" / f"{receipt.resource_id}.txt").read_text(encoding="utf-8")
        assert "item 02999 of the long list" in kept and not shown.is_error


async def test_a_huge_snapshot_with_no_workspace_to_keep_it_is_reported_without_failing() -> None:
    async with browsing() as web:
        shown = await web.call(action="navigate", url=f"{SHOP}/huge")

        assert not shown.is_error and shown.effects.committed_outputs == ()
        assert shown.text_content.startswith(f"[browser: navigate | page: {SHOP}/huge")
        assert "The action completed, but its full snapshot is unavailable" in shown.text_content
        assert "Do not repeat the action; use find to locate an element." in shown.text_content
        assert "item 00000 of the long list" in shown.text_content
        assert shown.protected_text == ""


async def test_a_screenshot_spends_the_image_budget_and_is_never_evidence() -> None:
    async with browsing(images=1) as web:
        await web.call(action="navigate", url=f"{SHOP}/")

        shot = await web.call(action="screenshot")
        again = await web.call(action="screenshot", full_page=True)

        (attachment,) = tool_content_attachments(shot.parts)
        (stored,) = shot.effects.attached_resources
        assert attachment.media_type.startswith("image/") and attachment.data
        assert (stored.resource_id, stored.content) == (attachment.resource_id, attachment.data)
        assert (stored.source_locator, stored.resource_kind) == ("shop.example/", "tool_attachment")
        assert shot.effects.evidence_sources == ()
        assert "Screenshot of the visible page" in shot.text_content
        assert "its pixels are context, not evidence" in shot.text_content
        # The Run's one image is spent: a second screenshot is refused, and attaches nothing.
        assert again.is_error and "remaining image budget" in again.text_content
        assert tool_content_attachments(again.parts) == ()
        assert again.effects.attached_resources == ()


async def test_upload_puts_a_workspace_file_into_a_file_input(tmp_path: Path) -> None:
    (tmp_path / "report.txt").write_text("quarterly numbers", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    async with browsing(workspace=tmp_path) as web:
        opened = await web.call(action="navigate", url=f"{SHOP}/upload")
        attachment = ref_of(opened.text_content, "Attachment")

        sent = await web.call(action="upload", ref=attachment, files=["report.txt"])
        assert "chosen: report.txt" in sent.text_content

        escaping = await web.call(action="upload", ref=attachment, files=["../outside.txt"])
        directory = await web.call(action="upload", ref=attachment, files=["folder"])
        missing = await web.call(action="upload", ref=attachment, files=["nothing.txt"])

        assert escaping.is_error and escaping.text_content == "path must not escape the workspace"
        assert directory.is_error and "upload needs regular workspace files: folder" in (
            directory.text_content
        )
        assert missing.is_error and "nothing.txt" in missing.text_content


class ReadRecorder(LocalExecutionEnvironment):
    """A workspace that keeps the names of the files read from it."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.read: list[str] = []

    def read_bytes(self, path: Path) -> bytes:
        self.read.append(path.name)
        return super().read_bytes(path)


async def test_an_upload_that_waited_for_the_workspace_checks_it_again_when_it_gets_it(
    tmp_path: Path,
) -> None:
    (tmp_path / "report.txt").write_text("quarterly numbers", encoding="utf-8")
    workspace, scheduler = LocalExecutionEnvironment(tmp_path), AccessScheduler()
    bash = {tool.name: tool for tool in path_tools(workspace, scheduler=scheduler)}["bash"]
    async with browsing(workspace=workspace, scheduler=scheduler) as web:
        opened = await web.call(action="navigate", url=f"{SHOP}/upload")
        attachment = ref_of(opened.text_content, "Attachment")

        # The command holds the workspace, and leaves an unsafe entry in it as it ends.
        command = BashArgs(command="ln -s /etc/passwd link; sleep 1")
        running = asyncio.ensure_future(bash.execute(command, recording_tool_runtime([])))
        await asyncio.sleep(0.4)
        refused = await web.call(action="upload", ref=attachment, files=["report.txt"])
        await running

        assert refused.is_error and "workspace integrity latched" in refused.text_content
        assert "chosen" not in refused.text_content


async def test_upload_refuses_what_is_over_the_limit_before_it_reads_it(tmp_path: Path) -> None:
    # Sparse files: they have the size and cost no disk.
    for name, mebibytes in (("huge.bin", 60), ("first.bin", 30), ("second.bin", 30)):
        with (tmp_path / name).open("wb") as handle:
            handle.truncate(mebibytes * 1024 * 1024)
    workspace = ReadRecorder(tmp_path)
    async with browsing(workspace=workspace) as web:
        opened = await web.call(action="navigate", url=f"{SHOP}/upload")
        attachment = ref_of(opened.text_content, "Attachment")

        huge = await web.call(action="upload", ref=attachment, files=["huge.bin"])
        both = await web.call(action="upload", ref=attachment, files=["first.bin", "second.bin"])

        for refused in (huge, both):
            assert refused.is_error
            assert refused.text_content == "upload sends at most 50 MiB in one call."
        # Only the first of two files that together are too big was ever read.
        assert workspace.read == ["first.bin"]


async def test_wait_for_text_and_back() -> None:
    async with browsing(navigation=1.5) as web:
        await web.call(action="navigate", url=f"{SHOP}/")
        await web.call(action="navigate", url=f"{SHOP}/ready")

        appeared = await web.call(action="wait", text="Ready now")
        assert "Ready now" in appeared.text_content

        timed_out = await web.call(action="wait", text="Never")
        assert timed_out.is_error
        assert timed_out.text_content == '"Never" did not appear within 1.5 seconds.'
        paused = await web.call(action="wait", seconds=0.2)
        assert not paused.is_error

        previous = await web.call(action="back")
        assert f"page: {SHOP}/ |" in previous.text_content
        blank = await web.call(action="back")
        assert "page: about:blank" in blank.text_content
        first = await web.call(action="back")
        assert first.is_error and "no earlier page" in first.text_content


async def test_failures_read_as_sentences_a_model_can_act_on() -> None:
    async with browsing(action=1.5) as web:
        await web.call(action="navigate", url=f"{SHOP}/")

        key = await web.call(action="press", key="Nonsense")
        assert key.is_error and key.text_content.startswith("Nonsense is not a key")

        heading = ref_of((await web.call(action="snapshot")).text_content, "heading")
        typed = await web.call(action="type", ref=heading, text="x")
        assert typed.is_error and typed.text_content.startswith(
            "The browser could not type into element"
        )
        assert "Call log" not in typed.text_content

        unreachable = await web.call(action="navigate", url="http://unrouted.example/")
        assert not unreachable.is_error and "HTTP 404" in unreachable.text_content


async def test_sessions_are_isolated_and_everything_they_load_goes_through_the_proxy() -> None:
    async with browsing() as web:
        await web.call("alpha", action="navigate", url=f"{SHOP}/cookie#alpha")
        again = await web.call("alpha", action="navigate", url=f"{SHOP}/cookie")
        other = await web.call("beta", action="navigate", url=f"{SHOP}/cookie")

        assert "cookie=who=#alpha" in again.text_content
        assert "cookie=who" not in other.text_content
        assert len(web.leases.claimed) == 1
        assert {request.method for request in web.proxy.requests} == {"GET"}
        assert [request.target for request in web.proxy.requests] == [f"{SHOP}/cookie"] * 3


async def test_a_browser_launched_with_a_dead_proxy_reaches_nothing() -> None:
    async with browsing(proxy_url="http://127.0.0.1:9") as web:
        failed = await web.call(action="navigate", url=f"{SHOP}/")

        assert failed.is_error
        assert failed.text_content == (
            "The Agent Browser could not load the page (net::ERR_PROXY_CONNECTION_FAILED)."
        )
        assert web.proxy.requests == []


async def test_a_disconnect_loses_every_page_and_the_next_navigate_leases_again() -> None:
    async with browsing() as web:
        await web.call("a", action="navigate", url="http://alpha.example/")
        await web.call("b", action="navigate", url="http://beta.example/")
        port = urlsplit(web.server.endpoint).port
        assert port is not None

        await web.server.stop()
        a_saw = await web.call("a", action="snapshot")
        b_saw = await web.call("b", action="snapshot")

        assert a_saw.is_error and "every open page of this Run was lost" in a_saw.text_content
        assert b_saw.is_error and "page was lost when the Agent Browser disconnected" in (
            b_saw.text_content
        )
        assert len(web.leases.claimed) == 1

        # The pool member came back at its endpoint, and the Run's next page asks for it again.
        async with run_server(port):
            again = await web.call("b", action="navigate", url="http://beta.example/")

            assert "Beta stock" in again.text_content
            assert len(web.leases.claimed) == 2


async def test_a_browser_that_disconnects_while_a_page_settles_is_reported_and_leased_afresh() -> (
    None
):
    held = asyncio.Event()
    pages = {**PAGES, f"{SHOP}/held": Served("held", hold=held)}
    async with browsing(pages, settle=30) as web:
        await web.call(action="navigate", url=f"{SHOP}/busy")
        port = urlsplit(web.server.endpoint).port
        assert port is not None

        # The image's request is never answered, so the call is still settling the page, past
        # the moment it waits for a popup, when its browser goes away.
        clicking = asyncio.create_task(web.call(action="click", ref=await web.ref("Open slow")))
        await until(lambda: web.proxy.fetched(f"{SHOP}/held"))
        await asyncio.sleep(0.6)
        await web.server.stop()
        lost = await asyncio.wait_for(clicking, 30)

        assert lost.is_error and "every open page of this Run was lost" in lost.text_content
        async with run_server(port):
            again = await web.call(action="navigate", url="http://alpha.example/")

            assert "Alpha stock" in again.text_content
            assert len(web.leases.claimed) == 2


async def read(reader: ResourceReader, resource_id: str) -> str:
    request = ResourceReadRequest(resource_id=resource_id, url=None, focus=None, cursor=None)
    result = await reader(request, recording_tool_runtime([], tool_name="read"))
    return result.text_content
