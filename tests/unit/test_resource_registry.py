# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the run-scoped answer Resource Registry."""

from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Callable
from typing import Any

import pytest

from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.answer.agent_browser import (
    AgentBrowserError,
    RenderedPage,
    browser_failure,
)
from dlightrag.engine.answer.resources.formatting import format_resource_read
from dlightrag.engine.answer.resources.models import (
    RenderedReadTargetError,
    ResourceAdmissionError,
    ResourceCursorError,
    ResourceDecodeError,
    ResourceInput,
    ResourceNotFoundError,
)
from dlightrag.engine.answer.resources.registry import (
    AgentBrowserRender,
    BrowserResourceInput,
    HostedExtract,
    ResourceEffectOwner,
    ResourceStateMismatchError,
    browser_capture_filename,
    browser_download_filename,
    browser_download_media_type,
)
from dlightrag.engine.answer.resources.registry import (
    ResourceRegistry as _ResourceRegistry,
)
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from dlightrag.engine.answer.web_sources import WebExtractResult, WebSourceUnavailable
from dlightrag.engine.public_http import (
    PublicHttpFetch,
    PublicHttpPolicyError,
    PublicHttpPresentation,
)
from tests.support.agent_browser import RecordingRenderer
from tests.support.dns import public_dns


class ResourceRegistry(_ResourceRegistry):
    """Exercise registry behavior under a small explicit model window."""

    async def read(
        self,
        resource_id: str,
        *,
        max_window_tokens: int = 100,
        focus: str | None = None,
        cursor: str | None = None,
        rendered: bool = False,
        effect_owner: ResourceEffectOwner | None = None,
    ):
        return await super().read(
            resource_id,
            max_window_tokens=max_window_tokens,
            focus=focus,
            cursor=cursor,
            rendered=rendered,
            effect_owner=effect_owner,
        )


class _Fetch:
    """The public HTTP transport a link read goes through, answering with one body."""

    def __init__(
        self,
        content: bytes = b"hello\nworld",
        *,
        final_url: str | None = None,
        fail: type[BaseException] | None = None,
    ) -> None:
        self.content = content
        self.final_url = final_url
        self.fail = fail
        self.presentations: list[PublicHttpPresentation] = []

    @property
    def calls(self) -> int:
        return len(self.presentations)

    async def __call__(
        self, url: str, *, presentation: PublicHttpPresentation, **_kwargs: object
    ) -> PublicHttpFetch:
        self.presentations.append(presentation)
        if self.fail is not None:
            raise self.fail()
        return PublicHttpFetch(
            content=self.content,
            final_url=self.final_url or url,
            media_type=None,
            status_code=200,
        )


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], Any]:
    """Answer the registry's link fetches; hosts resolve public for the Extract check."""
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)

    def install(fetch: Any) -> Any:
        monkeypatch.setattr("dlightrag.engine.answer.resources.registry.fetch_public_http", fetch)
        return fetch

    return install


def test_register_returns_stable_opaque_id() -> None:
    registry = ResourceRegistry()
    first = registry.register(ResourceInput(filename="a.txt", content=b"same bytes"))
    second = registry.register(ResourceInput(filename="a.txt", content=b"same bytes"))

    assert first == second
    assert "a.txt" not in first
    assert "same bytes" not in first
    assert len(registry.manifest()) == 1


def test_duplicate_bytes_preserve_distinct_source_filenames() -> None:
    registry = ResourceRegistry()
    first = registry.register(ResourceInput(filename="a.txt", content=b"payload"))
    second = registry.register(ResourceInput(filename="b.txt", content=b"payload"))

    assert first != second
    assert len(registry.manifest()) == 2


def test_request_isolation_uses_distinct_ids() -> None:
    left = ResourceRegistry().register(ResourceInput(content=b"shared"))
    right = ResourceRegistry().register(ResourceInput(content=b"shared"))

    assert left != right


def test_admission_rejects_more_than_max_items() -> None:
    registry = ResourceRegistry(max_attachments=2)
    registry.register(ResourceInput(content=b"one"))
    registry.register(ResourceInput(content=b"two"))

    with pytest.raises(ResourceAdmissionError):
        registry.register(ResourceInput(content=b"three"))


def test_admission_rejects_oversized_attachment() -> None:
    registry = ResourceRegistry(max_attachment_bytes=4)

    with pytest.raises(ResourceAdmissionError):
        registry.register(ResourceInput(content=b"too many bytes"))


def test_admission_rejects_total_bytes() -> None:
    registry = ResourceRegistry(max_attachment_bytes=8, max_total_attachment_bytes=10)
    registry.register(ResourceInput(content=b"aaaaaa"))

    with pytest.raises(ResourceAdmissionError):
        registry.register(ResourceInput(content=b"bbbbbb"))


def test_register_accepts_public_http_link() -> None:
    registry = ResourceRegistry()

    resource_id = registry.register(ResourceInput(url="http://example.com/report.txt"))

    assert resource_id.startswith("res-")
    assert registry.evidence_source(resource_id)["source_uri"] == resource_id
    (entry,) = registry.manifest()
    assert entry.filename == "report.txt"


def test_register_rejects_non_http_link() -> None:
    registry = ResourceRegistry()

    with pytest.raises(ValueError):
        registry.register(ResourceInput(url="ftp://example.com/report.txt"))


def test_discovered_links_bypass_only_the_caller_attachment_count() -> None:
    registry = ResourceRegistry(max_attachments=1)
    registry.register(ResourceInput(content=b"caller attachment"))

    discovered = registry.register_discovered_link("https://example.com/article")

    assert discovered is not None
    assert len(registry.manifest()) == 2
    with pytest.raises(ResourceAdmissionError, match="too many attachments"):
        registry.register(ResourceInput(content=b"second caller attachment"))


def test_discovered_link_deduplicates_with_a_caller_link_and_stays_inert(serve) -> None:
    fetch = serve(_Fetch())
    registry = ResourceRegistry()
    discovered = registry.register_discovered_link("https://example.com/article#section")
    assert discovered is not None
    assert registry.evidence_source(discovered)["source_uri"] == "https://example.com/article"
    caller = registry.register(
        ResourceInput(
            url="https://example.com/article#other",
            filename="preferred.html",
        )
    )

    assert discovered == caller
    (entry,) = registry.manifest()
    assert entry.filename == "preferred.html"
    assert registry.evidence_source(caller) == {
        "source_type": "web_attachment",
        "resource_kind": "web",
        "admission_origin": "caller",
        "acquisition": "",
        "source_uri": caller,
        "source_download_locator": caller,
        "title": "preferred.html",
    }
    assert fetch.calls == 0


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/article",
        "https://user:secret@example.com/article",
        "https://localhost/article",
        "http://127.0.0.1/article",
        "https://127.0.0.1/article",
    ],
)
def test_discovered_link_drops_an_unsafe_search_result(url: str) -> None:
    registry = ResourceRegistry()

    assert registry.register_discovered_link(url) is None
    assert registry.manifest() == ()


def test_discovered_link_registers_public_http() -> None:
    registry = ResourceRegistry()

    discovered = registry.register_discovered_link("http://example.com/article")

    assert discovered is not None
    assert registry.evidence_source(discovered)["source_uri"] == "http://example.com/article"


def test_manifest_reports_link_without_size_until_read() -> None:
    registry = ResourceRegistry()
    registry.register(ResourceInput(url="https://data.example.com/report.txt"))

    entries = registry.manifest()
    assert len(entries) == 1
    assert entries[0].filename == "report.txt"
    assert entries[0].source == "link"
    assert entries[0].byte_size is None


async def test_url_fetch_is_lazy(serve) -> None:
    fetch = serve(_Fetch(b"remote body"))
    registry = ResourceRegistry()
    resource_id = registry.register(ResourceInput(url="https://data.example.com/report.txt"))

    assert fetch.calls == 0

    result = await registry.read(resource_id)
    assert result.content == "remote body"
    assert fetch.calls == 1


async def test_discovered_link_has_only_per_operation_not_cumulative_web_bound(serve) -> None:
    fetch = serve(_Fetch(b"12345"))
    registry = ResourceRegistry(max_attachment_bytes=10, max_total_attachment_bytes=6)
    registry.register(ResourceInput(content=b"123456"))
    resource_id = registry.register_discovered_link("https://data.example.com/report.txt")
    assert resource_id is not None

    result = await registry.read(resource_id)

    assert result.content == "12345"
    assert registry._total_bytes == 6
    assert fetch.calls == 1


async def test_successful_url_read_keeps_one_fixed_snapshot(serve) -> None:
    fetch = serve(_Fetch(b"safe"))
    registry = ResourceRegistry()
    resource_id = registry.register(ResourceInput(url="https://data.example.com/report.txt"))

    first = await registry.read(resource_id)
    second = await registry.read(resource_id)

    assert first.content == second.content == "safe"
    assert fetch.calls == 1


async def test_read_uses_settled_bytes_without_live_dns_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reject_validation(_url: str) -> None:
        raise AssertionError("settled bytes must not re-enter the network gate")

    monkeypatch.setattr(
        "dlightrag.engine.answer.resources.registry.avalidate_public_http_url",
        reject_validation,
    )
    registry = ResourceRegistry()
    resource_id = registry.register(ResourceInput(url="https://data.example.com/report.txt"))
    registry.restore_fetched_bytes(resource_id, b"durable body")

    result = await registry.read(resource_id)

    assert result.content == "durable body"


async def test_read_returns_structural_text_window() -> None:
    registry = ResourceRegistry()
    resource_id = registry.register(
        ResourceInput(filename="notes.txt", content=b"alpha\nbeta\ngamma")
    )

    result = await registry.read(resource_id)

    assert result.resource_id == resource_id
    assert result.content == "alpha\nbeta\ngamma"
    assert result.extraction_status == "text"
    assert result.locator is not None
    assert result.locator.start == 1
    assert result.locator.end == 3
    assert result.has_more is False
    assert result.next_cursor is None
    assert result.visual_handles == ()


async def test_read_continues_above_observation_budget() -> None:
    registry = ResourceRegistry()
    text = "\n".join(f"line {index} " + "x" * 30 for index in range(2000))
    resource_id = registry.register(ResourceInput(content=text.encode("utf-8")))

    first = await registry.read(resource_id)
    assert first.has_more is True
    assert first.next_cursor is not None

    second = await registry.read(resource_id, cursor=first.next_cursor)
    combined = first.content + second.content
    while second.has_more:
        second = await registry.read(resource_id, cursor=second.next_cursor)
        combined = combined + second.content
    assert combined == text


@pytest.mark.parametrize("first_budget,next_budget", [(100, 40), (40, 100)])
async def test_cursor_is_stable_across_changing_window_budgets(
    first_budget: int,
    next_budget: int,
) -> None:
    registry = ResourceRegistry()
    text = "".join(f"line {index} " + "x" * 30 + "\n" for index in range(400))
    resource_id = registry.register(ResourceInput(content=text.encode("utf-8")))

    current = await registry.read(resource_id, max_window_tokens=first_budget)
    chunks = [current.content]
    while current.has_more:
        current = await registry.read(
            resource_id,
            cursor=current.next_cursor,
            max_window_tokens=next_budget,
        )
        chunks.append(current.content)

    assert "".join(chunks) == text


async def test_cursor_pages_do_not_rebuild_whole_resource_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from dlightrag.engine.answer.resources import registry as registry_module

    input_lengths: list[int] = []
    span_threads: list[int] = []
    loop_thread = threading.get_ident()
    build = registry_module.build_text_windows
    read_cursor_span = registry_module._read_cursor_span

    def count_builds(text: str, *, max_window_tokens: int):
        input_lengths.append(len(text))
        return build(text, max_window_tokens=max_window_tokens)

    def record_span(*args, **kwargs):
        span_threads.append(threading.get_ident())
        return read_cursor_span(*args, **kwargs)

    monkeypatch.setattr(registry_module, "build_text_windows", count_builds)
    monkeypatch.setattr(registry_module, "_read_cursor_span", record_span)
    registry = ResourceRegistry()
    text = "".join(f"line {index} " + "x" * 30 + "\n" for index in range(400))
    resource_id = registry.register(ResourceInput(content=text.encode("utf-8")))

    current = await registry.read(resource_id, max_window_tokens=100)
    for _ in range(3):
        assert current.next_cursor is not None
        current = await registry.read(
            resource_id,
            cursor=current.next_cursor,
            max_window_tokens=40,
        )

    assert input_lengths.count(len(text)) == 1
    assert all(length < len(text) for length in input_lengths[1:])
    assert span_threads and all(thread_id != loop_thread for thread_id in span_threads)


async def test_read_continues_within_single_oversized_line() -> None:
    from dlightrag.engine.ai.tokens import estimate_tokens

    registry = ResourceRegistry()
    # A minified single-line JSON payload with no newline, far over one budget.
    payload = '{"data":[' + ",".join(f'"{"v" * 40}"' for _ in range(4000)) + "]}"
    assert "\n" not in payload
    assert estimate_tokens(payload) > 100
    resource_id = registry.register(ResourceInput(content=payload.encode("utf-8")))

    first = await registry.read(resource_id)
    assert estimate_tokens(first.content) <= 100
    assert first.has_more is True
    assert first.next_cursor is not None

    combined = first.content
    current = first
    while current.has_more:
        current = await registry.read(resource_id, cursor=current.next_cursor)
        assert estimate_tokens(current.content) <= 100
        combined = combined + current.content
    assert combined == payload


async def test_cursor_is_bound_to_its_resource() -> None:
    registry = ResourceRegistry()
    big = "\n".join(f"line {index} " + "x" * 30 for index in range(2000))
    big_id = registry.register(ResourceInput(content=big.encode("utf-8")))
    small_id = registry.register(ResourceInput(content=b"tiny"))

    first = await registry.read(big_id)
    assert first.next_cursor is not None

    with pytest.raises(ResourceCursorError):
        await registry.read(small_id, cursor=first.next_cursor)


async def test_cursor_inherits_focus_and_rejects_a_conflict() -> None:
    registry = ResourceRegistry()
    text = "\n".join(f"line {index} " + "x" * 30 for index in range(2000))
    resource_id = registry.register(ResourceInput(content=text.encode("utf-8")))

    first = await registry.read(resource_id, focus="line 1999")
    assert first.next_cursor is not None
    second = await registry.read(resource_id, cursor=first.next_cursor)
    assert second.content

    with pytest.raises(ResourceCursorError):
        await registry.read(resource_id, focus="different", cursor=first.next_cursor)


async def test_focused_cursor_order_survives_a_smaller_window_budget() -> None:
    registry = ResourceRegistry()
    lines = [f"record {index:03d} value\n" for index in range(200)]
    lines[150] = "record 150 unique-needle\n"
    text = "".join(lines)
    resource_id = registry.register(ResourceInput(content=text.encode("utf-8")))

    current = await registry.read(
        resource_id,
        focus="unique-needle",
        max_window_tokens=100,
    )
    assert "unique-needle" in current.content
    chunks = [current.content]
    while current.has_more:
        current = await registry.read(
            resource_id,
            cursor=current.next_cursor,
            max_window_tokens=40,
        )
        chunks.append(current.content)

    assert sorted("".join(chunks).splitlines(keepends=True)) == sorted(lines)


async def test_cursor_is_isolated_across_registries() -> None:
    big = "\n".join(f"line {index} " + "x" * 30 for index in range(2000)).encode("utf-8")
    left = ResourceRegistry()
    right = ResourceRegistry()
    left_id = left.register(ResourceInput(content=big))
    right_id = right.register(ResourceInput(content=big))

    first = await left.read(left_id)
    assert first.next_cursor is not None

    with pytest.raises(ResourceCursorError):
        await right.read(right_id, cursor=first.next_cursor)


async def test_read_unknown_resource_raises() -> None:
    registry = ResourceRegistry()
    with pytest.raises(ResourceNotFoundError):
        await registry.read("res-does-not-exist")


async def test_cancellation_during_fetch_propagates_and_cleans_up(serve) -> None:
    serve(_Fetch(fail=asyncio.CancelledError))
    registry = ResourceRegistry()
    resource_id = registry.register(ResourceInput(url="https://data.example.com/report.txt"))

    with pytest.raises(asyncio.CancelledError):
        await registry.read(resource_id)

    await registry.aclose()


async def test_cancelled_waiter_does_not_cancel_shared_loader() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def loader() -> bytes:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return b"shared text"

    registry = ResourceRegistry()
    resource_id = registry.register(ResourceInput(loader=loader))
    cancelled_waiter = asyncio.create_task(registry.read(resource_id))
    surviving_waiter = asyncio.create_task(registry.read(resource_id))
    await started.wait()

    try:
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        release.set()
        result = await surviving_waiter

        assert result.content == "shared text"
        assert calls == 1
    finally:
        release.set()
        await asyncio.gather(cancelled_waiter, surviving_waiter, return_exceptions=True)
        await registry.aclose()


async def test_aclose_cancels_and_joins_pending_loader() -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def loader() -> bytes:
        started.set()
        try:
            await asyncio.Event().wait()
            return b"unreachable"
        finally:
            stopped.set()

    registry = ResourceRegistry()
    resource_id = registry.register(ResourceInput(loader=loader))
    read_task = asyncio.create_task(registry.read(resource_id))
    await started.wait()

    try:
        await registry.aclose()

        assert stopped.is_set()
        assert read_task.done()
        with pytest.raises(asyncio.CancelledError):
            await read_task
    finally:
        read_task.cancel()
        await asyncio.gather(read_task, return_exceptions=True)


# ---------------------------------------------------------------------------
# URL extraction fallback and Web-specific operation bounds
# ---------------------------------------------------------------------------


class _CountingFallback:
    def __init__(self, text: str | None, *, provider: str = "exa") -> None:
        self.text = text
        self.provider = provider
        self.calls = 0
        self.urls: list[str] = []

    async def __call__(self, url: str) -> WebExtractResult:
        self.calls += 1
        self.urls.append(url)
        if self.text is None:
            raise WebSourceUnavailable("all", "extract", "empty")
        acquisition = "exa_extract" if self.provider == "exa" else "tavily_extract"
        return WebExtractResult(
            url=url,
            text=self.text,
            provider=self.provider,
            acquisition=acquisition,  # type: ignore[arg-type]
        )


async def test_redirect_final_url_becomes_citable_identity_and_alias(serve) -> None:
    serve(_Fetch(b"final body", final_url="https://final.example/report"))
    admitted = []

    async def persist(fetched, _owner) -> None:
        admitted.append(fetched)

    identity_secret = b"r" * 32
    registry = ResourceRegistry(fetched_bytes_sink=persist, resource_secret=identity_secret)
    final_id = registry.register_agent_url("https://final.example/report")
    requested_id = registry.register_agent_url("https://start.example/report")
    assert requested_id != final_id

    result = await registry.read(requested_id)

    assert result.resource_id == final_id
    assert result.content == "final body"
    assert registry.evidence_source(requested_id)["source_uri"] == "https://final.example/report"
    assert registry.register_agent_url("https://start.example/report") == final_id
    assert registry.register_agent_url("https://final.example/report") == final_id
    assert admitted[0].aliases == (requested_id,)

    recovered = ResourceRegistry(resource_secret=identity_secret)
    recovered.restore_fetched_resource(
        resource_id=final_id,
        ordinal=admitted[0].ordinal,
        filename=admitted[0].filename,
        mime_type=admitted[0].mime_type,
        url=admitted[0].url,
        content=admitted[0].content,
        admission_origin="agent",
        acquisition="direct_http",
        aliases=admitted[0].aliases,
    )
    assert recovered.evidence_source(requested_id)["source_uri"] == admitted[0].url
    assert recovered.register_agent_url("https://start.example/report") == final_id
    recovered.restore_discovered_resources(
        {
            "chunks": [
                {
                    "metadata": {
                        "admission_origin": "search",
                        "source_uri": "https://start.example/report",
                        "resource_id": requested_id,
                    }
                }
            ]
        }
    )


async def test_redirect_recovery_preserves_final_provenance_without_predeclared_final(
    serve,
) -> None:
    serve(_Fetch(b"final body", final_url="https://final.example/report"))
    admitted = []

    async def persist(fetched, _owner) -> None:
        admitted.append(fetched)

    identity_secret = b"r" * 32
    registry = ResourceRegistry(fetched_bytes_sink=persist, resource_secret=identity_secret)
    requested_id = registry.register_agent_url("https://start.example/report")
    await registry.read(requested_id)

    recovered = ResourceRegistry(resource_secret=identity_secret)
    recovered.restore_fetched_resource(
        resource_id=requested_id,
        ordinal=admitted[0].ordinal,
        filename=admitted[0].filename,
        mime_type=admitted[0].mime_type,
        url=admitted[0].url,
        content=admitted[0].content,
        admission_origin="agent",
        acquisition="direct_http",
    )

    assert recovered.register_agent_url("https://start.example/report") == requested_id
    assert recovered.evidence_source(requested_id)["source_uri"] == ("https://final.example/report")


async def test_failed_agent_read_can_retry_with_new_presentation_headers(serve) -> None:
    fetch = serve(_Fetch(fail=RuntimeError))
    registry = ResourceRegistry()
    resource_id = registry.register_agent_url(
        "https://data.example.com/report.txt",
        presentation=PublicHttpPresentation(user_agent="First/1"),
    )

    first = await registry.read(resource_id)
    assert first.evidence_available is False
    fetch.fail = None
    assert (
        registry.register_agent_url(
            "https://data.example.com/report.txt",
            presentation=PublicHttpPresentation(user_agent="Second/2"),
        )
        == resource_id
    )

    second = await registry.read(resource_id)

    assert second.content == "hello\nworld"
    assert [presentation.user_agent for presentation in fetch.presentations] == [
        "First/1",
        "Second/2",
    ]


async def test_a_redirect_and_a_direct_read_of_its_page_share_one_representation(
    serve,
) -> None:
    """The first bytes bound to a page win, so two concurrent reads cannot disagree.

    The page renders differently on every fetch. The direct read's fetch is still
    streaming when the redirect binds its own render to the same Resource; letting
    the direct fetch overwrite it made one read refuse the page and every later read
    of it fail.
    """
    old, new = "https://site.example/old", "https://site.example/new"
    direct_started, release = asyncio.Event(), asyncio.Event()
    renders = 0

    async def render(url: str, **_kwargs: object) -> PublicHttpFetch:
        # Every fetch renders the page anew; the old address redirects to it.
        nonlocal renders
        renders += 1
        if renders == 1:
            # The direct read's render arrives only once released.
            direct_started.set()
            await release.wait()
        return PublicHttpFetch(f"render {renders}".encode(), new, None, 200)

    serve(render)

    async def persist(_fetched, _owner) -> None:
        return None

    def owner() -> ResourceEffectOwner:
        return ResourceEffectOwner(execution_scope="session", intent_id=IntentId.new())

    registry = ResourceRegistry(fetched_bytes_sink=persist)
    page = registry.register_agent_url(new)
    redirected = registry.register_agent_url(old)
    direct = asyncio.create_task(registry.read(page, effect_owner=owner()))
    await direct_started.wait()

    via_redirect = await registry.read(redirected, effect_owner=owner())
    release.set()
    directly = await direct

    assert (via_redirect.resource_id, directly.resource_id) == (page, page)
    assert via_redirect.content == directly.content == "render 2"
    assert (await registry.read(page, effect_owner=owner())).content == "render 2"


async def test_a_fetch_in_flight_keeps_the_presentation_it_started_with(serve) -> None:
    """A later read of the same page cannot change the headers of a fetch under way.

    Once a fetch has started, a read without headers shares it, and a read asking for
    other headers is refused, exactly as both are once that fetch has finished.
    """
    url = "https://data.example.com/report.txt"
    german = PublicHttpPresentation(accept_language="de")

    fetch = serve(_Fetch())
    registry = ResourceRegistry()
    resource_id = registry.register_agent_url(url, presentation=german)
    reading = asyncio.create_task(registry.read(resource_id))
    await asyncio.sleep(0)  # the read has started its fetch, which has not run yet
    assert registry.register_agent_url(url) == resource_id
    await reading
    assert [presentation.accept_language for presentation in fetch.presentations] == ["de"]

    fetch = serve(_Fetch())
    registry = ResourceRegistry()
    resource_id = registry.register_agent_url(url)
    reading = asyncio.create_task(registry.read(resource_id))
    await asyncio.sleep(0)
    with pytest.raises(ResourceAdmissionError, match="cannot replace"):
        registry.register_agent_url(url, presentation=german)
    await reading
    assert [presentation.accept_language for presentation in fetch.presentations] == [None]


async def test_direct_success_skips_url_text_fallback(serve) -> None:
    fallback = _CountingFallback("EXTRACTED TEXT")
    serve(_Fetch(b"good body"))
    registry = ResourceRegistry(extract_chain=(HostedExtract(fallback),))
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    result = await registry.read(resource_id)

    assert result.content == "good body"
    assert result.evidence_available is True
    assert registry.evidence_source(resource_id)["acquisition"] == "direct_http"
    assert fallback.calls == 0


async def test_direct_decode_failure_uses_one_extract_fallback(serve) -> None:
    fallback = _CountingFallback("recovered text\nsecond line", provider="tavily")
    serve(_Fetch(b"\x00\x01\x02\x03binary\x00\x00"))
    registry = ResourceRegistry(extract_chain=(HostedExtract(fallback),))
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    result = await registry.read(resource_id)
    again = await registry.read(resource_id)

    assert "recovered text" in result.content
    assert again.content == result.content
    assert fallback.calls == 1
    # The bytes stay what the Resource holds; the text a read cites came from Extract.
    assert registry.evidence_source(resource_id)["acquisition"] == "direct_http"
    assert registry.evidence_source(resource_id, text=True)["acquisition"] == "tavily_extract"


async def test_binary_direct_snapshot_is_retained_without_hosted_substitution(serve) -> None:
    admitted = []

    async def persist(fetched, _owner) -> None:
        admitted.append(fetched)

    fallback = _CountingFallback("provider replacement")
    content = b"\x00\x01\x02\x03binary\x00\x00"
    serve(_Fetch(content))
    registry = ResourceRegistry(
        extract_chain=(HostedExtract(fallback),), fetched_bytes_sink=persist
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.bin")

    with pytest.raises(ResourceDecodeError):
        await registry.read(resource_id)

    assert fallback.calls == 0
    assert len(admitted) == 1
    assert admitted[0].content == content
    assert admitted[0].acquisition == "direct_http"


async def test_shared_extract_snapshot_is_admitted_for_each_effect_owner(serve) -> None:
    owners = []

    async def persist(_fetched, owner) -> None:
        owners.append(owner)

    fallback = _CountingFallback("shared provider text")
    serve(_Fetch(b""))
    registry = ResourceRegistry(
        extract_chain=(HostedExtract(fallback),), fetched_bytes_sink=persist
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")
    first = ResourceEffectOwner("session-a", IntentId.new())
    second = ResourceEffectOwner("session-b", IntentId.new())

    await asyncio.gather(
        registry.read(resource_id, effect_owner=first),
        registry.read(resource_id, effect_owner=second),
    )

    assert fallback.calls == 1
    assert set(owners) == {first, second}


async def test_a_read_that_decodes_late_keeps_the_snapshot_another_read_admitted(
    serve, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two reads share one empty fetch, and one decodes it after the other's Extract.

    A read whose direct bytes decode to nothing forgets them and hands the URL to
    the shared Extract. Forgetting them late dropped the snapshot the Extract had
    admitted in their place, so neither read made it durable.
    """
    from dlightrag.engine.answer.resources import registry as registry_module

    decode = registry_module.decode_text
    extracted = threading.Event()
    lock = threading.Lock()
    decodes = 0
    waited: list[bool] = []

    def decode_the_second_after_the_extract(content: bytes, **kwargs: Any) -> str:
        nonlocal decodes
        with lock:
            decodes += 1
            late = decodes == 2
        if late:
            # Recorded, not asserted: the read swallows a decoder's own exception.
            waited.append(extracted.wait(5))
        return decode(content, **kwargs)

    class _Extract(_CountingFallback):
        async def __call__(self, url: str) -> WebExtractResult:
            extracted.set()
            return await super().__call__(url)

    monkeypatch.setattr(registry_module, "decode_text", decode_the_second_after_the_extract)
    owners = []

    async def persist(_fetched, owner) -> None:
        owners.append(owner)

    fallback = _Extract("shared provider text")
    serve(_Fetch(b""))
    registry = ResourceRegistry(
        extract_chain=(HostedExtract(fallback),), fetched_bytes_sink=persist
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")
    first = ResourceEffectOwner("session-a", IntentId.new())
    second = ResourceEffectOwner("session-b", IntentId.new())

    results = await asyncio.gather(
        registry.read(resource_id, effect_owner=first),
        registry.read(resource_id, effect_owner=second),
    )

    assert waited == [True], "the second read decoded after the Extract"
    assert [result.content for result in results] == ["shared provider text"] * 2
    assert fallback.calls == 1
    assert set(owners) == {first, second}


async def test_a_failed_fetch_leaves_the_extract_snapshot_another_read_admitted(serve) -> None:
    """A read whose own fetch failed has no bytes of its own to drop.

    The first read's fetch is refused and its Extract admits the page text. A second
    read began fetching while that Extract ran, and its fetch fails just after: it
    must answer with the admitted text and settle it for itself, as must every later
    read of the Run.
    """
    extract_started, second_fetching, extract_returned = (asyncio.Event() for _ in range(3))
    fetches = 0

    async def fetch(url: str, **_kwargs: object) -> PublicHttpFetch:
        nonlocal fetches
        fetches += 1
        if fetches == 2:
            second_fetching.set()
            await extract_returned.wait()
        raise RuntimeError("HTTP 403")

    class _Extract(_CountingFallback):
        async def __call__(self, url: str) -> WebExtractResult:
            extract_started.set()
            await second_fetching.wait()
            result = await super().__call__(url)
            extract_returned.set()
            return result

    owners = []

    async def persist(_fetched, owner) -> None:
        owners.append(owner)

    serve(fetch)
    fallback = _Extract("extracted page text")
    registry = ResourceRegistry(
        extract_chain=(HostedExtract(fallback),), fetched_bytes_sink=persist
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")
    first, second, third = (ResourceEffectOwner(name, IntentId.new()) for name in "abc")

    first_read = asyncio.create_task(registry.read(resource_id, effect_owner=first))
    await extract_started.wait()
    second_read = asyncio.create_task(registry.read(resource_id, effect_owner=second))
    results = [await first_read, await second_read]
    results.append(await registry.read(resource_id, effect_owner=third))

    assert [result.content for result in results] == ["extracted page text"] * 3
    assert (fetches, fallback.calls) == (2, 1)
    assert set(owners) == {first, second, third}


async def test_an_extract_text_view_settles_beside_the_bytes_it_reads(serve) -> None:
    """The fetched bytes stay the one representation; Extract text is their view.

    The bytes settle through the sink, the view with the read that took it, and
    recovery restores both, so the resumed read returns the same text unextracted.
    """
    admitted = []

    async def persist(fetched, _owner) -> None:
        admitted.append(fetched)

    fallback = _CountingFallback("  provider body text\n")
    serve(_Fetch(b""))
    registry = ResourceRegistry(
        extract_chain=(HostedExtract(fallback),), fetched_bytes_sink=persist
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.html")

    result = await registry.read(resource_id)

    assert result.content == "  provider body text\n"
    assert [(fetched.content, fetched.acquisition) for fetched in admitted] == [
        (b"", "direct_http")
    ]
    view = registry.conversion_effects(resource_id)

    recovered = ResourceRegistry()
    (fetched,) = admitted
    recovered.restore_fetched_resource(
        resource_id=resource_id,
        ordinal=fetched.ordinal,
        filename=fetched.filename,
        mime_type=fetched.mime_type,
        url=fetched.url,
        content=fetched.content,
        admission_origin="agent",
        acquisition=fetched.acquisition,
    )
    stored = {row.resource_id: row.content for row in view}
    (snapshot,) = [row for row in view if row.resource_kind == "conversion_snapshot"]
    recovered.adopt_conversion_snapshot(ConversionSnapshot.restore(snapshot.content, stored))
    assert (await recovered.read(resource_id)).content == "  provider body text\n"
    assert fallback.calls == 1


async def test_direct_empty_triggers_extract_fallback(serve) -> None:
    fallback = _CountingFallback("provider body text")
    serve(_Fetch(b""))
    registry = ResourceRegistry(extract_chain=(HostedExtract(fallback),))
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    result = await registry.read(resource_id)

    assert result.content == "provider body text"
    assert fallback.calls == 1


async def test_invalid_private_url_never_calls_extract_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dlightrag.engine.network_admission.socket.getaddrinfo",
        lambda host, port, *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port))
        ],
    )
    fallback = _CountingFallback("should never appear")
    # The real transport refuses the private address before any connection.
    registry = ResourceRegistry(extract_chain=(HostedExtract(fallback),))
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    with pytest.raises(ValueError):
        await registry.read(resource_id)
    assert fallback.calls == 0


async def test_exhausted_extract_returns_no_evidence_and_does_not_pin_failure(serve) -> None:
    fallback = _CountingFallback(None)
    fetch = serve(_Fetch(b"\x00\x01\x02\x03binary\x00\x00"))
    registry = ResourceRegistry(extract_chain=(HostedExtract(fallback),))
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    result = await registry.read(resource_id)
    again = await registry.read(resource_id)

    assert result.extraction_status == "unavailable"
    assert result.evidence_available is False
    assert "produced no citable text" in result.content
    assert again.content == result.content
    assert fallback.calls == 2
    assert fetch.calls == 2


async def test_fallback_text_windows_are_cursor_paginated(serve) -> None:
    big = "\n".join(f"line {index} " + "x" * 30 for index in range(2000))
    fallback = _CountingFallback(big)
    serve(_Fetch(b""))
    registry = ResourceRegistry(extract_chain=(HostedExtract(fallback),))
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    current = await registry.read(resource_id)
    combined = current.content
    while current.has_more:
        current = await registry.read(resource_id, cursor=current.next_cursor)
        combined += current.content

    assert combined == big
    assert fallback.calls == 1


async def test_web_reads_do_not_consume_attachment_cumulative_budget(serve) -> None:
    serve(_Fetch(b"0123456789"))
    registry = ResourceRegistry(max_attachment_bytes=100, max_total_attachment_bytes=8)
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    result = await registry.read(resource_id)

    assert result.content == "0123456789"
    assert registry._total_bytes == 0


async def test_loader_bytes_still_use_attachment_cumulative_budget() -> None:
    async def left() -> bytes:
        return b"12345678"

    async def right() -> bytes:
        return b"12345678"

    registry = ResourceRegistry(max_attachment_bytes=100, max_total_attachment_bytes=12)
    left_id = registry.register(ResourceInput(loader=left))
    right_id = registry.register(ResourceInput(loader=right))

    assert (await registry.read(left_id)).content == "12345678"
    with pytest.raises(ResourceAdmissionError):
        await registry.read(right_id)


async def test_concurrent_reads_same_link_share_one_fixed_fetch(serve) -> None:
    fetch = serve(_Fetch(b"0123456789"))
    registry = ResourceRegistry()
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    results = await asyncio.gather(*(registry.read(resource_id) for _ in range(3)))

    assert all(result.content == "0123456789" for result in results)
    assert fetch.calls == 1


async def test_durable_representation_and_cursor_survive_registry_recovery() -> None:
    identity_secret = b"i" * 32
    cursor_secret = b"c" * 32
    text = "\n".join(f"line {index} " + "x" * 30 for index in range(300))
    first = ResourceRegistry(
        resource_secret=identity_secret,
        cursor_secret=cursor_secret,
    )
    resource_id = first.register_agent_url("https://example.com/article")
    first.restore_fetched_resource(
        resource_id=resource_id,
        ordinal=0,
        filename="article.html",
        mime_type="text/plain",
        url="https://example.com/article",
        content=text.encode(),
        admission_origin="agent",
        acquisition="direct_http",
    )
    page = await first.read(resource_id, max_window_tokens=100)
    assert page.next_cursor is not None

    recovered = ResourceRegistry(
        resource_secret=identity_secret,
        cursor_secret=cursor_secret,
    )
    recovered.restore_fetched_resource(
        resource_id=resource_id,
        ordinal=0,
        filename="article.html",
        mime_type="text/plain",
        url="https://example.com/article",
        content=text.encode(),
        admission_origin="agent",
        acquisition="direct_http",
    )

    continued = await recovered.read(
        resource_id,
        cursor=page.next_cursor,
        max_window_tokens=100,
    )

    assert continued.content
    assert continued.content not in page.content


async def test_inline_read_does_not_double_count() -> None:
    registry = ResourceRegistry(max_attachment_bytes=100, max_total_attachment_bytes=100)
    resource_id = registry.register(ResourceInput(content=b"inline bytes"))
    before = registry._total_bytes

    await registry.read(resource_id)
    await registry.read(resource_id)

    assert registry._total_bytes == before


async def test_text_decode_windowing_and_focus_ranking_run_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from dlightrag.engine.answer.resources import registry as registry_module

    loop_thread = threading.get_ident()
    worker_threads: list[int] = []

    def record(real):
        def wrapper(*args: object, **kwargs: object):
            worker_threads.append(threading.get_ident())
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(registry_module, "decode_text", record(registry_module.decode_text))
    monkeypatch.setattr(
        registry_module, "build_text_windows", record(registry_module.build_text_windows)
    )
    monkeypatch.setattr(registry_module, "bm25_rank", record(registry_module.bm25_rank))

    registry = ResourceRegistry()
    text = "\n".join(f"line {index} " + "x" * 30 for index in range(2000))
    resource_id = registry.register(
        ResourceInput(filename="notes.txt", content=text.encode("utf-8"))
    )

    result = await registry.read(resource_id, focus="line 1999")

    assert result.content
    assert len(worker_threads) >= 3
    assert loop_thread not in worker_threads
    await registry.aclose()


# -- Rendered Reads (ADR 0032) --------------------------------------------------------------

_SPA = "https://spa.example.com/app.html"
_SHELL = b"<html><body><div id='app'></div><script>render()</script></body></html>"
_QUOTES = (
    "<html><body><h1>Quotes</h1>"
    "<p>A day without sunshine is like, you know, night. Albert Einstein</p>"
    "<p>It is our choices that show what we truly are. J.K. Rowling</p></body></html>"
)


def _long_page(label: str, lines: int = 400) -> str:
    return (
        "<html><body>"
        + "".join(f"<p>{label} line {n} " + "x" * 30 + "</p>" for n in range(lines))
        + "</body></html>"
    )


async def test_a_fresh_url_renders_once_without_a_direct_fetch_and_is_reused(serve) -> None:
    fetch = serve(_Fetch(b"never fetched"))
    renderer = RecordingRenderer({_SPA: _QUOTES})
    registry = ResourceRegistry(page_renderer=renderer)
    resource_id = registry.register_agent_url(_SPA)

    first = await registry.read(resource_id, rendered=True, max_window_tokens=2000)
    again = await registry.read(resource_id, rendered=True, max_window_tokens=2000)
    plain = await registry.read(resource_id, max_window_tokens=2000)

    assert "Albert Einstein" in first.content and "J.K. Rowling" in first.content
    assert (first.rendered, again.rendered, plain.rendered) == (True, True, True)
    assert again.content == first.content == plain.content
    assert (renderer.calls, fetch.calls) == ([_SPA], 0)
    # Results print the Resource's own handle, and the header says it is the rendering.
    assert first.resource_id == resource_id
    assert format_resource_read(first).startswith(f"[resource: {resource_id} | rendered | lines ")
    source = registry.evidence_source(resource_id, text=True, rendered=True)
    assert (source["acquisition"], source["source_uri"]) == ("browser_render", _SPA)
    assert registry.evidence_source(resource_id, text=True)["acquisition"] == ""


async def test_a_rendered_read_says_whose_view_it_is_and_where_the_page_ended(serve) -> None:
    moved_to = "https://spa.example.com/done"
    stays = "https://spa.example.com/stays.html"
    renderer = RecordingRenderer(
        {
            _SPA: RenderedPage(_SPA, moved_to, _long_page("rendered").encode(), 200),
            stays: _QUOTES,
        }
    )
    serve(_Fetch(b"direct"))
    registry = ResourceRegistry(page_renderer=renderer)
    moved = registry.register_agent_url(_SPA)
    stayed = registry.register_agent_url(stays)
    plain = registry.register_agent_url("https://spa.example.com/plain.html")

    first = await registry.read(moved, rendered=True, max_window_tokens=400)
    assert first.next_cursor is not None
    later = await registry.read(moved, cursor=first.next_cursor, max_window_tokens=400)
    unmoved = await registry.read(stayed, rendered=True, max_window_tokens=2000)
    direct = await registry.read(plain, max_window_tokens=2000)

    view = "Rendered view from the Agent Browser (browser_render)"
    # The note opens every page of the rendering, and names where the page ended only when
    # that is not the Resource's own URL.
    for page in (first, later):
        assert page.note is not None and page.note.startswith(
            f"{view}; the page ended at {moved_to}."
        )
    assert unmoved.note is not None and unmoved.note.startswith(f"{view}.")
    assert "the page ended" not in unmoved.note
    assert f"[{first.note}]" in format_resource_read(first)
    assert direct.note is None or "Rendered view" not in direct.note


async def test_a_rendering_is_appended_to_the_direct_snapshot_and_each_cursor_names_its_own(
    serve,
) -> None:
    direct_text = "\n".join(f"direct line {n} " + "x" * 30 for n in range(400))
    serve(_Fetch(direct_text.encode()))
    renderer = RecordingRenderer({"https://data.example.com/report.html": _long_page("rendered")})
    registry = ResourceRegistry(page_renderer=renderer)
    resource_id = registry.register_agent_url("https://data.example.com/report.html")

    window = 400
    direct = await registry.read(resource_id, max_window_tokens=window)
    rendered = await registry.read(resource_id, rendered=True, max_window_tokens=window)
    plain = await registry.read(resource_id, max_window_tokens=window)

    # The snapshot is never replaced: a plain read still returns what the URL served.
    assert not direct.rendered and not plain.rendered
    assert plain.content == direct.content and "direct line" in plain.content
    assert rendered.rendered and "rendered line" in rendered.content
    assert direct.next_cursor is not None and rendered.next_cursor is not None
    assert rendered.next_cursor.startswith("r.") and not direct.next_cursor.startswith("r.")

    # A cursor alone selects the representation it continues, flagged or not.
    continued = await registry.read(
        resource_id, cursor=rendered.next_cursor, max_window_tokens=window
    )
    flagged = await registry.read(
        resource_id, cursor=rendered.next_cursor, rendered=True, max_window_tokens=window
    )
    assert continued.rendered and "rendered line" in continued.content
    assert flagged.content == continued.content
    still_direct = await registry.read(
        resource_id, cursor=direct.next_cursor, max_window_tokens=window
    )
    assert not still_direct.rendered and "direct line" in still_direct.content
    with pytest.raises(ResourceCursorError, match="continues the direct representation"):
        await registry.read(
            resource_id, cursor=direct.next_cursor, rendered=True, max_window_tokens=window
        )


async def test_a_cursor_is_bound_to_its_representation_and_its_resource(serve) -> None:
    serve(_Fetch(b"direct"))
    renderer = RecordingRenderer({_SPA: _long_page("page")})
    registry = ResourceRegistry(page_renderer=renderer)
    resource_id = registry.register_agent_url(_SPA)
    other = registry.register_agent_url("https://spa.example.com/other.html")
    first = await registry.read(resource_id, rendered=True, max_window_tokens=400)
    assert first.next_cursor is not None

    # The token without its prefix is not a direct cursor, and it names no other Resource.
    with pytest.raises(ResourceCursorError):
        await registry.read(resource_id, cursor=first.next_cursor.removeprefix("r."))
    with pytest.raises(ResourceCursorError):
        await registry.read(other, cursor=first.next_cursor)


async def test_the_automatic_chain_asks_hosted_providers_before_the_browser(serve) -> None:
    hosted = _CountingFallback(None)
    renderer = RecordingRenderer({_SPA: _QUOTES})
    fetch = serve(_Fetch(fail=RuntimeError))
    registry = ResourceRegistry(
        extract_chain=(HostedExtract(hosted), AgentBrowserRender()), page_renderer=renderer
    )
    resource_id = registry.register_agent_url(_SPA)

    result = await registry.read(resource_id, max_window_tokens=2000)

    assert (hosted.calls, renderer.calls, fetch.calls) == (1, [_SPA], 1)
    assert result.rendered and "Albert Einstein" in result.content
    # The page is read as it was rendered: nothing is bound to the Resource as a snapshot.
    assert registry.evidence_source(resource_id)["acquisition"] == ""


async def test_the_browser_runs_only_when_nothing_before_it_in_the_chain_gave_text(serve) -> None:
    hosted = _CountingFallback("hosted text")
    renderer = RecordingRenderer({_SPA: _QUOTES})
    serve(_Fetch(fail=RuntimeError))
    registry = ResourceRegistry(
        extract_chain=(HostedExtract(hosted), AgentBrowserRender()), page_renderer=renderer
    )
    resource_id = registry.register_agent_url(_SPA)

    result = await registry.read(resource_id)

    assert (result.content, result.rendered, renderer.calls) == ("hosted text", False, [])


async def test_a_chain_that_names_the_browser_first_renders_before_asking_hosted_providers(
    serve,
) -> None:
    hosted = _CountingFallback("hosted text")
    renderer = RecordingRenderer({_SPA: _QUOTES})
    serve(_Fetch(fail=RuntimeError))
    registry = ResourceRegistry(
        extract_chain=(AgentBrowserRender(), HostedExtract(hosted)), page_renderer=renderer
    )
    resource_id = registry.register_agent_url(_SPA)

    result = await registry.read(resource_id, max_window_tokens=2000)

    assert result.rendered and (renderer.calls, hosted.calls) == ([_SPA], 0)


async def test_an_empty_javascript_shell_keeps_its_bytes_and_reads_as_its_rendering(serve) -> None:
    admitted = []

    async def persist(fetched, _owner) -> None:
        admitted.append(fetched)

    renderer = RecordingRenderer({_SPA: _QUOTES})
    hosted = _CountingFallback(None)
    fetch = serve(_Fetch(_SHELL))
    registry = ResourceRegistry(
        extract_chain=(HostedExtract(hosted), AgentBrowserRender()),
        page_renderer=renderer,
        fetched_bytes_sink=persist,
    )
    resource_id = registry.register_agent_url(_SPA)

    first = await registry.read(resource_id, max_window_tokens=2000)
    again = await registry.read(resource_id, max_window_tokens=2000)

    assert first.rendered and again.rendered and "Albert Einstein" in again.content
    # The shell is the Resource's snapshot, and it settles beside the rendering.
    assert [(f.content, f.acquisition) for f in admitted] == [(_SHELL, "direct_http")]
    effects = registry.rendered_effects(resource_id)
    assert [effect.resource_kind for effect in effects] == ["web_render", "conversion_snapshot"]
    assert effects[0].resource_id == f"{resource_id}-rendered"
    assert effects[0].source_locator == resource_id
    # The walk is made once; the shell is not fetched, converted, or sent to a provider again.
    assert (fetch.calls, hosted.calls, renderer.calls) == (1, 1, [_SPA])


async def test_after_an_explicit_render_a_plain_read_fetches_nothing_and_asks_no_provider(
    serve,
) -> None:
    hosted = _CountingFallback("never asked")
    renderer = RecordingRenderer({_SPA: _QUOTES})
    fetch = serve(_Fetch(b"never fetched"))
    registry = ResourceRegistry(
        extract_chain=(HostedExtract(hosted), AgentBrowserRender()), page_renderer=renderer
    )
    resource_id = registry.register_agent_url(_SPA)
    await registry.read(resource_id, rendered=True, max_window_tokens=2000)

    plain = await registry.read(resource_id, max_window_tokens=2000)

    assert plain.rendered and "Albert Einstein" in plain.content
    assert (fetch.calls, hosted.calls, renderer.calls) == (0, 0, [_SPA])


async def test_a_busy_browser_and_an_empty_page_are_not_pinned(serve) -> None:
    empty = "<html><body><script>nothing()</script></body></html>"
    renderer = RecordingRenderer(
        {_SPA: [browser_failure("busy"), empty, _QUOTES], "https://spa.example.com/b.html": empty}
    )
    serve(_Fetch(fail=RuntimeError))
    registry = ResourceRegistry(extract_chain=(AgentBrowserRender(),), page_renderer=renderer)
    resource_id = registry.register_agent_url(_SPA)

    with pytest.raises(AgentBrowserError) as busy:
        await registry.read(resource_id, rendered=True)
    with pytest.raises(AgentBrowserError) as no_text:
        await registry.read(resource_id, rendered=True)
    # Nothing of the empty rendering is left behind.
    assert (busy.value.reason, no_text.value.reason) == ("busy", "no_text")
    assert registry.rendered_effects(resource_id) == ()
    assert not registry.holds_rendered(f"{resource_id}-rendered")

    result = await registry.read(resource_id, rendered=True, max_window_tokens=2000)
    assert result.rendered and "Albert Einstein" in result.content
    assert len(renderer.calls) == 3


async def test_an_automatic_render_that_fails_never_raises_and_says_why(serve) -> None:
    renderer = RecordingRenderer({_SPA: browser_failure("busy")})
    serve(_Fetch(fail=RuntimeError))
    registry = ResourceRegistry(extract_chain=(AgentBrowserRender(),), page_renderer=renderer)
    resource_id = registry.register_agent_url(_SPA)

    result = await registry.read(resource_id, max_window_tokens=400)

    assert result.extraction_status == "unavailable" and not result.evidence_available
    assert result.note is not None and "Agent Browser: Every Agent Browser is in use" in result.note
    # A failed attempt pins nothing: the next read tries the browser again.
    await registry.read(resource_id, max_window_tokens=400)
    assert renderer.calls == [_SPA, _SPA]


async def test_a_page_that_ends_at_a_url_the_deployment_refuses_admits_nothing(serve) -> None:
    ended = RenderedPage(_SPA, "https://spa.example.com/done?token=secret", _QUOTES.encode(), 200)
    renderer = RecordingRenderer({_SPA: ended})
    serve(_Fetch(b"direct"))
    registry = ResourceRegistry(page_renderer=renderer)
    resource_id = registry.register_agent_url(_SPA)

    with pytest.raises(AgentBrowserError) as refused:
        await registry.read(resource_id, rendered=True)

    assert refused.value.reason == "final_url_refused"
    assert "secret" not in refused.value.public_message
    assert registry.rendered_effects(resource_id) == ()


async def test_a_rendering_over_the_attachment_limit_is_refused(serve) -> None:
    renderer = RecordingRenderer({_SPA: _long_page("page")})
    serve(_Fetch(b"direct"))
    registry = ResourceRegistry(page_renderer=renderer, max_attachment_bytes=1000)
    resource_id = registry.register_agent_url(_SPA)

    with pytest.raises(AgentBrowserError) as large:
        await registry.read(resource_id, rendered=True)

    assert large.value.reason == "too_large"
    assert "1000 bytes" in large.value.public_message


async def test_only_a_url_or_a_web_resource_can_be_read_rendered(serve) -> None:
    renderer = RecordingRenderer({})
    registry = ResourceRegistry(page_renderer=renderer)
    resource_id = registry.register(ResourceInput(filename="a.txt", content=b"caller bytes"))

    with pytest.raises(RenderedReadTargetError, match=f"{resource_id} is not a Web Resource"):
        await registry.read(resource_id, rendered=True)

    assert renderer.calls == []
    assert (await registry.read(resource_id)).content == "caller bytes"


async def test_a_url_that_resolves_to_a_private_address_never_reaches_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dlightrag.engine.network_admission.socket.getaddrinfo",
        lambda host, port, *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port))
        ],
    )
    renderer = RecordingRenderer({_SPA: _QUOTES})
    registry = ResourceRegistry(extract_chain=(AgentBrowserRender(),), page_renderer=renderer)
    resource_id = registry.register_agent_url(_SPA)

    with pytest.raises(PublicHttpPolicyError):
        await registry.read(resource_id, rendered=True)

    assert renderer.calls == []


async def test_concurrent_rendered_reads_share_one_render(serve) -> None:
    release = asyncio.Event()
    started = asyncio.Event()

    class _Slow(RecordingRenderer):
        async def __call__(self, url: str) -> RenderedPage:
            started.set()
            await release.wait()
            return await super().__call__(url)

    renderer = _Slow({_SPA: _QUOTES})
    serve(_Fetch(b"direct"))
    registry = ResourceRegistry(page_renderer=renderer)
    resource_id = registry.register_agent_url(_SPA)

    reads = [
        asyncio.create_task(registry.read(resource_id, rendered=True, max_window_tokens=2000))
        for _ in range(3)
    ]
    await started.wait()
    release.set()
    results = await asyncio.gather(*reads)

    assert len({result.content for result in results}) == 1
    assert renderer.calls == [_SPA]


async def test_closing_the_registry_cancels_a_render_in_flight(serve) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def render(url: str) -> RenderedPage:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    serve(_Fetch(b"direct"))
    registry = ResourceRegistry(page_renderer=render)
    resource_id = registry.register_agent_url(_SPA)
    reading = asyncio.create_task(registry.read(resource_id, rendered=True))
    await started.wait()

    await registry.aclose()

    assert cancelled.is_set()
    with pytest.raises(asyncio.CancelledError):
        await reading


async def test_a_resumed_run_reads_a_rendering_it_restored_without_rendering_again(serve) -> None:
    first_renderer = RecordingRenderer({_SPA: _QUOTES})
    serve(_Fetch(b"never fetched"))
    secret = b"rendered-run"
    async with ResourceRegistry(
        page_renderer=first_renderer, resource_secret=secret, cursor_secret=secret
    ) as registry:
        resource_id = registry.register_agent_url(_SPA)
        before = await registry.read(resource_id, rendered=True, max_window_tokens=2000)
        effects = registry.rendered_effects(resource_id)

    (rendering, snapshot, *assets) = effects
    stored = {effect.resource_id: effect.content for effect in effects}
    forbidden = RecordingRenderer({})
    async with ResourceRegistry(
        page_renderer=forbidden, resource_secret=secret, cursor_secret=secret
    ) as resumed:
        # Nothing registered the URL in this process: only the settled rows exist.
        resumed.restore_rendered(
            rendering.source_locator,
            url=_SPA,
            admission_origin="agent",
            final_url=_SPA,
            content=rendering.content,
        )
        assert resumed.holds_rendered(rendering.resource_id)
        resumed.adopt_conversion_snapshot(ConversionSnapshot.restore(snapshot.content, stored))

        after = await resumed.read(resource_id, max_window_tokens=2000)
        explicit = await resumed.read(resource_id, rendered=True, max_window_tokens=2000)

    assert after.rendered and after.content == explicit.content == before.content
    assert forbidden.calls == [] and not assets


async def test_a_restored_rendering_of_a_caller_link_the_request_does_not_hold_is_a_mismatch() -> (
    None
):
    registry = ResourceRegistry(resource_secret=b"run")

    with pytest.raises(Exception, match="caller link"):
        registry.restore_rendered(
            "res-0123456789abcdef01234567",
            url=_SPA,
            admission_origin="caller",
            final_url=_SPA,
            content=b"<p>x</p>",
        )
    with pytest.raises(Exception, match="does not match"):
        registry.restore_rendered(
            "res-0123456789abcdef01234567",
            url=_SPA,
            admission_origin="agent",
            final_url=_SPA,
            content=b"<p>x</p>",
        )


async def test_a_rendering_keeps_its_utf8_whatever_its_own_meta_charset_says(serve) -> None:
    page = '<html><head><meta charset="iso-8859-1"></head><body><p>café ☕</p></body></html>'
    renderer = RecordingRenderer({_SPA: page})
    serve(_Fetch(b"direct"))
    registry = ResourceRegistry(page_renderer=renderer)
    resource_id = registry.register_agent_url(_SPA)

    result = await registry.read(resource_id, rendered=True, max_window_tokens=2000)

    assert "café ☕" in result.content


async def test_a_direct_page_is_read_in_the_charset_its_server_declared(serve) -> None:
    body = '<html><head><meta charset="iso-8859-1"></head><body><p>café</p></body></html>'.encode()

    async def fetch(url: str, **_kwargs: object) -> PublicHttpFetch:
        return PublicHttpFetch(body, url, "text/html; charset=utf-8", 200)

    serve(fetch)
    registry = ResourceRegistry()
    resource_id = registry.register_agent_url("https://data.example.com/page.html")

    assert "café" in (await registry.read(resource_id, max_window_tokens=2000)).content


async def test_a_rendered_image_is_served_from_the_run_without_fetching(serve) -> None:
    import base64

    from tests.support.resources import png

    image = base64.b64encode(png()).decode()
    page = f"<html><body><p>Chart below</p><img alt='chart' src='data:image/png;base64,{image}'></body></html>"
    renderer = RecordingRenderer({_SPA: page})
    fetch = serve(_Fetch(b"direct"))
    registry = ResourceRegistry(page_renderer=renderer)
    resource_id = registry.register_agent_url(_SPA)

    result = await registry.read(resource_id, rendered=True, max_window_tokens=2000)

    (handle,) = result.visual_handles
    held = registry.held_visual_asset(resource_id, handle.handle_id)
    assert held is not None and held[1] is True and held[0].data == png()
    assert (await registry.visual_asset(resource_id, handle.handle_id)).data == png()
    assert registry.held_visual_asset(resource_id, "vis-unknown") is None
    assert fetch.calls == 0


# -- captures and downloads of an Agent Session's page ----------------------------------------

_SEARCH = "https://example.com/search?q=a"
_SEARCH_PAGE = "<html><body><h1>Results</h1><p>Albert Einstein said it first.</p></body></html>"


def _capture(url: str | None = _SEARCH, html: str = _SEARCH_PAGE) -> BrowserResourceInput:
    return BrowserResourceInput(
        "browser_capture", html.encode(), "search.html", "text/html; charset=utf-8", url
    )


def _download(
    content: bytes, *, name: str = "report.csv", mime: str = "text/csv"
) -> BrowserResourceInput:
    return BrowserResourceInput("browser_download", content, name, mime, "https://example.com/r")


class _Admissions:
    """The sink of a Run: every Resource a call admitted, with the call that admitted it."""

    def __init__(self) -> None:
        self.admitted: list[Any] = []

    async def __call__(self, fetched: Any, owner: ResourceEffectOwner | None) -> None:
        self.admitted.append((fetched, owner))


async def _forbidden(*_args: object, **_kwargs: object) -> Any:
    raise AssertionError("a captured page is read from its bytes, never fetched or extracted")


def _owner(scope: str = "parent") -> ResourceEffectOwner:
    return ResourceEffectOwner(scope, IntentId.new())


async def test_a_capture_is_a_new_agent_resource_cited_by_the_url_the_page_ended_at() -> None:
    sink = _Admissions()
    owner = _owner()
    registry = ResourceRegistry(
        fetched_bytes_sink=sink,
        extract_chain=(HostedExtract(_forbidden), AgentBrowserRender()),
        page_renderer=_forbidden,
    )

    resource_id = await registry.admit_browser_resource(_capture(), effect_owner=owner, index=0)
    result = await registry.read(resource_id, max_window_tokens=2000)

    # The bytes are kept with the call that made them, under the page's public URL.
    ((fetched, seen),) = sink.admitted
    assert seen == owner
    assert (fetched.resource_id, fetched.url, fetched.filename, fetched.content) == (
        resource_id,
        _SEARCH,
        "search.html",
        _SEARCH_PAGE.encode(),
    )
    assert (fetched.admission_origin, fetched.acquisition) == ("agent", "browser_capture")
    assert "Albert Einstein said it first." in result.content and result.evidence_available
    assert registry.evidence_source(resource_id, text=True) == {
        "source_type": "web_search",
        "resource_kind": "web",
        "admission_origin": "agent",
        "acquisition": "browser_capture",
        "source_uri": _SEARCH,
        "source_download_locator": _SEARCH,
        "title": "search.html",
    }
    assert registry.conversion_effects(resource_id)


async def test_each_capture_is_its_own_resource_and_never_rebinds_the_snapshot_of_its_url(
    serve,
) -> None:
    fetch = serve(_Fetch(b"<html><body>the snapshot</body></html>"))
    registry = ResourceRegistry()
    owner = _owner()

    first = await registry.admit_browser_resource(_capture(), effect_owner=owner, index=0)
    second = await registry.admit_browser_resource(_capture(), effect_owner=owner, index=1)
    other_call = await registry.admit_browser_resource(_capture(), effect_owner=_owner(), index=0)
    snapshot = registry.register_agent_url(_SEARCH)
    read = await registry.read(snapshot, max_window_tokens=2000)

    assert len({first, second, other_call, snapshot}) == 4
    assert "the snapshot" in read.content and fetch.calls == 1
    assert "Albert Einstein" in (await registry.read(first, max_window_tokens=2000)).content


@pytest.mark.parametrize(
    "locator",
    [
        "https://example.com/signed?token=abc",
        "blob:https://example.com/0f1e2d3c",
        "data:text/csv;base64,YSxiCjEsMgo=",
        "about:blank",
        "http://127.0.0.1/admin",
        None,
    ],
)
async def test_a_url_adr_0005_keeps_private_is_never_stored_and_the_handle_is_the_citation(
    locator: str | None,
) -> None:
    sink = _Admissions()
    registry = ResourceRegistry(fetched_bytes_sink=sink)

    resource_id = await registry.admit_browser_resource(
        _capture(locator), effect_owner=_owner(), index=0
    )

    ((fetched, _),) = sink.admitted
    source = registry.evidence_source(resource_id, text=True)
    assert fetched.url == resource_id
    assert (source["source_type"], source["source_uri"]) == ("web_attachment", resource_id)
    assert source["source_download_locator"] == resource_id
    assert (source["resource_kind"], source["acquisition"]) == ("web", "browser_capture")


async def test_a_capture_with_no_text_reads_as_none_and_never_walks_the_extract_chain() -> None:
    registry = ResourceRegistry(
        extract_chain=(HostedExtract(_forbidden), AgentBrowserRender()), page_renderer=_forbidden
    )
    resource_id = await registry.admit_browser_resource(
        _capture(html="<html><body><div></div></body></html>"), effect_owner=_owner(), index=0
    )

    result = await registry.read(resource_id, max_window_tokens=2000)

    assert result.extraction_status == "no_extracted_text" and not result.evidence_available


async def test_a_download_is_read_by_what_its_bytes_are_not_by_the_name_the_page_gave_it() -> None:
    from tests.support.resources import pdf_bytes

    pdf = pdf_bytes(2)
    registry = ResourceRegistry()
    name = browser_download_filename("")
    mime = browser_download_media_type(name, pdf)

    resource_id = await registry.admit_browser_resource(
        _download(pdf, name=name, mime=mime), effect_owner=_owner(), index=0
    )
    result = await registry.read(resource_id, max_window_tokens=2000)
    target = await registry.visual_target(resource_id)

    assert (name, mime) == ("download", "application/pdf")
    assert result.note is not None and "Physical PDF page count: 2" in result.note
    assert target.kind == "pdf"
    assert registry.evidence_source(resource_id)["acquisition"] == "browser_download"


async def test_a_capture_or_download_over_the_limit_is_refused_in_words_a_model_can_act_on() -> (
    None
):
    sink = _Admissions()
    registry = ResourceRegistry(max_attachment_bytes=1000, fetched_bytes_sink=sink)

    with pytest.raises(ResourceAdmissionError, match="the captured page exceeds 1000 bytes"):
        await registry.admit_browser_resource(
            _capture(html="x" * 1001), effect_owner=_owner(), index=0
        )
    with pytest.raises(ResourceAdmissionError, match="the download exceeds 1000 bytes"):
        await registry.admit_browser_resource(
            _download(b"x" * 1001), effect_owner=_owner(), index=1
        )

    assert sink.admitted == [] and registry.manifest() == ()


async def test_a_capture_or_download_takes_no_attachment_slot_and_no_share_of_the_total() -> None:
    registry = ResourceRegistry(max_attachments=1, max_total_attachment_bytes=10)
    owner = _owner()

    for index in range(3):
        await registry.admit_browser_resource(
            _download(b"x" * 100), effect_owner=owner, index=index
        )
    attached = registry.register(ResourceInput(filename="a.txt", content=b"small"))

    assert registry.canonical_resource_id(attached) == attached


async def test_a_resource_whose_bytes_could_not_be_kept_is_not_admitted() -> None:
    async def failing(_fetched: Any, _owner: ResourceEffectOwner | None) -> None:
        raise ConnectionError("the database is down")

    registry = ResourceRegistry(fetched_bytes_sink=failing)

    with pytest.raises(ConnectionError):
        await registry.admit_browser_resource(_capture(), effect_owner=_owner(), index=0)

    assert registry.manifest() == ()


@pytest.mark.parametrize("cited", ["url", "handle"])
async def test_a_resumed_run_restores_a_capture_without_the_browser_and_cites_it_as_before(
    cited: str,
) -> None:
    sink = _Admissions()
    secret = b"browser-run"
    locator = _SEARCH if cited == "url" else "https://example.com/signed?token=abc"
    async with ResourceRegistry(fetched_bytes_sink=sink, resource_secret=secret) as first:
        resource_id = await first.admit_browser_resource(
            _capture(locator), effect_owner=_owner(), index=0
        )
        before = await first.read(resource_id, max_window_tokens=2000)
        provenance = first.evidence_source(resource_id, text=True)
        view = first.conversion_effects(resource_id)
    ((fetched, _),) = sink.admitted

    async with ResourceRegistry(
        resource_secret=secret, extract_chain=(HostedExtract(_forbidden),)
    ) as resumed:
        resumed.restore_browser_resource(
            resource_id=fetched.resource_id,
            ordinal=fetched.ordinal,
            filename=fetched.filename,
            mime_type=fetched.mime_type,
            locator=fetched.url,
            content=fetched.content,
            acquisition=fetched.acquisition,
        )
        resumed.adopt_conversion_snapshot(
            ConversionSnapshot.restore(view[0].content, {resource_id: fetched.content})
        )
        after = await resumed.read(resource_id, max_window_tokens=2000)

        assert resumed.evidence_source(resource_id, text=True) == provenance
        assert await resumed.materialize(resource_id) == fetched.content
        assert after.content == before.content
        # The slot the capture settled under is not handed out again.
        assert resumed.allocate_fetched_ordinal("res-next") == fetched.ordinal + 1


@pytest.mark.parametrize("locator", ["http://127.0.0.1/x", "https://example.com/a?token=abc"])
def test_a_settled_capture_naming_a_private_locator_is_a_catalog_that_does_not_describe_the_run(
    locator: str,
) -> None:
    registry = ResourceRegistry()

    with pytest.raises(ResourceStateMismatchError, match="private locator"):
        registry.restore_browser_resource(
            resource_id="res-0123456789abcdef01234567",
            ordinal=0,
            filename="capture.html",
            mime_type="text/html",
            locator=locator,
            content=b"<p>x</p>",
            acquisition="browser_capture",
        )


@pytest.mark.parametrize(
    ("locator", "name"),
    [
        ("https://example.com/search?q=a", "search.html"),
        ("https://example.com/a/b/report.pdf", "report.html"),
        ("https://example.com/a/b/", "b.html"),
        ("https://example.com/", "example.com.html"),
        ("https://example.com/" + "x" * 200, "x" * 100 + ".html"),
        ("https://example.com/search?token=abc", "capture.html"),
        ("blob:https://example.com/0f1e", "capture.html"),
        (None, "capture.html"),
    ],
)
def test_a_capture_is_named_for_its_page_and_always_routed_to_the_html_converter(
    locator: str | None, name: str
) -> None:
    assert browser_capture_filename(locator) == name


@pytest.mark.parametrize(
    ("suggested", "name"),
    [
        ("report.csv", "report.csv"),
        ("../../etc/passwd", "passwd"),
        ("  ", "download"),
        ("", "download"),
    ],
)
def test_a_download_keeps_the_safe_part_of_the_name_its_page_gave_it(
    suggested: str, name: str
) -> None:
    assert browser_download_filename(suggested) == name


@pytest.mark.parametrize(
    ("name", "content", "media_type"),
    [
        ("report.csv", b"a,b", "text/csv"),
        ("figure.png", b"x", "image/png"),
        ("download", b"%PDF-1.7 ...", "application/pdf"),
        ("download", b"PK\x03\x04", "application/octet-stream"),
    ],
)
def test_a_download_is_typed_by_its_name_and_then_by_a_pdf_signature(
    name: str, content: bytes, media_type: str
) -> None:
    assert browser_download_media_type(name, content) == media_type
