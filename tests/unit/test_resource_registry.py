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
from dlightrag.engine.answer.resources.models import (
    ResourceAdmissionError,
    ResourceCursorError,
    ResourceDecodeError,
    ResourceInput,
    ResourceNotFoundError,
)
from dlightrag.engine.answer.resources.registry import (
    ResourceEffectOwner,
)
from dlightrag.engine.answer.resources.registry import (
    ResourceRegistry as _ResourceRegistry,
)
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from dlightrag.engine.answer.web_sources import WebExtractResult, WebSourceUnavailable
from dlightrag.engine.public_http import PublicHttpFetch, PublicHttpPresentation
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
        effect_owner: ResourceEffectOwner | None = None,
    ):
        return await super().read(
            resource_id,
            max_window_tokens=max_window_tokens,
            focus=focus,
            cursor=cursor,
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
    registry = ResourceRegistry(url_text_fallback=fallback)
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    result = await registry.read(resource_id)

    assert result.content == "good body"
    assert result.evidence_available is True
    assert registry.evidence_source(resource_id)["acquisition"] == "direct_http"
    assert fallback.calls == 0


async def test_direct_decode_failure_uses_one_extract_fallback(serve) -> None:
    fallback = _CountingFallback("recovered text\nsecond line", provider="tavily")
    serve(_Fetch(b"\x00\x01\x02\x03binary\x00\x00"))
    registry = ResourceRegistry(url_text_fallback=fallback)
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
    registry = ResourceRegistry(url_text_fallback=fallback, fetched_bytes_sink=persist)
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
    registry = ResourceRegistry(url_text_fallback=fallback, fetched_bytes_sink=persist)
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
    registry = ResourceRegistry(url_text_fallback=fallback, fetched_bytes_sink=persist)
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
    registry = ResourceRegistry(url_text_fallback=fallback, fetched_bytes_sink=persist)
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
    registry = ResourceRegistry(url_text_fallback=fallback, fetched_bytes_sink=persist)
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
    registry = ResourceRegistry(url_text_fallback=fallback)
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
    registry = ResourceRegistry(url_text_fallback=fallback)
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    with pytest.raises(ValueError):
        await registry.read(resource_id)
    assert fallback.calls == 0


async def test_exhausted_extract_returns_no_evidence_and_does_not_pin_failure(serve) -> None:
    fallback = _CountingFallback(None)
    fetch = serve(_Fetch(b"\x00\x01\x02\x03binary\x00\x00"))
    registry = ResourceRegistry(url_text_fallback=fallback)
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
    registry = ResourceRegistry(url_text_fallback=fallback)
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
