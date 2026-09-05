# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the run-scoped answer Resource Registry."""

from __future__ import annotations

import asyncio
import socket

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
from dlightrag.engine.answer.web_sources import WebExtractResult, WebSourceUnavailable
from dlightrag.engine.public_http import PublicHttpPresentation


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


class _StreamResponse:
    def __init__(
        self,
        content: bytes,
        url: str,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        fail: type[BaseException] | None = None,
    ) -> None:
        self._content = content
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}
        self._fail = fail

    async def __aenter__(self) -> _StreamResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        if self._fail is not None:
            raise self._fail()
        yield self._content


class _LinkClient:
    def __init__(
        self,
        *,
        content: bytes = b"hello\nworld",
        final_url: str = "https://data.example.com/report.txt",
        fail: type[BaseException] | None = None,
    ) -> None:
        self.content = content
        self.final_url = final_url
        self.fail = fail
        self.calls = 0
        self.closed = False
        self.headers: list[dict[str, str]] = []

    def stream(self, method: str, url: str, **_kwargs) -> _StreamResponse:
        assert method == "GET"
        self.calls += 1
        self.headers.append(dict(_kwargs.get("headers") or {}))
        return _StreamResponse(self.content, self.final_url, fail=self.fail)

    async def aclose(self) -> None:
        self.closed = True


def _public_getaddrinfo(host: str, port: int, *args: object, **kwargs: object):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


def test_register_returns_stable_opaque_id() -> None:
    registry = ResourceRegistry()
    first = registry.register(ResourceInput(filename="a.txt", content=b"same bytes"))
    second = registry.register(ResourceInput(filename="a.txt", content=b"same bytes"))

    assert first == second
    assert "a.txt" not in first
    assert "same bytes" not in first
    assert len(registry.manifest()) == 1


def test_duplicate_content_deduplicated_across_filenames() -> None:
    registry = ResourceRegistry()
    first = registry.register(ResourceInput(filename="a.txt", content=b"payload"))
    second = registry.register(ResourceInput(filename="b.txt", content=b"payload"))

    assert first == second
    assert len(registry.manifest()) == 1


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


def test_discovered_link_deduplicates_with_a_caller_link_and_stays_inert() -> None:
    client = _LinkClient()
    registry = ResourceRegistry(url_client=client)
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
    assert client.calls == 0


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
    registry = ResourceRegistry(url_client=_LinkClient())
    registry.register(ResourceInput(url="https://data.example.com/report.txt"))

    entries = registry.manifest()
    assert len(entries) == 1
    assert entries[0].filename == "report.txt"
    assert entries[0].source == "link"
    assert entries[0].byte_size is None


async def test_url_fetch_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    client = _LinkClient(content=b"remote body")
    registry = ResourceRegistry(url_client=client)
    resource_id = registry.register(ResourceInput(url="https://data.example.com/report.txt"))

    assert client.calls == 0

    result = await registry.read(resource_id)
    assert result.content == "remote body"
    assert client.calls == 1


async def test_discovered_link_has_only_per_operation_not_cumulative_web_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    client = _LinkClient(content=b"12345")
    registry = ResourceRegistry(
        max_attachment_bytes=10,
        max_total_attachment_bytes=6,
        url_client=client,
    )
    registry.register(ResourceInput(content=b"123456"))
    resource_id = registry.register_discovered_link("https://data.example.com/report.txt")
    assert resource_id is not None

    result = await registry.read(resource_id)

    assert result.content == "12345"
    assert registry._total_bytes == 6
    assert client.calls == 1


async def test_successful_url_read_keeps_one_fixed_snapshot_without_new_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def resolver(host: str, port: int, *args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", resolver)
    client = _LinkClient(content=b"safe")
    registry = ResourceRegistry(url_client=client)
    resource_id = registry.register(ResourceInput(url="https://data.example.com/report.txt"))

    first = await registry.read(resource_id)
    second = await registry.read(resource_id)

    assert first.content == second.content == "safe"
    assert calls == 1
    assert client.calls == 1


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


async def test_direct_text_read_uses_no_temp_file() -> None:
    registry = ResourceRegistry()
    resource_id = registry.register(ResourceInput(content=b"just text"))

    await registry.read(resource_id)

    assert registry.has_temp_storage is False


async def test_ensure_path_materializes_temp_and_aclose_cleans_up() -> None:
    registry = ResourceRegistry()
    resource_id = registry.register(ResourceInput(filename="d.txt", content=b"bytes"))

    path = await registry.ensure_path(resource_id)
    assert path.exists()
    assert path.read_bytes() == b"bytes"
    assert registry.has_temp_storage is True

    await registry.aclose()
    assert not path.exists()


async def test_cancellation_during_fetch_propagates_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    client = _LinkClient(fail=asyncio.CancelledError)
    registry = ResourceRegistry(url_client=client)
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


async def test_async_context_manager_closes_owned_resources() -> None:
    registry = ResourceRegistry()
    async with registry as active:
        resource_id = active.register(ResourceInput(content=b"payload"))
        path = await active.ensure_path(resource_id)
        assert path.exists()

    assert not path.exists()
    assert registry.has_temp_storage is False


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


async def test_redirect_final_url_becomes_citable_identity_and_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    responses = iter(
        (
            _StreamResponse(
                b"",
                "https://start.example/report",
                status_code=302,
                headers={"location": "https://final.example/report"},
            ),
            _StreamResponse(b"final body", "https://final.example/report"),
        )
    )

    class RedirectClient:
        def stream(self, method: str, url: str, **kwargs) -> _StreamResponse:
            return next(responses)

    admitted = []

    async def persist(fetched, _owner) -> None:
        admitted.append(fetched)

    identity_secret = b"r" * 32
    registry = ResourceRegistry(
        url_client=RedirectClient(),
        fetched_bytes_sink=persist,
        resource_secret=identity_secret,
    )
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    responses = iter(
        (
            _StreamResponse(
                b"",
                "https://start.example/report",
                status_code=302,
                headers={"location": "https://final.example/report"},
            ),
            _StreamResponse(b"final body", "https://final.example/report"),
        )
    )

    class RedirectClient:
        def stream(self, method: str, url: str, **kwargs) -> _StreamResponse:
            return next(responses)

    admitted = []

    async def persist(fetched, _owner) -> None:
        admitted.append(fetched)

    identity_secret = b"r" * 32
    registry = ResourceRegistry(
        url_client=RedirectClient(),
        fetched_bytes_sink=persist,
        resource_secret=identity_secret,
    )
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


async def test_failed_agent_read_can_retry_with_new_presentation_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    client = _LinkClient(fail=RuntimeError)
    registry = ResourceRegistry(url_client=client)
    resource_id = registry.register_agent_url(
        "https://data.example.com/report.txt",
        presentation=PublicHttpPresentation(user_agent="First/1"),
    )

    first = await registry.read(resource_id)
    assert first.evidence_available is False
    client.fail = None
    assert (
        registry.register_agent_url(
            "https://data.example.com/report.txt",
            presentation=PublicHttpPresentation(user_agent="Second/2"),
        )
        == resource_id
    )

    second = await registry.read(resource_id)

    assert second.content == "hello\nworld"
    assert [headers["user-agent"] for headers in client.headers] == ["First/1", "Second/2"]


async def test_direct_success_skips_url_text_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    fallback = _CountingFallback("EXTRACTED TEXT")
    registry = ResourceRegistry(
        url_client=_LinkClient(content=b"good body"),
        url_text_fallback=fallback,
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    result = await registry.read(resource_id)

    assert result.content == "good body"
    assert result.evidence_available is True
    assert registry.evidence_source(resource_id)["acquisition"] == "direct_http"
    assert fallback.calls == 0


async def test_direct_decode_failure_uses_one_extract_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    fallback = _CountingFallback("recovered text\nsecond line", provider="tavily")
    registry = ResourceRegistry(
        url_client=_LinkClient(content=b"\x00\x01\x02\x03binary\x00\x00"),
        url_text_fallback=fallback,
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    result = await registry.read(resource_id)
    again = await registry.read(resource_id)

    assert "recovered text" in result.content
    assert again.content == result.content
    assert fallback.calls == 1
    assert registry.evidence_source(resource_id)["acquisition"] == "tavily_extract"


async def test_binary_direct_snapshot_is_retained_without_hosted_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    admitted = []

    async def persist(fetched, _owner) -> None:
        admitted.append(fetched)

    fallback = _CountingFallback("provider replacement")
    content = b"\x00\x01\x02\x03binary\x00\x00"
    registry = ResourceRegistry(
        url_client=_LinkClient(content=content),
        url_text_fallback=fallback,
        fetched_bytes_sink=persist,
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.bin")

    with pytest.raises(ResourceDecodeError):
        await registry.read(resource_id)

    assert fallback.calls == 0
    assert len(admitted) == 1
    assert admitted[0].content == content
    assert admitted[0].acquisition == "direct_http"


async def test_shared_extract_snapshot_is_admitted_for_each_effect_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    owners = []

    async def persist(_fetched, owner) -> None:
        owners.append(owner)

    fallback = _CountingFallback("shared provider text")
    registry = ResourceRegistry(
        url_client=_LinkClient(content=b""),
        url_text_fallback=fallback,
        fetched_bytes_sink=persist,
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


async def test_extract_fallback_persists_only_the_admitted_text_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    admitted = []

    async def persist(fetched, _owner) -> None:
        admitted.append(fetched)

    fallback = _CountingFallback("  provider body text\n")
    registry = ResourceRegistry(
        url_client=_LinkClient(content=b""),
        url_text_fallback=fallback,
        fetched_bytes_sink=persist,
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.html")

    result = await registry.read(resource_id)

    assert result.content == "  provider body text\n"
    assert len(admitted) == 1
    assert admitted[0].content == b"  provider body text\n"
    assert admitted[0].acquisition == "exa_extract"

    recovered = ResourceRegistry()
    recovered.restore_fetched_resource(
        resource_id=resource_id,
        ordinal=admitted[0].ordinal,
        filename="report.html",
        mime_type="text/markdown; charset=utf-8",
        url=admitted[0].url,
        content=admitted[0].content,
        admission_origin="agent",
        acquisition="exa_extract",
    )
    assert (await recovered.read(resource_id)).content == "  provider body text\n"


async def test_direct_empty_triggers_extract_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    fallback = _CountingFallback("provider body text")
    registry = ResourceRegistry(
        url_client=_LinkClient(content=b""),
        url_text_fallback=fallback,
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    result = await registry.read(resource_id)

    assert result.content == "provider body text"
    assert fallback.calls == 1


async def test_invalid_private_url_never_calls_extract_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dlightrag.engine.public_http.socket.getaddrinfo",
        lambda host, port, *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port))
        ],
    )
    fallback = _CountingFallback("should never appear")
    registry = ResourceRegistry(
        url_client=_LinkClient(content=b"x"),
        url_text_fallback=fallback,
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    with pytest.raises(ValueError):
        await registry.read(resource_id)
    assert fallback.calls == 0


async def test_exhausted_extract_returns_no_evidence_and_does_not_pin_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    fallback = _CountingFallback(None)
    client = _LinkClient(content=b"\x00\x01\x02\x03binary\x00\x00")
    registry = ResourceRegistry(
        url_client=client,
        url_text_fallback=fallback,
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    result = await registry.read(resource_id)
    again = await registry.read(resource_id)

    assert result.extraction_status == "unavailable"
    assert result.evidence_available is False
    assert "produced no citable text" in result.content
    assert again.content == result.content
    assert fallback.calls == 2
    assert client.calls == 2


async def test_fallback_text_windows_are_cursor_paginated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    big = "\n".join(f"line {index} " + "x" * 30 for index in range(2000))
    fallback = _CountingFallback(big)
    registry = ResourceRegistry(
        url_client=_LinkClient(content=b""),
        url_text_fallback=fallback,
    )
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    current = await registry.read(resource_id)
    combined = current.content
    while current.has_more:
        current = await registry.read(resource_id, cursor=current.next_cursor)
        combined += current.content

    assert combined == big
    assert fallback.calls == 1


async def test_web_reads_do_not_consume_attachment_cumulative_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    registry = ResourceRegistry(
        max_attachment_bytes=100,
        max_total_attachment_bytes=8,
        url_client=_LinkClient(content=b"0123456789"),
    )
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


async def test_concurrent_reads_same_link_share_one_fixed_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dlightrag.engine.public_http.socket.getaddrinfo", _public_getaddrinfo)
    client = _LinkClient(content=b"0123456789")
    registry = ResourceRegistry(url_client=client)
    resource_id = registry.register_agent_url("https://data.example.com/report.txt")

    results = await asyncio.gather(*(registry.read(resource_id) for _ in range(3)))

    assert all(result.content == "0123456789" for result in results)
    assert client.calls == 1


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
