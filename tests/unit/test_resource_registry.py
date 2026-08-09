# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the request-local answer resource registry."""

from __future__ import annotations

import asyncio
import socket

import pytest

from dlightrag.core.resources.models import (
    ResourceAdmissionError,
    ResourceCursorError,
    ResourceInput,
    ResourceNotFoundError,
)
from dlightrag.core.resources.registry import ResourceRegistry


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

    def stream(self, method: str, url: str) -> _StreamResponse:
        assert method == "GET"
        self.calls += 1
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


def test_register_rejects_non_https_link() -> None:
    registry = ResourceRegistry()

    with pytest.raises(ValueError):
        registry.register(ResourceInput(url="http://example.com/report.txt"))


def test_manifest_reports_link_without_size_until_read() -> None:
    registry = ResourceRegistry(url_client=_LinkClient())
    registry.register(ResourceInput(url="https://data.example.com/report.txt"))

    entries = registry.manifest()
    assert len(entries) == 1
    assert entries[0].source == "link"
    assert entries[0].byte_size is None


async def test_url_fetch_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dlightrag.sourcing.url.socket.getaddrinfo", _public_getaddrinfo)
    client = _LinkClient(content=b"remote body")
    registry = ResourceRegistry(url_client=client)
    resource_id = registry.register(ResourceInput(url="https://data.example.com/report.txt"))

    assert client.calls == 0

    result = await registry.read(resource_id)
    assert result.content == "remote body"
    assert client.calls == 1


async def test_read_revalidates_host_resolution_each_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def resolver(host: str, port: int, *args: object, **kwargs: object):
        calls["n"] += 1
        ip = "93.184.216.34" if calls["n"] == 1 else "10.0.0.5"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr("dlightrag.sourcing.url.socket.getaddrinfo", resolver)
    registry = ResourceRegistry(url_client=_LinkClient(content=b"safe"))
    resource_id = registry.register(ResourceInput(url="https://data.example.com/report.txt"))

    first = await registry.read(resource_id)
    assert first.content == "safe"
    with pytest.raises(ValueError):
        await registry.read(resource_id)


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
    combined = first.content + "\n" + second.content
    while second.has_more:
        second = await registry.read(resource_id, cursor=second.next_cursor)
        combined = combined + "\n" + second.content
    assert combined == text


async def test_cursor_is_bound_to_its_resource() -> None:
    registry = ResourceRegistry()
    big = "\n".join(f"line {index} " + "x" * 30 for index in range(2000))
    big_id = registry.register(ResourceInput(content=big.encode("utf-8")))
    small_id = registry.register(ResourceInput(content=b"tiny"))

    first = await registry.read(big_id)
    assert first.next_cursor is not None

    with pytest.raises(ResourceCursorError):
        await registry.read(small_id, cursor=first.next_cursor)


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
    monkeypatch.setattr("dlightrag.sourcing.url.socket.getaddrinfo", _public_getaddrinfo)
    client = _LinkClient(fail=asyncio.CancelledError)
    registry = ResourceRegistry(url_client=client)
    resource_id = registry.register(ResourceInput(url="https://data.example.com/report.txt"))

    with pytest.raises(asyncio.CancelledError):
        await registry.read(resource_id)

    await registry.aclose()


async def test_async_context_manager_closes_owned_resources() -> None:
    registry = ResourceRegistry()
    async with registry as active:
        resource_id = active.register(ResourceInput(content=b"payload"))
        path = await active.ensure_path(resource_id)
        assert path.exists()

    assert not path.exists()
    assert registry.has_temp_storage is False
