# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the shared robust document embedding executor."""

import asyncio
import io
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from PIL import Image

import dlightrag.core.document_embedding as document_embedding
from dlightrag.core.document_embedding import (
    DocumentEmbeddingInput,
    DocumentEmbeddingTrace,
    DocumentEmbeddingVector,
    RobustDocumentEmbedder,
)


def _png_bytes(*, size: tuple[int, int] = (8, 8), mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _embedder(
    *,
    fused: object = None,
    text: object = None,
    dimension: int = 3,
) -> SimpleNamespace:
    fused_result = fused if fused is not None else [[1.0, 0.0, 0.0]]
    text_result = text if text is not None else [[0.0, 1.0, 0.0]]
    return SimpleNamespace(
        dim=dimension,
        embed_index_fused=AsyncMock(return_value=fused_result),
        embed_texts=AsyncMock(return_value=text_result),
        aclose=AsyncMock(),
    )


def _executor(
    embedder: Any,
    *,
    image_enabled: bool = True,
    batch_size: int = 8,
    max_concurrency: int = 4,
    min_image_pixel: int = 2,
    image_cleanup_timeout: float = 0.1,
) -> RobustDocumentEmbedder:
    return RobustDocumentEmbedder(
        embedder=embedder,
        image_enabled=image_enabled,
        dimension=3,
        min_image_pixel=min_image_pixel,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        image_cleanup_timeout=image_cleanup_timeout,
    )


async def test_image_capability_uses_fused_embedding() -> None:
    embedder = _embedder()
    executor = _executor(embedder)

    vectors, trace = await executor.aembed_documents(
        [DocumentEmbeddingInput(key="figure", text="a chart", image_bytes=_png_bytes())]
    )

    assert vectors == [DocumentEmbeddingVector(key="figure", vector=[1.0, 0.0, 0.0], mode="fused")]
    assert trace == DocumentEmbeddingTrace(fused=1, text=0, fused_to_text_fallback=0, failed=0)
    embedder.embed_index_fused.assert_awaited_once()
    assert embedder.embed_index_fused.await_args.args[0][0][0] == "a chart"
    embedder.embed_texts.assert_not_awaited()
    embedder.aclose.assert_not_awaited()


async def test_image_path_uses_fused_embedding(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(_png_bytes())
    embedder = _embedder()

    vectors, _trace = await _executor(embedder).aembed_documents(
        [DocumentEmbeddingInput(key="figure", text="a chart", image_path=image_path)]
    )

    assert [vector.key for vector in vectors] == ["figure"]
    embedder.embed_index_fused.assert_awaited_once()


async def test_text_only_capability_never_calls_fused() -> None:
    embedder = _embedder()
    executor = _executor(embedder, image_enabled=False)

    vectors, trace = await executor.aembed_documents(
        [DocumentEmbeddingInput(key="figure", text="a chart", image_bytes=_png_bytes())]
    )

    assert vectors == [DocumentEmbeddingVector(key="figure", vector=[0.0, 1.0, 0.0], mode="text")]
    assert trace == DocumentEmbeddingTrace(fused=0, text=1, fused_to_text_fallback=0, failed=0)
    embedder.embed_index_fused.assert_not_awaited()
    embedder.embed_texts.assert_awaited_once_with(["a chart"], context="document")


async def test_fused_batch_failure_falls_back_to_text_once() -> None:
    embedder = _embedder(text=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    embedder.embed_index_fused.side_effect = RuntimeError("fused unavailable")
    executor = _executor(embedder)
    items = [
        DocumentEmbeddingInput(key="a", text="first", image_bytes=_png_bytes()),
        DocumentEmbeddingInput(key="b", text="second", image_bytes=_png_bytes()),
    ]

    vectors, trace = await executor.aembed_documents(items)

    assert vectors == [
        DocumentEmbeddingVector(key="a", vector=[1.0, 0.0, 0.0], mode="text"),
        DocumentEmbeddingVector(key="b", vector=[0.0, 1.0, 0.0], mode="text"),
    ]
    assert trace == DocumentEmbeddingTrace(fused=0, text=2, fused_to_text_fallback=2, failed=0)
    embedder.embed_texts.assert_awaited_once_with(["first", "second"], context="document")


async def test_failed_text_fallback_records_fallback_and_failure() -> None:
    embedder = _embedder()
    embedder.embed_index_fused.side_effect = RuntimeError("fused unavailable")
    embedder.embed_texts.side_effect = RuntimeError("text unavailable")

    vectors, trace = await _executor(embedder).aembed_documents(
        [DocumentEmbeddingInput(key="a", text="first", image_bytes=_png_bytes())]
    )

    assert vectors == []
    assert trace == DocumentEmbeddingTrace(fused=0, text=0, fused_to_text_fallback=1, failed=1)


async def test_unreadable_or_small_image_falls_back_to_text() -> None:
    embedder = _embedder(text=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    executor = _executor(embedder, min_image_pixel=4)
    items = [
        DocumentEmbeddingInput(key="bad", text="unreadable", image_bytes=b"not an image"),
        DocumentEmbeddingInput(
            key="small",
            text="too small",
            image_bytes=_png_bytes(size=(2, 2)),
        ),
    ]

    vectors, trace = await executor.aembed_documents(items)

    assert [vector.mode for vector in vectors] == ["text", "text"]
    assert trace == DocumentEmbeddingTrace(fused=0, text=2, fused_to_text_fallback=0, failed=0)
    embedder.embed_index_fused.assert_not_awaited()
    embedder.embed_texts.assert_awaited_once_with(["unreadable", "too small"], context="document")


async def test_image_open_warning_includes_safe_key_and_source(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bad_path = tmp_path / "bad\nimage.png"
    bad_path.write_bytes(b"not an image")
    caplog.set_level(logging.WARNING, logger=document_embedding.__name__)

    await _executor(_embedder()).aembed_documents(
        [DocumentEmbeddingInput(key="chunk\none", text="fallback", image_path=bad_path)]
    )

    message = next(
        record.getMessage()
        for record in caplog.records
        if "Document image could not be opened" in record.getMessage()
    )
    assert "key=chunk\\none" in message
    assert f"source=path:{str(bad_path).replace(chr(10), '\\n')}" in message


async def test_text_embedding_failure_omits_only_failed_batch() -> None:
    embedder = _embedder()
    embedder.embed_texts.side_effect = [
        RuntimeError("first batch failed"),
        [[0.0, 0.0, 1.0]],
    ]
    executor = _executor(embedder, image_enabled=False, batch_size=2)
    items = [
        DocumentEmbeddingInput(key="a", text="first"),
        DocumentEmbeddingInput(key="b", text="second"),
        DocumentEmbeddingInput(key="c", text="third"),
    ]

    vectors, trace = await executor.aembed_documents(items)

    assert vectors == [DocumentEmbeddingVector(key="c", vector=[0.0, 0.0, 1.0], mode="text")]
    assert trace == DocumentEmbeddingTrace(fused=0, text=1, fused_to_text_fallback=0, failed=2)
    assert embedder.embed_texts.await_count == 2


async def test_query_embedding_uses_query_context() -> None:
    embedder = _embedder(text=[[0.0, 0.0, 1.0]])
    executor = _executor(embedder)

    vector = await executor.aembed_query("find this")

    assert vector == [0.0, 0.0, 1.0]
    embedder.embed_texts.assert_awaited_once_with(["find this"], context="query")
    embedder.embed_index_fused.assert_not_awaited()


@pytest.mark.parametrize(
    "invalid_vectors",
    [
        [],
        [[1.0, 0.0]],
        [[float("nan"), 0.0, 1.0]],
        [[0.0, 0.0, 0.0]],
    ],
    ids=["wrong-count", "wrong-dimension", "nonfinite", "zero-norm"],
)
async def test_wrong_dimension_nonfinite_and_zero_norm_vectors_are_rejected(
    invalid_vectors: list[list[float]],
) -> None:
    embedder = _embedder(text=invalid_vectors)
    executor = _executor(embedder, image_enabled=False)

    vectors, trace = await executor.aembed_documents(
        [DocumentEmbeddingInput(key="a", text="first")]
    )

    assert vectors == []
    assert trace.failed == 1


@pytest.mark.parametrize(
    "result",
    [
        RuntimeError("provider failed"),
        [],
        [[1.0, 0.0]],
        [[float("inf"), 0.0, 1.0]],
        [[0.0, 0.0, 0.0]],
    ],
    ids=["failure", "wrong-count", "wrong-dimension", "nonfinite", "zero-norm"],
)
async def test_invalid_query_vector_returns_none(result: object) -> None:
    embedder = _embedder()
    if isinstance(result, Exception):
        embedder.embed_texts.side_effect = result
    else:
        embedder.embed_texts.return_value = result

    assert await _executor(embedder).aembed_query("find this") is None


@pytest.mark.parametrize("fused_fails", [False, True], ids=["success", "failure"])
async def test_all_pil_images_close_on_success_and_failure(fused_fails: bool) -> None:
    opened: list[Image.Image] = []
    embedder = _embedder()

    async def fused(items: list[tuple[str, Image.Image]]) -> list[list[float]]:
        opened.extend(image for _text, image in items)
        if fused_fails:
            raise RuntimeError("fused failed")
        return [[1.0, 0.0, 0.0] for _item in items]

    embedder.embed_index_fused.side_effect = fused

    await _executor(embedder).aembed_documents(
        [DocumentEmbeddingInput(key="a", text="first", image_bytes=_png_bytes())]
    )

    assert len(opened) == 1
    with pytest.raises(ValueError, match="closed image"):
        opened[0].getpixel((0, 0))


async def test_shared_semaphore_bounds_fused_and_text_calls() -> None:
    active = 0
    max_active = 0
    fused_calls = 0
    text_calls = 0
    at_capacity = asyncio.Event()
    release = asyncio.Event()

    async def enter(count: int) -> list[list[float]]:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            at_capacity.set()
        await release.wait()
        active -= 1
        return [[1.0, 0.0, 0.0] for _index in range(count)]

    async def fused(items: list[tuple[str, Image.Image]]) -> list[list[float]]:
        nonlocal fused_calls
        fused_calls += 1
        return await enter(len(items))

    async def text(texts: list[str], *, context: str) -> list[list[float]]:
        nonlocal text_calls
        assert context in {"document", "query"}
        text_calls += 1
        return await enter(len(texts))

    embedder = SimpleNamespace(
        dim=3,
        embed_index_fused=AsyncMock(side_effect=fused),
        embed_texts=AsyncMock(side_effect=text),
    )
    executor = _executor(embedder, batch_size=1, max_concurrency=2)
    calls = [
        executor.aembed_documents(
            [DocumentEmbeddingInput(key="image", text="chart", image_bytes=_png_bytes())]
        ),
        executor.aembed_query("query one"),
        executor.aembed_query("query two"),
        executor.aembed_query("query three"),
    ]
    task = asyncio.gather(*calls)

    await asyncio.wait_for(at_capacity.wait(), timeout=1)
    release.set()
    await task

    assert max_active == 2
    assert fused_calls == 1
    assert text_calls == 3


async def test_embedding_cancellation_reraises_and_closes_open_images() -> None:
    opened: list[Image.Image] = []
    provider_started = asyncio.Event()
    never_finish = asyncio.Event()
    embedder = _embedder()

    async def fused(items: list[tuple[str, Image.Image]]) -> list[list[float]]:
        opened.extend(image for _text, image in items)
        provider_started.set()
        await never_finish.wait()
        return [[1.0, 0.0, 0.0]]

    embedder.embed_index_fused.side_effect = fused
    task = asyncio.create_task(
        _executor(embedder).aembed_documents(
            [DocumentEmbeddingInput(key="a", text="first", image_bytes=_png_bytes())]
        )
    )
    await asyncio.wait_for(provider_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(opened) == 1
    with pytest.raises(ValueError, match="closed image"):
        opened[0].getpixel((0, 0))


async def test_image_open_cancellation_waits_for_worker_and_closes_returned_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    opened: list[Image.Image] = []

    def blocking_open_image(
        _item: DocumentEmbeddingInput,
        *,
        min_image_pixel: int,
    ) -> Image.Image:
        assert min_image_pixel == 2
        image = Image.new("RGB", (8, 8), "white")
        opened.append(image)
        worker_started.set()
        assert release_worker.wait(timeout=5)
        return image

    monkeypatch.setattr(document_embedding, "_open_valid_image", blocking_open_image)
    task = asyncio.create_task(
        _executor(_embedder()).aembed_documents(
            [DocumentEmbeddingInput(key="a", text="first", image_bytes=_png_bytes())]
        )
    )
    assert await asyncio.to_thread(worker_started.wait, 1)

    try:
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release_worker.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(opened) == 1
    with pytest.raises(ValueError, match="closed image"):
        opened[0].getpixel((0, 0))


async def test_image_open_cancellation_timeout_closes_late_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    callback_finished = asyncio.Event()
    opened: list[Image.Image] = []
    callback_calls = 0
    close_task_result = document_embedding._close_image_task_result

    def blocking_open_image(
        _item: DocumentEmbeddingInput,
        *,
        min_image_pixel: int,
    ) -> Image.Image:
        assert min_image_pixel == 2
        image = Image.new("RGB", (8, 8), "white")
        opened.append(image)
        worker_started.set()
        assert release_worker.wait(timeout=5)
        return image

    def observe_close(task: asyncio.Task[Image.Image | None]) -> None:
        nonlocal callback_calls
        callback_calls += 1
        close_task_result(task)
        callback_finished.set()

    monkeypatch.setattr(document_embedding, "_open_valid_image", blocking_open_image)
    monkeypatch.setattr(document_embedding, "_close_image_task_result", observe_close)
    task = asyncio.create_task(
        _executor(_embedder(), image_cleanup_timeout=0.01).aembed_documents(
            [DocumentEmbeddingInput(key="a", text="first", image_bytes=_png_bytes())]
        )
    )
    assert await asyncio.to_thread(worker_started.wait, 1)

    try:
        task.cancel("original cancellation")
        async with asyncio.timeout(0.2):
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
        assert exc_info.value.args == ("original cancellation",)
        assert not release_worker.is_set()
    finally:
        release_worker.set()

    await asyncio.wait_for(callback_finished.wait(), timeout=1)
    assert callback_calls == 1
    assert len(opened) == 1
    with pytest.raises(ValueError, match="closed image"):
        opened[0].getpixel((0, 0))


async def test_image_open_cancellation_callback_consumes_late_worker_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    callback_finished = asyncio.Event()
    callback_calls = 0
    close_task_result = document_embedding._close_image_task_result
    loop = asyncio.get_running_loop()
    unhandled_contexts: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()

    def blocking_open_image(
        _item: DocumentEmbeddingInput,
        *,
        min_image_pixel: int,
    ) -> Image.Image:
        assert min_image_pixel == 2
        worker_started.set()
        assert release_worker.wait(timeout=5)
        raise RuntimeError("late decode failure")

    def observe_close(task: asyncio.Task[Image.Image | None]) -> None:
        nonlocal callback_calls
        callback_calls += 1
        close_task_result(task)
        callback_finished.set()

    monkeypatch.setattr(document_embedding, "_open_valid_image", blocking_open_image)
    monkeypatch.setattr(document_embedding, "_close_image_task_result", observe_close)
    loop.set_exception_handler(lambda _loop, context: unhandled_contexts.append(context))
    task = asyncio.create_task(
        _executor(_embedder(), image_cleanup_timeout=0.01).aembed_documents(
            [DocumentEmbeddingInput(key="a", text="first", image_bytes=_png_bytes())]
        )
    )
    assert await asyncio.to_thread(worker_started.wait, 1)

    try:
        task.cancel()
        async with asyncio.timeout(0.2):
            with pytest.raises(asyncio.CancelledError):
                await task
        release_worker.set()
        await asyncio.wait_for(callback_finished.wait(), timeout=1)
        await asyncio.sleep(0)
    finally:
        release_worker.set()
        loop.set_exception_handler(previous_handler)

    assert callback_calls == 1
    assert unhandled_contexts == []


async def test_concurrent_image_open_cancellation_closes_completed_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = threading.Event()
    release_worker = threading.Event()
    first_open_completed = asyncio.Event()
    opened: list[Image.Image] = []

    def partly_blocking_open_image(
        item: DocumentEmbeddingInput,
        *,
        min_image_pixel: int,
    ) -> Image.Image | None:
        assert min_image_pixel == 2
        if item.key == "ready":
            image = Image.new("RGB", (8, 8), "white")
            opened.append(image)
            return image
        worker_started.set()
        assert release_worker.wait(timeout=5)
        return None

    monkeypatch.setattr(document_embedding, "_open_valid_image", partly_blocking_open_image)
    executor = _executor(_embedder(), batch_size=2, max_concurrency=2, image_cleanup_timeout=0.01)
    open_indexed_image = executor._aopen_indexed_image

    async def observe_open(
        indexed_item: tuple[int, DocumentEmbeddingInput],
    ) -> tuple[int, DocumentEmbeddingInput, Image.Image | None]:
        result = await open_indexed_image(indexed_item)
        if indexed_item[1].key == "ready":
            first_open_completed.set()
        return result

    monkeypatch.setattr(executor, "_aopen_indexed_image", observe_open)
    task = asyncio.create_task(
        executor.aembed_documents(
            [
                DocumentEmbeddingInput(key="ready", text="first", image_bytes=_png_bytes()),
                DocumentEmbeddingInput(key="wedged", text="second", image_bytes=_png_bytes()),
            ]
        )
    )
    await asyncio.wait_for(first_open_completed.wait(), timeout=1)
    assert await asyncio.to_thread(worker_started.wait, 1)

    try:
        task.cancel()
        async with asyncio.timeout(0.2):
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(opened) == 1
        with pytest.raises(ValueError, match="closed image"):
            opened[0].getpixel((0, 0))
    finally:
        release_worker.set()


async def test_image_opening_is_bounded_and_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()
    two_started = threading.Event()
    release_workers = threading.Event()

    def blocking_open_image(
        _item: DocumentEmbeddingInput,
        *,
        min_image_pixel: int,
    ) -> Image.Image:
        nonlocal active, max_active
        assert min_image_pixel == 2
        with lock:
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                two_started.set()
        assert release_workers.wait(timeout=5)
        with lock:
            active -= 1
        return Image.new("RGB", (8, 8), "white")

    monkeypatch.setattr(document_embedding, "_open_valid_image", blocking_open_image)
    embedder = _embedder(fused=[[1.0, 0.0, 0.0]] * 3)
    task = asyncio.create_task(
        _executor(embedder, batch_size=3, max_concurrency=2).aembed_documents(
            [
                DocumentEmbeddingInput(key="a", text="first", image_bytes=_png_bytes()),
                DocumentEmbeddingInput(key="b", text="second", image_bytes=_png_bytes()),
                DocumentEmbeddingInput(key="c", text="third", image_bytes=_png_bytes()),
            ]
        )
    )

    started_in_parallel = await asyncio.to_thread(two_started.wait, 0.2)
    release_workers.set()
    vectors, _trace = await task

    assert started_in_parallel
    assert max_active == 2
    assert [vector.key for vector in vectors] == ["a", "b", "c"]
    assert [text for text, _image in embedder.embed_index_fused.await_args.args[0]] == [
        "first",
        "second",
        "third",
    ]
