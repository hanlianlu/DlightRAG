# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared robust document and query embedding over one borrowed provider."""

import asyncio
import io
import logging
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image

from dlightrag.utils import log_safe
from dlightrag.utils.concurrency import bounded_map
from dlightrag.utils.images import flatten_image_to_rgb

if TYPE_CHECKING:
    from dlightrag.models.multimodal_embedding import MultimodalEmbedder

logger = logging.getLogger(__name__)

DocumentEmbeddingMode = Literal["fused", "text"]

# Let a near-complete decode return its PIL resource without materially delaying cancellation.
_IMAGE_OPEN_CLEANUP_TIMEOUT_SECONDS = 0.1
# Decoded images are uncompressed, so cap their parallel memory footprint below provider capacity.
_MAX_IMAGE_DECODE_CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class DocumentEmbeddingInput:
    """One document chunk and its optional visual source."""

    key: str
    text: str
    image_bytes: bytes | None = None
    image_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DocumentEmbeddingVector:
    """One validated document vector and its effective embedding mode."""

    key: str
    vector: list[float]
    mode: DocumentEmbeddingMode


@dataclass(frozen=True, slots=True)
class DocumentEmbeddingTrace:
    """Per-item outcomes from one document embedding request."""

    fused: int
    text: int
    fused_to_text_fallback: int
    failed: int
    error_type: str | None = None


class RobustDocumentEmbedder:
    """Validate and batch document embeddings without owning the provider."""

    def __init__(
        self,
        *,
        embedder: MultimodalEmbedder,
        image_enabled: bool,
        dimension: int,
        min_image_pixel: int,
        batch_size: int,
        max_concurrency: int,
        image_cleanup_timeout: float = _IMAGE_OPEN_CLEANUP_TIMEOUT_SECONDS,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if min_image_pixel < 1:
            raise ValueError("min_image_pixel must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if not math.isfinite(image_cleanup_timeout) or image_cleanup_timeout <= 0:
            raise ValueError("image_cleanup_timeout must be positive and finite")
        self._embedder = embedder
        self._image_enabled = image_enabled
        self._dimension = dimension
        self._min_image_pixel = min_image_pixel
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._image_decode_concurrency = min(
            batch_size,
            max_concurrency,
            _MAX_IMAGE_DECODE_CONCURRENCY,
        )
        self._image_cleanup_timeout = image_cleanup_timeout

    @property
    def image_enabled(self) -> bool:
        """Return whether valid images may use fused embedding."""
        return self._image_enabled

    @property
    def dimension(self) -> int:
        """Return the configured output vector dimension."""
        return self._dimension

    @property
    def asymmetric(self) -> bool:
        """Return the provider's resolved asymmetric embedding mode."""
        return bool(self._embedder.asymmetric)

    @property
    def min_image_pixel(self) -> int:
        """Return the effective minimum image side admitted for fused embedding."""
        return self._min_image_pixel

    async def aembed_documents(
        self,
        items: list[DocumentEmbeddingInput],
    ) -> tuple[list[DocumentEmbeddingVector], DocumentEmbeddingTrace]:
        """Embed document chunks with batch-local fused-to-text fallback."""
        resolved: dict[int, DocumentEmbeddingVector] = {}
        fused_count = 0
        text_count = 0
        fallback_count = 0
        failed_count = 0
        error_type: str | None = None

        for start in range(0, len(items), self._batch_size):
            batch = list(enumerate(items[start : start + self._batch_size], start=start))
            fused_items: list[tuple[int, DocumentEmbeddingInput, Image.Image]] = []
            text_items: list[tuple[int, DocumentEmbeddingInput]] = []
            opened_images: list[Image.Image] = []
            try:
                opened = await bounded_map(
                    batch,
                    partial(self._aopen_tracked_image, opened_images=opened_images),
                    max_concurrent=self._image_decode_concurrency,
                    task_name="document image open",
                )
                for position, result in enumerate(opened):
                    index, item = batch[position]
                    image = None if isinstance(result, Exception) else result[2]
                    if image is None:
                        text_items.append((index, item))
                    else:
                        fused_items.append((index, item, image))

                if text_items:
                    text_vectors, text_error_type = await self._aembed_text_batch(text_items)
                    error_type = error_type or text_error_type
                    if text_vectors is None:
                        failed_count += len(text_items)
                    else:
                        for (index, item), vector in zip(text_items, text_vectors, strict=True):
                            resolved[index] = DocumentEmbeddingVector(item.key, vector, "text")
                        text_count += len(text_items)

                if fused_items:
                    try:
                        fused_vectors = await self._aembed_fused_batch(fused_items)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        error_type = error_type or type(exc).__name__
                        logger.warning(
                            "Fused document embedding failed; falling back to text for %d item(s)",
                            len(fused_items),
                            exc_info=True,
                        )
                        _close_images(opened_images)
                        opened_images.clear()
                        fallback_items = [(index, item) for index, item, _image in fused_items]
                        fallback_count += len(fallback_items)
                        fallback_vectors, fallback_error_type = await self._aembed_text_batch(
                            fallback_items
                        )
                        error_type = error_type or fallback_error_type
                        if fallback_vectors is None:
                            failed_count += len(fallback_items)
                        else:
                            for (index, item), vector in zip(
                                fallback_items, fallback_vectors, strict=True
                            ):
                                resolved[index] = DocumentEmbeddingVector(item.key, vector, "text")
                            text_count += len(fallback_items)
                    else:
                        for (index, item, _image), vector in zip(
                            fused_items, fused_vectors, strict=True
                        ):
                            resolved[index] = DocumentEmbeddingVector(item.key, vector, "fused")
                        fused_count += len(fused_items)
            finally:
                _close_images(opened_images)

        vectors = [resolved[index] for index in range(len(items)) if index in resolved]
        return vectors, DocumentEmbeddingTrace(
            fused=fused_count,
            text=text_count,
            fused_to_text_fallback=fallback_count,
            failed=failed_count,
            error_type=error_type,
        )

    async def aembed_query(self, query: str) -> list[float] | None:
        """Return one validated query-context vector, or ``None`` on failure."""
        try:
            async with self._semaphore:
                vectors = await self._embedder.embed_texts([query], context="query")
            return self._validate_vectors(vectors, expected_count=1)[0]
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("Query embedding failed validation or provider execution", exc_info=True)
            return None

    async def _aembed_text_batch(
        self,
        items: list[tuple[int, DocumentEmbeddingInput]],
    ) -> tuple[list[list[float]] | None, str | None]:
        try:
            async with self._semaphore:
                vectors = await self._embedder.embed_texts(
                    [item.text for _index, item in items],
                    context="document",
                )
            return self._validate_vectors(vectors, expected_count=len(items)), None
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Text document embedding failed; omitting %d item(s)",
                len(items),
                exc_info=True,
            )
            return None, type(exc).__name__

    async def _aembed_fused_batch(
        self,
        items: list[tuple[int, DocumentEmbeddingInput, Image.Image]],
    ) -> list[list[float]]:
        async with self._semaphore:
            vectors = await self._embedder.embed_index_fused(
                [(item.text, image) for _index, item, image in items]
            )
        return self._validate_vectors(vectors, expected_count=len(items))

    async def _aopen_indexed_image(
        self,
        indexed_item: tuple[int, DocumentEmbeddingInput],
    ) -> tuple[int, DocumentEmbeddingInput, Image.Image | None]:
        index, item = indexed_item
        image = await self._aopen_valid_image(item) if self._image_enabled else None
        return index, item, image

    async def _aopen_tracked_image(
        self,
        indexed_item: tuple[int, DocumentEmbeddingInput],
        *,
        opened_images: list[Image.Image],
    ) -> tuple[int, DocumentEmbeddingInput, Image.Image | None]:
        result = await self._aopen_indexed_image(indexed_item)
        if result[2] is not None:
            opened_images.append(result[2])
        return result

    async def _aopen_valid_image(self, item: DocumentEmbeddingInput) -> Image.Image | None:
        if item.image_bytes is None and item.image_path is None:
            return None
        task = asyncio.create_task(
            _aopen_image_result(
                item,
                min_image_pixel=self._min_image_pixel,
            )
        )
        try:
            result = await asyncio.shield(task)
            if isinstance(result, BaseException):
                raise result
            return result
        except asyncio.CancelledError:
            waiter = asyncio.create_task(asyncio.wait({task}, timeout=self._image_cleanup_timeout))
            while True:
                try:
                    done, _pending = await asyncio.shield(waiter)
                except asyncio.CancelledError:
                    continue
                break
            if task in done:
                _close_image_task_result(task)
            else:
                task.add_done_callback(_close_image_task_result)
            raise
        except Exception:  # noqa: BLE001
            source = f"path:{log_safe(item.image_path)}" if item.image_path is not None else "bytes"
            logger.warning(
                "Document image could not be opened; using text embedding key=%s source=%s",
                log_safe(item.key),
                source,
                exc_info=True,
            )
            return None

    def _validate_vectors(
        self,
        vectors: object,
        *,
        expected_count: int,
    ) -> list[list[float]]:
        if not isinstance(vectors, list) or len(vectors) != expected_count:
            actual_count = len(vectors) if isinstance(vectors, list) else "non-list"
            raise ValueError(f"Expected {expected_count} embedding vectors, got {actual_count}")
        validated: list[list[float]] = []
        for index, vector in enumerate(vectors):
            if not isinstance(vector, list) or len(vector) != self._dimension:
                actual_dimension = len(vector) if isinstance(vector, list) else "non-list"
                raise ValueError(
                    f"Expected embedding dim {self._dimension}, got {actual_dimension} "
                    f"at index {index}"
                )
            if not all(isinstance(value, int | float) and math.isfinite(value) for value in vector):
                raise ValueError(f"Embedding vector at index {index} contains non-finite values")
            normalized = [float(value) for value in vector]
            norm = math.hypot(*normalized)
            if not math.isfinite(norm) or norm == 0.0:
                raise ValueError(f"Embedding vector at index {index} has invalid norm")
            validated.append(normalized)
        return validated


def _open_valid_image(
    item: DocumentEmbeddingInput,
    *,
    min_image_pixel: int,
) -> Image.Image | None:
    source: io.BytesIO | Path
    source = io.BytesIO(item.image_bytes) if item.image_bytes is not None else item.image_path  # type: ignore[assignment]
    image = Image.open(source)
    try:
        image.load()
        if image.width < min_image_pixel or image.height < min_image_pixel:
            image.close()
            return None
        flattened = flatten_image_to_rgb(image)
        if flattened is not image:
            image.close()
        return flattened
    except BaseException:
        image.close()
        raise


async def _aopen_image_result(
    item: DocumentEmbeddingInput,
    *,
    min_image_pixel: int,
) -> Image.Image | None | BaseException:
    try:
        return await asyncio.to_thread(
            _open_valid_image,
            item,
            min_image_pixel=min_image_pixel,
        )
    except BaseException as exc:
        return exc


def _close_images(images: list[Image.Image]) -> None:
    for image in images:
        image.close()


def _close_image_task_result(task: asyncio.Task[Image.Image | None | BaseException]) -> None:
    if task.cancelled():
        return
    try:
        image = task.result()
    except BaseException:
        return
    if isinstance(image, Image.Image):
        image.close()


__all__ = [
    "DocumentEmbeddingInput",
    "DocumentEmbeddingTrace",
    "DocumentEmbeddingVector",
    "RobustDocumentEmbedder",
]
