# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Direct image->image retrieval leg over the fused visual chunk vectors."""

import asyncio
import io
import logging
import math
from dataclasses import dataclass
from typing import Any

from PIL import Image

from dlightrag.engine.ai.concurrency import bounded_map
from dlightrag.engine.ai.fingerprints import normalized_endpoint_fingerprint
from dlightrag.engine.ai.media import (
    decode_image_base64,
    image_url_block,
    verify_web_image_bytes,
)
from dlightrag.engine.rag.retrieval import ContextRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VisualEmbeddingDomain:
    """Non-secret identity of one query-image vector space."""

    provider: str
    model: str
    endpoint_fingerprint: str | None
    dimension: int
    input_modality: str

    def __post_init__(self) -> None:
        if not self.provider or not self.model:
            raise ValueError("Visual embedding provider and model are required")
        if self.dimension < 1:
            raise ValueError("Visual embedding dimension must be positive")
        if not self.input_modality:
            raise ValueError("Visual embedding input modality is required")


@dataclass(frozen=True, slots=True)
class PreparedVisualQuery:
    """Immutable query-image vectors that are valid only in ``domain``."""

    domain: VisualEmbeddingDomain
    vectors: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        for vector in self.vectors:
            if len(vector) != self.domain.dimension:
                raise ValueError("Prepared visual query vector dimension does not match its domain")
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Prepared visual query vectors must contain finite values")


class DirectVisualRetriever:
    """Rank chunks by raw query-image similarity beside semantic and BM25 legs.

    Preparation owns image decoding and the provider request. Store search accepts
    only immutable prepared vectors and rejects a different embedding domain. Every
    provider, decode, or store failure still degrades to an empty ranking.
    """

    def __init__(self, *, embedder: Any, stores: Any, top_k: int) -> None:
        self._embedder = embedder
        self._stores = stores
        self._top_k = max(0, int(top_k))
        self._domain = _embedding_domain(embedder) if self._top_k > 0 else None

    @property
    def embedding_domain(self) -> VisualEmbeddingDomain | None:
        """Return compatibility facts only while this retrieval leg is enabled."""
        return self._domain if self._top_k > 0 else None

    async def prepare(
        self, query_image_blocks: list[dict[str, Any]] | None
    ) -> PreparedVisualQuery | None:
        """Decode, validate, embed once, and close all query images."""
        domain = self.embedding_domain
        if domain is None or not query_image_blocks:
            return None
        images = await asyncio.to_thread(_extract_images, query_image_blocks)
        if not images:
            return None
        try:
            raw_vectors = await self._embedder.embed_query_images(images)
            vectors = _immutable_vectors(raw_vectors, expected=len(images), domain=domain)
        except Exception:
            logger.warning("Direct visual query embedding failed", exc_info=True)
            return None
        finally:
            for image in images:
                image.close()
        if not vectors:
            return None
        return PreparedVisualQuery(domain=domain, vectors=vectors)

    async def search_prepared(self, prepared: PreparedVisualQuery) -> list[ContextRow]:
        """Search this workspace VDB without decoding or embedding query images."""
        domain = self.embedding_domain
        if domain is None or prepared.domain != domain:
            logger.warning("Prepared visual query embedding domain mismatch")
            return []
        if not prepared.vectors:
            return []

        async def _search(vector: tuple[float, ...]) -> list[ContextRow]:
            try:
                return (
                    await self._stores.chunks_vdb.query(
                        query="", top_k=self._top_k, query_embedding=list(vector)
                    )
                    or []
                )
            except Exception:
                logger.warning("Direct visual vector search failed", exc_info=True)
                return []

        results = await bounded_map(
            list(prepared.vectors),
            _search,
            max_concurrent=min(8, max(1, len(prepared.vectors))),
            task_name="direct-visual-query",
        )
        merged: dict[str, ContextRow] = {}
        for raw_chunks in results:
            if isinstance(raw_chunks, Exception):
                continue
            for chunk in raw_chunks:
                cid = chunk.get("id")
                if not cid:
                    continue
                distance = chunk.get("distance")
                existing = merged.get(cid)
                existing_distance = existing.get("relevance_score") if existing else None
                if existing is None or (
                    distance is not None
                    and (existing_distance is None or distance < existing_distance)
                ):
                    merged[cid] = {
                        "chunk_id": cid,
                        "content": chunk.get("content", ""),
                        "file_path": chunk.get("file_path", ""),
                        "reference_id": "",
                        "relevance_score": distance,
                    }
                    if chunk.get("full_doc_id"):
                        merged[cid]["full_doc_id"] = chunk["full_doc_id"]
        return sorted(
            merged.values(),
            key=lambda chunk: (
                chunk["relevance_score"]
                if chunk.get("relevance_score") is not None
                else float("inf")
            ),
        )[: self._top_k]

    async def search(self, query_image_blocks: list[dict[str, Any]] | None) -> list[ContextRow]:
        """Convenience path for non-application callers: prepare, then search."""
        prepared = await self.prepare(query_image_blocks)
        return await self.search_prepared(prepared) if prepared is not None else []


def _embedding_domain(embedder: Any) -> VisualEmbeddingDomain:
    fingerprint = getattr(embedder, "fingerprint", None)
    provider = getattr(fingerprint, "provider", None)
    if not isinstance(provider, str) or not provider:
        provider_value = getattr(embedder, "provider", None)
        if isinstance(provider_value, str) and provider_value:
            provider = provider_value
        elif provider_value is not None and not _is_mock_like(provider_value):
            provider = f"{type(provider_value).__module__}.{type(provider_value).__qualname__}"
        else:
            provider = f"{type(embedder).__module__}.{type(embedder).__qualname__}"

    model = getattr(embedder, "model", None)
    if not isinstance(model, str) or not model:
        raise ValueError("Direct visual embedder must expose a non-empty model")
    dimension = getattr(embedder, "dim", None)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Direct visual embedder must expose a positive integer dimension")

    input_modality = getattr(embedder, "input_modality", None)
    if not isinstance(input_modality, str) or not input_modality:
        # Direct visual retrieval is constructed only after image capability is
        # resolved; lightweight test embedders may omit the resolved attribute.
        input_modality = "multimodal"

    request_url = getattr(embedder, "request_url", None)
    endpoint_fingerprint = normalized_endpoint_fingerprint(
        request_url if isinstance(request_url, str) else None
    )
    if endpoint_fingerprint is None:
        fingerprint_endpoint = getattr(fingerprint, "endpoint_fingerprint", None)
        endpoint_fingerprint = (
            fingerprint_endpoint if isinstance(fingerprint_endpoint, str) else None
        )
    if endpoint_fingerprint is None:
        base_url = getattr(embedder, "base_url", None)
        endpoint_fingerprint = normalized_endpoint_fingerprint(
            base_url if isinstance(base_url, str) else None
        )

    return VisualEmbeddingDomain(
        provider=provider,
        model=model,
        endpoint_fingerprint=endpoint_fingerprint,
        dimension=dimension,
        input_modality=input_modality,
    )


def _is_mock_like(value: object) -> bool:
    return type(value).__module__.startswith("unittest.mock")


def _immutable_vectors(
    raw_vectors: Any,
    *,
    expected: int,
    domain: VisualEmbeddingDomain,
) -> tuple[tuple[float, ...], ...]:
    vectors = tuple(tuple(float(value) for value in vector) for vector in raw_vectors)
    if len(vectors) != expected:
        raise ValueError("Direct visual embedding returned an unexpected vector count")
    # PreparedVisualQuery repeats this invariant at the typed boundary; checking
    # here keeps malformed provider output in failure-to-empty semantics.
    for vector in vectors:
        if len(vector) != domain.dimension or not all(math.isfinite(value) for value in vector):
            raise ValueError("Direct visual embedding returned an invalid vector")
    return vectors


def _extract_images(blocks: list[dict[str, Any]] | None) -> list[Image.Image]:
    images: list[Image.Image] = []
    for item in blocks or []:
        if item.get("type") != "image_url":
            continue
        block = image_url_block(item)
        if block is None:
            continue
        image_url = block.get("image_url")
        if not isinstance(image_url, dict):
            continue
        url = image_url.get("url")
        if not isinstance(url, str) or not url.strip().startswith("data:"):
            continue
        try:
            raw, _ = decode_image_base64(url)
            verify_web_image_bytes(raw)
            images.append(Image.open(io.BytesIO(raw)))
        except Exception:
            logger.warning("Failed to decode direct visual query image", exc_info=True)
    return images


__all__ = ["DirectVisualRetriever", "PreparedVisualQuery", "VisualEmbeddingDomain"]
