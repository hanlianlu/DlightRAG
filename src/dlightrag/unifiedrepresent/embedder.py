# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Visual embedding for unified representational RAG.

Embeds page images into visual vectors via OpenAI-compatible multimodal
embedding API (e.g., qwen3-vl-embedding).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging

import httpx
from PIL import Image

from dlightrag.models.providers.embed_providers import (
    OpenAICompatEmbedProvider,
    VoyageEmbedProvider,
)

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """Embed page images (and text queries) via an OpenAI-compatible multimodal embedding API.

    Reuses a single ``httpx.AsyncClient`` for connection pooling across requests.
    Call ``aclose()`` when done to release resources.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        dim: int,
        batch_size: int = 4,
        timeout: float = 120.0,
        provider: OpenAICompatEmbedProvider | VoyageEmbedProvider | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dim = dim
        self.batch_size = batch_size
        self.provider = provider or OpenAICompatEmbedProvider()
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=httpx.AsyncHTTPTransport(retries=2),
        )

    async def aclose(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_pages(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed a list of PIL images into visual vectors.

        Multimodal embedding APIs treat a list of images as a single
        composite input (returning 1 vector), so each image must be
        embedded individually.  ``batch_size`` controls concurrency.

        Returns:
            List of embedding vectors, one per input image.
        """
        if not images:
            return []

        sem = asyncio.Semaphore(self.batch_size)

        async def _embed_one(img: Image.Image) -> list[float]:
            async with sem:
                image_b64 = self._image_to_b64(img)
                payload = self._build_image_payload(image_b64)
                resp: httpx.Response | None = None
                for attempt in range(4):  # 1 initial + 3 retries
                    resp = await self._client.post(
                        f"{self.base_url}{self.provider.endpoint}",
                        json=payload,
                    )
                    if resp.status_code == 429 and attempt < 3:
                        retry_after = float(resp.headers.get("retry-after", 2 ** (attempt + 1)))
                        logger.warning(
                            "429 rate-limited, retrying in %.1fs (attempt %d/3)",
                            retry_after,
                            attempt + 1,
                        )
                        await asyncio.sleep(retry_after)
                        continue
                    resp.raise_for_status()
                    break
                if resp is None:
                    raise RuntimeError("Embedding API: no response after retries")
                embeddings = self.provider.parse_response(resp.json())
                vec = embeddings[0]
                if len(vec) != self.dim:
                    raise ValueError(f"Expected embedding dim {self.dim}, got {len(vec)}")
                return vec

        results = await asyncio.gather(*[_embed_one(img) for img in images])
        return list(results)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the same multimodal embedding API.

        Used as the LightRAG ``embedding_func`` in unified mode so that
        entity/relationship embedding goes through the same httpx client
        (and ``encoding_format: "float"``) as visual embedding — avoiding
        LightRAG's ``openai_embed`` which uses ``encoding_format: "base64"``
        and the ``openai`` Python client, both of which can fail with
        Xinference VL embedding models.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []
        payload = self.provider.build_payload(self.model, texts)
        resp: httpx.Response | None = None
        for attempt in range(4):  # 1 initial + 3 retries
            resp = await self._client.post(
                f"{self.base_url}{self.provider.endpoint}",
                json=payload,
            )
            if resp.status_code == 429 and attempt < 3:
                retry_after = float(resp.headers.get("retry-after", 2 ** (attempt + 1)))
                logger.warning(
                    "429 rate-limited, retrying in %.1fs (attempt %d/3)", retry_after, attempt + 1
                )
                await asyncio.sleep(retry_after)
                continue
            resp.raise_for_status()
            break
        if resp is None:
            raise RuntimeError("Embedding API: no response after retries")
        embeddings = self.provider.parse_response(resp.json())
        if len(embeddings) > 0 and len(embeddings[0]) != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {len(embeddings[0])}")
        return embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_image_payload(self, image_b64: str) -> dict:
        """Build payload for image embedding based on provider type."""
        if isinstance(self.provider, VoyageEmbedProvider):
            return {
                "model": self.model,
                "inputs": [{"content": [{"type": "image_base64", "image_base64": image_b64}]}],
            }
        # OpenAICompatEmbedProvider and others
        return {"model": self.model, "input": image_b64, "encoding_format": "float"}

    @staticmethod
    def _image_to_b64(image: Image.Image) -> str:
        """Convert a PIL Image to a ``data:image/png;base64,...`` URI string."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
