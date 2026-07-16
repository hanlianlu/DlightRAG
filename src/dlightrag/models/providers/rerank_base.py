# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Abstract base for HTTP rerank providers.

One transport driver (``_run_http_rerank`` in :mod:`dlightrag.models.rerank`)
runs every provider against this contract; a subclass supplies only the wire
shape that differs (URL, auth, request body, response key). This mirrors the
embedding side's :class:`~dlightrag.models.providers.embed_base.EmbedProvider`.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

InputModality = Literal["auto", "text", "multimodal"]
ResolvedInputModality = Literal["text", "multimodal"]

# One prepared candidate: its text plus an optional bounded image data URI.
PreparedDocument = tuple[str, str | None]


def resolve_rerank_input_modality(input_modality: InputModality) -> ResolvedInputModality:
    """Resolve the configured rerank modality with no per-model capability table.

    ``input_modality`` is the whole signal. ``auto`` defaults to ``text`` (never
    errors); ``multimodal`` is an explicit opt-in that a text-only provider
    rejects loudly at build time. Unlike embedding, rerank has no reliable
    capability probe -- the API returns a score whether or not it read the image
    -- so a safe default plus explicit opt-in replaces name-matching guesses.
    """
    return "text" if input_modality == "auto" else input_modality


class RerankProvider(ABC):
    """Strategy for a provider-specific HTTP rerank protocol."""

    #: Full default endpoint URL, or ``None`` when ``base_url`` is required.
    default_base_url: str | None = None
    #: Default model when the config omits one.
    default_model: str = ""
    #: Whether this API protocol accepts image documents at all. Text-only
    #: protocols reject ``input_modality='multimodal'`` at build time.
    accepts_images: bool = False
    #: Whether the strategy requires an explicit ``base_url``.
    requires_base_url: bool = False
    #: Whether the strategy requires an API key.
    requires_api_key: bool = True

    @abstractmethod
    def build_payload(
        self,
        *,
        model: str,
        query: str,
        documents: list[PreparedDocument],
        top_n: int,
    ) -> dict[str, Any]:
        """Build the provider-specific rerank request body."""

    def request_url(self, base_url: str | None, model: str) -> str:
        """Return the full request URL for this provider."""
        url = base_url or self.default_base_url
        if url is None:
            raise ValueError(f"{self.__class__.__name__} requires base_url")
        return url

    def request_headers(self, api_key: str | None) -> dict[str, str]:
        """Return provider-specific request headers (Bearer auth by default)."""
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def parse_results(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract the ``[{index, relevance_score}]`` list from the response."""
        return data["results"]
