# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tavily REST adapter for provider-neutral Web search and extraction."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from dlightrag.engine.answer.web_sources.contracts import (
    WebEffort,
    WebExtractResult,
    WebSearchHit,
    WebSearchRequest,
    WebSearchResult,
    WebSourceUnavailable,
)
from dlightrag.engine.public_http import PublicHttpPolicyError, validate_agent_public_url

logger = logging.getLogger(__name__)

_SEARCH_ENDPOINT = "https://api.tavily.com/search"
_EXTRACT_ENDPOINT = "https://api.tavily.com/extract"
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_TIMEOUT_SECONDS = 20.0
_SEARCH_DEPTH: dict[WebEffort, str] = {
    "fast": "basic",
    "balanced": "advanced",
    "deep": "advanced",
}
_EXTRACT_DEPTH: dict[WebEffort, str] = {
    "fast": "basic",
    "balanced": "advanced",
    "deep": "advanced",
}
_STATUS_REASONS = {401: "unauthorized", 402: "payment_required", 429: "rate_limited"}


class TavilyWebSource:
    """Tavily Search and Extract through their direct REST APIs."""

    name = "tavily"

    def __init__(self, api_key: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._api_key = api_key
        self._client = client if client is not None else _default_client()
        self._owns_client = client is None

    async def search(self, request: WebSearchRequest) -> WebSearchResult:
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": request.query,
            "max_results": request.max_results,
            "search_depth": _SEARCH_DEPTH[request.effort],
            "include_answer": False,
            "include_raw_content": False,
        }
        if request.effort == "deep":
            payload["chunks_per_source"] = 3
        if request.include_domains:
            payload["include_domains"] = list(request.include_domains)
        if request.exclude_domains:
            payload["exclude_domains"] = list(request.exclude_domains)
        if request.start_date:
            payload["start_date"] = request.start_date
        if request.end_date:
            payload["end_date"] = request.end_date
        response = await self._post(_SEARCH_ENDPOINT, payload, operation="search")
        return _read_search_result(response)

    async def extract(self, url: str, *, effort: WebEffort) -> WebExtractResult:
        response = await self._post(
            _EXTRACT_ENDPOINT,
            {
                "api_key": self._api_key,
                "urls": [url],
                "extract_depth": _EXTRACT_DEPTH[effort],
                "include_images": False,
                "format": "markdown",
            },
            operation="extract",
        )
        payload = response
        results = payload["results"]
        dropped = 0
        passages: list[str] = []
        final_url = url
        for item in results or ():
            if not isinstance(item, dict):
                dropped += 1
                continue
            candidate_url = _text_or_none(item.get("url"))
            body = _body_or_none(item.get("raw_content") or item.get("content"))
            if candidate_url is None or body is None:
                dropped += 1
                continue
            try:
                validate_agent_public_url(candidate_url)
            except PublicHttpPolicyError:
                dropped += 1
                continue
            if passages:
                dropped += 1
                continue
            final_url = candidate_url
            passages.append(body)
        if not passages:
            raise WebSourceUnavailable(self.name, "extract", "empty")
        return WebExtractResult(
            url=final_url,
            text=passages[0],
            provider=self.name,
            acquisition="tavily_extract",
            dropped_results=dropped,
        )

    async def _post(
        self, endpoint: str, payload: dict[str, Any], *, operation: str
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout(_TIMEOUT_SECONDS):
                async with self._client.stream("POST", endpoint, json=payload) as response:
                    if response.is_error:
                        reason = _STATUS_REASONS.get(response.status_code, "error")
                        logger.warning(
                            "Tavily %s returned HTTP %d", operation, response.status_code
                        )
                        raise WebSourceUnavailable(self.name, operation, reason)
                    raw = await _bounded_response_bytes(response, operation=operation)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise WebSourceUnavailable(self.name, operation, "timeout") from exc
        except httpx.HTTPError as exc:
            raise WebSourceUnavailable(self.name, operation, "unreachable") from exc
        try:
            decoded = json.loads(raw)
        except (UnicodeError, ValueError) as exc:
            raise WebSourceUnavailable(self.name, operation, "invalid_response") from exc
        if not isinstance(decoded, dict) or not isinstance(decoded.get("results"), list):
            raise WebSourceUnavailable(self.name, operation, "invalid_response")
        return decoded

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _default_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)


def _read_search_result(payload: dict[str, Any]) -> WebSearchResult:
    results = payload["results"]
    hits: list[WebSearchHit] = []
    dropped = 0
    for item in results or ():
        if not isinstance(item, dict):
            dropped += 1
            continue
        url = _text_or_none(item.get("url"))
        text = _text_or_none(item.get("content"))
        if url is None or text is None:
            dropped += 1
            continue
        try:
            validate_agent_public_url(url)
        except PublicHttpPolicyError:
            dropped += 1
            continue
        hits.append(
            WebSearchHit(
                url=url,
                title=_text_or_none(item.get("title")) or url,
                text=text,
                published_date=_text_or_none(item.get("published_date")),
                acquisition="tavily_search",
            )
        )
    if dropped:
        logger.warning("Tavily search dropped %d malformed or unusable result(s)", dropped)
    return WebSearchResult(hits=tuple(hits), provider="tavily", dropped_results=dropped)


async def _bounded_response_bytes(response: httpx.Response, *, operation: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _MAX_RESPONSE_BYTES:
            raise WebSourceUnavailable("tavily", operation, "response_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def _body_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _text_or_none(value: Any) -> str | None:
    text = value.strip() if isinstance(value, str) else ""
    return text or None


__all__ = ["TavilyWebSource"]
