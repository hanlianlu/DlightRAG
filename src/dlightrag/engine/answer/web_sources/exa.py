# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Exa REST adapter for provider-neutral Web search and extraction."""

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

_SEARCH_ENDPOINT = "https://api.exa.ai/search"
_EXTRACT_ENDPOINT = "https://api.exa.ai/contents"
_HIGHLIGHT_MAX_CHARACTERS = 4000
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_TIMEOUT_SECONDS = 15.0
_EFFORT_TYPE: dict[WebEffort, str] = {
    "fast": "fast",
    "balanced": "auto",
    "deep": "deep",
}
_STATUS_REASONS = {401: "unauthorized", 402: "payment_required", 429: "rate_limited"}


class ExaWebSource:
    """Exa Search and Contents through their direct REST APIs."""

    name = "exa"

    def __init__(self, api_key: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._api_key = api_key
        self._client = client if client is not None else _default_client()
        self._owns_client = client is None

    async def search(self, request: WebSearchRequest) -> WebSearchResult:
        payload: dict[str, Any] = {
            "query": request.query,
            "type": _EFFORT_TYPE[request.effort],
            "numResults": request.max_results,
            "contents": {"highlights": {"maxCharacters": _HIGHLIGHT_MAX_CHARACTERS}},
        }
        if request.include_domains:
            payload["includeDomains"] = list(request.include_domains)
        if request.exclude_domains:
            payload["excludeDomains"] = list(request.exclude_domains)
        if request.start_date:
            payload["startPublishedDate"] = request.start_date
        if request.end_date:
            payload["endPublishedDate"] = request.end_date
        response = await self._post(_SEARCH_ENDPOINT, payload, operation="search")
        return _read_search_result(response)

    async def extract(self, url: str, *, effort: WebEffort) -> WebExtractResult:
        response = await self._post(
            _EXTRACT_ENDPOINT,
            {
                "urls": [url],
                "text": {"maxCharacters": 10_000},
                "livecrawl": "always" if effort == "deep" else "fallback",
            },
            operation="extract",
        )
        payload = response
        results = payload["results"]
        dropped = 0
        passages: list[str] = []
        title: str | None = None
        final_url = url
        for item in results or ():
            if not isinstance(item, dict):
                dropped += 1
                continue
            candidate_url = _text_or_none(item.get("url"))
            body = _body_or_none(item.get("text"))
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
            title = _text_or_none(item.get("title"))
            passages.append(body)
        if not passages:
            raise WebSourceUnavailable(self.name, "extract", "empty")
        return WebExtractResult(
            url=final_url,
            title=title,
            text=passages[0],
            provider=self.name,
            acquisition="exa_extract",
            dropped_results=dropped,
        )

    async def _post(
        self, endpoint: str, payload: dict[str, Any], *, operation: str
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout(_TIMEOUT_SECONDS):
                async with self._client.stream(
                    "POST",
                    endpoint,
                    headers={"x-api-key": self._api_key},
                    json=payload,
                ) as response:
                    if response.is_error:
                        reason = _STATUS_REASONS.get(response.status_code, "error")
                        logger.warning("Exa %s returned HTTP %d", operation, response.status_code)
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
        if url is None:
            dropped += 1
            continue
        try:
            validate_agent_public_url(url)
        except PublicHttpPolicyError:
            dropped += 1
            continue
        page = {
            "url": url,
            "title": _text_or_none(item.get("title")) or url,
            "published_date": _text_or_none(item.get("publishedDate")),
            "image_url": _text_or_none(item.get("image")),
            "acquisition": "exa_search",
        }
        passages = 0
        for passage in item.get("highlights") or ():
            text = _text_or_none(passage)
            if text is not None:
                hits.append(WebSearchHit(text=text, **page))  # type: ignore[arg-type]
                passages += 1
        body = _text_or_none(item.get("text"))
        if body is not None:
            hits.append(WebSearchHit(text=body, **page))  # type: ignore[arg-type]
            passages += 1
        if passages == 0:
            dropped += 1
    cost = payload.get("costDollars") if isinstance(payload, dict) else None
    total = cost.get("total") if isinstance(cost, dict) else None
    if dropped:
        logger.warning("Exa search dropped %d malformed or unusable result(s)", dropped)
    return WebSearchResult(
        hits=tuple(hits),
        cost_dollars=float(total) if isinstance(total, int | float) else 0.0,
        provider="exa",
        dropped_results=dropped,
    )


async def _bounded_response_bytes(response: httpx.Response, *, operation: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _MAX_RESPONSE_BYTES:
            raise WebSourceUnavailable("exa", operation, "response_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def _body_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _text_or_none(value: Any) -> str | None:
    text = value.strip() if isinstance(value, str) else ""
    return text or None


__all__ = ["ExaWebSource"]
