# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DeepSeek REST adapter for provider-neutral Web search and extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any
from urllib.parse import urlsplit

import httpx

from dlightrag.engine.answer.web_sources.contracts import (
    WebExtractResult,
    WebSearchHit,
    WebSearchRequest,
    WebSearchResult,
    WebSourceUnavailable,
)
from dlightrag.engine.public_http import PublicHttpPolicyError, validate_agent_public_url

logger = logging.getLogger(__name__)

_SEARCH_ENDPOINT = "https://api.deepseek.com/v1/tools/search"
_FETCH_ENDPOINT = "https://api.deepseek.com/v1/tools/fetch"
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_SEARCH_TIMEOUT_SECONDS = 20.0
_FETCH_TIMEOUT_SECONDS = 30.0
#: Search takes at most this many Unicode code points of query, operators included.
_MAX_QUERY_CHARACTERS = 128
#: What a long query text keeps when domain operators compete with it for the budget.
_MIN_TEXT_CHARACTERS = 32
#: Fetch refuses a longer URL, so a longer one is left to the next provider.
_MAX_URL_CHARACTERS = 8192
#: The account's QPS limit is per second, so one short wait is what a 429 asks for.
_THROTTLE_WAIT_SECONDS = 1.0
_THROTTLE_JITTER_SECONDS = 0.25
_STATUS_REASONS = {401: "unauthorized", 402: "payment_required", 429: "throttled"}
#: Fetch answers 403 and 404 about the page rather than about the call.
_FETCH_STATUS_REASONS = {403: "refused", 404: "empty"}


class DeepSeekWebSource:
    """DeepSeek Search and Fetch through their direct REST APIs."""

    name = "deepseek"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = _default_client()

    async def search(self, request: WebSearchRequest) -> WebSearchResult:
        # No depth knob exists, so ``fast`` and ``balanced`` make the same request.
        payload: dict[str, Any] = {
            "query": _search_query(request),
            "count": request.max_results,
            "need_content": False,
        }
        response = await self._post(
            _SEARCH_ENDPOINT, payload, operation="search", timeout=_SEARCH_TIMEOUT_SECONDS
        )
        if not isinstance(response.get("results"), list):
            raise WebSourceUnavailable(self.name, "search", "invalid_response")
        return _read_search_result(response, request)

    async def extract(self, url: str) -> WebExtractResult:
        if len(url) > _MAX_URL_CHARACTERS:
            raise WebSourceUnavailable(self.name, "extract", "unsupported_url")
        response = await self._post(
            _FETCH_ENDPOINT, {"url": url}, operation="extract", timeout=_FETCH_TIMEOUT_SECONDS
        )
        content = response.get("content")
        if not isinstance(content, str):
            raise WebSourceUnavailable(self.name, "extract", "invalid_response")
        body = _body_or_none(content)
        if body is None:
            raise WebSourceUnavailable(self.name, "extract", "empty")
        # Fetch reports no final URL, so the page stays under the one that was asked for.
        return WebExtractResult(
            url=url,
            title=_text_or_none(response.get("title")),
            text=body,
            provider=self.name,
            acquisition="deepseek_extract",
        )

    async def _post(
        self, endpoint: str, payload: dict[str, Any], *, operation: str, timeout: float
    ) -> dict[str, Any]:
        try:
            # The one timeout covers a throttled first attempt, the wait and the retry.
            async with asyncio.timeout(timeout):
                raw = await self._post_bytes(endpoint, payload, operation=operation)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise WebSourceUnavailable(self.name, operation, "timeout") from exc
        except httpx.HTTPError as exc:
            raise WebSourceUnavailable(self.name, operation, "unreachable") from exc
        try:
            decoded = json.loads(raw)
        except (UnicodeError, ValueError) as exc:
            raise WebSourceUnavailable(self.name, operation, "invalid_response") from exc
        if not isinstance(decoded, dict):
            raise WebSourceUnavailable(self.name, operation, "invalid_response")
        return decoded

    async def _post_bytes(self, endpoint: str, payload: dict[str, Any], *, operation: str) -> bytes:
        """One body, asked for twice at most: a 429 is answered with a wait and one more try."""
        retried = False
        while True:
            async with self._client.stream(
                "POST",
                endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            ) as response:
                status = response.status_code
                if not response.is_error:
                    return await _bounded_response_bytes(response, operation=operation)
            if status == 429 and not retried:
                retried = True
                jitter = random.uniform(0.0, _THROTTLE_JITTER_SECONDS)  # noqa: S311 - retry jitter
                await asyncio.sleep(_THROTTLE_WAIT_SECONDS + jitter)
                continue
            raise WebSourceUnavailable(
                self.name, operation, _failure_reason(status, operation), status=status
            )

    async def aclose(self) -> None:
        await self._client.aclose()


def _default_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS)


def _failure_reason(status: int, operation: str) -> str:
    reason = _STATUS_REASONS.get(status)
    if reason is None and operation == "extract":
        reason = _FETCH_STATUS_REASONS.get(status)
    return reason or "error"


def _search_query(request: WebSearchRequest) -> str:
    """The query text and its operators, within the 128 code points Search accepts.

    The dates are always asked for. A domain operator is asked for when it leaves the text at
    least ``_MIN_TEXT_CHARACTERS`` (or all of it, when it is shorter), and the text takes the rest
    of the budget. An operator that does not fit is not lost: ``_read_search_result`` holds every
    result to the domains whichever way they were asked for.
    """
    text = request.query.strip()
    dates: list[str] = []
    if request.start_date:
        dates.append(f"after:{request.start_date}")
    if request.end_date:
        dates.append(f"before:{request.end_date}")
    domains: list[str] = []
    floor = min(len(text), _MIN_TEXT_CHARACTERS)
    for group in (
        " OR ".join(f"site:{domain}" for domain in request.include_domains),
        " ".join(f"-site:{domain}" for domain in request.exclude_domains),
    ):
        if not group:
            continue
        if _MAX_QUERY_CHARACTERS - len(" ".join([*domains, group, *dates])) - 1 >= floor:
            domains.append(group)
    suffix = " ".join([*domains, *dates])
    budget = _MAX_QUERY_CHARACTERS - (len(suffix) + 1 if suffix else 0)
    return f"{_fit(text, budget)} {suffix}".strip()


def _fit(text: str, budget: int) -> str:
    """``text`` cut to ``budget`` code points, at a word when that keeps half of them.

    A language without spaces has no word to cut at, and is cut where the budget ends.
    """
    if len(text) <= budget:
        return text
    spaces = [index for index, char in enumerate(text[: budget + 1]) if char.isspace()]
    boundary = spaces[-1] if spaces else -1
    return text[: boundary if boundary * 2 >= budget else budget].strip()


def _within_domains(url: str, request: WebSearchRequest) -> bool:
    """Whether ``url`` is on an included domain, when any is named, and on no excluded one."""
    try:
        host = (urlsplit(url).hostname or "").rstrip(".")
    except ValueError:
        host = ""
    if request.include_domains and not _on_domain(host, request.include_domains):
        return False
    return not _on_domain(host, request.exclude_domains)


def _on_domain(host: str, domains: tuple[str, ...]) -> bool:
    names = (domain.lower().rstrip(".") for domain in domains)
    return any(host == name or host.endswith(f".{name}") for name in names)


def _read_search_result(payload: dict[str, Any], request: WebSearchRequest) -> WebSearchResult:
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
        if not _within_domains(url, request):
            # Filtered, not malformed: the query could not always carry the operators.
            continue
        text = _text_or_none(item.get("summary")) or _text_or_none(item.get("snippet"))
        if text is None:
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
                published_date=_text_or_none(item.get("date")),
                acquisition="deepseek_search",
            )
        )
    if dropped:
        logger.warning("DeepSeek search dropped %d malformed or unusable result(s)", dropped)
    return WebSearchResult(hits=tuple(hits), provider="deepseek", dropped_results=dropped)


async def _bounded_response_bytes(response: httpx.Response, *, operation: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _MAX_RESPONSE_BYTES:
            raise WebSourceUnavailable("deepseek", operation, "response_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def _body_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _text_or_none(value: Any) -> str | None:
    text = value.strip() if isinstance(value, str) else ""
    return text or None


__all__ = ["DeepSeekWebSource"]
