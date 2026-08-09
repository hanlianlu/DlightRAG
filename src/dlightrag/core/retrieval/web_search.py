# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web search over the Exa API.

Passages come back already chosen against the query and already scored, so this
module hands them over as they are: ranking and packing belong to the caller.
"""

import hashlib
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.exa.ai/search"
_CONTENTS_ENDPOINT = "https://api.exa.ai/contents"

# A rejected key or an empty balance is not a blip, so asking again next turn
# only buys a round trip and another warning. Long enough to top up an account,
# short enough that nobody has to restart the service to get search back.
_PARK_SECONDS = 15 * 60
_PARKING_STATUS = {401: "unauthorized", 402: "payment_required"}

# Left alone, one excerpt measures around 7k characters. This is the length the
# provider recommends: ten results come to roughly 10k tokens, which a survey of
# the open web can be worth beside the corpus without displacing it.
_HIGHLIGHT_MAX_CHARACTERS = 4000

# Without a cached copy the provider crawls the page live, and its own crawl
# budget is ten seconds, so httpx's five-second default would call a working
# search slow. One retry covers a dropped connection without waiting on one.
_TIMEOUT_SECONDS = 15.0
_CONNECT_RETRIES = 1

# Web passages belong to no workspace, and the sentinel keeps them out of every
# workspace-routed path the way Composer documents already are.
_WEB_SEARCH_WORKSPACE = "__web_search__"


class WebSearchUnavailable(Exception):
    """A search could not run; ``reason`` is a stable code the caller can report."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class WebSearchHit:
    """One passage from the open web, carrying the page it was taken from."""

    url: str
    title: str
    text: str
    published_date: str | None = None
    image_url: str | None = None


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    hits: tuple[WebSearchHit, ...]
    cost_dollars: float


class ExaSearch:
    """Search the open web, and stop asking once the account says no."""

    def __init__(self, api_key: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._api_key = api_key
        self._client = client if client is not None else _default_client()
        self._owns_client = client is None
        self._parked: tuple[str, float] | None = None

    async def search(self, query: str) -> WebSearchResult:
        """Return the passages a search found, or say why it could not run."""
        parked = self._parked_reason()
        if parked is not None:
            raise WebSearchUnavailable(parked)

        contents: dict[str, Any] = {"highlights": {"maxCharacters": _HIGHLIGHT_MAX_CHARACTERS}}
        try:
            response = await self._client.post(
                _ENDPOINT,
                headers={"x-api-key": self._api_key},
                json={"query": query, "contents": contents},
            )
        except httpx.TimeoutException as exc:
            raise WebSearchUnavailable("timeout") from exc
        except httpx.HTTPError as exc:
            raise WebSearchUnavailable("unreachable") from exc

        reason = _PARKING_STATUS.get(response.status_code)
        if reason is not None:
            self._parked = (reason, time.monotonic() + _PARK_SECONDS)
            logger.warning("Web search parked for %d minutes: %s", _PARK_SECONDS // 60, reason)
            raise WebSearchUnavailable(reason)
        if response.is_error:
            logger.warning("Web search returned HTTP %d", response.status_code)
            raise WebSearchUnavailable("error")
        return _read_result(response.json())

    async def contents(self, url: str) -> WebSearchResult:
        """Fetch the text of one known URL through Exa Contents.

        This is the single known-URL fallback used only after a safe direct
        fetch or local conversion has failed or come back empty. It never
        crawls discovered links on its own; the caller registers any returned
        page as an inert resource handle and reads it explicitly.
        """
        parked = self._parked_reason()
        if parked is not None:
            raise WebSearchUnavailable(parked)

        try:
            response = await self._client.post(
                _CONTENTS_ENDPOINT,
                headers={"x-api-key": self._api_key},
                json={
                    "urls": [url],
                    "text": True,
                    "highlights": {"maxCharacters": _HIGHLIGHT_MAX_CHARACTERS},
                },
            )
        except httpx.TimeoutException as exc:
            raise WebSearchUnavailable("timeout") from exc
        except httpx.HTTPError as exc:
            raise WebSearchUnavailable("unreachable") from exc

        reason = _PARKING_STATUS.get(response.status_code)
        if reason is not None:
            self._parked = (reason, time.monotonic() + _PARK_SECONDS)
            logger.warning("Web search parked for %d minutes: %s", _PARK_SECONDS // 60, reason)
            raise WebSearchUnavailable(reason)
        if response.is_error:
            logger.warning("Web contents returned HTTP %d", response.status_code)
            raise WebSearchUnavailable("error")
        return _read_result(response.json())

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _parked_reason(self) -> str | None:
        if self._parked is None:
            return None
        reason, until = self._parked
        if time.monotonic() >= until:
            self._parked = None
            return None
        return reason


def _default_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=_TIMEOUT_SECONDS,
        transport=httpx.AsyncHTTPTransport(retries=_CONNECT_RETRIES),
    )


def _read_result(payload: Any) -> WebSearchResult:
    results = payload.get("results") if isinstance(payload, dict) else None
    if results is None:
        # The provider declares this field required, so its absence means the
        # payload has moved on without us rather than that nothing was found.
        logger.warning("Web search answered without a results field")
    hits: list[WebSearchHit] = []
    for result in results or []:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url") or "")
        if not url:
            continue
        page = {
            "url": url,
            "title": str(result.get("title") or url),
            "published_date": _text_or_none(result.get("publishedDate")),
            "image_url": _text_or_none(result.get("image")),
        }
        for passage in result.get("highlights") or []:
            hits.append(WebSearchHit(text=str(passage), **page))
        body = result.get("text")
        if body:
            hits.append(WebSearchHit(text=str(body), **page))
    cost = payload.get("costDollars") if isinstance(payload, dict) else None
    total = cost.get("total") if isinstance(cost, dict) else None
    if results and not hits:
        # Pages came back and none of them read, so the payload has moved on
        # without us. Silence here is indistinguishable from finding nothing.
        logger.warning("Web search returned %d results and no usable passage", len(results))
    return WebSearchResult(tuple(hits), float(total) if isinstance(total, (int, float)) else 0.0)


def _text_or_none(value: Any) -> str | None:
    text = str(value).strip() if value else ""
    return text or None


def web_context_rows(hits: Iterable[WebSearchHit]) -> list[dict[str, Any]]:
    """Project passages into answer-context rows, one source per page.

    Two waves searching different angles reach the same page for different
    reasons, so a repeat of the same passage is dropped while a new passage on
    a page already seen is kept.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for hit in hits:
        if not hit.text.strip() or (hit.url, hit.text) in seen:
            continue
        seen.add((hit.url, hit.text))
        reference_id = _reference_id(hit.url)
        index = counts[reference_id] = counts.get(reference_id, 0) + 1
        rows.append(
            {
                "chunk_id": f"{reference_id}-{index}",
                "reference_id": reference_id,
                "full_doc_id": reference_id,
                "file_path": hit.title,
                "content": hit.text,
                "page_number": None,
                "_workspace": _WEB_SEARCH_WORKSPACE,
                "metadata": {
                    # The prompt has to be able to say where this came from: a
                    # page the model chose carries none of an upload's warrant.
                    "source_type": "web_search",
                    "source_uri": hit.url,
                    "source_download_locator": hit.url,
                    "title": hit.title,
                    "published_date": hit.published_date,
                    "remote_image_url": hit.image_url,
                },
            }
        )
    return rows


def _reference_id(url: str) -> str:
    return "web-" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "ExaSearch",
    "WebSearchHit",
    "WebSearchResult",
    "WebSearchUnavailable",
    "web_context_rows",
]
