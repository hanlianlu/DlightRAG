# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reader-activated official players, separate from arbitrary Artifact HTML.

Recognition is a local projection. Provider records own URL differences; they
never authorize fetching arbitrary oEmbed endpoints or inserting returned HTML.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
from pydantic import Field

from dlightrag.adapters.http.browser.oembed_catalogue import oembed_provider, permitted_player
from dlightrag.engine.answer.client_contracts import ClientContractModel
from dlightrag.engine.public_http import (
    PublicHttpFetch,
    fetch_public_http,
    validate_agent_public_url,
)


class VideoPlaybackLink(ClientContractModel):
    url: str
    provider: str
    player_domains: list[str]


class VideoPlayer(ClientContractModel):
    embed_url: str
    aspect_ratio: float = Field(default=16 / 9, ge=0.25, le=4)


@dataclass(frozen=True)
class _Provider:
    name: str
    canonical: str
    player: str
    oembed: str | None = None


@dataclass(frozen=True)
class _Rule:
    provider: _Provider
    hosts: frozenset[str]
    path: str
    query_id: str | None = None


_YOUTUBE = _Provider(
    "YouTube",
    "https://www.youtube.com/watch?v={id}",
    "https://www.youtube-nocookie.com/embed/{id}",
    "https://www.youtube.com/oembed",
)
_VIMEO = _Provider(
    "Vimeo",
    "https://vimeo.com/{id}",
    "https://player.vimeo.com/video/{id}",
    "https://vimeo.com/api/oembed.json",
)
_BILIBILI = _Provider(
    "Bilibili",
    "https://www.bilibili.com/video/{id}",
    "https://player.bilibili.com/player.html?bvid={id}",
)
_RULES = (
    _Rule(_YOUTUBE, frozenset({"youtube.com", "www.youtube.com", "m.youtube.com"}), r"/watch", "v"),
    _Rule(
        _YOUTUBE,
        frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "www.youtube-nocookie.com"}),
        r"/(?:shorts|live|embed)/(?P<id>[\w-]{11})/?",
    ),
    _Rule(_YOUTUBE, frozenset({"youtu.be", "www.youtu.be"}), r"/(?P<id>[\w-]{11})/?"),
    _Rule(_VIMEO, frozenset({"vimeo.com", "www.vimeo.com"}), r"/(?P<id>[0-9]{1,12})/?"),
    _Rule(_VIMEO, frozenset({"player.vimeo.com"}), r"/video/(?P<id>[0-9]{1,12})/?"),
    _Rule(
        _BILIBILI,
        frozenset({"bilibili.com", "www.bilibili.com", "m.bilibili.com"}),
        r"/video/(?P<id>BV[0-9A-Za-z]{10})/?",
    ),
)


@dataclass(frozen=True)
class _Target:
    url: str
    provider: _Provider
    identifier: str

    @property
    def canonical(self) -> str:
        return self.provider.canonical.format(id=self.identifier)

    @property
    def player(self) -> str:
        return self.provider.player.format(id=self.identifier)


def _target(url: str) -> _Target | None:
    if len(url) > 2048:
        return None
    try:
        normalized = validate_agent_public_url(url)
        parsed = urlsplit(normalized)
        if parsed.port not in (None, 80 if parsed.scheme == "http" else 443):
            return None
        query = parse_qs(parsed.query, max_num_fields=32)
    except ValueError:
        return None
    for rule in _RULES:
        if parsed.hostname not in rule.hosts:
            continue
        match = re.fullmatch(rule.path, parsed.path, re.ASCII)
        if match is None:
            continue
        if rule.query_id:
            values = query.get(rule.query_id, [])
            if len(values) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]{11}", values[0]):
                return None
            identifier = values[0]
        else:
            identifier = match.group("id")
        return _Target(normalized, rule.provider, identifier)
    return None


@dataclass(frozen=True)
class _Candidate:
    link: VideoPlaybackLink
    endpoint: str | None
    lookup_url: str
    fallback: _Target | None


def _candidate(url: str) -> _Candidate | None:
    try:
        validate_agent_public_url(url)
        parsed = urlsplit(url)
        if len(url) > 2048 or parsed.port not in (None, 80 if parsed.scheme == "http" else 443):
            return None
    except ValueError:
        return None
    target = _target(url)
    # The explicit adapters also reject malformed ids, playlists and unlisted
    # path credentials; a broad registry scheme must not undo those decisions.
    if target is None:
        if parsed.hostname in {
            host for rule in _RULES if rule.provider == _YOUTUBE for host in rule.hosts
        }:
            return None
        if parsed.hostname in {"vimeo.com", "www.vimeo.com"} and re.fullmatch(
            r"/[0-9]+/[^/]+/?", parsed.path
        ):
            return None
    lookup = target.canonical if target else url
    registered = oembed_provider(lookup)
    if target:
        name = target.provider.name
        endpoint = registered.endpoint if registered else target.provider.oembed
    elif registered:
        name, endpoint = registered.name, registered.endpoint
    else:
        return None
    domains = set(registered.player_domains if registered else ())
    if target:
        domains.add(urlsplit(target.player).hostname or "")
    link = VideoPlaybackLink(url=url, provider=name, player_domains=sorted(domains))
    return _Candidate(link, endpoint, lookup, target)


def video_playback_link(url: str) -> VideoPlaybackLink | None:
    """Recognize a registry candidate without reading metadata or promising playback."""
    candidate = _candidate(url)
    return candidate.link if candidate else None


class _OEmbedFrame(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "iframe":
            self.sources.extend(value for name, value in attrs if name == "src" and value)


def _video_response(payload: bytes, domains: tuple[str, ...]) -> VideoPlayer | None:
    """Extract one permitted video frame, never execute provider HTML or scripts."""
    data = json.loads(payload)
    if not isinstance(data, dict) or data.get("type") != "video":
        return None
    markup = data.get("html")
    if not isinstance(markup, str) or len(markup) > 8192:
        return None
    frame = _OEmbedFrame()
    frame.feed(markup)
    if len(frame.sources) != 1 or not permitted_player(frame.sources[0], domains):
        return None
    ratio = 16 / 9
    width, height = data.get("width"), data.get("height")
    if (
        isinstance(width, (int, float))
        and isinstance(height, (int, float))
        and all(not isinstance(value, bool) and 0 < value <= 16384 for value in (width, height))
    ):
        candidate = width / height
        if 0.25 <= candidate <= 4:
            ratio = candidate
    return VideoPlayer(embed_url=frame.sources[0], aspect_ratio=ratio)


def _start(value: str) -> int:
    if value.isascii() and value.isdecimal():
        return min(int(value), 604800) if len(value) <= 8 else 0
    match = re.fullmatch(r"(?:(\d{1,3})h)?(?:(\d{1,3})m)?(?:(\d{1,3})s)?", value)
    return (
        min(
            sum(
                int(part or 0) * scale
                for part, scale in zip(match.groups(), (3600, 60, 1), strict=True)
            ),
            604800,
        )
        if match
        else 0
    )


def _player_url(target: _Target) -> str:
    query = parse_qs(urlsplit(target.url).query, max_num_fields=32)
    options: dict[str, str | int] = {"autoplay": 1}
    start = _start(query.get("start", query.get("t", [""]))[0])
    if target.provider == _YOUTUBE:
        options["playsinline"] = 1
        if start:
            options["start"] = start
    elif target.provider == _BILIBILI:
        part = query.get("p", [""])[0]
        if re.fullmatch(r"[1-9][0-9]{0,3}", part):
            options["p"] = part
        if start:
            options["t"] = start
    source = target.player + ("&" if "?" in target.player else "?") + urlencode(options)
    if target.provider == _VIMEO:
        start = start or _start(urlsplit(target.url).fragment.removeprefix("t="))
        if start:
            source += f"#t={start}s"
    return source


async def resolve_video_playback(
    url: str,
    *,
    fetch: Callable[..., Awaitable[PublicHttpFetch]] = fetch_public_http,
) -> VideoPlayer | None:
    """Use the shared oEmbed path; small official mappings survive metadata failure."""
    candidate = _candidate(url)
    if candidate is None:
        return None
    player = None
    if candidate.endpoint:
        endpoint = (
            candidate.endpoint
            + ("&" if "?" in candidate.endpoint else "?")
            + urlencode({"url": candidate.lookup_url, "format": "json"})
        )
        try:
            # Includes waiting for shared network admission, not just reading.
            async with asyncio.timeout(4.0):
                result = await fetch(endpoint, max_bytes=65536, timeout=4.0, agent_url=True)
                media_type = (result.media_type or "").partition(";")[0].strip().lower()
                if result.status_code == 200 and media_type == "application/json":
                    player = _video_response(result.content, tuple(candidate.link.player_domains))
        except ValueError, OSError, httpx.HTTPError, TimeoutError, RecursionError:
            pass
    target = candidate.fallback
    if target is None:
        return player
    # Known mappings retain offset/privacy parameters even if metadata is absent
    # or declares a different video. oEmbed never overrides that chosen identity.
    declared = _target(player.embed_url) if player else None
    ratio = (
        player.aspect_ratio
        if player
        and declared
        and (declared.provider, declared.identifier) == (target.provider, target.identifier)
        else 16 / 9
    )
    return VideoPlayer(embed_url=_player_url(target), aspect_ratio=ratio)
