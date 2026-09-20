# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pinned oembed.com publisher data, not an online aggregation service.

Registry schemes recognize candidates locally. They do not prove a response is
video, authorize arbitrary returned HTML, or grant a publisher unrelated origins.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from dlightrag.engine.public_http import validate_agent_public_url


@dataclass(frozen=True)
class OEmbedProvider:
    name: str
    endpoint: str
    player_domains: tuple[str, ...]


@dataclass(frozen=True)
class _Scheme:
    provider: OEmbedProvider
    scheme: str
    host: re.Pattern[str]
    resource: re.Pattern[str]


def _pattern(value: str) -> re.Pattern[str]:
    # oEmbed schemes use * only; ? is a literal query separator, not a glob.
    return re.compile(re.escape(value).replace(r"\*", ".*"), re.ASCII)


def _domain(url: str) -> str:
    return (urlsplit(url).hostname or "").removeprefix("www.")


@lru_cache(maxsize=1)
def _schemes() -> tuple[_Scheme, ...]:
    data = json.loads(Path(__file__).with_name("oembed-providers.json").read_text())
    rules = []
    for item in data["providers"]:
        for entry in item["endpoints"]:
            endpoint = entry["url"].replace("{format}", "json")
            # Only fixed, anonymous HTTPS publisher endpoints. Discovery-only,
            # XML-only and templated endpoints need no unsafe guess or SaaS fallback.
            if "{" in endpoint or "json" not in entry.get("formats", ["json"]):
                continue
            try:
                validate_agent_public_url(endpoint)
                validate_agent_public_url(item["provider_url"])
            except ValueError:
                continue
            if urlsplit(endpoint).scheme != "https":
                continue
            domains = tuple(
                sorted(
                    {
                        _domain(item["provider_url"]),
                        _domain(endpoint),
                        *data.get("additional_player_domains", {}).get(item["provider_name"], []),
                    }
                )
            )
            if any("." not in domain or "*" in domain for domain in domains):
                continue
            provider = OEmbedProvider(item["provider_name"], endpoint, domains)
            for scheme in entry.get("schemes", []):
                parsed = urlsplit(scheme)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    continue
                resource = parsed.path + ("?" + parsed.query if parsed.query else "")
                rules.append(
                    _Scheme(
                        provider,
                        parsed.scheme,
                        _pattern(parsed.hostname.removeprefix("www.")),
                        _pattern(resource),
                    )
                )
    return tuple(rules)


@lru_cache(maxsize=1024)
def oembed_provider(url: str) -> OEmbedProvider | None:
    """Match a public URL against the bundled registry without any network read."""
    try:
        validate_agent_public_url(url)
        parsed = urlsplit(url)
        if len(url) > 2048 or parsed.port not in (None, 80 if parsed.scheme == "http" else 443):
            return None
    except ValueError:
        return None
    resource = parsed.path + ("?" + parsed.query if parsed.query else "")
    for rule in _schemes():
        # Match the authority separately: * must never consume a slash and turn
        # an attacker's path mentioning a publisher into that publisher's URL.
        if (
            parsed.scheme == rule.scheme
            and (
                rule.host.fullmatch(parsed.hostname or "")
                or rule.host.fullmatch((parsed.hostname or "").removeprefix("www."))
            )
            and rule.resource.fullmatch(resource)
        ):
            return rule.provider
    return None


def permitted_player(url: str, domains: tuple[str, ...]) -> bool:
    """Only public HTTPS within the selected publisher's recorded domain families."""
    try:
        validate_agent_public_url(url)
        parsed = urlsplit(url)
        return (
            parsed.scheme == "https"
            and parsed.port in (None, 443)
            and any(
                parsed.hostname == domain or (parsed.hostname or "").endswith("." + domain)
                for domain in domains
            )
        )
    except ValueError:
        return False
