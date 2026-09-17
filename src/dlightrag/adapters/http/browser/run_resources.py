# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Same-origin addresses for the bytes one Answer Run recorded.

A run's images may reach the answer text as the external URL the model read.
This surface serves the copy the run actually stored, so what a conversation
shows does not depend on a third-party URL outliving the run: each mapped
address resolves through the one run-resource reader, and a URL this run holds
no bytes for is left exactly as the answer wrote it.
"""

import re
from collections.abc import Mapping
from urllib.parse import quote, urlsplit

from dlightrag.engine.network_admission import normalize_public_http_url_identity

RUN_RESOURCE_URL_BASE = "/web/api/runs"

_IMAGE_SOURCE = re.compile(
    r'(<img\b[^>]*?\bsrc\s*=\s*)(["\'])([^"\']+)(\2)',
    re.IGNORECASE,
)


def _public_url_identity(url: str) -> str | None:
    """Normalize a public HTTP(S) URL, or return None for anything else.

    Only such a URL can name bytes this deployment fetched, so a relative path,
    a ``data:`` payload, or a script URL is never addressed by a stored copy even
    if some map key happened to spell the same text.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return None
    return normalize_public_http_url_identity(url)


def run_resource_url(run_id: str, resource_id: str) -> str:
    """Return the one same-origin address for one stored run resource."""
    return (
        f"{RUN_RESOURCE_URL_BASE}/{quote(run_id, safe='')}/resources/{quote(resource_id, safe='')}"
    )


def image_rewrites(run_id: str, sources: Mapping[str, str]) -> dict[str, str]:
    """Address each stored source URL by its same-origin Run resource."""
    rewrites: dict[str, str] = {}
    for url, resource_id in sources.items():
        identity = _public_url_identity(url)
        if identity is not None:
            rewrites[identity] = run_resource_url(run_id, resource_id)
    return rewrites


def rewrite_image_sources(html: str, rewrites: Mapping[str, str]) -> str:
    """Point every stored image at its run-resource address.

    Only ``<img src>`` is rewritten: a link the answer cites keeps pointing at
    the page it names. A source the run stored no bytes for, and any URL that is
    not public HTTP(S), is returned untouched.
    """
    if not rewrites or "<img" not in html.casefold():
        return html

    def replace(match: re.Match[str]) -> str:
        head, quote_char, url, tail = match.groups()
        identity = _public_url_identity(url)
        replacement = None if identity is None else rewrites.get(identity)
        if replacement is None:
            return match.group(0)
        return f"{head}{quote_char}{replacement}{tail}"

    return _IMAGE_SOURCE.sub(replace, html)


__all__ = [
    "RUN_RESOURCE_URL_BASE",
    "image_rewrites",
    "rewrite_image_sources",
    "run_resource_url",
]
