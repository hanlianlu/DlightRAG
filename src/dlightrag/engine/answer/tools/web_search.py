# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Project provider-neutral Web search passages into Answer evidence rows."""

import hashlib
from collections.abc import Iterable
from typing import Any

from dlightrag.engine.answer.web_sources import WebSearchHit
from dlightrag.engine.public_http import normalize_public_http_url_identity

_WEB_SEARCH_WORKSPACE = "__web_search__"


def web_context_rows(hits: Iterable[WebSearchHit]) -> list[dict[str, Any]]:
    """Project passages into evidence rows, deduplicated by final URL and text."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for hit in hits:
        url = normalize_public_http_url_identity(hit.url)
        if not hit.text.strip() or (url, hit.text) in seen:
            continue
        seen.add((url, hit.text))
        reference_id = _reference_id(url)
        index = counts[reference_id] = counts.get(reference_id, 0) + 1
        metadata = {
            "source_type": "web_search",
            "resource_kind": "web",
            "admission_origin": "search",
            "acquisition": hit.acquisition,
            "source_uri": url,
            "source_download_locator": url,
            "title": hit.title,
            "published_date": hit.published_date,
            "remote_image_url": hit.image_url,
        }
        rows.append(
            {
                "chunk_id": f"{reference_id}-{index}",
                "reference_id": reference_id,
                "full_doc_id": reference_id,
                "file_path": hit.title,
                "content": hit.text,
                "page_number": None,
                "_workspace": _WEB_SEARCH_WORKSPACE,
                "metadata": metadata,
            }
        )
    return rows


def _reference_id(url: str) -> str:
    return "web-" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


__all__ = ["web_context_rows"]
