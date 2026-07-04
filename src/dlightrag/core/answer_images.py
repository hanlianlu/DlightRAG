# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer LLM image budgeting."""

import ipaddress
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from dlightrag.utils.image_budget import ImagePayloadBudget
from dlightrag.utils.images import image_url_block

logger = logging.getLogger(__name__)

# Schemes that answer_image_budget allows as image sources.
# - data: → decoded, resized, re-encoded through the byte budget
# - https: → provider's server fetches; only blocked when the host is an
#   IP literal in a private / loopback / link-local range
# - dlightrag-image:// → internal checkpoint references, no fetch
# Everything else (http://, file://, ftp://, custom:) is rejected.
_ALLOWED_SCHEMES = frozenset({"data", "https", "dlightrag-image"})


# Non-standard IP representations that ``ipaddress`` rejects but some
# HTTP clients may still resolve.  No legitimate image URL uses these.
_NONSTANDARD_IP_RE = re.compile(r"^(?:0[xX][0-9a-fA-F]+|0[0-7]+|[1-9]\d{8,})$")


def _is_unsafe_host(host: str | None) -> bool:
    """Return True if *host* is an IP literal in a dangerous range.

    Only called when ``urlparse`` succeeds — no DNS resolution.
    A trailing dot (DNS FQDN marker) is stripped so that ``127.0.0.1.``
    is treated the same as ``127.0.0.1``.
    """
    if not host:
        return False
    host = host.rstrip(".")

    # Catch non-standard IP representations — hex (0x7f000001),
    # octal (0177.0.0.1), integer/DWORD (2130706433) — that
    # ``ipaddress.ip_address()`` rejects but some HTTP clients resolve.
    if _NONSTANDARD_IP_RE.match(host):
        return True

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # not an IP literal — safe (hostname)
    return (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_unspecified  # 0.0.0.0 / ::
    )


def _validate_image_url(text: str, *, label: str) -> str | None:
    """Return *text* if it is safe to pass to a model provider, else None.

    Whitelist approach: only ``data:``, ``https:``, and
    ``dlightrag-image://`` are permitted.  For ``https:`` URLs whose
    host is an IP literal we additionally reject private / loopback /
    link-local addresses (SSRF guard).
    """
    # Fast path — no parse needed for these
    if text.startswith("data:"):
        return text
    if text.startswith("dlightrag-image://"):
        return text

    colon = text.find(":")
    if colon < 0:
        logger.warning("Rejected image source without scheme · %s · %.80s", label, text)
        return None
    scheme = text[:colon].lower().rstrip("/")
    if scheme not in _ALLOWED_SCHEMES:
        logger.warning(
            "Rejected image source with unsafe scheme '%s' · %s · %.80s",
            scheme,
            label,
            text,
        )
        return None

    # https: — parse and validate host
    parsed = urlparse(text)
    if _is_unsafe_host(parsed.hostname):
        logger.warning(
            "Rejected image URL with unsafe host '%s' · %s",
            parsed.hostname,
            label,
        )
        return None
    return text


@dataclass
class AnswerImageBudget:
    """Bound image payloads sent to an answer model."""

    max_images: int
    max_total_bytes: int
    max_bytes_per_image: int
    max_px: int
    min_px: int
    quality: int
    min_quality: int
    count: int = 0
    used_bytes: int = 0

    def add_base64(self, value: str, *, label: str) -> dict[str, Any] | None:
        """Add a raw base64/data URI image if it fits the remaining budget."""
        bounded = self._bound_base64(value, label=label)
        if bounded is None:
            return None
        uri, _ = bounded
        return {"type": "image_url", "image_url": {"url": uri}}

    def _bound_base64(self, value: str, *, label: str) -> tuple[str, int] | None:
        """Bound a raw base64/data URI image and record consumed bytes."""
        budget = ImagePayloadBudget(
            max_images=self.max_images,
            max_total_bytes=self.max_total_bytes,
            max_bytes_per_image=self.max_bytes_per_image,
            max_px=self.max_px,
            min_px=self.min_px,
            quality=self.quality,
            min_quality=self.min_quality,
            count=self.count,
            used_bytes=self.used_bytes,
        )
        bounded = budget.add_base64(value, label=label)
        if bounded is None:
            return None
        self.count = budget.count
        self.used_bytes = budget.used_bytes
        return bounded

    def add_user_image(self, value: str | dict[str, Any], *, label: str) -> dict[str, Any] | None:
        """Add a user image after validating its source URL.

        ``data:`` URIs and bare base64 strings are decoded, resized, and
        re-encoded through the byte budget.  ``https:`` URLs are passed
        through (the provider's server fetches them) — private / loopback /
        link-local IP literals are rejected.  ``dlightrag-image://``
        references are passed through without fetching.  All other URI
        schemes (including ``http:``) are rejected.
        """
        if self.count >= self.max_images:
            return None
        if isinstance(value, dict):
            return self._add_image_url_block(value, label=label)
        text = value.strip()

        # data: URIs → base64 budget pipeline (resize + compress)
        if text.startswith("data:"):
            return self.add_base64(text, label=label)
        # dlightrag-image:// → internal reference, pass through
        if text.startswith("dlightrag-image://"):
            self.count += 1
            return {"type": "image_url", "image_url": {"url": text}}

        # Only validate strings that look like URLs (have a scheme://).
        # Bare base64, raw bytes, or unrecognized formats fall through
        # to the byte-budget pipeline.
        colon = text.find(":")
        if colon < 0 or "//" not in text:
            return self.add_base64(text, label=label)

        scheme = text[:colon].lower().rstrip("/")
        if scheme in ("http", "https"):
            safe = _validate_image_url(text, label=label)
            if safe is None:
                return None
            self.count += 1
            return {"type": "image_url", "image_url": {"url": safe}}

        # Unknown scheme — not safe to pass through as a URL
        if scheme != "data":  # data: already handled above
            logger.warning(
                "Rejected image source with unsafe scheme '%s' · %s · %.80s",
                scheme,
                label,
                text,
            )
            return None

        return self.add_base64(text, label=label)

    def _add_image_url_block(self, value: dict[str, Any], *, label: str) -> dict[str, Any] | None:
        """Add an OpenAI-style image block after validating the inner URL.

        ``image_url.url`` is validated through the same whitelist as
        ``add_user_image``.  ``data:`` URIs inside blocks go through
        the byte budget; non-``data:`` safe schemes pass through.
        """
        block = image_url_block(value)
        if block is None:
            return None
        image_url = block.get("image_url")
        if not isinstance(image_url, dict):
            return None
        url = image_url.get("url")
        if not isinstance(url, str) or not url.strip():
            return None
        text = url.strip()

        # data: inside a dict block → still goes through budget
        if text.startswith("data:"):
            bounded = self._bound_base64(text, label=label)
            if bounded is None:
                return None
            bounded_url, _ = bounded
            bounded_block = dict(block)
            bounded_image_url = dict(image_url)
            bounded_image_url["url"] = bounded_url
            bounded_block["image_url"] = bounded_image_url
            return bounded_block

        # dlightrag-image:// → pass through
        if text.startswith("dlightrag-image://"):
            self.count += 1
            return block

        # All other schemes → validate through whitelist gate
        safe = _validate_image_url(text, label=label)
        if safe is None:
            return None
        self.count += 1
        return block


__all__ = ["AnswerImageBudget"]
