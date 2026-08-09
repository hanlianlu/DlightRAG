# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-local answer resource registry: admission, materialization, reads.

The registry owns every resource for the lifetime of one answer request. Inline
bytes stay in memory; HTTPS links are fetched lazily and revalidated on every
read. Full bytes never enter model context — only bounded text windows do.
Continuation cursors are opaque, request-local tokens bound to a resource and
focus so they expose no path, offset, or provider locator and never cross
requests. Temporary files are created only when a caller explicitly needs a
filesystem path; direct text reads never spill to disk. ``aclose`` deterministic-
ally releases the fetch client and any temporary storage.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx

from dlightrag.core.resources.converters import (
    ExtractedVisual,
    convert_resource,
    is_convertible,
)
from dlightrag.core.resources.lexical import bm25_rank, mixed_script_terms
from dlightrag.core.resources.models import (
    EXTRACTION_TEXT,
    ResourceAdmissionError,
    ResourceCursorError,
    ResourceInput,
    ResourceManifestEntry,
    ResourceNotFoundError,
    ResourceReadResult,
    TextWindowLocator,
    VisualHandle,
)
from dlightrag.core.resources.text import build_text_windows, decode_text
from dlightrag.sourcing.source_contract import safe_source_filename
from dlightrag.sourcing.url import afetch_public_https_bytes, validate_public_https_url
from dlightrag.utils.images import verify_web_image_bytes

_DEFAULT_MAX_ATTACHMENTS = 6
_DEFAULT_MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024
_DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES = 128 * 1024 * 1024

_PDF_MIME = "application/pdf"


@dataclass(frozen=True)
class InspectionTarget:
    """Materialized bytes plus the visual class of one inspectable resource."""

    resource_id: str
    kind: Literal["image", "pdf", "document", "opaque"]
    content: bytes
    media_type: str | None


@dataclass
class _Registered:
    resource_id: str
    filename: str | None
    declared_mime: str | None
    source: str
    content: bytes | None
    url: str | None
    byte_size: int | None
    loader: Any | None = None


@dataclass(frozen=True)
class _CursorState:
    resource_id: str
    focus: str | None
    window_index: int


@dataclass
class _ConvertedResource:
    windows: list[tuple[TextWindowLocator, str]]
    handles: tuple[VisualHandle, ...]


class ResourceRegistry:
    """Own answer resources for one request and expose bounded reads."""

    def __init__(
        self,
        *,
        max_attachments: int = _DEFAULT_MAX_ATTACHMENTS,
        max_attachment_bytes: int = _DEFAULT_MAX_ATTACHMENT_BYTES,
        max_total_attachment_bytes: int = _DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES,
        url_client: Any | None = None,
        url_timeout: float = 120.0,
    ) -> None:
        self._max_attachments = max_attachments
        self._max_attachment_bytes = max(1, int(max_attachment_bytes))
        self._max_total_attachment_bytes = max(1, int(max_total_attachment_bytes))
        self._url_client = url_client
        self._owns_url_client = url_client is None
        self._url_timeout = url_timeout
        self._secret = secrets.token_bytes(32)

        self._resources: dict[str, _Registered] = {}
        self._ids_by_dedup: dict[tuple[str, bytes], str] = {}
        self._fetched: dict[str, bytes] = {}
        self._cursors: dict[str, _CursorState] = {}
        self._paths: dict[str, Path] = {}
        self._converted: dict[str, _ConvertedResource] = {}
        self._visual_assets: dict[str, ExtractedVisual] = {}
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._total_bytes = 0
        self._closed = False

    async def __aenter__(self) -> ResourceRegistry:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def register(self, resource: ResourceInput) -> str:
        self._ensure_open()
        provided = sum(
            1 for value in (resource.content, resource.url, resource.loader) if value is not None
        )
        if provided != 1:
            raise ResourceAdmissionError("resource requires exactly one of content, url, or loader")

        if resource.loader is not None:
            # Durable server-owned bytes stay lazy: no eager fetch, no byte-size
            # admission until the model actually reads/inspects the resource.
            dedup_key = ("loader", secrets.token_bytes(16))
            source = "bytes"
            byte_size = None
            content = None
        elif resource.url is not None:
            # Cheap scheme/credential check now; full DNS/redirect happens on read.
            validate_public_https_url(resource.url)
            dedup_key = ("link", _normalized_url(resource.url).encode("utf-8"))
            source = "link"
            byte_size = None
            content = None
        else:
            content = resource.content
            if content is None:  # pragma: no cover - exactly-one check guarantees bytes
                raise ResourceAdmissionError("resource requires content bytes")
            if len(content) > self._max_attachment_bytes:
                raise ResourceAdmissionError("attachment exceeds per-attachment byte limit")
            dedup_key = ("bytes", hashlib.sha256(content).digest())
            source = "bytes"
            byte_size = len(content)

        existing = self._ids_by_dedup.get(dedup_key)
        if existing is not None:
            return existing

        if len(self._resources) >= self._max_attachments:
            raise ResourceAdmissionError("too many attachments")
        if byte_size is not None and self._total_bytes + byte_size > (
            self._max_total_attachment_bytes
        ):
            raise ResourceAdmissionError("total attachment bytes exceeded")

        resource_id = self._mint_resource_id(dedup_key)
        self._resources[resource_id] = _Registered(
            resource_id=resource_id,
            filename=resource.filename,
            declared_mime=resource.declared_mime,
            source=source,
            content=content,
            url=resource.url,
            byte_size=byte_size,
            loader=resource.loader,
        )
        self._ids_by_dedup[dedup_key] = resource_id
        if byte_size is not None:
            self._total_bytes += byte_size
        return resource_id

    def manifest(self) -> tuple[ResourceManifestEntry, ...]:
        return tuple(
            ResourceManifestEntry(
                resource_id=item.resource_id,
                filename=item.filename,
                declared_mime=item.declared_mime,
                source=item.source,  # type: ignore[arg-type]
                byte_size=item.byte_size,
            )
            for item in self._resources.values()
        )

    async def materialize(self, resource_id: str) -> bytes:
        """Return the full bytes for a resource, fetching a link if needed."""
        return await self._materialize_bytes(self._require(resource_id))

    async def ensure_path(self, resource_id: str) -> Path:
        """Materialize a resource to a request-local temporary file and return it.

        Only binary readers that need a filesystem path call this; direct text
        reads never do, so text answers never create temporary storage.
        """
        resource = self._require(resource_id)
        existing = self._paths.get(resource_id)
        if existing is not None:
            return existing
        content = await self._materialize_bytes(resource)
        if self._tempdir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="dlrag-res-")
        name = safe_source_filename(resource.filename or resource_id)
        path = Path(self._tempdir.name) / f"{resource_id}-{name}"
        path.write_bytes(content)
        self._paths[resource_id] = path
        return path

    async def read(
        self,
        resource_id: str,
        *,
        focus: str | None = None,
        cursor: str | None = None,
    ) -> ResourceReadResult:
        resource = self._require(resource_id)
        position = 0
        if cursor is not None:
            position = self._resolve_cursor(
                cursor, resource_id=resource_id, focus=focus
            ).window_index

        windows, resource_handles = await self._read_windows(resource)
        if not windows:
            return ResourceReadResult(
                resource_id=resource_id,
                locator=None,
                content="",
                extraction_status=EXTRACTION_TEXT,
                has_more=False,
                next_cursor=None,
                visual_handles=resource_handles,
            )

        # Focus reorders the read sequence without changing any window's bytes or
        # the <=16K contract. Visual handles ride the first *returned* window so
        # they are never lost when focus selects a nonzero physical window.
        order = _focus_order(windows, focus)
        position = min(position, len(order) - 1)
        locator, chunk = windows[order[position]]
        has_more = position + 1 < len(order)
        next_cursor = self._mint_cursor(resource_id, focus, position + 1) if has_more else None
        return ResourceReadResult(
            resource_id=resource_id,
            locator=locator,
            content=chunk,
            extraction_status=EXTRACTION_TEXT,
            has_more=has_more,
            next_cursor=next_cursor,
            visual_handles=resource_handles if position == 0 else (),
        )

    async def _read_windows(
        self, resource: _Registered
    ) -> tuple[list[tuple[TextWindowLocator, str]], tuple[VisualHandle, ...]]:
        if is_convertible(resource.filename, resource.declared_mime):
            converted = await self._ensure_converted(resource)
            return converted.windows, converted.handles
        content = await self._materialize_bytes(resource)
        text = decode_text(content, declared_charset=_charset_of(resource.declared_mime))
        return build_text_windows(text), ()

    async def _ensure_converted(self, resource: _Registered) -> _ConvertedResource:
        cached = self._converted.get(resource.resource_id)
        if cached is not None:
            return cached
        content = await self._materialize_bytes(resource)
        converted = await convert_resource(
            content, filename=resource.filename, declared_mime=resource.declared_mime
        )
        handles: list[VisualHandle] = []
        for visual in converted.visuals:
            self._visual_assets[visual.handle_id] = visual
            handles.append(VisualHandle(handle_id=visual.handle_id, label=visual.anchor))
        entry = _ConvertedResource(
            windows=build_text_windows(converted.text),
            handles=tuple(handles),
        )
        self._converted[resource.resource_id] = entry
        return entry

    async def inspection_target(self, resource_id: str) -> InspectionTarget:
        """Materialize a resource and classify how it can be visually inspected."""
        resource = self._require(resource_id)
        content = await self._materialize_bytes(resource)
        if _is_pdf(resource.filename, resource.declared_mime):
            return InspectionTarget(resource_id, "pdf", content, _PDF_MIME)
        if is_convertible(resource.filename, resource.declared_mime):
            return InspectionTarget(resource_id, "document", content, resource.declared_mime)
        try:
            media = verify_web_image_bytes(content)
        except ValueError:
            media = None
        if media is not None:
            return InspectionTarget(resource_id, "image", content, media)
        return InspectionTarget(resource_id, "opaque", content, resource.declared_mime)

    async def visual_asset(self, resource_id: str, handle_id: str) -> ExtractedVisual:
        """Return an embedded visual asset by handle, converting on demand."""
        resource = self._require(resource_id)
        if is_convertible(resource.filename, resource.declared_mime):
            await self._ensure_converted(resource)
        asset = self._visual_assets.get(handle_id)
        if asset is None:
            raise ResourceNotFoundError(f"unknown visual handle: {handle_id}")
        return asset

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._owns_url_client and self._url_client is not None:
                await self._url_client.aclose()
        finally:
            if self._tempdir is not None:
                self._tempdir.cleanup()
                self._tempdir = None
            self._paths.clear()
            self._converted.clear()
            self._visual_assets.clear()

    @property
    def has_temp_storage(self) -> bool:
        return self._tempdir is not None

    def _require(self, resource_id: str) -> _Registered:
        self._ensure_open()
        try:
            return self._resources[resource_id]
        except KeyError as exc:
            raise ResourceNotFoundError(f"unknown resource id: {resource_id}") from exc

    async def _materialize_bytes(self, resource: _Registered) -> bytes:
        if resource.content is not None:
            return resource.content
        if resource.loader is not None:
            cached = self._fetched.get(resource.resource_id)
            if cached is not None:
                return cached
            data = await resource.loader()
            if len(data) > self._max_attachment_bytes:
                raise ResourceAdmissionError("attachment exceeds per-attachment byte limit")
            self._fetched[resource.resource_id] = data
            return data
        url = resource.url
        if url is None:  # pragma: no cover - a resource is always bytes or a link
            raise ResourceNotFoundError(f"resource {resource.resource_id} has no content")
        # Per-read SSRF revalidation: rerun full scheme/credential/host/DNS checks
        # even when bytes are already cached from an earlier read.
        validate_public_https_url(url, resolve_host=True)
        cached = self._fetched.get(resource.resource_id)
        if cached is not None:
            return cached
        data = await afetch_public_https_bytes(
            url,
            max_bytes=self._max_attachment_bytes,
            timeout=self._url_timeout,
            client=self._ensure_url_client(),
        )
        self._fetched[resource.resource_id] = data
        return data

    def _ensure_url_client(self) -> Any:
        if self._url_client is None:
            self._url_client = httpx.AsyncClient(
                follow_redirects=False,
                timeout=httpx.Timeout(self._url_timeout),
            )
        return self._url_client

    def _mint_resource_id(self, dedup_key: tuple[str, bytes]) -> str:
        kind, payload = dedup_key
        digest = hmac.new(
            self._secret, kind.encode("utf-8") + b"|" + payload, hashlib.sha256
        ).hexdigest()
        return f"res-{digest[:24]}"

    def _mint_cursor(self, resource_id: str, focus: str | None, window_index: int) -> str:
        token = secrets.token_urlsafe(18)
        self._cursors[token] = _CursorState(resource_id, focus, window_index)
        return token

    def _resolve_cursor(self, cursor: str, *, resource_id: str, focus: str | None) -> _CursorState:
        state = self._cursors.get(cursor)
        if state is None or state.resource_id != resource_id or state.focus != focus:
            raise ResourceCursorError("cursor is not valid for this resource read")
        return state

    def _ensure_open(self) -> None:
        if self._closed:
            raise ResourceRegistryClosedError("resource registry is closed")


class ResourceRegistryClosedError(RuntimeError):
    """Raised when a closed registry is used again."""


def _normalized_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def _is_pdf(filename: str | None, declared_mime: str | None) -> bool:
    if filename and Path(filename).suffix.lower() == ".pdf":
        return True
    if declared_mime and declared_mime.split(";", 1)[0].strip().lower() == _PDF_MIME:
        return True
    return False


def _focus_order(windows: list[tuple[TextWindowLocator, str]], focus: str | None) -> list[int]:
    """Return read positions ordered by focus relevance, physical order otherwise."""
    count = len(windows)
    if not focus or count <= 1:
        return list(range(count))
    query_terms = mixed_script_terms(focus)
    if not query_terms:
        return list(range(count))
    documents = [mixed_script_terms(text) for _, text in windows]
    ranked = bm25_rank(query_terms, documents, limit=count)
    order = [index for index, _ in ranked]
    seen = set(order)
    order.extend(index for index in range(count) if index not in seen)
    return order


def _charset_of(declared_mime: str | None) -> str | None:
    if not declared_mime:
        return None
    for token in declared_mime.split(";")[1:]:
        key, _, value = token.strip().partition("=")
        if key.strip().lower() == "charset" and value:
            return value.strip().strip('"')
    return None


__all__ = ["InspectionTarget", "ResourceRegistry", "ResourceRegistryClosedError"]
