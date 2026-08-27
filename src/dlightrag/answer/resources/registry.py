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

import asyncio
import hashlib
import hmac
import secrets
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from dlightrag.answer.resources.converters import (
    ExtractedVisual,
    convert_resource,
    is_convertible,
)
from dlightrag.answer.resources.formatting import format_resource_read
from dlightrag.answer.resources.lexical import bm25_rank, mixed_script_terms
from dlightrag.answer.resources.models import (
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
from dlightrag.answer.resources.text import build_text_windows, decode_text
from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.ai.media import verify_web_image_bytes
from dlightrag.engine.ai.tokens import estimate_tokens
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename
from dlightrag.engine.rag.corpus.sources.url import (
    afetch_public_https_bytes,
    avalidate_public_https_url,
    normalize_https_url_identity,
    validate_public_https_url,
)

_DEFAULT_MAX_ATTACHMENTS = 6
_DEFAULT_MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024
_DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES = 128 * 1024 * 1024

_PDF_MIME = "application/pdf"
_CURSOR_TOKEN_BYTES = 18
_CURSOR_PLACEHOLDER = "x" * 24

# A request-local, provider-neutral fallback that returns already-usable text for
# a public URL, or ``None`` when it cannot. Exa owns its adapter; the registry
# never imports any web-search provider.
UrlTextFallback = Callable[[str], Awaitable[str | None]]


@dataclass(frozen=True, slots=True)
class FetchedResourceBytes:
    """Validated run-scoped web bytes plus the replay slot they were bound to."""

    resource_id: str
    ordinal: int
    filename: str
    mime_type: str
    url: str
    content: bytes


@dataclass(frozen=True, slots=True)
class ResourceEffectOwner:
    """Explicit Agent effect identity for a resource materialization."""

    execution_scope: str
    intent_id: IntentId


# Persist validated fetched bytes before their ToolResult settles in the Session.
FetchedBytesSink = Callable[
    [FetchedResourceBytes, ResourceEffectOwner | None],
    Awaitable[None],
]


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
    plan_window_tokens: int
    plan_position: int
    char_offset: int


@dataclass
class _ConvertedResource:
    text: str
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
        url_text_fallback: UrlTextFallback | None = None,
        fetched_bytes_sink: FetchedBytesSink | None = None,
    ) -> None:
        self._max_attachments = max_attachments
        self._max_attachment_bytes = max(1, int(max_attachment_bytes))
        self._max_total_attachment_bytes = max(1, int(max_total_attachment_bytes))
        self._url_client = url_client
        self._owns_url_client = url_client is None
        self._url_timeout = url_timeout
        self._url_text_fallback = url_text_fallback
        self._fetched_bytes_sink = fetched_bytes_sink
        self._secret = secrets.token_bytes(32)

        self._resources: dict[str, _Registered] = {}
        self._ids_by_dedup: dict[tuple[str, bytes], str] = {}
        self._caller_dedup: set[tuple[str, bytes]] = set()
        self._fetched: dict[str, bytes] = {}
        self._cursors: dict[str, _CursorState] = {}
        self._cursor_plans: dict[tuple[str, str | None, int], tuple[tuple[int, int], ...]] = {}
        self._paths: dict[str, Path] = {}
        self._converted: dict[str, _ConvertedResource] = {}
        self._visual_assets: dict[str, ExtractedVisual] = {}
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._total_bytes = 0
        self._closed = False
        # Durable replay slots for run-scoped fetched bytes. An ordinal is minted
        # once per fetched resource and never reused, so a later turn cannot
        # rebind a slot a previous settlement already made durable.
        self._fetched_ordinals: dict[str, int] = {}
        self._next_fetched_ordinal = 0
        # Fetched bytes a recovery restored. They are already durable, so they
        # are never refetched, revalidated, or persisted again.
        self._durable_fetched: set[str] = set()
        # Fetched (url/loader) bytes are materialized exactly once per resource
        # and charged against the request-wide total under a lock. A separate
        # single-flight guards the URL text fallback so it runs at most once per
        # resource and caches both success and failure.
        self._fetch_lock = asyncio.Lock()
        self._total_lock = asyncio.Lock()
        self._fetch_tasks: dict[str, asyncio.Future[bytes]] = {}
        self._text_views: dict[str, _ConvertedResource] = {}
        self._fallback_lock = asyncio.Lock()
        self._fallback_tasks: dict[str, asyncio.Future[str | None]] = {}
        self._fallback_done: set[str] = set()

    async def __aenter__(self) -> ResourceRegistry:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def register(self, resource: ResourceInput) -> str:
        return self._register(resource, caller=True)

    def register_discovered_link(self, url: str) -> str | None:
        """Register one inert search-discovered HTTPS link outside caller count."""
        try:
            return self._register(ResourceInput(url=url), caller=False)
        except ValueError:
            return None

    def _register(self, resource: ResourceInput, *, caller: bool) -> str:
        self._ensure_open()
        filename = resource.filename
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
            filename = _link_filename(resource.url, filename)
            dedup_key = ("link", normalize_https_url_identity(resource.url).encode("utf-8"))
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
        if caller:
            self._admit_caller_key(dedup_key)
        if existing is not None:
            registered = self._resources[existing]
            if caller:
                if registered.source == "web_search":
                    registered.source = "link"
                if resource.url is not None:
                    registered.url = normalize_https_url_identity(resource.url)
                if resource.filename:
                    registered.filename = safe_source_filename(resource.filename)
            return existing

        if byte_size is not None and self._total_bytes + byte_size > (
            self._max_total_attachment_bytes
        ):
            if caller:
                self._caller_dedup.remove(dedup_key)
            raise ResourceAdmissionError("total attachment bytes exceeded")

        resource_id = self._mint_resource_id(dedup_key)
        self._resources[resource_id] = _Registered(
            resource_id=resource_id,
            filename=filename,
            declared_mime=resource.declared_mime,
            source="web_search" if source == "link" and not caller else source,
            content=content,
            url=normalize_https_url_identity(resource.url) if resource.url is not None else None,
            byte_size=byte_size,
            loader=resource.loader,
        )
        self._ids_by_dedup[dedup_key] = resource_id
        if byte_size is not None:
            self._total_bytes += byte_size
        return resource_id

    def allocate_fetched_ordinal(self, resource_id: str) -> int:
        """Return this fetched resource's durable replay slot, minting it once.

        Slots are never handed out twice, and the next slot is settled durably, so a
        later turn cannot rebind bytes a committed settlement already made
        durable. Re-executing the same unsettled turn reuses its own slots.
        """
        existing = self._fetched_ordinals.get(resource_id)
        if existing is not None:
            return existing
        ordinal = self._next_fetched_ordinal
        self._next_fetched_ordinal = ordinal + 1
        self._fetched_ordinals[resource_id] = ordinal
        return ordinal

    def fetched_replay_slots(self) -> dict[str, int]:
        """Return each fetched resource's durable replay slot."""
        return dict(self._fetched_ordinals)

    def restore_fetched_bytes(self, resource_id: str, content: bytes) -> None:
        """Restore one settled fetch so a resumed read never repeats it.

        These bytes are durable run state, not a cache: they are charged once
        against the request total and their replay slot is frozen, so a resumed
        run can neither read a page that changed underneath it nor rebind the
        slot a committed settlement depends on.
        """
        self._ensure_open()
        if resource_id not in self._resources:
            raise ResourceStateMismatchError(
                "settled fetched bytes name a resource the catalog does not describe"
            )
        self._durable_fetched.add(resource_id)
        if resource_id in self._fetched:
            return
        self._fetched[resource_id] = content
        self._total_bytes += len(content)

    def _admit_caller_key(self, dedup_key: tuple[str, bytes]) -> None:
        if dedup_key in self._caller_dedup:
            return
        if len(self._caller_dedup) >= self._max_attachments:
            raise ResourceAdmissionError("too many attachments")
        self._caller_dedup.add(dedup_key)

    def manifest(self) -> tuple[ResourceManifestEntry, ...]:
        return tuple(
            ResourceManifestEntry(
                resource_id=item.resource_id,
                filename=item.filename,
                declared_mime=item.declared_mime,
                source="link" if item.source == "web_search" else item.source,  # type: ignore[arg-type]
                byte_size=item.byte_size,
            )
            for item in self._resources.values()
        )

    def evidence_source(self, resource_id: str) -> dict[str, str]:
        """Return stable private provenance for evidence derived from a resource."""
        resource = self._require(resource_id)
        source_uri = (
            resource.url if resource.source == "web_search" and resource.url else resource_id
        )
        return {
            "source_type": "web_search" if resource.source == "web_search" else "web_attachment",
            "source_uri": source_uri,
            "source_download_locator": source_uri,
            "title": safe_source_filename(resource.filename or source_uri),
        }

    async def materialize(
        self,
        resource_id: str,
        *,
        effect_owner: ResourceEffectOwner | None = None,
    ) -> bytes:
        """Return full bytes, attributing any fetch to an explicit effect."""
        return await self._materialize_bytes(self._require(resource_id), effect_owner=effect_owner)

    async def ensure_path(
        self,
        resource_id: str,
        *,
        effect_owner: ResourceEffectOwner | None = None,
    ) -> Path:
        """Materialize a resource to a request-local temporary file and return it.

        Only binary readers that need a filesystem path call this; direct text
        reads never do, so text answers never create temporary storage.
        """
        resource = self._require(resource_id)
        existing = self._paths.get(resource_id)
        if existing is not None:
            return existing
        content = await self._materialize_bytes(resource, effect_owner=effect_owner)
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
        max_window_tokens: int,
        focus: str | None = None,
        cursor: str | None = None,
        effect_owner: ResourceEffectOwner | None = None,
    ) -> ResourceReadResult:
        """Return one page whose complete model-visible envelope fits the budget."""
        if max_window_tokens < 1:
            raise ResourceAdmissionError("resource read has no residual model capacity")
        resource = self._require(resource_id)
        effective_focus = focus
        cursor_state: _CursorState | None = None
        if cursor is not None:
            cursor_state = self._resolve_cursor(cursor, resource_id=resource_id)
            if focus is not None and focus != cursor_state.focus:
                raise ResourceCursorError("cursor is not valid for this resource read")
            effective_focus = cursor_state.focus

        view = await self._read_text_view(resource, effect_owner=effect_owner)
        text = view.text
        resource_handles = view.handles
        if not text:
            result = ResourceReadResult(
                resource_id=resource_id,
                locator=None,
                content="",
                extraction_status=EXTRACTION_TEXT,
                has_more=False,
                next_cursor=None,
                visual_handles=resource_handles,
            )
            if estimate_tokens(format_resource_read(result)) > max_window_tokens:
                raise ResourceAdmissionError(
                    "resource read envelope exceeds residual model capacity"
                )
            return result

        plan_window_tokens = (
            cursor_state.plan_window_tokens if cursor_state is not None else max_window_tokens
        )
        plan = await self._cursor_plan(
            resource_id,
            text,
            effective_focus,
            plan_window_tokens=plan_window_tokens,
        )
        plan_position = cursor_state.plan_position if cursor_state is not None else 0
        char_offset = cursor_state.char_offset if cursor_state is not None else plan[0][0]
        visual_handles = () if cursor_state is not None else resource_handles
        locator, chunk, next_position, next_offset = await asyncio.to_thread(
            _read_cursor_span,
            text,
            plan,
            resource_id=resource_id,
            plan_position=plan_position,
            char_offset=char_offset,
            max_window_tokens=max_window_tokens,
            visual_handles=visual_handles,
        )
        has_more = next_position < len(plan)
        if cursor is not None:
            self._consume_cursor(cursor, cursor_state)
        next_cursor = None
        if has_more:
            next_cursor = self._mint_cursor(
                _CursorState(
                    resource_id=resource_id,
                    focus=effective_focus,
                    plan_window_tokens=plan_window_tokens,
                    plan_position=next_position,
                    char_offset=next_offset,
                )
            )
        return ResourceReadResult(
            resource_id=resource_id,
            locator=locator,
            content=chunk,
            extraction_status=EXTRACTION_TEXT,
            has_more=has_more,
            next_cursor=next_cursor,
            visual_handles=visual_handles,
        )

    async def _read_text_view(
        self,
        resource: _Registered,
        *,
        effect_owner: ResourceEffectOwner | None,
    ) -> _ConvertedResource:
        if resource.url is not None:
            return await self._read_link_text_view(resource, effect_owner=effect_owner)
        cached = self._text_views.get(resource.resource_id)
        if cached is not None:
            return cached
        content = await self._materialize_bytes(resource, effect_owner=effect_owner)
        return await self._text_view_from_content(resource, content)

    async def _text_view_from_content(
        self,
        resource: _Registered,
        content: bytes,
    ) -> _ConvertedResource:
        if is_convertible(resource.filename, resource.declared_mime):
            view = await self._ensure_converted(resource, content)
        else:
            text = await asyncio.to_thread(
                decode_text,
                content,
                declared_charset=_charset_of(resource.declared_mime),
            )
            view = _ConvertedResource(text=text, handles=())
        if view.text:
            self._text_views[resource.resource_id] = view
        return view

    async def _read_link_text_view(
        self,
        resource: _Registered,
        *,
        effect_owner: ResourceEffectOwner | None,
    ) -> _ConvertedResource:
        """Read a link, falling back to provider text only when direct fails/empty.

        Settled bytes already passed the complete fetch gate and
        never re-enter the network path. Otherwise SSRF revalidation runs before
        any direct, cached, or fallback path, so a private, invalid, or credential
        URL raises here and never reaches the fallback.
        The Exa Contents fallback is tried at most once per resource — only after
        a direct fetch/decode/conversion fails or yields empty — and its text
        enters the same bounded window pipeline under the original resource id.
        """
        url = resource.url
        if url is None:  # pragma: no cover - only link resources are routed here
            raise ResourceNotFoundError(f"resource {resource.resource_id} has no link")
        restored = self._restored_bytes(resource.resource_id)
        if restored is not None:
            cached = self._text_views.get(resource.resource_id)
            if cached is not None:
                return cached
            return await self._text_view_from_content(resource, restored)
        await avalidate_public_https_url(url)
        cached = self._text_views.get(resource.resource_id)
        if cached is not None:
            return cached
        try:
            content = await self._materialize_fetched(
                resource.resource_id,
                lambda: self._fetch_link(url),
                effect_owner=effect_owner,
            )
            view = await self._text_view_from_content(resource, content)
        except ResourceAdmissionError:
            # A per-attachment or request-wide byte limit is a real bound, not a
            # direct-extraction failure; never mask it by fetching provider text.
            raise
        except Exception:
            fallback = await self._fallback_text_view(resource, url)
            if fallback is not None:
                return fallback
            raise
        if not view.text:
            fallback = await self._fallback_text_view(resource, url)
            if fallback is not None:
                return fallback
        return view

    async def _ensure_converted(
        self,
        resource: _Registered,
        content: bytes | None = None,
        *,
        effect_owner: ResourceEffectOwner | None = None,
    ) -> _ConvertedResource:
        cached = self._converted.get(resource.resource_id)
        if cached is not None:
            return cached
        if content is None:
            content = await self._materialize_bytes(resource, effect_owner=effect_owner)
        converted = await convert_resource(
            content, filename=resource.filename, declared_mime=resource.declared_mime
        )
        handles: list[VisualHandle] = []
        for visual in converted.visuals:
            self._visual_assets[visual.handle_id] = visual
            handles.append(VisualHandle(handle_id=visual.handle_id, label=visual.anchor))
        entry = _ConvertedResource(
            text=converted.text,
            handles=tuple(handles),
        )
        self._converted[resource.resource_id] = entry
        return entry

    async def inspection_target(
        self,
        resource_id: str,
        *,
        effect_owner: ResourceEffectOwner | None = None,
    ) -> InspectionTarget:
        """Materialize a resource and classify how it can be visually inspected."""
        resource = self._require(resource_id)
        content = await self._materialize_bytes(resource, effect_owner=effect_owner)
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

    async def visual_asset(
        self,
        resource_id: str,
        handle_id: str,
        *,
        effect_owner: ResourceEffectOwner | None = None,
    ) -> ExtractedVisual:
        """Return an embedded visual asset by handle, converting on demand."""
        resource = self._require(resource_id)
        if is_convertible(resource.filename, resource.declared_mime):
            await self._ensure_converted(resource, effect_owner=effect_owner)
        asset = self._visual_assets.get(handle_id)
        if asset is None:
            raise ResourceNotFoundError(f"unknown visual handle: {handle_id}")
        return asset

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks: list[asyncio.Future[Any]] = [
            *self._fetch_tasks.values(),
            *self._fallback_tasks.values(),
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
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
            self._text_views.clear()
            self._cursor_plans.clear()
            self._fetch_tasks.clear()
            self._fallback_tasks.clear()
            self._fallback_done.clear()

    @property
    def has_temp_storage(self) -> bool:
        return self._tempdir is not None

    def _require(self, resource_id: str) -> _Registered:
        self._ensure_open()
        try:
            return self._resources[resource_id]
        except KeyError as exc:
            raise ResourceNotFoundError(f"unknown resource id: {resource_id}") from exc

    async def _materialize_bytes(
        self,
        resource: _Registered,
        *,
        effect_owner: ResourceEffectOwner | None,
    ) -> bytes:
        if resource.content is not None:
            # Inline bytes were charged against the total at registration and are
            # never re-counted on read.
            return resource.content
        restored = self._restored_bytes(resource.resource_id)
        if restored is not None:
            return restored
        if resource.loader is not None:
            return await self._materialize_fetched(
                resource.resource_id,
                resource.loader,
                effect_owner=effect_owner,
            )
        url = resource.url
        if url is None:  # pragma: no cover - a resource is always bytes or a link
            raise ResourceNotFoundError(f"resource {resource.resource_id} has no content")
        # Per-read SSRF revalidation: rerun full scheme/credential/host/DNS checks
        # even when bytes are already cached from an earlier read.
        await avalidate_public_https_url(url)
        return await self._materialize_fetched(
            resource.resource_id,
            lambda: self._fetch_link(url),
            effect_owner=effect_owner,
        )

    def _restored_bytes(self, resource_id: str) -> bytes | None:
        """Return settled bytes, which never re-enter the network path.

        Revalidation guards a fetch; a restored read makes no request at all, so
        a host that stopped resolving cannot fail a run whose bytes are durable.
        """
        if resource_id not in self._durable_fetched:
            return None
        return self._fetched.get(resource_id)

    async def _fetch_link(self, url: str) -> bytes:
        return await afetch_public_https_bytes(
            url,
            max_bytes=self._max_attachment_bytes,
            timeout=self._url_timeout,
            client=self._ensure_url_client(),
        )

    async def _materialize_fetched(
        self,
        resource_id: str,
        producer: Callable[[], Awaitable[bytes]],
        *,
        effect_owner: ResourceEffectOwner | None,
    ) -> bytes:
        """Fetch bytes once per resource and charge them against the total.

        Concurrent reads of the same resource share a single fetch task, so the
        bytes are produced and charged exactly once. A failed or over-limit fetch
        is neither cached nor charged. Cancelling one waiter does not cancel the
        shared producer needed by other readers.
        """
        cached = self._fetched.get(resource_id)
        if cached is not None:
            return cached
        async with self._fetch_lock:
            cached = self._fetched.get(resource_id)
            if cached is not None:
                return cached
            task = self._fetch_tasks.get(resource_id)
            if task is None:
                task = asyncio.ensure_future(
                    self._fetch_and_charge(
                        resource_id,
                        producer,
                        effect_owner=effect_owner,
                    )
                )
                self._fetch_tasks[resource_id] = task
        try:
            data = await asyncio.shield(task)
        except BaseException:
            if task.done():
                async with self._fetch_lock:
                    if self._fetch_tasks.get(resource_id) is task:
                        self._fetch_tasks.pop(resource_id, None)
            raise
        async with self._fetch_lock:
            self._fetch_tasks.pop(resource_id, None)
        return data

    async def _fetch_and_charge(
        self,
        resource_id: str,
        producer: Callable[[], Awaitable[bytes]],
        *,
        effect_owner: ResourceEffectOwner | None,
    ) -> bytes:
        data = await producer()
        if len(data) > self._max_attachment_bytes:
            raise ResourceAdmissionError("attachment exceeds per-attachment byte limit")
        async with self._total_lock:
            if self._total_bytes + len(data) > self._max_total_attachment_bytes:
                raise ResourceAdmissionError("total attachment bytes exceeded")
            self._total_bytes += len(data)
            self._fetched[resource_id] = data
        await self._persist_fetched(resource_id, data, effect_owner=effect_owner)
        return data

    async def _persist_fetched(
        self,
        resource_id: str,
        content: bytes,
        *,
        effect_owner: ResourceEffectOwner | None,
    ) -> None:
        """Bind validated web bytes to a durable replay slot before they are used.

        The sink runs after every HTTPS, redirect, DNS, SSRF, and byte check has
        passed and before the ToolResult can settle in the Session, so a resumed
        run never silently re-fetches a page that changed underneath it.
        """
        resource = self._resources.get(resource_id)
        if self._fetched_bytes_sink is None or resource is None or not resource.url:
            return
        if resource_id in self._durable_fetched:
            # Rebinding a settled slot could delete the bytes it depends on.
            return
        await self._fetched_bytes_sink(
            FetchedResourceBytes(
                resource_id=resource_id,
                ordinal=self.allocate_fetched_ordinal(resource_id),
                filename=safe_source_filename(resource.filename or resource_id),
                mime_type=resource.declared_mime or "application/octet-stream",
                url=resource.url,
                content=content,
            ),
            effect_owner,
        )

    async def _fallback_text_view(
        self, resource: _Registered, url: str
    ) -> _ConvertedResource | None:
        """Return a text view built from the provider fallback, at most once.

        The fallback runs a single time per resource; both a usable result and a
        failure are cached so a later read never re-invokes it. Returned text is
        kept distinct from any raw fetched bytes and is windowed like any other
        resource so it remains citable under the original resource id.
        """
        if self._url_text_fallback is None:
            return None
        resource_id = resource.resource_id
        cached = self._text_views.get(resource_id)
        if cached is not None:
            return cached
        async with self._fallback_lock:
            if resource_id in self._fallback_done:
                return self._text_views.get(resource_id)
            task = self._fallback_tasks.get(resource_id)
            if task is None:
                task = asyncio.ensure_future(self._url_text_fallback(url))
                self._fallback_tasks[resource_id] = task
        try:
            text = await asyncio.shield(task)
        except BaseException:
            if task.done():
                async with self._fallback_lock:
                    if self._fallback_tasks.get(resource_id) is task:
                        self._fallback_tasks.pop(resource_id, None)
            raise
        async with self._fallback_lock:
            self._fallback_done.add(resource_id)
            self._fallback_tasks.pop(resource_id, None)
            if text and text.strip() and resource_id not in self._text_views:
                self._text_views[resource_id] = _ConvertedResource(
                    text=text,
                    handles=(),
                )
            return self._text_views.get(resource_id)

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

    async def _cursor_plan(
        self,
        resource_id: str,
        text: str,
        focus: str | None,
        *,
        plan_window_tokens: int,
    ) -> tuple[tuple[int, int], ...]:
        key = (resource_id, focus, plan_window_tokens)
        cached = self._cursor_plans.get(key)
        if cached is not None:
            return cached
        plan = await asyncio.to_thread(
            _build_cursor_plan,
            text,
            focus,
            max_window_tokens=plan_window_tokens,
        )
        self._cursor_plans[key] = plan
        return plan

    def _mint_cursor(self, state: _CursorState) -> str:
        token = secrets.token_urlsafe(_CURSOR_TOKEN_BYTES)
        self._cursors[token] = state
        return token

    def _consume_cursor(self, cursor: str, state: _CursorState | None) -> None:
        if state is None or self._cursors.get(cursor) != state:
            raise ResourceCursorError("cursor is not valid for this resource read")
        self._cursors.pop(cursor)

    def _resolve_cursor(self, cursor: str, *, resource_id: str) -> _CursorState:
        state = self._cursors.get(cursor)
        if state is None or state.resource_id != resource_id:
            raise ResourceCursorError("cursor is not valid for this resource read")
        return state

    def _ensure_open(self) -> None:
        if self._closed:
            raise ResourceRegistryClosedError("resource registry is closed")


class ResourceRegistryClosedError(RuntimeError):
    """Raised when a closed registry is used again."""


class ResourceStateMismatchError(RuntimeError):
    """Raised when a settled catalog cannot describe the replayed request."""


def _link_filename(url: str, explicit: str | None) -> str:
    filename = safe_source_filename(explicit or url)
    return filename if Path(filename).suffix else f"{filename}.html"


def _is_pdf(filename: str | None, declared_mime: str | None) -> bool:
    if filename and Path(filename).suffix.lower() == ".pdf":
        return True
    if declared_mime and declared_mime.split(";", 1)[0].strip().lower() == _PDF_MIME:
        return True
    return False


def _line_bounds(text: str, offset: int) -> tuple[int, int, int]:
    position = 0
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        line_end = position + len(line)
        if offset < line_end:
            return line_number, position, line_end
        position = line_end
    raise ValueError("text offset is outside the resource")


def _build_cursor_plan(
    text: str,
    focus: str | None,
    *,
    max_window_tokens: int,
) -> tuple[tuple[int, int], ...]:
    windows = build_text_windows(text, max_window_tokens=max_window_tokens)
    spans: list[tuple[int, int]] = []
    offset = 0
    for _, chunk in windows:
        end = offset + len(chunk)
        spans.append((offset, end))
        offset = end
    order = _focus_order(windows, focus)
    return tuple(spans[index] for index in order)


def _read_cursor_span(
    text: str,
    plan: tuple[tuple[int, int], ...],
    *,
    resource_id: str,
    plan_position: int,
    char_offset: int,
    max_window_tokens: int,
    visual_handles: tuple[VisualHandle, ...],
) -> tuple[TextWindowLocator, str, int, int]:
    if plan_position < 0 or plan_position >= len(plan):
        raise ResourceCursorError("cursor has no remaining resource text")
    span_start, end = plan[plan_position]
    start = char_offset
    if start < span_start or start >= end:
        raise ResourceCursorError("cursor does not match the current resource text")
    if start < 0 or end <= start or end > len(text):
        raise ResourceCursorError("cursor does not match the current resource text")
    content_budget = max_window_tokens
    while content_budget >= 1:
        consumed_end = _bounded_span_end(
            text,
            start,
            end,
            max_window_tokens=content_budget,
        )
        if consumed_end < end:
            next_position = plan_position
            next_offset = consumed_end
        else:
            next_position = plan_position + 1
            next_offset = plan[next_position][0] if next_position < len(plan) else consumed_end
        has_more = next_position < len(plan)
        locator = _locator_for_span(text, start, consumed_end)
        result = ResourceReadResult(
            resource_id=resource_id,
            locator=locator,
            content=text[start:consumed_end],
            extraction_status=EXTRACTION_TEXT,
            has_more=has_more,
            next_cursor=_CURSOR_PLACEHOLDER if has_more else None,
            visual_handles=visual_handles,
        )
        used = estimate_tokens(format_resource_read(result))
        if used <= max_window_tokens:
            return locator, result.content, next_position, next_offset
        content_budget -= max(1, used - max_window_tokens)
    raise ResourceAdmissionError("resource read envelope exceeds residual model capacity")


def _bounded_span_end(
    text: str,
    start: int,
    end: int,
    *,
    max_window_tokens: int,
) -> int:
    """Return a bounded prefix end without crossing from a partial line."""
    _, line_start, line_end = _line_bounds(text, start)
    candidate_end = min(end, line_end) if start != line_start else end
    windows = build_text_windows(
        text[start:candidate_end],
        max_window_tokens=max_window_tokens,
    )
    if not windows:
        raise ValueError("cursor span contains no resource text")
    return start + len(windows[0][1])


def _locator_for_span(text: str, start: int, end: int) -> TextWindowLocator:
    start_line, start_line_offset, start_line_end = _line_bounds(text, start)
    end_line, end_line_offset, end_line_end = _line_bounds(text, end - 1)
    if start_line == end_line:
        if start == start_line_offset and end == start_line_end:
            return TextWindowLocator(unit="line", start=start_line, end=end_line)
        return TextWindowLocator(
            unit="line",
            start=start_line,
            end=end_line,
            char_start=start - start_line_offset + 1,
            char_end=end - end_line_offset,
        )
    if start != start_line_offset or end != end_line_end:
        raise ValueError("multi-line cursor spans must align to physical lines")
    return TextWindowLocator(unit="line", start=start_line, end=end_line)


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


__all__ = [
    "FetchedBytesSink",
    "FetchedResourceBytes",
    "InspectionTarget",
    "ResourceEffectOwner",
    "ResourceRegistry",
    "ResourceRegistryClosedError",
    "ResourceStateMismatchError",
    "UrlTextFallback",
]
