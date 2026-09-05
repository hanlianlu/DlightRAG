# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Run-scoped answer resource registry: admission, materialization, and reads.

The registry owns every resource for one Answer Run. Inline bytes stay in memory;
public HTTP(S) locators are fetched lazily and the first successful acquisition
becomes a fixed durable snapshot. Full bytes never enter model context — only
bounded text windows do. Continuation cursors are opaque, run-scoped tokens bound
to a Resource Handle and focus so they expose no path, offset, or provider
locator. Temporary files are created only when a caller explicitly needs a
filesystem path; direct text reads never spill to disk. ``aclose``
deterministically releases the fetch client and temporary storage.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import secrets
import struct
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.ai.media import verify_web_image_bytes
from dlightrag.engine.ai.tokens import estimate_tokens
from dlightrag.engine.answer.resources.converters import (
    ExtractedVisual,
    convert_resource,
    is_convertible,
)
from dlightrag.engine.answer.resources.formatting import format_resource_read
from dlightrag.engine.answer.resources.lexical import bm25_rank, mixed_script_terms
from dlightrag.engine.answer.resources.models import (
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
from dlightrag.engine.answer.resources.text import build_text_windows, decode_text
from dlightrag.engine.answer.web_sources import WebExtractResult
from dlightrag.engine.public_http import (
    PublicHttpPolicyError,
    PublicHttpPresentation,
    avalidate_public_http_url,
    fetch_public_http,
    normalize_public_http_url_identity,
    validate_agent_public_url,
    validate_public_http_url,
)
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename

_DEFAULT_MAX_ATTACHMENTS = 6
_DEFAULT_MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024
_DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES = 128 * 1024 * 1024

_PDF_MIME = "application/pdf"
_CURSOR_VERSION = 1
_CURSOR_SIGNATURE_BYTES = 8
_CURSOR_PLACEHOLDER = "x" * 34

# A run-scoped, provider-neutral fallback that returns already-usable text for
# a public URL, or ``None`` when it cannot. Exa owns its adapter; the registry
# never imports any web-search provider.
UrlTextFallback = Callable[[str], Awaitable[WebExtractResult]]


@dataclass(frozen=True, slots=True)
class FetchedResourceBytes:
    """Validated run-scoped web bytes plus the replay slot they were bound to."""

    resource_id: str
    ordinal: int
    filename: str
    mime_type: str
    url: str
    content: bytes
    admission_origin: Literal["caller", "search", "agent"]
    acquisition: str
    aliases: tuple[str, ...] = ()


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
    admission_origin: Literal["caller", "search", "agent"] = "caller"
    acquisition: str | None = None
    presentation: PublicHttpPresentation = PublicHttpPresentation()
    degradation: str | None = None


@dataclass(frozen=True)
class _CursorState:
    resource_id: str
    plan_window_tokens: int
    plan_position: int
    char_offset: int
    anchor_offset: int


@dataclass
class _ConvertedResource:
    text: str
    handles: tuple[VisualHandle, ...]
    evidence_available: bool = True
    note: str | None = None
    extraction_status: str = EXTRACTION_TEXT


class _RedirectAlias(Exception):
    def __init__(self, resource_id: str) -> None:
        self.resource_id = resource_id
        super().__init__(resource_id)


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
        resource_secret: bytes | None = None,
        cursor_secret: bytes | None = None,
    ) -> None:
        self._max_attachments = max_attachments
        self._max_attachment_bytes = max(1, int(max_attachment_bytes))
        self._max_total_attachment_bytes = max(1, int(max_total_attachment_bytes))
        self._url_client = url_client
        self._url_timeout = url_timeout
        self._url_text_fallback = url_text_fallback
        self._fetched_bytes_sink = fetched_bytes_sink
        self._secret = resource_secret or secrets.token_bytes(32)
        self._cursor_secret = cursor_secret or secrets.token_bytes(32)

        self._resources: dict[str, _Registered] = {}
        self._aliases: dict[str, str] = {}
        self._ids_by_dedup: dict[tuple[str, bytes], str] = {}
        self._caller_dedup: set[tuple[str, bytes]] = set()
        self._fetched: dict[str, bytes] = {}
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
        self._next_loader_ordinal = 0
        # Fetched bytes a recovery restored. They are already durable, so they
        # are never refetched, revalidated, or persisted again.
        self._durable_fetched: set[str] = set()
        self._admitted_fetched: dict[str, bytes] = {}
        self._admitted_effects: set[tuple[str, str | None, str | None]] = set()
        # Fetched (url/loader) bytes are materialized exactly once per resource
        # and charged against the request-wide total under a lock. A separate
        # single-flight guards the URL text fallback so it runs at most once per
        # resource and caches both success and failure.
        self._fetch_lock = asyncio.Lock()
        self._persist_lock = asyncio.Lock()
        self._total_lock = asyncio.Lock()
        self._fetch_tasks: dict[str, asyncio.Future[bytes]] = {}
        self._text_views: dict[str, _ConvertedResource] = {}
        self._fallback_lock = asyncio.Lock()
        self._fallback_tasks: dict[str, asyncio.Future[_ConvertedResource]] = {}

    async def __aenter__(self) -> ResourceRegistry:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def register(self, resource: ResourceInput) -> str:
        return self._register(resource, admission_origin="caller")

    def register_discovered_link(self, url: str) -> str | None:
        """Register one inert search-discovered public link outside caller count."""
        try:
            validate_agent_public_url(url)
            return self._register(ResourceInput(url=url), admission_origin="search")
        except ValueError:
            return None

    def register_agent_url(
        self,
        url: str,
        *,
        presentation: PublicHttpPresentation = PublicHttpPresentation(),
    ) -> str:
        """Admit an arbitrary anonymous public URL chosen by the Agent."""
        validate_agent_public_url(url)
        return self._register(
            ResourceInput(url=url),
            admission_origin="agent",
            presentation=presentation,
        )

    def _register(
        self,
        resource: ResourceInput,
        *,
        admission_origin: Literal["caller", "search", "agent"],
        presentation: PublicHttpPresentation = PublicHttpPresentation(),
    ) -> str:
        self._ensure_open()
        filename = resource.filename
        provided = sum(
            1 for value in (resource.content, resource.url, resource.loader) if value is not None
        )
        if provided != 1:
            raise ResourceAdmissionError("resource requires exactly one of content, url, or loader")

        caller = admission_origin == "caller"
        if resource.loader is not None:
            # Durable server-owned bytes stay lazy: no eager fetch, no byte-size
            # admission until the model actually reads/inspects the resource.
            loader_ordinal = self._next_loader_ordinal
            self._next_loader_ordinal += 1
            loader_identity = (
                f"{loader_ordinal}\0{resource.filename or ''}\0{resource.declared_mime or ''}"
            ).encode()
            dedup_key = ("loader", loader_identity)
            source = "bytes"
            byte_size = None
            content = None
        elif resource.url is not None:
            # Cheap scheme/credential check now; full DNS/redirect happens on read.
            validate_public_http_url(resource.url)
            filename = _link_filename(resource.url, filename)
            dedup_key = (
                "link",
                normalize_public_http_url_identity(resource.url).encode("utf-8"),
            )
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
            existing = self._canonical_resource_id(existing)
            registered = self._resources[existing]
            if admission_origin == "agent" and existing not in self._fetched:
                registered.presentation = presentation
            if caller:
                if registered.source == "web":
                    registered.source = "link"
                    registered.admission_origin = "caller"
                if resource.url is not None:
                    registered.url = normalize_public_http_url_identity(resource.url)
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
        aliased = self._aliases.get(resource_id)
        if aliased is not None:
            canonical = self._canonical_resource_id(aliased)
            self._ids_by_dedup[dedup_key] = canonical
            return canonical
        restored = self._resources.get(resource_id)
        if restored is not None:
            if resource.url is None or restored.url is None:
                raise ResourceStateMismatchError("resource identity collides with durable state")
            self._ids_by_dedup[dedup_key] = resource_id
            return resource_id
        self._resources[resource_id] = _Registered(
            resource_id=resource_id,
            filename=filename,
            declared_mime=resource.declared_mime,
            source="web" if source == "link" and not caller else source,
            content=content,
            url=(
                normalize_public_http_url_identity(resource.url)
                if resource.url is not None
                else None
            ),
            byte_size=byte_size,
            loader=resource.loader,
            admission_origin=admission_origin,
            presentation=presentation,
        )
        self._ids_by_dedup[dedup_key] = resource_id
        if byte_size is not None:
            self._total_bytes += byte_size
        return resource_id

    def canonical_resource_id(self, resource_id: str) -> str:
        """Return the durable canonical handle for a known Resource alias."""
        canonical = self._canonical_resource_id(resource_id)
        self._require(canonical)
        return canonical

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

    def restore_fetched_resource(
        self,
        *,
        resource_id: str,
        ordinal: int,
        filename: str,
        mime_type: str,
        url: str,
        content: bytes,
        admission_origin: Literal["caller", "search", "agent"],
        acquisition: str,
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Hydrate one durable Web catalog entry and its fixed representation."""
        self._ensure_open()
        if admission_origin in {"search", "agent"}:
            validate_agent_public_url(url)
        else:
            validate_public_http_url(url)
        normalized_url = normalize_public_http_url_identity(url)
        existing = self._resources.get(resource_id)
        if existing is None:
            existing = _Registered(
                resource_id=resource_id,
                filename=filename,
                declared_mime=mime_type,
                source="web" if admission_origin != "caller" else "link",
                content=None,
                url=normalized_url,
                byte_size=None,
                admission_origin=admission_origin,
                acquisition=acquisition,
            )
            self._resources[resource_id] = existing
        else:
            existing.filename = filename
            existing.declared_mime = mime_type
            existing.url = normalized_url
            existing.admission_origin = admission_origin
            existing.acquisition = acquisition
        self._ids_by_dedup[("link", normalized_url.encode("utf-8"))] = resource_id
        self._fetched_ordinals[resource_id] = ordinal
        self._next_fetched_ordinal = max(self._next_fetched_ordinal, ordinal + 1)
        for alias in aliases:
            if not alias.startswith("res-"):
                raise ResourceStateMismatchError("durable Web resource alias is invalid")
            if alias == resource_id:
                continue
            if alias in self._resources or alias in self._aliases:
                raise ResourceStateMismatchError("durable Web resource alias collides")
            self._aliases[alias] = resource_id
        self.restore_fetched_bytes(resource_id, content)

    def restore_discovered_resources(self, contexts: dict[str, Any]) -> None:
        """Rebuild search handles from the already durable Evidence ledger."""
        for row in contexts.get("chunks") or ():
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata") or {}
            if metadata.get("admission_origin") != "search":
                continue
            url = str(metadata.get("source_uri") or "")
            expected = str(metadata.get("resource_id") or "")
            restored = self.register_discovered_link(url)
            canonical_expected = self._canonical_resource_id(expected) if expected else ""
            if canonical_expected and restored != canonical_expected:
                raise ResourceStateMismatchError(
                    "durable search evidence does not match the resource catalog"
                )

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
        if self._resources[resource_id].url is None:
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
                source="link" if item.source == "web" else item.source,  # type: ignore[arg-type]
                byte_size=item.byte_size,
            )
            for item in self._resources.values()
        )

    def evidence_source(self, resource_id: str) -> dict[str, str]:
        """Return stable private provenance for evidence derived from a resource."""
        resource = self._require(resource_id)
        source_uri = resource.url if resource.source == "web" and resource.url else resource_id
        return {
            "source_type": "web_search" if resource.source == "web" else "web_attachment",
            "resource_kind": "web" if resource.url else "attachment",
            "admission_origin": resource.admission_origin,
            "acquisition": resource.acquisition or "",
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
        """Materialize a Resource to an ephemeral temporary file and return it.

        Only binary readers that need a filesystem path call this; direct text
        reads never do, so text answers never create temporary storage.
        """
        resource = self._require(resource_id)
        existing = self._paths.get(resource_id)
        if existing is not None:
            return existing
        content = await self._materialize_bytes(resource, effect_owner=effect_owner)
        resource = self._require(resource_id)
        resource_id = resource.resource_id
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
        resource_id = resource.resource_id
        effective_focus = focus
        cursor_state: _CursorState | None = None
        if cursor is not None:
            cursor_state = self._resolve_cursor(cursor, resource_id=resource_id)
            if focus is not None:
                raise ResourceCursorError("cursor is not valid for this resource read")
            effective_focus = None

        view = await self._read_text_view(resource, effect_owner=effect_owner)
        resource_id = self._canonical_resource_id(resource_id)
        text = view.text
        resource_handles = view.handles
        if not text:
            result = ResourceReadResult(
                resource_id=resource_id,
                locator=None,
                content="",
                extraction_status=view.extraction_status,
                has_more=False,
                next_cursor=None,
                visual_handles=resource_handles,
                evidence_available=view.evidence_available,
                note=view.note,
            )
            if estimate_tokens(format_resource_read(result)) > max_window_tokens:
                raise ResourceAdmissionError(
                    "resource read envelope exceeds residual model capacity"
                )
            return result

        if cursor_state is None:
            # A failed Web acquisition is deliberately not pinned. A fresh retry
            # may therefore produce different text and must replace any plan made
            # for the earlier bounded failure summary.
            self._cursor_plans = {
                key: plan for key, plan in self._cursor_plans.items() if key[0] != resource_id
            }
        plan_window_tokens = (
            cursor_state.plan_window_tokens if cursor_state is not None else max_window_tokens
        )
        plan = await self._cursor_plan(
            resource_id,
            text,
            effective_focus,
            plan_window_tokens=plan_window_tokens,
        )
        if cursor_state is not None:
            plan = _rotate_plan(plan, cursor_state.anchor_offset)
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
            evidence_available=view.evidence_available,
            note=view.note,
            extraction_status=view.extraction_status,
        )
        has_more = next_position < len(plan)
        next_cursor = None
        if has_more:
            next_cursor = self._mint_cursor(
                _CursorState(
                    resource_id=resource_id,
                    plan_window_tokens=plan_window_tokens,
                    plan_position=next_position,
                    char_offset=next_offset,
                    anchor_offset=plan[0][0],
                )
            )
        return ResourceReadResult(
            resource_id=resource_id,
            locator=locator,
            content=chunk,
            extraction_status=view.extraction_status,
            has_more=has_more,
            next_cursor=next_cursor,
            visual_handles=visual_handles,
            evidence_available=view.evidence_available,
            note=view.note,
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
        if resource.acquisition in {"exa_extract", "tavily_extract"}:
            view = _ConvertedResource(text=content.decode("utf-8"), handles=())
        elif is_convertible(resource.filename, resource.declared_mime):
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
        """Read one fixed URL snapshot, using the Extract chain only on failure."""
        url = resource.url
        if url is None:  # pragma: no cover - only link resources are routed here
            raise ResourceNotFoundError(f"resource {resource.resource_id} has no link")
        restored = self._restored_bytes(resource.resource_id)
        if restored is not None:
            cached = self._text_views.get(resource.resource_id)
            if cached is not None:
                return cached
            return await self._text_view_from_content(resource, restored)
        cached = self._text_views.get(resource.resource_id)
        if cached is not None:
            content = self._fetched.get(resource.resource_id)
            if content is not None:
                await self._persist_fetched(
                    resource.resource_id,
                    content,
                    effect_owner=effect_owner,
                )
            return cached
        try:
            content = await self._materialize_fetched(
                resource.resource_id,
                lambda: self._fetch_link(resource),
                effect_owner=effect_owner,
                charge_total=False,
            )
        except _RedirectAlias as alias:
            return await self._read_link_text_view(
                self._require(alias.resource_id),
                effect_owner=effect_owner,
            )
        except PublicHttpPolicyError:
            # Never send a URL rejected by the local public/anonymous policy to
            # an external extraction provider.
            raise
        except Exception:
            self._fetched.pop(resource.resource_id, None)
            self._converted.pop(resource.resource_id, None)
            return await self._fallback_text_view(
                resource,
                resource.url or url,
                effect_owner=effect_owner,
            )
        try:
            view = await self._text_view_from_content(resource, content)
        except Exception:
            if _is_textual_web_resource(resource):
                self._fetched.pop(resource.resource_id, None)
                self._converted.pop(resource.resource_id, None)
                return await self._fallback_text_view(
                    resource,
                    resource.url or url,
                    effect_owner=effect_owner,
                )
            await self._persist_fetched(
                resource.resource_id,
                content,
                effect_owner=effect_owner,
            )
            raise
        if not view.text:
            if _is_textual_web_resource(resource):
                self._fetched.pop(resource.resource_id, None)
                self._converted.pop(resource.resource_id, None)
                return await self._fallback_text_view(
                    resource,
                    resource.url or url,
                    effect_owner=effect_owner,
                )
            await self._persist_fetched(
                resource.resource_id,
                content,
                effect_owner=effect_owner,
            )
            return _unavailable_web_view()
        try:
            await self._persist_fetched(
                resource.resource_id,
                content,
                effect_owner=effect_owner,
            )
        except BaseException:
            self._text_views.pop(resource.resource_id, None)
            self._converted.pop(resource.resource_id, None)
            self._fetched.pop(resource.resource_id, None)
            raise
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

    def is_declared_image(self, resource_id: str) -> bool:
        resource = self._require(resource_id)
        media = (resource.declared_mime or "").split(";", 1)[0].strip().lower()
        suffix = Path(resource.filename or "").suffix.lower()
        return media.startswith("image/") or suffix in {
            ".avif",
            ".bmp",
            ".gif",
            ".jpeg",
            ".jpg",
            ".png",
            ".tif",
            ".tiff",
            ".webp",
        }

    async def inspection_target(
        self,
        resource_id: str,
        *,
        effect_owner: ResourceEffectOwner | None = None,
    ) -> InspectionTarget:
        """Materialize a resource and classify how it can be visually inspected."""
        resource = self._require(resource_id)
        content = await self._materialize_bytes(resource, effect_owner=effect_owner)
        resource = self._require(resource_id)
        resource_id = resource.resource_id
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

    @property
    def has_temp_storage(self) -> bool:
        return self._tempdir is not None

    def _canonical_resource_id(self, resource_id: str) -> str:
        seen: set[str] = set()
        while resource_id in self._aliases:
            if resource_id in seen:  # pragma: no cover - aliases are only created acyclically
                raise ResourceStateMismatchError("resource alias cycle")
            seen.add(resource_id)
            resource_id = self._aliases[resource_id]
        return resource_id

    def _require(self, resource_id: str) -> _Registered:
        self._ensure_open()
        canonical = self._canonical_resource_id(resource_id)
        try:
            return self._resources[canonical]
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
            data = await self._materialize_fetched(
                resource.resource_id,
                resource.loader,
                effect_owner=effect_owner,
                charge_total=True,
            )
            await self._persist_fetched(resource.resource_id, data, effect_owner=effect_owner)
            return data
        url = resource.url
        if url is None:  # pragma: no cover - a resource is always bytes or a link
            raise ResourceNotFoundError(f"resource {resource.resource_id} has no content")
        try:
            data = await self._materialize_fetched(
                resource.resource_id,
                lambda: self._fetch_link(resource),
                effect_owner=effect_owner,
                charge_total=False,
            )
        except _RedirectAlias as alias:
            return await self._materialize_bytes(
                self._require(alias.resource_id),
                effect_owner=effect_owner,
            )
        canonical_id = self._canonical_resource_id(resource.resource_id)
        await self._persist_fetched(canonical_id, data, effect_owner=effect_owner)
        return data

    def _restored_bytes(self, resource_id: str) -> bytes | None:
        """Return settled bytes, which never re-enter the network path.

        Revalidation guards a fetch; a restored read makes no request at all, so
        a host that stopped resolving cannot fail a run whose bytes are durable.
        """
        if resource_id not in self._durable_fetched:
            return None
        return self._fetched.get(resource_id)

    async def _fetch_link(self, resource: _Registered) -> bytes:
        url = resource.url
        if url is None:  # pragma: no cover - caller routes only links here
            raise ResourceNotFoundError(f"resource {resource.resource_id} has no link")
        result = await fetch_public_http(
            url,
            max_bytes=self._max_attachment_bytes,
            timeout=self._url_timeout,
            presentation=resource.presentation,
            client=self._url_client,
            agent_url=resource.source == "web",
        )
        resource.acquisition = "direct_http"
        resource.declared_mime = resource.declared_mime or result.media_type
        canonical = self._bind_final_url(resource, result.final_url)
        if canonical is not resource:
            if canonical.resource_id not in self._fetched:
                canonical.acquisition = resource.acquisition
                canonical.declared_mime = canonical.declared_mime or resource.declared_mime
                self._fetched[canonical.resource_id] = result.content
            raise _RedirectAlias(canonical.resource_id)
        return result.content

    def _bind_final_url(self, resource: _Registered, final_url: str) -> _Registered:
        if resource.source == "web":
            validate_agent_public_url(final_url)
        normalized = normalize_public_http_url_identity(final_url)
        dedup_key = ("link", normalized.encode("utf-8"))
        existing_id = self._ids_by_dedup.get(dedup_key)
        if existing_id is not None:
            existing_id = self._canonical_resource_id(existing_id)
        if existing_id is not None and existing_id != resource.resource_id:
            canonical = self._resources[existing_id]
            self._aliases[resource.resource_id] = existing_id
            self._resources.pop(resource.resource_id, None)
            return canonical
        resource.url = normalized
        self._ids_by_dedup[dedup_key] = resource.resource_id
        return resource

    async def _materialize_fetched(
        self,
        resource_id: str,
        producer: Callable[[], Awaitable[bytes]],
        *,
        effect_owner: ResourceEffectOwner | None,
        charge_total: bool,
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
                        charge_total=charge_total,
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
        charge_total: bool,
    ) -> bytes:
        data = await producer()
        if len(data) > self._max_attachment_bytes:
            raise ResourceAdmissionError("attachment exceeds per-attachment byte limit")
        async with self._total_lock:
            if charge_total and self._total_bytes + len(data) > self._max_total_attachment_bytes:
                raise ResourceAdmissionError("total attachment bytes exceeded")
            if charge_total:
                self._total_bytes += len(data)
            self._fetched[resource_id] = data
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
        async with self._persist_lock:
            if resource_id in self._durable_fetched:
                # Rebinding a settled slot could delete the bytes it depends on.
                return
            admitted = self._admitted_fetched.get(resource_id)
            if admitted is not None and admitted != content:
                raise ResourceStateMismatchError(
                    "one Web resource cannot bind two admitted representations"
                )
            effect_key = (
                resource_id,
                effect_owner.execution_scope if effect_owner is not None else None,
                effect_owner.intent_id.value if effect_owner is not None else None,
            )
            if effect_key in self._admitted_effects:
                return
            await self._fetched_bytes_sink(
                FetchedResourceBytes(
                    resource_id=resource_id,
                    ordinal=self.allocate_fetched_ordinal(resource_id),
                    filename=safe_source_filename(resource.filename or resource_id),
                    mime_type=resource.declared_mime or "application/octet-stream",
                    url=resource.url,
                    content=content,
                    admission_origin=resource.admission_origin,
                    acquisition=resource.acquisition or "direct_http",
                    aliases=tuple(
                        alias
                        for alias in self._aliases
                        if self._canonical_resource_id(alias) == resource_id
                    ),
                ),
                effect_owner,
            )
            self._admitted_fetched[resource_id] = content
            self._admitted_effects.add(effect_key)

    async def _fallback_text_view(
        self,
        resource: _Registered,
        url: str,
        *,
        effect_owner: ResourceEffectOwner | None,
    ) -> _ConvertedResource:
        """Share one Extract attempt; cache only a successfully admitted snapshot."""
        resource_id = resource.resource_id
        cached = self._text_views.get(resource_id)
        if cached is not None:
            content = self._fetched.get(self._canonical_resource_id(resource_id))
            if content is not None:
                await self._persist_fetched(
                    resource_id,
                    content,
                    effect_owner=effect_owner,
                )
            return cached
        async with self._fallback_lock:
            cached = self._text_views.get(resource_id)
            if cached is not None:
                content = self._fetched.get(self._canonical_resource_id(resource_id))
                if content is not None:
                    await self._persist_fetched(
                        resource_id,
                        content,
                        effect_owner=effect_owner,
                    )
                return cached
            task = self._fallback_tasks.get(resource_id)
            if task is None:
                task = asyncio.ensure_future(self._run_fallback(resource, url))
                self._fallback_tasks[resource_id] = task
        try:
            view = await asyncio.shield(task)
        except BaseException:
            if task.done():
                async with self._fallback_lock:
                    if self._fallback_tasks.get(resource_id) is task:
                        self._fallback_tasks.pop(resource_id, None)
            raise
        async with self._fallback_lock:
            self._fallback_tasks.pop(resource_id, None)
            if not view.evidence_available:
                return view
            cache_id = self._canonical_resource_id(resource_id)
            self._text_views.setdefault(cache_id, view)
            admitted = self._text_views[cache_id]
        content = self._fetched.get(cache_id)
        if content is not None:
            await self._persist_fetched(
                cache_id,
                content,
                effect_owner=effect_owner,
            )
        return admitted

    async def _run_fallback(
        self,
        resource: _Registered,
        url: str,
    ) -> _ConvertedResource:
        if self._url_text_fallback is None:
            return _unavailable_web_view()
        # Extraction providers must never receive a private or credential-bearing
        # locator, even when a caller attachment used private transport metadata.
        validate_agent_public_url(url)
        await avalidate_public_http_url(url)
        try:
            extracted = await self._url_text_fallback(url)
        except asyncio.CancelledError:
            raise
        except Exception:
            return _unavailable_web_view()
        text = extracted.text
        if not text.strip():
            return _unavailable_web_view()
        data = text.encode("utf-8")
        if len(data) > self._max_attachment_bytes:
            return _unavailable_web_view()
        canonical = self._bind_final_url(resource, extracted.url)
        if canonical is not resource:
            existing_content = self._fetched.get(canonical.resource_id)
            if existing_content is not None:
                return await self._text_view_from_content(canonical, existing_content)
        resource = canonical
        resource.acquisition = extracted.acquisition
        resource.declared_mime = "text/markdown; charset=utf-8"
        notes: list[str] = []
        if extracted.dropped_results:
            notes.append(f"Dropped {extracted.dropped_results} malformed extraction result(s).")
        if extracted.degradation:
            notes.append(extracted.degradation)
        resource.degradation = " ".join(notes) or None
        self._fetched[resource.resource_id] = data
        return _ConvertedResource(
            text=text,
            handles=(),
            note=resource.degradation,
        )

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
        payload = struct.pack(
            ">BIIII",
            _CURSOR_VERSION,
            state.plan_window_tokens,
            state.plan_position,
            state.char_offset,
            state.anchor_offset,
        )
        signature = hmac.new(
            self._cursor_secret,
            state.resource_id.encode("utf-8") + b"|" + payload,
            hashlib.sha256,
        ).digest()[:_CURSOR_SIGNATURE_BYTES]
        return base64.urlsafe_b64encode(payload + signature).rstrip(b"=").decode("ascii")

    def _resolve_cursor(self, cursor: str, *, resource_id: str) -> _CursorState:
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            payload, signature = raw[:-_CURSOR_SIGNATURE_BYTES], raw[-_CURSOR_SIGNATURE_BYTES:]
            expected = hmac.new(
                self._cursor_secret,
                resource_id.encode("utf-8") + b"|" + payload,
                hashlib.sha256,
            ).digest()[:_CURSOR_SIGNATURE_BYTES]
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            if len(payload) != struct.calcsize(">BIIII"):
                raise ValueError
            version, window, position, offset, anchor_offset = struct.unpack(">BIIII", payload)
            if version != _CURSOR_VERSION:
                raise ValueError
        except (ValueError, UnicodeError, struct.error) as exc:
            raise ResourceCursorError("cursor is not valid for this resource read") from exc
        return _CursorState(
            resource_id=resource_id,
            plan_window_tokens=window,
            plan_position=position,
            char_offset=offset,
            anchor_offset=anchor_offset,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ResourceRegistryClosedError("resource registry is closed")


def _unavailable_web_view() -> _ConvertedResource:
    return _ConvertedResource(
        text=(
            "This public URL produced no citable text: direct HTTP failed or "
            "returned no usable textual representation, and the configured "
            "extraction chain returned no usable representation."
        ),
        handles=(),
        evidence_available=False,
        note="Web acquisition degraded: no citable evidence was admitted.",
        extraction_status="unavailable",
    )


class ResourceRegistryClosedError(RuntimeError):
    """Raised when a closed registry is used again."""


class ResourceStateMismatchError(RuntimeError):
    """Raised when a settled catalog cannot describe the replayed request."""


def _link_filename(url: str, explicit: str | None) -> str:
    filename = safe_source_filename(explicit or url)
    return filename if Path(filename).suffix else f"{filename}.html"


def _is_textual_web_resource(resource: _Registered) -> bool:
    media_type = (resource.declared_mime or "").split(";", 1)[0].strip().lower()
    if media_type.startswith("text/") or media_type in {
        "application/json",
        "application/ld+json",
        "application/xhtml+xml",
        "application/xml",
    }:
        return True
    return Path(resource.filename or "").suffix.lower() in {
        ".csv",
        ".html",
        ".htm",
        ".json",
        ".log",
        ".md",
        ".rst",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }


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
    if not focus or order == list(range(len(spans))):
        return tuple(spans)
    best = order[0]
    start, _end = spans[best]
    match = windows[best][1].casefold().find(focus.casefold())
    anchor = start if match < 0 else text.rfind("\n", 0, start + match) + 1
    return _rotate_plan(tuple(spans), anchor)


def _rotate_plan(
    plan: tuple[tuple[int, int], ...], anchor_offset: int
) -> tuple[tuple[int, int], ...]:
    for index, (start, end) in enumerate(plan):
        if start <= anchor_offset < end:
            head = ((anchor_offset, end),) if anchor_offset < end else ()
            tail = ((start, anchor_offset),) if start < anchor_offset else ()
            return (*head, *plan[index + 1 :], *plan[:index], *tail)
    raise ResourceCursorError("cursor does not match the current resource text")


def _read_cursor_span(
    text: str,
    plan: tuple[tuple[int, int], ...],
    *,
    resource_id: str,
    plan_position: int,
    char_offset: int,
    max_window_tokens: int,
    visual_handles: tuple[VisualHandle, ...],
    evidence_available: bool,
    note: str | None,
    extraction_status: str,
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
            extraction_status=extraction_status,
            has_more=has_more,
            next_cursor=_CURSOR_PLACEHOLDER if has_more else None,
            visual_handles=visual_handles,
            evidence_available=evidence_available,
            note=note,
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
    """Start at the best focus window, then cover the resource in physical order."""
    count = len(windows)
    if not focus or count <= 1:
        return list(range(count))
    query_terms = mixed_script_terms(focus)
    if not query_terms:
        return list(range(count))
    documents = [mixed_script_terms(text) for _, text in windows]
    ranked = bm25_rank(query_terms, documents, limit=1)
    if not ranked:
        return list(range(count))
    best = ranked[0][0]
    return [*range(best, count), *range(0, best)]


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
