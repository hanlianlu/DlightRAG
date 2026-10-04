# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Run-scoped answer resource registry: admission, materialization, and reads.

The registry owns every resource for one Answer Run. Inline bytes stay in memory;
public HTTP(S) locators are fetched lazily and the first successful acquisition
becomes a fixed durable snapshot. A Web Resource may also hold one rendered
representation, the page as the Agent Browser serialized it after its scripts ran,
which is appended to the Resource and never replaces its snapshot (ADR 0032). Full
bytes never enter model context — only bounded text windows do. Continuation cursors
are opaque, run-scoped tokens bound to a Resource Handle, the representation they
page, and focus so they expose no path, offset, or provider locator. ``aclose``
deterministically cancels pending fetches and renders and joins the conversions still
running.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import mimetypes
import secrets
import struct
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.agent.tools import ResourceAttachmentBytes
from dlightrag.engine.ai.media import verify_web_image_bytes
from dlightrag.engine.ai.tokens import estimate_tokens
from dlightrag.engine.answer.agent_browser import AgentBrowserError, RenderedPage, browser_failure
from dlightrag.engine.answer.resources.converters import (
    ConversionLimitError,
    ExtractedVisual,
    ResourceConversionError,
    UnsafeArchiveError,
    convert_resource,
    is_convertible,
)
from dlightrag.engine.answer.resources.formatting import format_resource_read
from dlightrag.engine.answer.resources.lexical import bm25_rank, mixed_script_terms
from dlightrag.engine.answer.resources.models import (
    EXTRACTION_TEXT,
    PREPARED_RESOURCE_HANDLE_PREFIX,
    RenderedReadTargetError,
    ResourceAdmissionError,
    ResourceCursorError,
    ResourceInput,
    ResourceManifestEntry,
    ResourceNotConvertedError,
    ResourceNotFoundError,
    ResourceReadResult,
    TextWindowLocator,
    VisualHandle,
    is_resource_handle,
)
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from dlightrag.engine.answer.resources.text import (
    build_text_windows,
    declared_charset,
    decode_text,
)
from dlightrag.engine.answer.resources.visual import ResourceViewError, pdf_page_count
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
_EXTRACT_ACQUISITIONS = frozenset({"exa_extract", "tavily_extract"})
#: What an Agent Session's page yields beside a Rendered Read: the page as it stands, and a
#: file it downloaded. Each is its own Resource, never a representation of a URL's snapshot.
BROWSER_CAPTURE = "browser_capture"
BROWSER_DOWNLOAD = "browser_download"
BROWSER_ACQUISITIONS = frozenset({BROWSER_CAPTURE, BROWSER_DOWNLOAD})

#: A Web Resource's rendered representation is named after it, and is never a handle
#: the model is shown: results and notes print the Resource's own.
RENDERED_REPRESENTATION_SUFFIX = "-rendered"
BROWSER_RENDER = "browser_render"
_RENDERED_FILENAME = "rendered.html"
#: The serialized DOM is UTF-8 whatever its own ``<meta charset>`` says.
_RENDERED_MIME = "text/html; charset=utf-8"
#: A cursor names the representation it pages, so one Resource's two cannot be confused.
_RENDERED_CURSOR_PREFIX = "r."
_VISUAL_CURSOR_KIND = "visual"
_RENDERED_VISUAL_CURSOR_KIND = "rvisual"

# A run-scoped, provider-neutral fallback that returns already-usable text for
# a public URL, or ``None`` when it cannot. Exa owns its adapter; the registry
# never imports any web-search provider.
UrlTextFallback = Callable[[str], Awaitable[WebExtractResult]]
# Renders one public URL in the Run's Agent Browser; the registry never imports a browser.
PageRenderer = Callable[[str], Awaitable[RenderedPage]]


@dataclass(frozen=True, slots=True)
class HostedExtract:
    """Ask hosted extraction providers for the URL's text."""

    extract: UrlTextFallback


@dataclass(frozen=True, slots=True)
class AgentBrowserRender:
    """Render the URL in the Run's Agent Browser."""


# One step of the automatic Extract chain, tried in order (ADR 0005, ADR 0032).
type ExtractStep = HostedExtract | AgentBrowserRender


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


@dataclass(frozen=True, slots=True)
class BrowserResourceInput:
    """A page capture or a downloaded file, as the Agent Browser delivered it."""

    acquisition: Literal["browser_capture", "browser_download"]
    content: bytes
    filename: str
    declared_mime: str
    locator: str | None
    """The page's final URL, or the download's URL, as the browser reported it."""


# Persist validated fetched bytes before their ToolResult settles in the Session.
FetchedBytesSink = Callable[
    [FetchedResourceBytes, ResourceEffectOwner | None],
    Awaitable[None],
]


@dataclass(frozen=True)
class VisualTarget:
    """Materialized bytes plus the visual class of one viewable resource."""

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
    stored_view_only: bool = False
    #: Where a rendered representation's page ended; provenance, never an identity.
    final_url: str | None = None
    #: The public URL a browser capture or download is cited by; none where ADR 0005 keeps
    #: its URL private, and the Resource is cited by its handle.
    citable_url: str | None = None


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


@dataclass(frozen=True, slots=True)
class _ChainOutcome:
    """What walking the Extract chain produced: text, a rendering, or why neither."""

    extracted: WebExtractResult | None = None
    rendered: _Registered | None = None
    browser_failure: AgentBrowserError | None = None


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
        url_timeout: float = 120.0,
        extract_chain: tuple[ExtractStep, ...] = (),
        page_renderer: PageRenderer | None = None,
        fetched_bytes_sink: FetchedBytesSink | None = None,
        resource_secret: bytes | None = None,
        cursor_secret: bytes | None = None,
    ) -> None:
        self._max_attachments = max_attachments
        self._max_attachment_bytes = max(1, int(max_attachment_bytes))
        self._max_total_attachment_bytes = max(1, int(max_total_attachment_bytes))
        self._url_timeout = url_timeout
        self._extract_chain = extract_chain
        self._page_renderer = page_renderer
        self._fetched_bytes_sink = fetched_bytes_sink
        self._secret = resource_secret or secrets.token_bytes(32)
        self._cursor_secret = cursor_secret or secrets.token_bytes(32)

        self._resources: dict[str, _Registered] = {}
        self._aliases: dict[str, str] = {}
        self._ids_by_dedup: dict[tuple[str, bytes], str] = {}
        self._caller_dedup: set[tuple[str, bytes]] = set()
        self._fetched: dict[str, bytes] = {}
        self._cursor_plans: dict[tuple[str, str | None, int], tuple[tuple[int, int], ...]] = {}
        self._converted: dict[str, _ConvertedResource] = {}
        self._visual_assets: dict[tuple[str, str], ExtractedVisual] = {}
        self._snapshots: dict[str, ConversionSnapshot] = {}
        self._pdf_counts: dict[str, int | None] = {}
        self._refused: dict[str, BaseException] = {}
        self._conversion_tasks: dict[str, asyncio.Task[_ConvertedResource]] = {}
        # Views of bytes this Run publishes; see ``conversion_view``.
        self._view_tasks: set[asyncio.Task[ConversionSnapshot]] = set()
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
        self._fallback_tasks: dict[str, asyncio.Future[tuple[_Registered, _ConvertedResource]]] = {}
        self._text_view_tasks: dict[
            str, asyncio.Future[tuple[_Registered, _ConvertedResource]]
        ] = {}
        # Adoptions of earlier Runs' Resources run one at a time; see ``adopt``.
        self._adoption_lock = asyncio.Lock()
        # A Web Resource's rendered representation, by the Resource's id. It is never in
        # ``_resources``, the manifest, or a handle the model is shown, and it is not
        # charged to the attachment total. One page renders at most once at a time.
        self._rendered: dict[str, _Registered] = {}
        self._render_lock = asyncio.Lock()
        self._render_tasks: dict[str, asyncio.Future[_Registered]] = {}
        # Resources whose plain read is the rendered text: their own bytes hold none.
        self._default_rendered: set[str] = set()

    async def __aenter__(self) -> ResourceRegistry:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def register(self, resource: ResourceInput) -> str:
        """Admit caller bytes, a link, or a lazily loaded upload."""
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
        stored_view_only: bool = False,
    ) -> str:
        """Admit one Resource; ``stored_view_only`` bytes are never converted here.

        Identical bytes already admitted keep the state of their first admission,
        so ``stored_view_only`` applies only to bytes this call admits first.
        """
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
            # admission until the model actually reads/views the resource.
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
            filename = safe_source_filename(filename or resource.url)
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
            dedup_key = _content_key(resource)
            source = "bytes"
            byte_size = len(content)

        existing = self._ids_by_dedup.get(dedup_key)
        if caller:
            self._admit_caller_key(dedup_key)
        if existing is not None:
            existing = self._canonical_resource_id(existing)
            registered = self._resources[existing]
            # A fetch in flight already chose its presentation, as a finished one has.
            fetched = existing in self._fetched or existing in self._fetch_tasks
            if any(
                (presentation.user_agent, presentation.accept, presentation.accept_language)
            ) and (fetched or existing in self._text_views):
                raise ResourceAdmissionError(
                    "HTTP presentation cannot replace an admitted snapshot"
                )
            if admission_origin == "agent" and not fetched:
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
            stored_view_only=stored_view_only,
        )
        self._ids_by_dedup[dedup_key] = resource_id
        if byte_size is not None:
            self._total_bytes += byte_size
        return resource_id

    def _bind_alias(self, alias: str, canonical: str) -> None:
        """Point one earlier durable handle at this Run's canonical Resource."""
        if self._alias_binds(alias, canonical):
            self._aliases[alias] = canonical

    def _alias_binds(self, alias: str, canonical: str) -> bool:
        """Whether ``alias`` is a handle to bind to ``canonical``, checking it may be.

        Aliases only ever name bytes this Run already holds, so following one can
        never reach content the Run did not adopt; a collision is therefore a state
        mismatch rather than a silent rebind.
        """
        if not is_resource_handle(alias) or alias == canonical:
            return False
        bound = self._aliases.get(alias)
        if bound is not None and self._canonical_resource_id(bound) != canonical:
            raise ResourceStateMismatchError("Resource alias collides with another Resource")
        if alias in self._resources:
            raise ResourceStateMismatchError("Resource alias collides with another Resource")
        return True

    async def adopt(
        self,
        resource: ResourceInput,
        *,
        alias: str,
        view: ConversionSnapshot | None,
        record: Callable[[str, ConversionSnapshot | None], Awaitable[None]],
        needs_text: bool = False,
    ) -> str:
        """Hold another Run's bytes under ``alias`` once ``record`` made them durable.

        Adoptions run one at a time, so the canonical handle, whether ``view``
        comes along, the durable record, and the binding see no other adoption in
        between, and one Resource never gains two views. ``view`` becomes the view
        of this Run's Resource, named by its handle, only for bytes that are new
        here or were adopted without one: bytes this Run can convert itself keep
        their own conversion. A caller that ``needs_text`` is refused before
        anything is admitted when the Resource would read text only through a view
        it does not have.

        Nothing a later call can reach, the alias or the view, exists before
        ``record`` returns. The bytes themselves are admitted first, because
        lazily loaded uploads charge the same request total while the record is
        written, and are withdrawn again when it does not complete. A record whose
        outcome is unknown may still have landed; a resume then restores it with
        ``restore_adopted``, as durable state rather than a new admission.
        """
        async with self._adoption_lock:
            if alias in self._aliases:
                return self._canonical_resource_id(alias)
            held = self._held_bytes(resource)
            has_view = held is not None and held.resource_id in self._snapshots
            converts_here = held is not None and not held.stored_view_only
            takes_view = view is not None and not has_view and not converts_here
            if (
                needs_text
                and not (takes_view or has_view or converts_here)
                and is_convertible(resource.filename, resource.declared_mime)
            ):
                raise ResourceNotConvertedError(resource.filename or alias, resource.declared_mime)
            canonical = self._register(resource, admission_origin="caller", stored_view_only=True)
            try:
                binds = self._alias_binds(alias, canonical)
                adopted = (
                    replace(view, resource_id=canonical)
                    if view is not None and takes_view
                    else None
                )
                await record(canonical, adopted)
            except BaseException:
                if held is None:
                    self._withdraw(canonical, resource)
                raise
            if binds:
                self._aliases[alias] = canonical
            if adopted is not None:
                self.adopt_conversion_snapshot(adopted)
            return canonical

    def _held_bytes(self, resource: ResourceInput) -> _Registered | None:
        """This Run's Resource that registering these caller bytes would return."""
        key = _content_key(resource)
        known = self._ids_by_dedup.get(key) or self._aliases.get(self._mint_resource_id(key))
        return None if known is None else self._resources[self._canonical_resource_id(known)]

    def _withdraw(self, resource_id: str, resource: ResourceInput) -> None:
        """Undo the first admission of caller bytes whose adoption was not recorded."""
        withdrawn = self._resources.pop(resource_id)
        key = _content_key(resource)
        self._ids_by_dedup.pop(key, None)
        self._caller_dedup.discard(key)
        self._total_bytes -= withdrawn.byte_size or 0

    def restore_adopted(
        self,
        resource_id: str,
        *,
        filename: str,
        mime_type: str,
        content: bytes,
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Hydrate one recorded adoption under the handle it was recorded with.

        A recorded adoption is durable Run state, like restored fetched bytes: it
        takes its allowance slot and its bytes from the request total but is never
        refused, even when a record whose outcome was unknown landed after the Run
        had given its slot to another adoption. Bytes this Run already holds keep
        their Resource and state, and gain the recorded handles as aliases.
        """
        self._ensure_open()
        resource = ResourceInput(filename=filename, declared_mime=mime_type, content=content)
        held = self._held_bytes(resource)
        if held is None:
            if resource_id in self._resources:
                raise ResourceStateMismatchError("recorded adoption collides with another Resource")
            key = _content_key(resource)
            self._resources[resource_id] = _Registered(
                resource_id=resource_id,
                filename=filename,
                declared_mime=mime_type,
                source="bytes",
                content=content,
                url=None,
                byte_size=len(content),
                stored_view_only=True,
            )
            self._ids_by_dedup[key] = resource_id
            self._caller_dedup.add(key)
            self._total_bytes += len(content)
        canonical = resource_id if held is None else held.resource_id
        for alias in (resource_id, *aliases):
            self._bind_alias(alias, canonical)

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
            if not is_resource_handle(alias):
                raise ResourceStateMismatchError("durable Web resource alias is invalid")
            if alias == resource_id:
                continue
            if alias in self._resources or alias in self._aliases:
                raise ResourceStateMismatchError("durable Web resource alias collides")
            self._aliases[alias] = resource_id
        self.restore_fetched_bytes(resource_id, content)

    async def admit_browser_resource(
        self, resource: BrowserResourceInput, *, effect_owner: ResourceEffectOwner, index: int
    ) -> str:
        """Admit a capture or a download as a new Agent Resource, kept with the call that made it.

        It is a new Resource every time and never touches the URL dedup map, so what a page
        showed after interaction never rebinds the snapshot its URL serves. It takes no
        attachment slot and no share of the request total, as a fetched URL takes none.
        Its bytes are inline, so a read converts or decodes them and never fetches,
        extracts, or renders. A URL ADR 0005 keeps private is never stored: the durable
        locator is the citable URL, or the handle.
        """
        self._ensure_open()
        if len(resource.content) > self._max_attachment_bytes:
            noun = "captured page" if resource.acquisition == BROWSER_CAPTURE else "download"
            raise ResourceAdmissionError(f"the {noun} exceeds {self._max_attachment_bytes} bytes")
        # The identity names the call and the file's place in it, so one call's admissions
        # never collide and a recovered Run mints the handles it already printed.
        resource_id = self._mint_resource_id(
            (
                "browser",
                f"{effect_owner.execution_scope}\0{effect_owner.intent_id.value}\0{index}".encode(),
            )
        )
        citable = _citable_agent_url(resource.locator)
        self._resources[resource_id] = _Registered(
            resource_id=resource_id,
            filename=resource.filename,
            declared_mime=resource.declared_mime,
            source="bytes",
            content=resource.content,
            url=None,
            byte_size=len(resource.content),
            admission_origin="agent",
            acquisition=resource.acquisition,
            citable_url=citable,
        )
        if self._fetched_bytes_sink is not None:
            try:
                await self._fetched_bytes_sink(
                    FetchedResourceBytes(
                        resource_id=resource_id,
                        ordinal=self.allocate_fetched_ordinal(resource_id),
                        filename=resource.filename,
                        mime_type=resource.declared_mime,
                        url=citable or resource_id,
                        content=resource.content,
                        admission_origin="agent",
                        acquisition=resource.acquisition,
                    ),
                    effect_owner,
                )
            except BaseException:
                self._resources.pop(resource_id, None)
                raise
        return resource_id

    def restore_browser_resource(
        self,
        *,
        resource_id: str,
        ordinal: int,
        filename: str,
        mime_type: str,
        locator: str,
        content: bytes,
        acquisition: str,
    ) -> None:
        """Hydrate one settled capture or download under the handle it was admitted with.

        Its locator is the citable URL or the handle itself. A row that says otherwise
        names a URL ADR 0005 keeps private, which admission never stores, so the catalog
        does not describe this Run.
        """
        self._ensure_open()
        citable = _citable_agent_url(locator)
        if citable is None and locator != resource_id:
            raise ResourceStateMismatchError("a durable browser resource names a private locator")
        if resource_id in self._resources or resource_id in self._aliases:
            raise ResourceStateMismatchError("a durable browser resource collides with another")
        self._resources[resource_id] = _Registered(
            resource_id=resource_id,
            filename=filename,
            declared_mime=mime_type,
            source="bytes",
            content=content,
            url=None,
            byte_size=len(content),
            admission_origin="agent",
            acquisition=acquisition,
            citable_url=citable,
        )
        self._fetched_ordinals[resource_id] = ordinal
        self._next_fetched_ordinal = max(self._next_fetched_ordinal, ordinal + 1)

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

    def evidence_source(
        self, resource_id: str, *, text: bool = False, rendered: bool = False
    ) -> dict[str, str]:
        """Return stable private provenance for evidence derived from a resource.

        ``text`` asks for the provenance of the text a read returns, which the
        Extract chain supplies when the Resource's own bytes hold none. ``rendered``
        says that text is the Agent Browser's rendering; everything else about the
        evidence stays the Resource's own.
        """
        resource = self._require(resource_id)
        if resource.acquisition in BROWSER_ACQUISITIONS:
            # A capture or a download is its own text and its own bytes, cited by its
            # public URL when it has one and by its handle when it has not.
            source_uri = resource.citable_url or resource_id
            return {
                "source_type": "web_search" if resource.citable_url else "web_attachment",
                "resource_kind": "web",
                "admission_origin": "agent",
                "acquisition": resource.acquisition or "",
                "source_uri": source_uri,
                "source_download_locator": source_uri,
                "title": safe_source_filename(resource.filename or source_uri),
            }
        source_uri = resource.url if resource.source == "web" and resource.url else resource_id
        acquisition = resource.acquisition or ""
        snapshot = self._snapshots.get(resource.resource_id)
        if text and snapshot is not None and snapshot.converter in _EXTRACT_ACQUISITIONS:
            acquisition = snapshot.converter
        if rendered:
            acquisition = BROWSER_RENDER
        return {
            "source_type": "web_search" if resource.source == "web" else "web_attachment",
            "resource_kind": "web" if resource.url else "attachment",
            "admission_origin": resource.admission_origin,
            "acquisition": acquisition,
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

    async def read(
        self,
        resource_id: str,
        *,
        max_window_tokens: int,
        focus: str | None = None,
        cursor: str | None = None,
        rendered: bool = False,
        effect_owner: ResourceEffectOwner | None = None,
    ) -> ResourceReadResult:
        """Return one page whose complete model-visible envelope fits the budget.

        ``rendered`` reads a Web Resource as the Agent Browser renders it, rendering the
        page when the Run holds no rendering of it yet. A cursor names the representation
        it continues, so alone it selects that one. Without either, the read is the
        Resource's own text, or its rendering where the Resource holds none of its own.
        """
        if max_window_tokens < 1:
            raise ResourceAdmissionError("resource read has no residual model capacity")
        resource = self._require(resource_id)
        resource_id = resource.resource_id
        continues_rendered = cursor is not None and cursor.startswith(
            (_RENDERED_CURSOR_PREFIX, f"{_RENDERED_VISUAL_CURSOR_KIND}.")
        )
        if rendered:
            if cursor is not None and not continues_rendered:
                raise ResourceCursorError(
                    "this cursor continues the direct representation; "
                    "call read without rendered=true"
                )
            if resource.url is None:
                raise RenderedReadTargetError(resource_id)
        rendered_id = f"{resource_id}{RENDERED_REPRESENTATION_SUFFIX}"
        effective_focus = focus
        cursor_state: _CursorState | None = None
        if cursor is not None and cursor.startswith(
            (f"{_VISUAL_CURSOR_KIND}.", f"{_RENDERED_VISUAL_CURSOR_KIND}.")
        ):
            if focus is not None:
                raise ResourceCursorError("visual inventory cursor does not accept focus")
            start = self.resolve_visual_cursor(
                cursor,
                rendered_id if continues_rendered else resource_id,
                _RENDERED_VISUAL_CURSOR_KIND if continues_rendered else _VISUAL_CURSOR_KIND,
            )
            representation, view = await self._view_of(
                resource, rendered=continues_rendered, effect_owner=effect_owner
            )
            self._require_continued(cursor, representation, continues_rendered)
            handles, note = self._discovery(
                resource, representation, view, max_window_tokens, start=start
            )
            result = ResourceReadResult(
                resource_id,
                None,
                "Visual inventory (not text evidence).",
                view.extraction_status,
                False,
                None,
                handles,
                False,
                note,
                continues_rendered,
            )
            if estimate_tokens(format_resource_read(result)) > max_window_tokens:
                raise ResourceAdmissionError(
                    "visual inventory envelope exceeds residual model capacity"
                )
            return result
        if cursor is not None:
            cursor_state = self._resolve_cursor(
                cursor.removeprefix(_RENDERED_CURSOR_PREFIX) if continues_rendered else cursor,
                resource_id=rendered_id if continues_rendered else resource_id,
            )
            if focus is not None:
                raise ResourceCursorError("cursor is not valid for this resource read")
            effective_focus = None

        representation, view = await self._view_of(
            resource, rendered=rendered or continues_rendered, effect_owner=effect_owner
        )
        self._require_continued(cursor, representation, continues_rendered)
        # A redirect may have bound this read to the Resource another URL already held.
        resource_id = self._canonical_resource_id(resource_id)
        is_rendered = _is_rendered(representation)
        plan_id = representation.resource_id
        text = view.text
        resource = self._require(resource_id)
        if (
            not is_rendered
            and _is_pdf(resource.filename, resource.declared_mime)
            and resource_id not in self._pdf_counts
        ):
            content = await self._materialize_bytes(resource, effect_owner=effect_owner)
            try:
                self._pdf_counts[resource_id] = await asyncio.to_thread(pdf_page_count, content)
            except ResourceViewError:
                self._pdf_counts[resource_id] = None
        resource_handles, discovery = self._discovery(
            resource, representation, view, max_window_tokens
        )
        view = replace(view, note=discovery)
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
                rendered=is_rendered,
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
                key: plan for key, plan in self._cursor_plans.items() if key[0] != plan_id
            }
        plan_window_tokens = (
            cursor_state.plan_window_tokens if cursor_state is not None else max_window_tokens
        )
        plan = await self._cursor_plan(
            plan_id,
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
            rendered=is_rendered,
        )
        has_more = next_position < len(plan)
        next_cursor = None
        if has_more:
            next_cursor = _cursor_prefix(is_rendered) + self._mint_cursor(
                _CursorState(
                    resource_id=plan_id,
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
            rendered=is_rendered,
        )

    async def _view_of(
        self,
        resource: _Registered,
        *,
        rendered: bool,
        effect_owner: ResourceEffectOwner | None,
    ) -> tuple[_Registered, _ConvertedResource]:
        """The representation a read pages, with its text view.

        ``rendered`` names the Agent Browser's rendering: a read that asked for it renders
        the page when the Run holds none, and one continuing it needs the Run to hold it.
        """
        if not rendered:
            return await self._read_text_view(resource, effect_owner=effect_owner)
        if resource.url is None:
            raise RenderedReadTargetError(resource.resource_id)
        representation = await self._render(resource)
        return representation, await self._rendered_view(representation)

    @staticmethod
    def _require_continued(
        cursor: str | None, representation: _Registered, continues_rendered: bool
    ) -> None:
        """A cursor continues the representation it names, or the read refuses it."""
        if cursor is not None and _is_rendered(representation) != continues_rendered:
            raise ResourceCursorError(
                "this cursor continues the "
                + ("rendered" if continues_rendered else "direct")
                + " representation, which this read does not return"
            )

    async def _read_text_view(
        self,
        resource: _Registered,
        *,
        effect_owner: ResourceEffectOwner | None,
    ) -> tuple[_Registered, _ConvertedResource]:
        """The representation a plain read pages, with its text view.

        It is the Resource's own text. A Resource whose own bytes hold none reads as the
        Agent Browser rendered it, when it did (ADR 0032).
        """
        if resource.resource_id in self._refused:
            # A cancelled read can finish native conversion after its own Tool
            # intent has stopped. Rebind the already-fetched source to the next
            # observing intent so its refusal terminal cannot settle alone.
            content = self._fetched.get(resource.resource_id)
            if content is not None:
                await self._persist_fetched(
                    resource.resource_id,
                    content,
                    effect_owner=effect_owner,
                )
            raise ResourceAdmissionError("resource refused by safety/resource limits")
        if resource.url is not None:
            return await self._read_link_text_view(resource, effect_owner=effect_owner)
        cached = self._text_views.get(resource.resource_id)
        if cached is not None:
            return resource, cached
        content = await self._materialize_bytes(resource, effect_owner=effect_owner)
        return resource, await self._text_view_from_content(resource, content)

    async def _text_view_from_content(
        self,
        resource: _Registered,
        content: bytes,
    ) -> _ConvertedResource:
        try:
            image_media = verify_web_image_bytes(content)
        except ValueError:
            image_media = None
        if image_media is not None:
            view = _ConvertedResource(
                text="",
                handles=(),
                evidence_available=False,
                extraction_status="image",
                note=f"Image ({image_media}, {len(content)} bytes). Use view(resource_id={resource.resource_id!r}).",
            )
        elif resource.acquisition in _EXTRACT_ACQUISITIONS:
            view = _ConvertedResource(text=content.decode("utf-8"), handles=())
        elif is_convertible(resource.filename, resource.declared_mime):
            view = await self._ensure_converted(resource, content)
        else:
            text = await asyncio.to_thread(
                decode_text,
                content,
                declared_charset=declared_charset(resource.declared_mime),
            )
            view = _ConvertedResource(text=text, handles=())
        if view.text or not (resource.url and _is_textual_web_resource(resource)):
            self._text_views[resource.resource_id] = view
        return view

    async def _read_link_text_view(
        self,
        resource: _Registered,
        *,
        effect_owner: ResourceEffectOwner | None,
    ) -> tuple[_Registered, _ConvertedResource]:
        """Read one fixed URL snapshot, taking text from the Extract chain when it has none."""
        url = resource.url
        if url is None:  # pragma: no cover - only link resources are routed here
            raise ResourceNotFoundError(f"resource {resource.resource_id} has no link")
        cached = self._text_views.get(resource.resource_id)
        if cached is not None:
            content = self._fetched.get(resource.resource_id)
            if content is not None:
                await self._persist_fetched(
                    resource.resource_id,
                    content,
                    effect_owner=effect_owner,
                )
            return resource, cached
        rendered = self._rendered_for_plain_read(resource)
        if rendered is not None:
            return rendered, await self._rendered_view(rendered)
        # Settled bytes never re-enter the network path, and read as fetched ones do.
        content = self._restored_bytes(resource.resource_id)
        if content is None:
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
            except PublicHttpPolicyError, ResourceAdmissionError:
                # Never send a URL rejected by the local public/anonymous policy to
                # an external extraction provider.
                raise
            except Exception:
                # A failed fetch bound nothing, so there is nothing of this read's to
                # forget: what the Resource holds now another read bound or admitted.
                return await self._fallback_text_view(
                    resource,
                    resource.url or url,
                    effect_owner=effect_owner,
                )
        try:
            view = await self._text_view_from_content(resource, content)
        except ResourceAdmissionError, UnsafeArchiveError, ConversionLimitError, MemoryError:
            # Acquisition admitted these bytes. Terminal conversion snapshots must
            # settle with their source even though it is not extracted evidence.
            await self._persist_fetched(resource.resource_id, content, effect_owner=effect_owner)
            raise
        except Exception:
            if _is_textual_web_resource(resource):
                return await self._extract_text_view(
                    resource, resource.url or url, content, effect_owner=effect_owner
                )
            await self._persist_fetched(
                resource.resource_id,
                content,
                effect_owner=effect_owner,
            )
            raise
        if not view.text and view.extraction_status != "image":
            if _is_textual_web_resource(resource):
                return await self._extract_text_view(
                    resource, resource.url or url, content, effect_owner=effect_owner
                )
            await self._persist_fetched(
                resource.resource_id,
                content,
                effect_owner=effect_owner,
            )
            return resource, view
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
        return resource, view

    def _rendered_for_plain_read(self, resource: _Registered) -> _Registered | None:
        """The rendering a plain read takes instead of fetching, when the Resource's own text is no use.

        The decision rests on what the Run holds, which is what recovery restores: a Resource
        with no bytes of its own, or one whose bytes were found to hold no text, reads as its
        rendering, without a fetch and without walking the chain again.
        """
        rendered = self._settled_rendered(resource.resource_id)
        if rendered is None:
            return None
        resource_id = resource.resource_id
        holds_bytes = resource_id in self._fetched or resource_id in self._fetch_tasks
        if resource_id in self._default_rendered or not holds_bytes:
            return rendered
        return None

    async def _extract_text_view(
        self,
        resource: _Registered,
        url: str,
        content: bytes,
        *,
        effect_owner: ResourceEffectOwner | None,
    ) -> tuple[_Registered, _ConvertedResource]:
        """Read bound bytes that hold no text through the Extract chain, keeping them.

        The bytes stay the Resource's one representation, so a view and a read of it
        agree whichever comes first (ADR 0029). The Extract text becomes their text
        view: a conversion snapshot of exactly these bytes, which settles with the
        read and is restored with them. When the chain's browser rendered the page
        instead, the rendering is the text and the bytes settle beside it. Concurrent
        reads share one walk. When it yields nothing, bytes no call has admitted are
        forgotten, so a later read may fetch them again as after a failed fetch.
        """
        resource_id = resource.resource_id
        representation: _Registered = resource
        view = self._text_views.get(resource_id)
        if view is None:
            representation, view = await _single_flight(
                self._text_view_tasks,
                self._fallback_lock,
                resource_id,
                lambda: self._adopt_extract_text_view(resource, url, content),
            )
        if _is_rendered(representation):
            self._default_rendered.add(resource_id)
            await self._persist_fetched(resource_id, content, effect_owner=effect_owner)
            return representation, view
        if not view.evidence_available:
            unadmitted = resource_id not in self._admitted_fetched
            if unadmitted and resource_id not in self._durable_fetched:
                if self._fetched.get(resource_id) is content:
                    del self._fetched[resource_id]
                self._converted.pop(resource_id, None)
            return resource, view
        await self._persist_fetched(resource_id, content, effect_owner=effect_owner)
        return resource, view

    async def _adopt_extract_text_view(
        self,
        resource: _Registered,
        url: str,
        content: bytes,
    ) -> tuple[_Registered, _ConvertedResource]:
        outcome = await self._walk_chain(resource, url)
        if outcome.rendered is not None:
            return outcome.rendered, await self._rendered_view(outcome.rendered)
        extracted = outcome.extracted
        if extracted is None:
            return resource, _unavailable_web_view(outcome.browser_failure)
        self.adopt_conversion_snapshot(
            ConversionSnapshot(
                resource_id=resource.resource_id,
                input_digest=hashlib.sha256(content).hexdigest(),
                text=extracted.text,
                visuals=(),
                extraction_status=EXTRACTION_TEXT,
                converter=extracted.acquisition,
                converter_version=extracted.provider,
                note=_extract_note(extracted),
            )
        )
        return resource, self._converted[resource.resource_id]

    async def _ensure_converted(
        self,
        resource: _Registered,
        content: bytes | None = None,
        *,
        effect_owner: ResourceEffectOwner | None = None,
    ) -> _ConvertedResource:
        if resource.resource_id in self._refused:
            raise ResourceAdmissionError("resource refused by safety/resource limits")
        cached = self._converted.get(resource.resource_id)
        if cached is not None:
            return cached
        if resource.stored_view_only:
            raise ResourceNotConvertedError(
                resource.filename or resource.resource_id, resource.declared_mime
            )
        if content is None:
            content = await self._materialize_bytes(resource, effect_owner=effect_owner)
        task = self._conversion_tasks.get(resource.resource_id)
        if task is None:
            task = asyncio.create_task(self._convert_and_adopt(resource, content))
            self._conversion_tasks[resource.resource_id] = task
        # Signal cancellation to the conversion budget without losing the worker
        # join or single-flight. A cancelled read cannot start a late fallback.
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelling():
                task.cancel()
            raise

    async def _convert_and_adopt(self, resource: _Registered, content: bytes) -> _ConvertedResource:
        try:
            snapshot = await self._snapshot_of(
                resource.resource_id,
                content,
                filename=resource.filename,
                declared_mime=resource.declared_mime,
            )
        except (
            UnsafeArchiveError,
            ConversionLimitError,
            ResourceAdmissionError,
            MemoryError,
        ) as exc:
            self.adopt_conversion_snapshot(
                _failure_snapshot(resource.resource_id, content, exc, safety_refused=True)
            )
            raise
        except ResourceConversionError as exc:
            if resource.url and _is_textual_web_resource(resource):
                raise
            self.adopt_conversion_snapshot(
                _failure_snapshot(resource.resource_id, content, exc, safety_refused=False)
            )
            return self._converted[resource.resource_id]
        if not snapshot.text and resource.url and _is_textual_web_resource(resource):
            return _ConvertedResource(
                text="", handles=(), evidence_available=False, extraction_status="no_extracted_text"
            )
        self.adopt_conversion_snapshot(snapshot)
        return self._converted[resource.resource_id]

    async def conversion_view(
        self,
        resource_id: str,
        content: bytes,
        *,
        filename: str | None,
        declared_mime: str | None,
        deadline: float | None = None,
    ) -> ConversionSnapshot:
        """Convert bytes this Run publishes into the view a read of them would adopt.

        The view is built as a read builds its own, by the same converters under the
        same limits, with image handles minted by this Run, and it names
        ``resource_id``. Nothing is registered: the Run's own Resources are unchanged.
        A conversion that fails, runs out of ``deadline``, or is refused raises, and
        the caller keeps the bytes without a view. Native work a cancelled caller
        leaves behind is joined when the registry closes, as a read's is.
        """
        self._ensure_open()
        task = asyncio.create_task(
            self._snapshot_of(
                resource_id,
                content,
                filename=filename,
                declared_mime=declared_mime,
                deadline=deadline,
            )
        )
        self._view_tasks.add(task)
        task.add_done_callback(self._view_tasks.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelling():
                task.cancel()
            raise

    async def _snapshot_of(
        self,
        resource_id: str,
        content: bytes,
        *,
        filename: str | None,
        declared_mime: str | None,
        deadline: float | None = None,
    ) -> ConversionSnapshot:
        """Convert one document into the snapshot this Run adopts as its view."""
        converted = await convert_resource(
            content, filename=filename, declared_mime=declared_mime, deadline=deadline
        )
        text = converted.text
        visuals = []
        for index, visual in enumerate(converted.visuals):
            handle = (
                "vis-"
                + hmac.new(
                    self._secret, f"{resource_id}:{index}".encode(), hashlib.sha256
                ).hexdigest()[:24]
            )
            text = text.replace(f"visual://{visual.handle_id}", f"visual://{handle}", 1)
            visuals.append(replace(visual, handle_id=handle))
        return ConversionSnapshot(
            resource_id=resource_id,
            input_digest=hashlib.sha256(content).hexdigest(),
            text=text,
            visuals=tuple(visuals),
            extraction_status=converted.extraction_status,
            converter=converted.converter,
            converter_version=converted.converter_version,
            fallback_reason=converted.fallback_reason,
            known_ocr_pages=converted.known_ocr_pages,
            known_page_count=converted.known_page_count,
            note=converted.note,
        )

    def adopt_conversion_snapshot(self, snapshot: ConversionSnapshot) -> None:
        resource = self._representation(snapshot.resource_id)
        content = resource.content or self._fetched.get(resource.resource_id)
        if content is not None and hashlib.sha256(content).hexdigest() != snapshot.input_digest:
            raise ResourceStateMismatchError("conversion snapshot input digest mismatch")
        previous = self._snapshots.get(resource.resource_id)
        if previous is not None and previous != snapshot:
            raise ResourceStateMismatchError("conversion snapshot is already adopted")
        self._snapshots[resource.resource_id] = snapshot
        if snapshot.extraction_status == "safety_refused":
            self._refused[resource.resource_id] = ResourceAdmissionError(
                "resource refused by safety/resource limits"
            )
        for visual in snapshot.visuals:
            self._visual_assets[(resource.resource_id, visual.handle_id)] = visual
        entry = _ConvertedResource(
            text=snapshot.text,
            handles=tuple(
                VisualHandle(
                    v.handle_id,
                    f"package part {v.origin_part}; location unknown"
                    if v.origin_part
                    else v.anchor,
                )
                for v in snapshot.visuals
            ),
            evidence_available=bool(snapshot.text.strip()),
            extraction_status=snapshot.extraction_status,
            note=" ".join(
                filter(
                    None,
                    (
                        snapshot.note,
                        f"Known OCR pages: {list(snapshot.known_ocr_pages)} of {snapshot.known_page_count}."
                        if snapshot.known_ocr_pages
                        else None,
                    ),
                )
            )
            or None,
        )
        self._converted[resource.resource_id] = entry
        self._text_views[resource.resource_id] = entry

    def loads_on_read(self, resource_id: str) -> bool:
        """Whether reading this Resource first loads bytes it holds lazily.

        That load spends this Run's byte allowance, which an adoption spends too, so
        a call running beside others waits for its source order before it reads.
        """
        resource = self._resources.get(self._canonical_resource_id(resource_id))
        return (
            resource is not None
            and resource.loader is not None
            and resource.resource_id not in self._fetched
        )

    def has_conversion_snapshot(self, resource_id: str) -> bool:
        """Whether this Run already reads the Resource through a conversion view."""
        return self._canonical_resource_id(resource_id) in self._snapshots

    def conversion_effects(self, resource_id: str) -> tuple[ResourceAttachmentBytes, ...]:
        snapshot = self._snapshots.get(self._canonical_resource_id(resource_id))
        return snapshot.effects() if snapshot is not None else ()

    def holds_rendered(self, representation_id: str) -> bool:
        """Whether ``representation_id`` names a rendered representation this Run holds."""
        parent = representation_id.removesuffix(RENDERED_REPRESENTATION_SUFFIX)
        rendered = self._rendered.get(parent)
        return rendered is not None and rendered.resource_id == representation_id

    def rendered_effects(self, resource_id: str) -> tuple[ResourceAttachmentBytes, ...]:
        """What settles a Web Resource's rendering: its bytes, then its conversion view.

        The rendering is its own Resource row, named for the Resource it renders and
        located by it, so recovery restores it without rendering again. A Resource with
        no rendering settles nothing.
        """
        parent = self._canonical_resource_id(resource_id)
        rendered = self._settled_rendered(parent)
        resource = self._resources.get(parent)
        if rendered is None or resource is None or rendered.content is None:
            return ()
        return (
            ResourceAttachmentBytes(
                resource_id=rendered.resource_id,
                filename=_RENDERED_FILENAME,
                mime_type=_RENDERED_MIME,
                source_locator=parent,
                content=rendered.content,
                resource_kind="web_render",
                attributes=(
                    ("acquisition", BROWSER_RENDER),
                    ("admission_origin", rendered.admission_origin),
                    ("url", resource.url or ""),
                    ("final_url", rendered.final_url or ""),
                ),
            ),
            *self.conversion_effects(rendered.resource_id),
        )

    def restore_rendered(
        self,
        parent_id: str,
        *,
        url: str,
        admission_origin: Literal["caller", "search", "agent"],
        final_url: str,
        content: bytes,
    ) -> None:
        """Hydrate one settled rendering under the Web Resource it renders.

        A Resource only ever rendered has no row of its own to restore, so a search or
        agent Resource is registered again from its URL: the handle is minted from the
        Run's identity and the URL, and must be the handle the rendering was recorded
        under. A caller Resource comes from the accepted request, so one that is missing
        is a catalog that does not describe the Run.
        """
        self._ensure_open()
        if self._resources.get(self._canonical_resource_id(parent_id)) is None:
            if admission_origin == "caller":
                raise ResourceStateMismatchError(
                    "a durable rendering names a caller link the request does not describe"
                )
            validate_agent_public_url(url)
            minted = self._register(ResourceInput(url=url), admission_origin=admission_origin)
            if minted != parent_id:
                raise ResourceStateMismatchError(
                    "a durable rendering does not match the Resource its URL registers"
                )
        parent = self._require(parent_id)
        if parent.url is None:
            raise ResourceStateMismatchError("a durable rendering names a Resource with no URL")
        validate_agent_public_url(final_url)
        held = self._rendered.get(parent.resource_id)
        if held is not None:
            if held.content != content:
                raise ResourceStateMismatchError("one Web resource cannot hold two renderings")
            return
        self._rendered[parent.resource_id] = _rendered_representation(
            parent, content, final_url=normalize_public_http_url_identity(final_url)
        )

    def canonical_resource_id(self, resource_id: str) -> str:
        """The handle a Resource goes by in results, whichever of its handles was named."""
        return self._canonical_resource_id(resource_id)

    def held_visual_asset(
        self, resource_id: str, handle_id: str
    ) -> tuple[ExtractedVisual, bool] | None:
        """An embedded image this Run already holds for the Resource, and whether it came from
        the rendering, or None; nothing is fetched or converted to find it."""
        resource = self._resources.get(self._canonical_resource_id(resource_id))
        if resource is None:
            return None
        asset = self._visual_assets.get((resource.resource_id, handle_id))
        if asset is not None:
            return asset, False
        rendered = self._settled_rendered(resource.resource_id)
        if rendered is not None:
            asset = self._visual_assets.get((rendered.resource_id, handle_id))
            if asset is not None:
                return asset, True
        return None

    def visual_cursor(self, resource_id: str, start: int, kind: str) -> str:
        payload = struct.pack(">I", start)
        return (
            kind
            + "."
            + self._encode_cursor_payload(
                payload, binding=f"{kind}:{resource_id}:".encode(), signature_bytes=16
            )
        )

    def resolve_visual_cursor(self, cursor: str, resource_id: str, kind: str) -> int:
        try:
            prefix, encoded = cursor.split(".", 1)
            if prefix != kind:
                raise ValueError
            payload = self._decode_cursor_payload(
                encoded,
                binding=f"{kind}:{resource_id}:".encode(),
                signature_bytes=16,
                payload_bytes=4,
            )
            return struct.unpack(">I", payload)[0]
        except (ValueError, UnicodeError, struct.error) as exc:
            raise ResourceCursorError("invalid visual continuation cursor") from exc

    def _discovery(
        self,
        resource: _Registered,
        representation: _Registered,
        view: _ConvertedResource,
        budget: int,
        *,
        start: int = 0,
    ) -> tuple[tuple[VisualHandle, ...], str | None]:
        """The visual handles of ``view``, named by the Resource and bound to its representation."""
        notes = [view.note] if view.note else []
        if _is_rendered(representation):
            notes.insert(0, _rendered_note(resource, representation))
        if _is_pdf(resource.filename, resource.declared_mime) and not _is_rendered(representation):
            count = self._pdf_counts.get(resource.resource_id)
            if count is not None:
                notes.append(f"Physical PDF page count: {count}.")
            notes.append(
                f"Extracted text view; physical pages are not mapped to text lines. Use view(resource_id={resource.resource_id!r}) for a bounded page overview, or locator='1' for page detail."
            )
        elif is_convertible(representation.filename, representation.declared_mime):
            notes.append(
                f"Extracted text view; coverage is unverified. Use view(resource_id={resource.resource_id!r}, locator=<handle>) for an embedded image, not a whole-page screenshot."
            )
        if start > len(view.handles):
            raise ResourceCursorError("visual inventory cursor is out of range")
        # Keep discovery within a fraction of the current text allowance.
        selected: list[VisualHandle] = []
        for handle in view.handles[start : start + 8]:
            if estimate_tokens(str(selected) + str(handle)) > max(32, budget // 4):
                break
            selected.append(handle)
        end = start + len(selected)
        if end < len(view.handles):
            if not selected:
                raise ResourceAdmissionError("visual inventory has no residual model capacity")
            cursor = self.visual_cursor(
                representation.resource_id,
                end,
                _RENDERED_VISUAL_CURSOR_KIND
                if _is_rendered(representation)
                else _VISUAL_CURSOR_KIND,
            )
            notes.append(
                f"Visuals {start + 1}-{end} of {len(view.handles)}; more: read(resource_id={resource.resource_id!r}, cursor={cursor!r})."
            )
        return tuple(selected), " ".join(notes) or None

    async def visual_target(
        self,
        resource_id: str,
        *,
        effect_owner: ResourceEffectOwner | None = None,
    ) -> VisualTarget:
        """Materialize a resource and classify its supported visual representation."""
        resource = self._require(resource_id)
        if resource.resource_id in self._refused:
            raise ResourceAdmissionError("resource previously refused by safety/resource limits")
        content = await self._materialize_bytes(resource, effect_owner=effect_owner)
        resource = self._require(resource_id)
        resource_id = resource.resource_id
        if resource.acquisition in _EXTRACT_ACQUISITIONS:
            return VisualTarget(resource_id, "opaque", content, resource.declared_mime)
        try:
            media = verify_web_image_bytes(content)
        except ValueError:
            media = None
        if media is not None:
            return VisualTarget(resource_id, "image", content, media)
        if _is_pdf(resource.filename, resource.declared_mime):
            return VisualTarget(resource_id, "pdf", content, _PDF_MIME)
        if is_convertible(resource.filename, resource.declared_mime):
            return VisualTarget(resource_id, "document", content, resource.declared_mime)
        return VisualTarget(resource_id, "opaque", content, resource.declared_mime)

    async def visual_asset(
        self,
        resource_id: str,
        handle_id: str,
        *,
        effect_owner: ResourceEffectOwner | None = None,
    ) -> ExtractedVisual:
        """Return an embedded visual asset by handle, converting on demand."""
        held = self.held_visual_asset(resource_id, handle_id)
        if held is not None:
            return held[0]
        resource = self._require(resource_id)
        if is_convertible(resource.filename, resource.declared_mime):
            await self._ensure_converted(resource, effect_owner=effect_owner)
        asset = self._visual_assets.get((resource.resource_id, handle_id))
        if asset is None:
            # The Resource is held; only the handle inside it is unknown, which is
            # a view it cannot give rather than a handle to look for elsewhere.
            raise ResourceViewError(
                f"unknown visual handle: {handle_id}; read the resource for the handles it holds"
            )
        return asset

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks: list[asyncio.Future[Any]] = [
            *self._fetch_tasks.values(),
            *self._fallback_tasks.values(),
            *self._text_view_tasks.values(),
            *self._render_tasks.values(),
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Native conversion may still be running after its caller was cancelled.
        # Join it before releasing adopted views; never launch fallback.
        conversions = [*self._conversion_tasks.values(), *self._view_tasks]
        if conversions:
            await asyncio.gather(*conversions, return_exceptions=True)
        self._conversion_tasks.clear()
        self._view_tasks.clear()
        self._snapshots.clear()
        self._pdf_counts.clear()
        self._refused.clear()
        self._converted.clear()
        self._visual_assets.clear()
        self._text_views.clear()
        self._cursor_plans.clear()
        self._fetch_tasks.clear()
        self._fallback_tasks.clear()
        self._text_view_tasks.clear()
        self._render_tasks.clear()
        self._rendered.clear()
        self._default_rendered.clear()

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

    def _representation(self, representation_id: str) -> _Registered:
        """The Resource, or the rendered representation, that ``representation_id`` names."""
        parent = representation_id.removesuffix(RENDERED_REPRESENTATION_SUFFIX)
        rendered = self._rendered.get(parent)
        if rendered is not None and rendered.resource_id == representation_id:
            return rendered
        return self._require(representation_id)

    def _settled_rendered(self, resource_id: str) -> _Registered | None:
        """The Resource's rendering once it has settled; one still being made is not yet."""
        if resource_id in self._render_tasks:
            return None
        return self._rendered.get(resource_id)

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
        return await _single_flight(
            self._fetch_tasks,
            self._fetch_lock,
            resource_id,
            lambda: self._fetch_and_charge(
                resource_id,
                producer,
                effect_owner=effect_owner,
                charge_total=charge_total,
            ),
            settled=lambda: self._fetched.get(resource_id),
        )

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
            # A redirect to this Resource may have bound its bytes while this fetch
            # ran. The first bytes win, so every reader sees one representation.
            first = self._fetched.get(resource_id)
            if first is not None:
                return first
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
    ) -> tuple[_Registered, _ConvertedResource]:
        """Share one walk of the Extract chain; cache only a successfully admitted snapshot.

        A hosted provider's text becomes the Resource's representation. The browser's
        rendering is appended to the Resource instead, so a plain read of it takes the
        rendering from then on and nothing is cached as the Resource's own text.

        This is not ``_single_flight``: a caller that waited for the lock reads the cache
        under it and persists the bytes there, and the walk's future leaves ``_fallback_tasks``
        in the same critical section that writes the cache.
        """
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
            return resource, cached
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
                return resource, cached
            task = self._fallback_tasks.get(resource_id)
            if task is None:
                task = asyncio.ensure_future(self._run_fallback(resource, url))
                self._fallback_tasks[resource_id] = task
        try:
            representation, view = await asyncio.shield(task)
        except BaseException:
            if task.done():
                async with self._fallback_lock:
                    if self._fallback_tasks.get(resource_id) is task:
                        self._fallback_tasks.pop(resource_id, None)
            raise
        async with self._fallback_lock:
            self._fallback_tasks.pop(resource_id, None)
            if not view.evidence_available:
                return representation, view
            if _is_rendered(representation):
                self._default_rendered.add(resource_id)
                return representation, view
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
        return resource, admitted

    async def _run_fallback(
        self,
        resource: _Registered,
        url: str,
    ) -> tuple[_Registered, _ConvertedResource]:
        """Bind the chain's text to a Resource whose fetch failed, as its representation."""
        outcome = await self._walk_chain(resource, url)
        if outcome.rendered is not None:
            return outcome.rendered, await self._rendered_view(outcome.rendered)
        extracted = outcome.extracted
        if extracted is None:
            return resource, _unavailable_web_view(outcome.browser_failure)
        canonical = self._bind_final_url(resource, extracted.url)
        existing_content = self._fetched.get(canonical.resource_id)
        if existing_content is not None:
            # Bytes bound first win, here or behind a redirect: the text yields to them.
            return resource, await self._text_view_from_content(canonical, existing_content)
        canonical.acquisition = extracted.acquisition
        canonical.declared_mime = "text/markdown; charset=utf-8"
        canonical.degradation = _extract_note(extracted)
        self._fetched[canonical.resource_id] = extracted.text.encode("utf-8")
        return resource, _ConvertedResource(
            text=extracted.text,
            handles=(),
            note=canonical.degradation,
        )

    async def _walk_chain(self, resource: _Registered, url: str) -> _ChainOutcome:
        """Try the Extract chain's steps in order for one URL; the first usable result wins.

        A rendering the Run already holds is the result, and no step runs. A browser that
        fails is recorded and the walk goes on: an automatic render never raises.
        """
        rendered = self._settled_rendered(resource.resource_id)
        if rendered is not None:
            return _ChainOutcome(rendered=rendered)
        failure: AgentBrowserError | None = None
        for step in self._extract_chain:
            if isinstance(step, HostedExtract):
                extracted = await self._extract(step.extract, url)
                if extracted is not None:
                    return _ChainOutcome(extracted=extracted)
                continue
            try:
                return _ChainOutcome(rendered=await self._render(resource))
            except AgentBrowserError as exc:
                failure = exc
        return _ChainOutcome(browser_failure=failure)

    async def _extract(self, extract: UrlTextFallback, url: str) -> WebExtractResult | None:
        """Text of one public URL from one hosted step, or None when it has none."""
        # Extraction providers must never receive a private or credential-bearing
        # locator, even when a caller attachment used private transport metadata.
        validate_agent_public_url(url)
        await avalidate_public_http_url(url)
        try:
            extracted = await extract(url)
        except asyncio.CancelledError:
            raise
        except Exception:
            return None
        if not extracted.text.strip():
            return None
        if len(extracted.text.encode("utf-8")) > self._max_attachment_bytes:
            return None
        return extracted

    async def _render(self, resource: _Registered) -> _Registered:
        """The Resource's rendering, made now unless the Run holds one; concurrent callers share it.

        A render that fails pins nothing, so a later call may try again.
        """
        self._ensure_open()
        resource_id = resource.resource_id
        settled = self._settled_rendered(resource_id)
        if settled is not None:
            return settled
        return await _single_flight(
            self._render_tasks,
            self._render_lock,
            resource_id,
            lambda: self._render_page(resource),
        )

    async def _render_page(self, resource: _Registered) -> _Registered:
        """Render one Web Resource's URL and convert the result, admitting it only with text."""
        url = resource.url
        if url is None:  # pragma: no cover - only Web Resources are routed here
            raise RenderedReadTargetError(resource.resource_id)
        if self._page_renderer is None:
            raise browser_failure("not_configured")
        # The browser is handed only a URL the direct read would accept; everything else
        # it loads is confined by the deployment's network, not by this check.
        validate_agent_public_url(url)
        await avalidate_public_http_url(url)
        page = await self._page_renderer(url)
        try:
            validate_agent_public_url(page.final_url)
        except ValueError:
            raise browser_failure("final_url_refused") from None
        if len(page.html) > self._max_attachment_bytes:
            raise browser_failure("too_large", limit=self._max_attachment_bytes)
        rendered = _rendered_representation(
            resource, page.html, final_url=normalize_public_http_url_identity(page.final_url)
        )
        # Registered first, because adopting its conversion names it by its own id.
        self._rendered[resource.resource_id] = rendered
        try:
            view = await self._text_view_from_content(rendered, page.html)
        except ResourceAdmissionError, UnsafeArchiveError, ConversionLimitError, MemoryError:
            # Refused for safety: the rendering stays, with its refusal, to settle with the read.
            raise
        except BaseException:
            self._drop_rendered(resource.resource_id)
            raise
        if not view.text.strip():
            self._drop_rendered(resource.resource_id)
            raise browser_failure("no_text")
        return rendered

    async def _rendered_view(self, rendered: _Registered) -> _ConvertedResource:
        """The text view of a rendering the Run holds, converting it on first use."""
        cached = self._text_views.get(rendered.resource_id)
        if cached is not None:
            return cached
        if rendered.resource_id in self._refused:
            raise ResourceAdmissionError("resource refused by safety/resource limits")
        if rendered.content is None:  # pragma: no cover - a rendering always holds its bytes
            raise ResourceNotFoundError(f"resource {rendered.resource_id} has no content")
        return await self._text_view_from_content(rendered, rendered.content)

    def _drop_rendered(self, resource_id: str) -> None:
        """Forget a rendering and everything the Run made of it, as if it never rendered."""
        rendered = self._rendered.pop(resource_id, None)
        if rendered is None:
            return
        representation_id = rendered.resource_id
        self._text_views.pop(representation_id, None)
        self._converted.pop(representation_id, None)
        self._snapshots.pop(representation_id, None)
        self._refused.pop(representation_id, None)
        self._pdf_counts.pop(representation_id, None)
        self._conversion_tasks.pop(representation_id, None)
        for key in [key for key in self._visual_assets if key[0] == representation_id]:
            del self._visual_assets[key]
        self._cursor_plans = {
            key: plan for key, plan in self._cursor_plans.items() if key[0] != representation_id
        }

    def _mint_resource_id(self, dedup_key: tuple[str, bytes]) -> str:
        kind, payload = dedup_key
        digest = hmac.new(
            self._secret, kind.encode("utf-8") + b"|" + payload, hashlib.sha256
        ).hexdigest()
        return f"{PREPARED_RESOURCE_HANDLE_PREFIX}{digest[:24]}"

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

    def _encode_cursor_payload(
        self, payload: bytes, *, binding: bytes, signature_bytes: int
    ) -> str:
        signature = hmac.new(self._cursor_secret, binding + payload, hashlib.sha256).digest()[
            :signature_bytes
        ]
        return base64.urlsafe_b64encode(payload + signature).rstrip(b"=").decode("ascii")

    def _decode_cursor_payload(
        self, encoded: str, *, binding: bytes, signature_bytes: int, payload_bytes: int
    ) -> bytes:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        if len(raw) != payload_bytes + signature_bytes:
            raise ValueError("invalid cursor length")
        payload = raw[:payload_bytes]
        expected = self._encode_cursor_payload(
            payload, binding=binding, signature_bytes=signature_bytes
        )
        if not hmac.compare_digest(encoded, expected):
            raise ValueError("invalid cursor signature or encoding")
        return payload

    def _mint_cursor(self, state: _CursorState) -> str:
        payload = struct.pack(
            ">BIIII",
            _CURSOR_VERSION,
            state.plan_window_tokens,
            state.plan_position,
            state.char_offset,
            state.anchor_offset,
        )
        return self._encode_cursor_payload(
            payload,
            binding=state.resource_id.encode("utf-8") + b"|",
            signature_bytes=_CURSOR_SIGNATURE_BYTES,
        )

    def _resolve_cursor(self, cursor: str, *, resource_id: str) -> _CursorState:
        try:
            payload = self._decode_cursor_payload(
                cursor,
                binding=resource_id.encode("utf-8") + b"|",
                signature_bytes=_CURSOR_SIGNATURE_BYTES,
                payload_bytes=struct.calcsize(">BIIII"),
            )
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


def _failure_snapshot(
    resource_id: str,
    content: bytes,
    error: Exception,
    *,
    safety_refused: bool,
) -> ConversionSnapshot:
    conversion_error = error if isinstance(error, ResourceConversionError) else None
    return ConversionSnapshot(
        resource_id=resource_id,
        input_digest=hashlib.sha256(content).hexdigest(),
        text="",
        visuals=(),
        extraction_status="safety_refused" if safety_refused else "conversion_failed",
        converter=conversion_error.converter if conversion_error else "resource-host",
        converter_version=conversion_error.converter_version
        if conversion_error
        else version("dlightrag"),
        fallback_reason=conversion_error.fallback_reason
        if conversion_error and not safety_refused
        else None,
        note=None if safety_refused else "Text conversion failed; no text evidence was extracted.",
    )


def _extract_note(extracted: WebExtractResult) -> str | None:
    notes: list[str] = []
    if extracted.dropped_results:
        notes.append(f"Dropped {extracted.dropped_results} malformed extraction result(s).")
    if extracted.degradation:
        notes.append(extracted.degradation)
    return " ".join(notes) or None


def _unavailable_web_view(browser_failure: AgentBrowserError | None = None) -> _ConvertedResource:
    note = "Web acquisition degraded: no citable evidence was admitted."
    if browser_failure is not None:
        note += f" Agent Browser: {browser_failure.public_message}"
    return _ConvertedResource(
        text=(
            "This public URL produced no citable text: direct HTTP failed or "
            "returned no usable textual representation, and the configured "
            "extraction chain returned no usable representation."
        ),
        handles=(),
        evidence_available=False,
        note=note,
        extraction_status="unavailable",
    )


def _is_rendered(representation: _Registered) -> bool:
    return representation.acquisition == BROWSER_RENDER


def _citable_agent_url(locator: str | None) -> str | None:
    """The URL a browser Resource is cited by, or None where ADR 0005 keeps it private.

    A URL that carries a credential or a signature, and one that is not an HTTP(S) URL of
    a public host (``blob:``, ``data:``, ``about:``), is cited by the Resource's handle.
    """
    if locator is None:
        return None
    try:
        validate_agent_public_url(locator)
    except ValueError:
        return None
    return normalize_public_http_url_identity(locator)


def browser_capture_filename(locator: str | None) -> str:
    """The name a capture is stored under.

    Its ``.html`` suffix routes the serialized DOM to the HTML converter whatever the
    URL's own path ends in, and the name carries the last path segment, or the host when
    the path has none.
    """
    url = _citable_agent_url(locator)
    if url is None:
        return "capture.html"
    parts = urlsplit(url)
    segment = PurePosixPath(parts.path).name
    stem = (
        PurePosixPath(safe_source_filename(segment)).stem
        if segment
        else safe_source_filename(parts.hostname)
    )
    return f"{stem[:100]}.html"


def browser_download_filename(suggested: str) -> str:
    return safe_source_filename(suggested) if suggested.strip() else "download"


def browser_download_media_type(filename: str, content: bytes) -> str:
    """A download's type: its filename's, then a PDF's signature, then an opaque file's."""
    guessed, _ = mimetypes.guess_type(filename, strict=False)
    return guessed or (
        "application/pdf" if content.startswith(b"%PDF-") else "application/octet-stream"
    )


def _cursor_prefix(rendered: bool) -> str:
    """What a cursor starts with to name the representation it continues."""
    return _RENDERED_CURSOR_PREFIX if rendered else ""


def _rendered_note(resource: _Registered, rendering: _Registered) -> str:
    """What a read of a rendering tells the model about its text: who produced it, and where
    the page ended when that is not the Resource's own URL."""
    note = f"Rendered view from the Agent Browser ({BROWSER_RENDER})"
    if rendering.final_url != resource.url:
        note += f"; the page ended at {rendering.final_url}"
    return f"{note}."


def _rendered_representation(parent: _Registered, html: bytes, *, final_url: str) -> _Registered:
    """The rendering of ``parent``: the page's serialized DOM, with the parent's admission."""
    return _Registered(
        resource_id=f"{parent.resource_id}{RENDERED_REPRESENTATION_SUFFIX}",
        filename=_RENDERED_FILENAME,
        declared_mime=_RENDERED_MIME,
        source="bytes",
        content=html,
        url=None,
        byte_size=len(html),
        admission_origin=parent.admission_origin,
        acquisition=BROWSER_RENDER,
        final_url=final_url,
    )


async def _single_flight[T](
    flights: dict[str, asyncio.Future[T]],
    lock: asyncio.Lock,
    key: str,
    start: Callable[[], Coroutine[Any, Any, T]],
    *,
    settled: Callable[[], T | None] | None = None,
) -> T:
    """Run ``start`` once for ``key`` however many callers ask while it runs.

    The first caller starts the work and later ones wait for the same future; cancelling a
    waiter does not cancel the work the others need. A future is forgotten once it is done,
    by whichever caller sees it end, and never while it still runs, so a waiter that gives
    up cannot drop a future other callers are still waiting on. ``settled`` reads what an
    earlier call left behind, under the lock, so a caller that waited for the lock does not
    start the work again.
    """
    async with lock:
        if settled is not None and (result := settled()) is not None:
            return result
        flight = flights.get(key)
        if flight is None:
            flight = asyncio.ensure_future(start())
            flights[key] = flight
    try:
        result = await asyncio.shield(flight)
    except BaseException:
        if flight.done():
            async with lock:
                if flights.get(key) is flight:
                    flights.pop(key, None)
        raise
    async with lock:
        if flights.get(key) is flight:
            flights.pop(key, None)
    return result


class ResourceRegistryClosedError(RuntimeError):
    """Raised when a closed registry is used again."""


class ResourceStateMismatchError(RuntimeError):
    """Raised when a settled catalog cannot describe the replayed request."""


def _content_key(resource: ResourceInput) -> tuple[str, bytes]:
    """The identity under which one Run admits caller bytes once."""
    return (
        "bytes",
        (resource.filename or "").encode()
        + b"\0"
        + (resource.declared_mime or "").encode()
        + b"\0"
        + hashlib.sha256(resource.content or b"").digest(),
    )


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
    for chunk in windows:
        end = offset + len(chunk)
        spans.append((offset, end))
        offset = end
    order = _focus_order(windows, focus)
    if not focus or order == list(range(len(spans))):
        return tuple(spans)
    best = order[0]
    start, _end = spans[best]
    match = windows[best].casefold().find(focus.casefold())
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
    rendered: bool = False,
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
            next_cursor=_cursor_prefix(rendered) + _CURSOR_PLACEHOLDER if has_more else None,
            visual_handles=visual_handles,
            evidence_available=evidence_available,
            note=note,
            rendered=rendered,
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
    return start + len(windows[0])


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


def _focus_order(windows: list[str], focus: str | None) -> list[int]:
    """Start at the best focus window, then cover the resource in physical order."""
    count = len(windows)
    if not focus or count <= 1:
        return list(range(count))
    query_terms = mixed_script_terms(focus)
    if not query_terms:
        return list(range(count))
    documents = [mixed_script_terms(text) for text in windows]
    ranked = bm25_rank(query_terms, documents, limit=1)
    if not ranked:
        return list(range(count))
    best = ranked[0][0]
    return [*range(best, count), *range(0, best)]


__all__ = [
    "BROWSER_ACQUISITIONS",
    "BROWSER_CAPTURE",
    "BROWSER_DOWNLOAD",
    "BROWSER_RENDER",
    "RENDERED_REPRESENTATION_SUFFIX",
    "AgentBrowserRender",
    "BrowserResourceInput",
    "ExtractStep",
    "FetchedBytesSink",
    "FetchedResourceBytes",
    "HostedExtract",
    "PageRenderer",
    "VisualTarget",
    "ResourceEffectOwner",
    "ResourceRegistry",
    "ResourceRegistryClosedError",
    "ResourceStateMismatchError",
    "UrlTextFallback",
    "browser_capture_filename",
    "browser_download_filename",
    "browser_download_media_type",
]
