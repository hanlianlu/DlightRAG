# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Validate and settle attached Agent Workspace Artifacts.

Agent paths are request-local input. This module is the only publication boundary
that may read them: structured attachments authorize roots, safe links discover
dependencies and place outputs. Settlement records document-scoped bindings to
stable resource ids without rewriting model-authored Markdown or HTML.

Validation scans the Agent Workspace and reads and decodes every candidate file,
so an event-loop caller runs ``validate_publication`` and
``prepare_artifact_attachment`` through ``run_artifact_check``, in a small thread
pool of their own. PDF checks take the process PDFium lock for that reason.
"""

from __future__ import annotations

import asyncio
import bz2
import contextvars
import functools
import gzip
import hashlib
import html
import json
import lzma
import re
import stat
import tarfile
import threading
import xml.etree.ElementTree as ET
import zipfile
from collections import deque
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote

from defusedxml import ElementTree as DefusedElementTree
from PIL import Image

from dlightrag.engine.agent.skill_protection import ProtectedTextIndex
from dlightrag.engine.agent.tools.contracts import DeclaredSource
from dlightrag.engine.answer.citations.contracts import SourceReference
from dlightrag.engine.answer.citations.finalization import finalize_answer
from dlightrag.engine.answer.citations.projection import link_public_citations
from dlightrag.engine.answer.markdown import answer_markdown, rendered_text
from dlightrag.engine.answer.reference import (
    classify_target,
    html_references,
    inline_references,
    markdown_references,
)
from dlightrag.engine.answer.resources.models import PUBLISHED_ARTIFACT_HANDLE_PREFIX
from dlightrag.engine.answer.resources.visual import ResourceViewError, pdf_page_count
from dlightrag.engine.rag.retrieval.results import (
    RetrievalContexts,
)

PresentationCapability = Literal["image", "video", "markdown", "html", "pdf", "text", "download"]
ArtifactIssueKind = Literal[
    "invalid_reference",
    "missing_file",
    "unsafe_file",
    "media_mismatch",
    "file_too_large",
    "answer_too_large",
    "too_many_artifacts",
    "image_too_large",
    "active_preview_too_large",
    "reference_cycle",
    "stale_attachment",
    "unattached_reference",
    "protected_content",
]

_HTML_LOADING_TAGS = frozenset({"script", "img", "audio", "video", "source", "iframe"})
# A url() that is not a data:, blob: or same-document (#id) reference, or an @import that is not
# data:. The whitespace is possessive so that giving it back cannot hide a data: or # behind it.
_CSS_EXTERNAL = re.compile(
    r"url\(\s*+(?![\"']?(?:data:|blob:|#))[^)]+\)|@import\s+(?!url\(\s*[\"']?data:)",
    re.IGNORECASE,
)
_SVG_RASTER_DATA_URL = re.compile(r"^data:image/(?:gif|jpeg|png|webp)(?:;[^,]*)?,", re.IGNORECASE)
_MEDIA_BY_EXTENSION: dict[str, tuple[str, PresentationCapability]] = {
    ".md": ("text/markdown", "markdown"),
    ".html": ("text/html", "html"),
    ".htm": ("text/html", "html"),
    ".xhtml": ("text/html", "html"),
    ".xht": ("text/html", "html"),
    ".pdf": ("application/pdf", "pdf"),
    # Video is presented by a native ``<video>`` element, never by a container
    # parser: the browser owns decoding, so the extension only has to name the
    # container the bytes actually are (see ``_identify_media_type``).
    ".mp4": ("video/mp4", "video"),
    ".m4v": ("video/mp4", "video"),
    ".mov": ("video/quicktime", "video"),
    ".webm": ("video/webm", "video"),
    ".png": ("image/png", "image"),
    ".jpg": ("image/jpeg", "image"),
    ".jpeg": ("image/jpeg", "image"),
    ".webp": ("image/webp", "image"),
    ".gif": ("image/gif", "image"),
    ".svg": ("image/svg+xml", "image"),
    ".csv": ("text/csv", "text"),
    ".json": ("application/json", "text"),
    ".txt": ("text/plain", "text"),
    ".js": ("text/javascript", "download"),
    ".mjs": ("text/javascript", "download"),
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "download",
    ),
    ".xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "download",
    ),
    ".pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "download",
    ),
    ".doc": ("application/msword", "download"),
    ".xls": ("application/vnd.ms-excel", "download"),
    ".ppt": ("application/vnd.ms-powerpoint", "download"),
    ".zip": ("application/zip", "download"),
}
_IMAGE_MEDIA = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_VIDEO_MEDIA = frozenset({"video/mp4", "video/quicktime", "video/webm"})
_OOXML_MEDIA = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
)
_OLE_MEDIA = frozenset(
    {
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
    }
)
_NON_PROSE_MEDIA = frozenset({"application/zip", *_IMAGE_MEDIA, *_VIDEO_MEDIA})
"""Media types whose bytes are not prose, and what the content gate reads out of each.

The form checks read structure, never words: a PDF is a page count, a docx is a zip shape.
A reader reads the words, so the gate reads them too — PDF text through the renderer the
image already ships, Office Open XML through its own XML, everything else as UTF-8 or
UTF-16. The legacy OLE formats have no reader in this process; they are stated as a limit in
ADR 0061 rather than pretended to be covered.
"""

MIN_PUBLISHED_REPRODUCTION_CHARS = 320
"""Shortest verified run of a protected document's own text that refuses publication.

Publication masks nothing: it either ships the file whole or withholds it while naming the
issue, so the run here is longer than one a reader surface replaces. A deliverable may quote a
Skill's line, repeat its terms, or carry a table row, and those are legitimate; a paragraph or
a copied body is a reproduction, and a Run publishes what it produced, not the material it
reasoned with (ADR 0046, ADR 0061).
"""

# Content identification for video containers. The instance carries a model, so
# it is built once per process and only when a video Artifact is actually
# validated.
_MEDIA_IDENTIFIER: Any = None
_MEDIA_IDENTIFIER_LOCK = threading.Lock()
_ANSWER_MARKDOWN = answer_markdown()
# A check can hold its thread through a whole Workspace scan, so checks queue
# on a pool of their own rather than take the event loop's default executor
# from short work such as DNS lookups and embedding request planning.
_CHECK_THREADS = 4
_CHECK_POOL = ThreadPoolExecutor(max_workers=_CHECK_THREADS, thread_name_prefix="artifact-check")


async def run_artifact_check[**P, T](
    check: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Run one publication or attachment check in the bounded check pool."""
    context = contextvars.copy_context()
    return await asyncio.get_running_loop().run_in_executor(
        _CHECK_POOL, functools.partial(context.run, check, *args, **kwargs)
    )


def _identify_media_type(content: bytes) -> str:
    """Return the media type the bytes themselves look like.

    Video has no standard-library container parser and this deployment ships no
    ffmpeg, so identification is delegated to magika, which already arrives with
    the binary converter dependency. That is a weaker guarantee than the
    structural checks published images, Office documents, and PDFs pass: a
    truncated container whose header is still recognizable is admitted, and a
    viewer of bytes the browser cannot decode sees the element's own error path
    rather than a publication issue. Recorded in
    ``docs/adr/0026-a-video-artifact-plays-and-a-video-link-is-a-card.md``.
    """
    global _MEDIA_IDENTIFIER
    with _MEDIA_IDENTIFIER_LOCK:
        if _MEDIA_IDENTIFIER is None:
            from dlightrag.engine.answer.resources.converters import disable_model_telemetry

            disable_model_telemetry()
            import magika

            _MEDIA_IDENTIFIER = magika.Magika()
    identified = _MEDIA_IDENTIFIER.identify_bytes(content).output.mime_type
    return str(identified)


class PublicationScanError(ValueError):
    """The Artifact root contains a linked, special, or unreadable entry."""


@dataclass(frozen=True, slots=True)
class PublicationLimits:
    max_artifacts: int = 20
    max_file_bytes: int = 30 * 1024 * 1024
    max_total_bytes: int = 100 * 1024 * 1024
    workspace_max_bytes: int = 1024 * 1024 * 1024
    preview_image_max_pixels: int = 16_000_000
    preview_image_max_edge: int = 4096
    original_image_max_pixels: int = 64_000_000
    original_image_max_edge: int = 8000
    active_html_max_bytes: int = 20 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ArtifactIssue:
    kind: ArtifactIssueKind
    description: str
    resource_id: str | None = None

    def as_dict(self) -> dict[str, str]:
        value = {"kind": self.kind, "description": self.description}
        if self.resource_id:
            value["resource_id"] = self.resource_id
        return value


@dataclass(frozen=True, slots=True)
class ArtifactAttachment:
    """One durable root attachment authorized for terminal publication."""

    relative_path: str
    label: str
    content_digest: str
    size_bytes: int
    presentation: PresentationCapability
    #: The files the root was built from, as Artifact-root-relative paths. They are published
    #: with the root and never placed in the Answer: a generated deliverable that cannot link
    #: to what produced it would otherwise reach a later turn with nothing that rebuilds it.
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StagedArtifact:
    """One validated attached root or dependency ready for publication."""

    relative_path: str
    media_type: str
    size_bytes: int
    resource_id: str = ""
    filename: str = ""
    digest: str = ""
    source_digest: str = ""
    source_size_bytes: int = 0
    presentation: PresentationCapability = "download"
    width: int | None = None
    height: int | None = None
    content: bytes = b""
    artifact_bindings: Mapping[str, str] = field(default_factory=dict)

    def descriptor(self, *, label: str = "") -> dict[str, object]:
        value: dict[str, object] = {
            "resource_id": self.resource_id,
            "media_type": self.media_type,
            "label": label or self.filename,
            "filename": self.filename,
            "byte_size": self.size_bytes,
            "digest": self.digest,
            "presentation": self.presentation,
            "status": "available",
            "artifact_bindings": dict(self.artifact_bindings),
        }
        if self.width is not None and self.height is not None:
            value["width"] = self.width
            value["height"] = self.height
        return value


@dataclass(frozen=True, slots=True)
class PublicationPlan:
    answer: str
    artifact_bindings: Mapping[str, str] = field(default_factory=dict)
    artifacts: tuple[StagedArtifact, ...] = ()
    descriptors: tuple[Mapping[str, object], ...] = ()
    issues: tuple[ArtifactIssue, ...] = ()
    artifact_sources: Mapping[str, Sequence[SourceReference]] = field(default_factory=dict)
    #: What each attached root was built from, by the root's resource id: the declared source's
    #: path and its resource id. Recorded on the published product, so reopening it returns the
    #: files that produce it again rather than only the bytes it received (ADR 0052).
    sources_by_root: Mapping[str, tuple[DeclaredSource, ...]] = field(default_factory=dict)

    @property
    def outcome(self) -> dict[str, object]:
        if not self.issues:
            status = "complete"
        elif self.artifacts:
            status = "partial"
        else:
            status = "failed"
        return {"status": status, "issues": [issue.as_dict() for issue in self.issues]}

    def correction_feedback(self) -> str:
        lines = [
            "Artifact publication validation failed. This is the one correction pass.",
            "Repair each root Artifact, then call attach_artifact after its final modification before returning the complete final answer again.",
            "Artifact links only place successfully attached roots or their validated dependencies; attach_artifact returns the canonical link.",
            "Apply the same inline Citation Contract to evidence-backed factual claims; citations are validated independently for each Markdown Artifact.",
        ]
        lines.extend(f"- {issue.kind}: {issue.description}" for issue in self.issues)
        return "\n".join(lines)


def is_substantive_text(text: str) -> bool:
    """True when visible content remains after actual Artifact affordances."""
    punctuation = " \t\r\n.,;:!?，。；：！？"
    for block in _ANSWER_MARKDOWN.parse(text):
        if block.type in {"fence", "code_block", "html_block"}:
            if block.content.strip(punctuation):
                return True
        if block.type != "inline":
            continue
        children = block.children or ()
        excluded: set[int] = set()
        for reference in inline_references(children):
            if classify_target(reference.target, image=reference.image) == "artifact":
                excluded.update(range(reference.start, reference.end))
        for index, token in enumerate(children):
            if index not in excluded and (
                token.type == "image" or token.content.strip(punctuation)
            ):
                return True
    return False


def is_empty_answer(*, answer: str, has_artifacts: bool) -> bool:
    """Fail only when neither visible text nor an available Artifact exists."""
    return not has_artifacts and not is_substantive_text(answer)


def prepare_artifact_attachment(
    artifacts_root: Path,
    *,
    path: str,
    label: str = "",
    sources: Sequence[str] = (),
    limits: PublicationLimits | None = None,
    protected_digests: Collection[str] = frozenset(),
    protected_index: Callable[[], ProtectedTextIndex] | None = None,
    protected_text_exempt_names: Collection[str] = (),
) -> ArtifactAttachment:
    """Validate one model-selected root, the files it declares, and bind the root's digest.

    What the root declares is published with it, so it is checked against the same budget the
    closure is: a kit that cannot fit is refused here, where its author is, rather than left
    behind at settlement where the product would ship without it.

    ``protected_index`` is asked for the process's content fingerprints only when a file is
    actually read: a caller that never publishes never builds the index.
    ``protected_text_exempt_names`` names built-in Skills' declared sealed artifacts. Those files
    are machine-required delivery data, not reader-facing prose: the exact-byte digest gate still
    protects them, but a reference passage the engine or verifier wrote into the case does not
    refuse the sealed artifact's own publication. Every other file, source and container member is
    still text-scanned.
    """
    limits = limits or PublicationLimits()
    try:
        relative = _normalize_reference(path, parent=None)
    except ValueError as exc:
        raise ArtifactValidationError(
            "invalid_reference", "Artifact path must be relative to the Artifact root."
        ) from exc
    try:
        files = _inventory(artifacts_root, limits=limits)
    except PublicationScanError as exc:
        raise ArtifactValidationError("unsafe_file", _safe_issue_text(str(exc))) from exc
    target = files.get(relative)
    if target is None:
        raise ArtifactValidationError(
            "missing_file", f"Artifact {Path(relative).name or 'file'} is missing."
        )
    staged = _validate_file(
        relative,
        target,
        limits=limits,
        protected_digests=protected_digests,
        protected_index=protected_index,
        protected_text_exempt_names=protected_text_exempt_names,
    )
    normalized_sources: list[str] = []
    for source in sources:
        try:
            normalized = _normalize_reference(source, parent=None)
        except ValueError as exc:
            raise ArtifactValidationError(
                "invalid_reference", "An Artifact source must be relative to the Artifact root."
            ) from exc
        if normalized == relative:
            raise ArtifactValidationError(
                "invalid_reference", f"Artifact {staged.filename} cannot be its own source."
            )
        if normalized in normalized_sources:
            continue
        if len(normalized_sources) + 1 >= limits.max_artifacts:
            raise ArtifactValidationError(
                "too_many_artifacts",
                f"At most {limits.max_artifacts - 1} sources publish with one Artifact, which "
                f"shares the {limits.max_artifacts}-file publication budget.",
            )
        source_path = files.get(normalized)
        if source_path is None:
            raise ArtifactValidationError(
                "missing_file",
                f"Artifact source {Path(normalized).name or 'file'} is missing from the "
                "Artifact root.",
            )
        staged_source = _validate_file(
            normalized,
            source_path,
            limits=limits,
            protected_digests=protected_digests,
            protected_index=protected_index,
            protected_text_exempt_names=protected_text_exempt_names,
        )
        if staged.size_bytes + staged_source.size_bytes > limits.max_total_bytes:
            raise ArtifactValidationError(
                "answer_too_large",
                f"{staged.filename} and the {len(normalized_sources) + 1} file(s) declared with "
                f"it exceed the {limits.max_total_bytes}-byte publication budget.",
            )
        normalized_sources.append(normalized)
    normalized_label = " ".join(label.split())[:200]
    return ArtifactAttachment(
        relative_path=relative,
        label=normalized_label or staged.filename,
        content_digest=staged.source_digest,
        size_bytes=staged.source_size_bytes,
        presentation=staged.presentation,
        sources=tuple(normalized_sources),
    )


def artifact_closure(
    artifacts_root: Path,
    roots: Sequence[str],
    *,
    limits: PublicationLimits | None = None,
) -> tuple[str, ...]:
    """Every Artifact-root-relative file reachable from ``roots`` through safe links and sources.

    The attach-time sealed-artifact gate has to judge what will ship, not only the root: a Markdown
    page may link the sealed case as a dependency, and a root that cannot embed its inputs may
    declare them as sources. This walks the same two edges publication walks — ``classify_target``
    references in Markdown/HTML and the already-declared sources — bounded by the same Artifact
    count, and returns the paths in discovery order. It reads text only; callers that need the bytes
    read them in their own workspace hold.
    """
    budget = limits or PublicationLimits()
    discovered: list[str] = []
    pending = deque(roots)
    while pending and len(discovered) < budget.max_artifacts:
        relative = pending.popleft()
        if relative in discovered:
            continue
        try:
            data = (artifacts_root / relative).read_bytes()
        except OSError:
            continue
        discovered.append(relative)
        suffix = PurePosixPath(relative).suffix.lower()
        if suffix not in {".md", ".html", ".htm"}:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        references = (
            html_references(text) if suffix in {".html", ".htm"} else markdown_references(text)
        )
        for reference in references:
            if classify_target(reference.target, image=reference.image) != "artifact":
                continue
            raw = reference.target.partition(":")[2]
            try:
                child = _normalize_reference(raw, parent=relative)
            except ValueError:
                continue
            if child not in discovered:
                pending.append(child)
    return tuple(discovered)


def artifact_read_call(relative_path: str) -> str:
    """Return the agent call that reaches one published Artifact again.

    Every product is read by ``read(resource_id=…)``. One whose type has no conversion
    route is decoded from the adopted bytes; one whose type routes to a converter reads
    the conversion view publication stored with it, because the adopting Run never
    converts what it adopts. A product whose view publication could not build refuses
    that read, as any convertible document without a view does.
    """
    return f"read(resource_id={artifact_resource_id(relative_path)!r})"


def validate_publication(
    artifacts_root: Path,
    *,
    answer: str,
    attachments: Sequence[ArtifactAttachment] = (),
    limits: PublicationLimits | None = None,
    contexts: RetrievalContexts | None = None,
    protected_digests: Collection[str] = frozenset(),
    protected_index: Callable[[], ProtectedTextIndex] | None = None,
    protected_text_exempt_names: Collection[str] = (),
) -> PublicationPlan:
    """Authorize roots, validate their closure, and bind actual document references.

    Each document keeps its own target-to-resource map and original source. Links
    place structured attachments or their dependencies; they cannot authorize an
    unattached root. Invalid targets bind to an unavailable descriptor in place.

    ``protected_digests`` names the Skill documents no Run may publish, so a root and
    every dependency it pulls in are refused the same way (ADR 0046). ``protected_index``
    adds the content gate beside it: a text deliverable that reproduces a protected
    document's own words is refused even though its bytes are the Run's own.
    ``protected_text_exempt_names`` is the same machine-artifact exemption
    ``prepare_artifact_attachment`` applies: the declared sealed artifact and a member of a
    container carrying that name are data, not reader prose; everything else stays scanned.
    """
    limits = limits or PublicationLimits()
    roots: list[str] = []
    attached: dict[str, ArtifactAttachment] = {}
    declared_sources: dict[str, tuple[str, ...]] = {}
    invalid_references: dict[tuple[str | None, str], tuple[str, ArtifactIssue]] = {}
    for attachment in attachments:
        try:
            relative = _normalize_reference(attachment.relative_path, parent=None)
        except ValueError:
            invalid_references.setdefault(
                (None, attachment.relative_path),
                (
                    attachment.label,
                    ArtifactIssue("invalid_reference", "An attached Artifact path is not safe."),
                ),
            )
            continue
        sources: list[str] = []
        for source in attachment.sources:
            try:
                normalized = _normalize_reference(source, parent=None)
            except ValueError:
                invalid_references.setdefault(
                    (None, source),
                    (
                        attachment.label,
                        ArtifactIssue(
                            "invalid_reference",
                            "A declared Artifact source is not a safe relative path.",
                        ),
                    ),
                )
                continue
            if normalized == relative:
                invalid_references.setdefault(
                    (None, source),
                    (
                        attachment.label,
                        ArtifactIssue(
                            "invalid_reference",
                            f"Artifact {relative} cannot be its own source.",
                        ),
                    ),
                )
                continue
            if normalized not in sources:
                sources.append(normalized)
        if sources:
            declared_sources[relative] = tuple(sources)
        if relative in attached:
            roots.remove(relative)
        roots.append(relative)
        attached[relative] = attachment

    paths_by_id = {artifact_resource_id(path): path for path in roots}
    placed: set[str] = set()
    for reference in markdown_references(answer):
        if classify_target(reference.target, image=reference.image) != "artifact":
            continue
        try:
            placed.add(_reference_path(reference.target, parent=None, paths_by_id=paths_by_id))
        except ValueError:
            continue
    omitted = [attached[path] for path in roots if path not in placed]
    if omitted:
        answer = _append_artifact_affordances(answer, omitted)

    try:
        files = _inventory(artifacts_root, limits=limits)
    except PublicationScanError as exc:
        issue = ArtifactIssue("unsafe_file", _safe_issue_text(str(exc)))
        bindings: dict[str, str] = {}
        descriptors: list[Mapping[str, object]] = []
        for reference in markdown_references(answer):
            if (
                classify_target(reference.target, image=reference.image)
                not in {"artifact", "unsupported"}
                or reference.target in bindings
            ):
                continue
            resource_id = _invalid_reference_id(reference.target, parent=None)
            bindings[reference.target] = resource_id
            descriptors.append(_unavailable_descriptor(resource_id, reference.label, issue))
        return PublicationPlan(
            answer=answer,
            artifact_bindings=bindings,
            descriptors=tuple(descriptors),
            issues=(issue,),
        )

    paths_by_id.update({artifact_resource_id(path): path for path in files})
    labels = {path: attachment.label for path, attachment in attached.items()}
    reference_paths: dict[str | None, dict[str, str]] = {}

    referenced: set[str] = set(roots)

    def collect_references(text: str, *, parent: str | None, html: bool = False) -> list[str]:
        references = html_references(text) if html else markdown_references(text)
        children: list[str] = []
        targets = reference_paths.setdefault(parent, {})
        for reference in references:
            kind = classify_target(reference.target, image=reference.image)
            if kind == "unsupported" and not html:
                invalid_references.setdefault(
                    (parent, reference.target),
                    (
                        reference.label,
                        ArtifactIssue(
                            "invalid_reference",
                            f"Link {reference.target!r} cannot be opened by the user. "
                            "Use an external HTTP(S) link, or call attach_artifact for the "
                            "completed file and use its returned artifact: URI.",
                        ),
                    ),
                )
                continue
            if kind != "artifact":
                continue
            try:
                relative = _reference_path(reference.target, parent=parent, paths_by_id=paths_by_id)
            except ValueError:
                invalid_references.setdefault(
                    (parent, reference.target),
                    (
                        reference.label,
                        ArtifactIssue(
                            "invalid_reference",
                            "An Artifact reference is not a safe relative path.",
                        ),
                    ),
                )
                continue
            targets[reference.target] = relative
            labels.setdefault(relative, reference.label)
            if relative not in children:
                children.append(relative)
        referenced.update(children)
        return children

    collect_references(answer, parent=None)
    discovery: deque[str] = deque(roots)
    candidates: dict[str, StagedArtifact] = {}
    graph: dict[str, list[str]] = {}
    issues_by_path: dict[str, ArtifactIssue] = {}
    artifact_sources: dict[str, list[SourceReference]] = {}
    while discovery:
        relative = discovery.popleft()
        if relative in candidates or relative in issues_by_path:
            continue
        if relative not in files:
            issues_by_path[relative] = ArtifactIssue(
                "missing_file", f"Referenced Artifact {Path(relative).name or 'file'} is missing."
            )
            continue
        try:
            staged = _validate_file(
                relative,
                files[relative],
                limits=limits,
                protected_digests=protected_digests,
                protected_index=protected_index,
                protected_text_exempt_names=protected_text_exempt_names,
            )
        except ArtifactValidationError as exc:
            issues_by_path[relative] = ArtifactIssue(exc.kind, exc.description)
            continue
        attachment = attached.get(relative)
        if attachment is not None and (
            attachment.content_digest != staged.source_digest
            or attachment.size_bytes != staged.source_size_bytes
        ):
            issues_by_path[relative] = ArtifactIssue(
                "stale_attachment",
                f"Attached Artifact {staged.filename} changed and must be attached again.",
            )
            continue
        if staged.media_type == "text/markdown":
            try:
                original = staged.content.decode("utf-8")
                cleaned = finalize_answer(original, contexts or {})
                prepared = link_public_citations(cleaned.answer, cleaned.sources)
                content = prepared.encode("utf-8")
                if len(content) > limits.max_file_bytes:
                    raise ArtifactValidationError(
                        "file_too_large",
                        f"Artifact {staged.filename} exceeds {limits.max_file_bytes} bytes after citation preparation.",
                    )
            except ArtifactValidationError as exc:
                issues_by_path[relative] = ArtifactIssue(exc.kind, exc.description)
                continue
            artifact_sources[staged.resource_id] = list(cleaned.sources)
            staged = replace(
                staged,
                content=content,
                size_bytes=len(content),
                digest=hashlib.sha256(content).hexdigest(),
            )
        candidates[relative] = staged
        children: list[str] = []
        if staged.media_type in {"text/markdown", "text/html"}:
            children = collect_references(
                staged.content.decode("utf-8"),
                parent=relative,
                html=staged.media_type == "text/html",
            )
        # The files this root was built from enter its closure as if it had linked them: a
        # workbook or a Word file cannot link, so the author declares them instead. They are
        # validated, admitted and published exactly like a linked dependency (ADR 0052).
        for source in declared_sources.get(relative, ()):
            if source not in children:
                children.append(source)
        if children:
            graph[relative] = children
            discovery.extend(children)

    for relative in _cycle_nodes(graph):
        issues_by_path[relative] = ArtifactIssue(
            "reference_cycle", "Artifact references must not contain a cycle."
        )

    admitted: dict[str, StagedArtifact] = {}
    pending = deque(roots)
    total = 0
    while pending:
        relative = pending.popleft()
        if relative in admitted or relative in issues_by_path:
            continue
        staged = candidates.get(relative)
        if staged is None:
            continue
        if len(admitted) >= limits.max_artifacts:
            issues_by_path[relative] = ArtifactIssue(
                "too_many_artifacts", f"At most {limits.max_artifacts} Artifacts may be published."
            )
            continue
        if total + staged.size_bytes > limits.max_total_bytes:
            issues_by_path[relative] = ArtifactIssue(
                "answer_too_large",
                f"Published Artifacts may total at most {limits.max_total_bytes} bytes.",
            )
            continue
        admitted[relative] = staged
        total += staged.size_bytes
        pending.extend(graph.get(relative, ()))

    # Every visible reference needs a disposition, including a dependency whose
    # root was rejected before the dependency could enter the admitted closure.
    visible_scopes = {None, *admitted}
    for parent, targets in reference_paths.items():
        if parent not in visible_scopes:
            continue
        for relative in targets.values():
            if relative not in admitted and relative not in issues_by_path:
                issues_by_path[relative] = ArtifactIssue(
                    "unattached_reference",
                    f"Artifact {Path(relative).name or 'file'} is not in the published attachment closure.",
                )

    invalid_references = {
        key: value for key, value in invalid_references.items() if key[0] in visible_scopes
    }
    resource_ids = {relative: item.resource_id for relative, item in admitted.items()}
    resource_ids.update({relative: _unavailable_id(relative) for relative in issues_by_path})
    bindings_by_scope: dict[str | None, dict[str, str]] = {
        parent: {
            target: resource_ids[path] for target, path in targets.items() if path in resource_ids
        }
        for parent, targets in reference_paths.items()
        if parent in visible_scopes
    }
    for parent, target in invalid_references:
        bindings_by_scope.setdefault(parent, {})[target] = _invalid_reference_id(
            target, parent=parent
        )

    settled_artifacts = tuple(
        replace(item, artifact_bindings=bindings_by_scope.get(relative, {}))
        for relative, item in admitted.items()
    )
    # A source is carried, not delivered: nothing in the report points at it, so listing it
    # would hand the reader a file the deliverable never mentions. One that the delivery does
    # reference as well is a dependency like any other and stays listed.
    carried_only = {
        source for sources in declared_sources.values() for source in sources
    } - referenced
    descriptors = [
        item.descriptor(label=labels.get(item.relative_path, ""))
        for item in settled_artifacts
        if item.relative_path not in carried_only
    ]
    issues: list[ArtifactIssue] = []
    for relative, issue in issues_by_path.items():
        resource_id = resource_ids[relative]
        issue = replace(issue, resource_id=resource_id)
        issues.append(issue)
        descriptors.append(
            _unavailable_descriptor(
                resource_id,
                labels.get(relative) or Path(relative).name,
                issue,
                filename=_safe_filename(relative),
            )
        )
    for (parent, target), (label, issue) in invalid_references.items():
        resource_id = bindings_by_scope[parent][target]
        issue = replace(issue, resource_id=resource_id)
        issues.append(issue)
        descriptors.append(_unavailable_descriptor(resource_id, label, issue))

    return PublicationPlan(
        answer=answer,
        artifact_bindings=bindings_by_scope.get(None, {}),
        artifacts=settled_artifacts,
        descriptors=tuple(descriptors),
        issues=tuple(_dedupe_issues(issues)),
        artifact_sources={
            item.resource_id: artifact_sources[item.resource_id]
            for item in settled_artifacts
            if item.resource_id in artifact_sources
        },
        sources_by_root={
            resource_ids[root]: tuple(
                DeclaredSource(path=source, resource_id=resource_ids[source])
                for source in sources
                # Only what was actually admitted: a source the closure refused keeps its own
                # issue, and recording it here would point the product at bytes nobody holds.
                if source in admitted and root in admitted
            )
            for root, sources in declared_sources.items()
            if root in admitted
        },
    )


@dataclass(frozen=True, slots=True)
class ArtifactValidationError(ValueError):
    kind: ArtifactIssueKind
    description: str


def _inventory(root: Path, *, limits: PublicationLimits) -> dict[str, Path]:
    if not root.exists():
        return {}
    if root.is_symlink() or not root.is_dir():
        raise PublicationScanError("Artifact root must be a real directory")
    workspace_bytes = 0
    files: dict[str, Path] = {}
    workspace = root.parent
    for path in sorted(workspace.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        _reject_special(path)
        size = path.stat().st_size
        workspace_bytes += size
        if path.is_relative_to(root):
            files[path.relative_to(root).as_posix()] = path
    if workspace_bytes > limits.workspace_max_bytes:
        raise PublicationScanError(
            f"Agent Workspace exceeds the {limits.workspace_max_bytes}-byte working-set limit"
        )
    return files


class _ExternalLoads(HTMLParser):
    """Find what an HTML document loads from outside its own file.

    Markup and CSS can name a resource, and CSS can sit in a style element or in any attribute:
    an SVG presentation attribute such as ``clip-path`` or ``filter`` takes a ``url()``. A script's
    source is code, so a bundled library that mentions ``toDataURL(`` or builds an ``<img src=``
    string loads nothing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loads_external = False
        self.css: list[str] = []
        self._in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values: dict[str, str] = {}
        for name, value in attrs:
            values.setdefault(name, value or "")  # a browser keeps the first copy of an attribute
        source = values.get("src")
        if (
            tag in _HTML_LOADING_TAGS
            and source is not None
            and not source.lower().startswith(("data:", "blob:", "artifact:"))
        ):
            self.loads_external = True
        href = values.get("href")
        if (
            tag == "link"
            and "stylesheet" in values.get("rel", "").lower()
            and href is not None
            and not href.lower().startswith("data:")
        ):
            self.loads_external = True
        self.css.extend(values.values())
        self._in_style = tag == "style"

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.css.append(data)


def _loads_externally(text: str) -> bool:
    scan = _ExternalLoads()
    scan.feed(text)
    scan.close()
    return scan.loads_external or any(_CSS_EXTERNAL.search(css) for css in scan.css)


def _container_kind(content: bytes) -> str | None:
    """What a container is by its own bytes, so a name that lies does not hide one.

    A Run names its own deliverables and the media map reads the name, so this is the second
    look. Recognised: a zip (which an Office file also is) and the wrappers a tar arrives in.
    """
    if content.startswith(b"PK\x03\x04"):
        return "zip"
    if len(content) > 262 and content[257:262] == b"ustar":
        return "tar"
    if content.startswith((b"\x1f\x8b", b"BZh", b"\xfd7zXZ", b"\x28\xb5\x2f\xfd")):
        return "compressed"
    return None


def _protected_member(protected_digests: Collection[str], name: str, content: bytes) -> None:
    """Refuse a container member that is one of a protected Skill's documents.

    A document inside a zip or an Office file is the same document: digesting only the outer
    bytes would leave the delivery archive — the case this rule exists for — open.
    """
    if hashlib.sha256(content).hexdigest() in protected_digests:
        raise ArtifactValidationError(
            "protected_content",
            f"Member {name} is a Skill document, which is its author's to publish and not this "
            "Run's. Deliver what the Run produced instead.",
        )


_MARKUP_TAG = re.compile(r"<[^>]{0,4096}>")
"""One markup tag, removed before a text file is read for its words.

A tag between two words of a copied paragraph would otherwise split the run at every
formatting boundary, and a reader of the rendered file would never see the tag at all.
Inline tags are removed with nothing in their place, because they take none: ``<b>T</b>he``
reads ``The``. A block boundary reads as whitespace.
"""

_HTML_BLOCK_TAG = re.compile(
    r"</?(?:p|div|br|hr|li|ul|ol|tr|td|th|table|thead|tbody|h[1-6]|section|article|header|"
    r"footer|blockquote|pre|figure|figcaption|dl|dt|dd)\b[^>]{0,4096}>",
    re.IGNORECASE,
)
_OOXML_BLOCK_TAG = re.compile(r"</(?:w:p|w:tr|w:tc|a:p|a:tr|w:tbl)>|<w:br\s*/>")

_OOXML_TEXT = re.compile(r"<[^>]{0,4096}>")
_UTF16_BOM = (b"\xff\xfe", b"\xfe\xff")
_MAX_READABLE_CHARS = 16 * 1024 * 1024
"""Longest text the content gate reads out of one file.

A document past it is refused as unverifiable rather than read part-way: the words after the
bound would be published unscanned, and a file too large to check is not one this gate can
pass.
"""


class _UnverifiableDocument(Exception):
    """A file whose words this process cannot read to the end."""


def _decode_text(content: bytes) -> str | None:
    """A text file's characters, or None when the bytes are not text this build can read."""
    if content.startswith(_UTF16_BOM):
        try:
            return content.decode("utf-16")
        except UnicodeDecodeError:
            return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        pass
    # A UTF-16 file written without a BOM. ASCII text in that encoding is half NUL bytes, but a
    # CJK document is not, so the order is decided by which one reads as text.
    if len(content) >= 2 and len(content) % 2 == 0:
        for codec in ("utf-16-le", "utf-16-be"):
            try:
                decoded = content.decode(codec)
            except UnicodeDecodeError:
                continue
            if _looks_like_text(decoded):
                return decoded
    return None


def _looks_like_text(text: str) -> bool:
    """Whether a decoded string is prose rather than the same bytes read as the wrong encoding."""
    sample = text[:4096]
    if not sample:
        return True
    printable = sum(1 for character in sample if character.isprintable() or character in "\t\n\r")
    return printable / len(sample) >= 0.95


def _office_open_xml_text(content: bytes) -> str | None:
    """The words of a Word, Excel or PowerPoint file, in the order its parts carry them.

    Every one of these is a zip of XML whose text is split into runs at formatting choices, so
    the words are read by stripping tags from each part and joining the parts: a run split into
    one element per word reads as the sentence it is. The layout parts and relationships hold
    no words.
    """
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            parts = [
                name
                for name in archive.namelist()
                if name.endswith(".xml")
                and (
                    name.startswith("word/")
                    or name.startswith("xl/")
                    or name.startswith("ppt/slides/")
                    or name == "xl/sharedStrings.xml"
                )
            ]
            chunks: list[str] = []
            for name in sorted(parts):
                try:
                    part = archive.read(name).decode("utf-8", errors="ignore")
                except OSError, KeyError, RuntimeError:
                    continue
                # Paragraph and row boundaries are words apart; a run boundary inside a word
                # is not, and the writer splits one word into several runs routinely.
                text_part = _OOXML_BLOCK_TAG.sub(" ", part)
                chunks.append(html.unescape(_OOXML_TEXT.sub("", text_part)))
                if sum(len(chunk) for chunk in chunks) > _MAX_READABLE_CHARS:
                    raise _UnverifiableDocument(name)
    except OSError, zipfile.BadZipFile, RuntimeError:
        return None
    return " ".join(chunks)


def _pdf_text(content: bytes) -> str | None:
    """The words a PDF shows, through the renderer the product already ships.

    Imported here rather than at module load: the library carries a rendering engine, and a
    deployment that publishes no PDF should not pay for it at boot.
    """
    try:
        import pypdfium2
    except ImportError:  # pragma: no cover - the dependency is part of the image
        return None
    try:
        document = pypdfium2.PdfDocument(content)
    except Exception:
        return None
    chunks: list[str] = []
    try:
        for page in document:
            try:
                text_page = page.get_textpage()
                chunks.append(text_page.get_text_range())
            finally:
                page.close()
            if sum(len(chunk) for chunk in chunks) > _MAX_READABLE_CHARS:
                raise _UnverifiableDocument("page")
    except Exception:
        return None
    finally:
        document.close()
    return "\n".join(chunks)


def _member_media_type(name: str) -> str | None:
    """The media type a container member must be read as.

    A member's name is chosen by whoever built the archive, so it cannot prove the member is
    binary: a text reproduction named ``notes.png`` is still words a reader would get. Only the
    formats that need their own reader (a PDF, an Office Open XML part) and a tagged page keep
    their type; everything else is read as text, and bytes that are not text are skipped by the
    decoder.
    """
    suffix = PurePosixPath(name).suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".xhtml", ".xht", ".html", ".htm"}:
        return "text/html"
    known = _MEDIA_BY_EXTENSION.get(suffix)
    return known[0] if known is not None and known[0] in _OOXML_MEDIA else None


def _readable_text(content: bytes, media_type: str | None) -> str | None:
    """What a reader would read out of one file, or None when this build cannot tell.

    The digest gate reads the file's bytes, so a copy of a Skill document under any name is
    already refused; this reads what the file *says*, which is the road the digest cannot see —
    the Run writing the document's method into a page, a note, or a report it produced (ADR
    0061). A media type the form checks read as structure is read here as words, because "PDF"
    and "docx" are formats a reader opens, not binary payloads.
    """
    try:
        if media_type in _OOXML_MEDIA:
            return _office_open_xml_text(content)
        if media_type == "application/pdf":
            return _pdf_text(content)
    except _UnverifiableDocument as exc:
        raise ArtifactValidationError(
            "protected_content",
            f"A document could not be read to its end ({exc}); deliver a smaller file.",
        ) from exc
    if media_type is not None and (
        media_type.startswith(("image/", "video/")) or media_type in _NON_PROSE_MEDIA
    ):
        return None
    if media_type in _OLE_MEDIA:
        # No reader for the legacy OLE formats in this process; ADR 0061 states the limit.
        return None
    text = _decode_text(content)
    if text is None:
        return None
    if media_type is not None and media_type.startswith(("text/html", "application/xhtml")):
        text = html.unescape(_MARKUP_TAG.sub("", _HTML_BLOCK_TAG.sub(" ", text)))
    elif media_type is None or media_type.startswith("text/"):
        # A page's own text is what a reader reads; Markdown links split a reproduction in the
        # source and hold it whole on the page, and the page is what the reader is handed. A
        # plain text file is rendered too: the renderer leaves it as it is, and a `.md` whose
        # detected type is plain text still gets its links folded into the words they show.
        text = rendered_text(text)
    return text


def _protected_text_in(
    protected_text: ProtectedTextIndex | None,
    subject: str,
    content: bytes,
    media_type: str | None = None,
) -> None:
    """Refuse a file whose words reproduce a protected Skill document's own text."""
    if protected_text is None or not protected_text:
        return
    text = _readable_text(content, media_type)
    if text is None and media_type is not None:
        # The declared reader declined: a member named for a format it is not. Words are
        # words under any name, so the bytes are tried as text before they are let through.
        text = _decode_text(content)
    if text is None:
        return
    if protected_text.contains(text, minimum=MIN_PUBLISHED_REPRODUCTION_CHARS):
        raise ArtifactValidationError(
            "protected_content",
            f"{subject} reproduces a Skill document's own text, which is its author's to "
            "publish and not this Run's. Deliver what the Run produced instead.",
        )


def _inspect_container(
    content: bytes,
    protected_digests: Collection[str],
    protected_text: ProtectedTextIndex | None = None,
    *,
    depth: int = 2,
    protected_text_exempt_names: Collection[str] = (),
) -> None:
    """Look inside a container as far as it can be opened, and refuse a protected member.

    An unrecognised or unopenable container is left alone: this is a rule about a Skill's own
    documents, not a ban on delivering archives, and
    [ADR 0046](../../../../docs/adr/0046-a-skills-documents-are-its-authors-to-publish.md) states
    that limit rather than pretending the check is exhaustive.
    """
    if (not protected_digests and not protected_text) or depth <= 0:
        return
    exempt_names = {name.casefold() for name in protected_text_exempt_names}
    kind = _container_kind(content)
    if kind is None:
        return
    if kind == "zip":
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                for entry in archive.infolist():
                    if entry.is_dir():
                        continue
                    member = archive.read(entry)
                    _protected_member(protected_digests, entry.filename, member)
                    if PurePosixPath(entry.filename).name.casefold() not in exempt_names:
                        _protected_text_in(
                            protected_text,
                            f"Member {entry.filename}",
                            member,
                            _member_media_type(entry.filename),
                        )
                    _inspect_container(
                        member,
                        protected_digests,
                        protected_text,
                        depth=depth - 1,
                        protected_text_exempt_names=exempt_names,
                    )
        except ArtifactValidationError:
            raise
        except OSError, RuntimeError, ValueError, zipfile.BadZipFile:
            return
        return
    opened: bytes | None = None
    try:
        with tarfile.open(fileobj=BytesIO(content), mode="r:*") as archive:
            for entry in archive:
                if not entry.isfile():
                    continue
                member = archive.extractfile(entry)
                if member is None:
                    continue
                payload = member.read()
                _protected_member(protected_digests, entry.name, payload)
                if PurePosixPath(entry.name).name.casefold() not in exempt_names:
                    _protected_text_in(
                        protected_text,
                        f"Member {entry.name}",
                        payload,
                        _member_media_type(entry.name),
                    )
                _inspect_container(
                    payload,
                    protected_digests,
                    protected_text,
                    depth=depth - 1,
                    protected_text_exempt_names=exempt_names,
                )
            return
    except ArtifactValidationError:
        raise
    except OSError, RuntimeError, ValueError, tarfile.TarError:
        pass
    # A stream that is not a tar: one file in a wrapper, which is still that file.
    for module, magic in ((gzip, b"\x1f\x8b"), (bz2, b"BZh"), (lzma, b"\xfd7zXZ")):
        if not content.startswith(magic):
            continue
        try:
            opened = module.decompress(content)
        except ArtifactValidationError:
            raise
        except OSError, EOFError, ValueError:
            return
        break
    if opened is None:
        return
    _protected_member(protected_digests, "the wrapped file", opened)
    _protected_text_in(protected_text, "Member the wrapped file", opened)
    _inspect_container(
        opened,
        protected_digests,
        protected_text,
        depth=depth - 1,
        protected_text_exempt_names=exempt_names,
    )


def _validate_file(
    relative: str,
    path: Path,
    *,
    limits: PublicationLimits,
    protected_digests: Collection[str] = frozenset(),
    protected_index: Callable[[], ProtectedTextIndex] | None = None,
    protected_text_exempt_names: Collection[str] = (),
) -> StagedArtifact:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ArtifactValidationError(
            "unsafe_file", "An Artifact could not be read safely."
        ) from exc
    if size > limits.max_file_bytes:
        raise ArtifactValidationError(
            "file_too_large",
            f"Artifact {Path(relative).name} exceeds {limits.max_file_bytes} bytes.",
        )
    suffix = PurePosixPath(relative).suffix.casefold()
    declared = _MEDIA_BY_EXTENSION.get(suffix)
    if declared is None:
        media_type, capability = "application/octet-stream", "download"
    else:
        media_type, capability = declared
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ArtifactValidationError(
            "unsafe_file", "An Artifact could not be read safely."
        ) from exc
    source_digest = hashlib.sha256(content).hexdigest()
    source_size_bytes = len(content)
    if source_digest in protected_digests:
        # A Skill's own document reached the Artifacts directory, by a copy of its bytes
        # under any name. A Run publishes what it produced, not its author's material
        # (ADR 0046); the file stays in the workspace, it is just not handed over.
        raise ArtifactValidationError(
            "protected_content",
            f"Artifact {Path(relative).name} is a Skill document, which is its author's to "
            "publish and not this Run's. Deliver what the Run produced instead.",
        )
    # The digest gate above reads bytes; this one reads what a text file says. Asked for only
    # once per file actually read, so a process that publishes nothing never builds the index.
    exempt_names = {name.casefold() for name in protected_text_exempt_names}
    protected_text = protected_index() if protected_index is not None else None
    if Path(relative).name.casefold() not in exempt_names:
        _protected_text_in(protected_text, f"Artifact {Path(relative).name}", content, media_type)
    width: int | None = None
    height: int | None = None
    try:
        if media_type in _IMAGE_MEDIA:
            with Image.open(BytesIO(content)) as image:
                image.verify()
            with Image.open(BytesIO(content)) as image:
                actual = Image.MIME.get(image.format or "", "")
                width, height = image.size
            if actual != media_type:
                raise ValueError("image type does not match extension")
            if (
                width * height > limits.original_image_max_pixels
                or max(width, height) > limits.original_image_max_edge
            ):
                raise ArtifactValidationError(
                    "image_too_large",
                    "Artifact image exceeds the 64-megapixel or 8000-pixel original limit.",
                )
            if (
                width * height > limits.preview_image_max_pixels
                or max(width, height) > limits.preview_image_max_edge
            ):
                capability = "download"
        elif media_type in _VIDEO_MEDIA:
            if _identify_media_type(content) != media_type:
                raise ValueError("video container does not match its extension")
        elif media_type == "image/svg+xml":
            content = _sanitize_svg(content)
        elif media_type == "application/pdf":
            if pdf_page_count(content) == 0:
                raise ValueError("PDF has no pages")
        elif media_type.startswith("application/vnd.openxmlformats-officedocument"):
            expected_root = {
                ".docx": "word/",
                ".xlsx": "xl/",
                ".pptx": "ppt/",
            }[suffix]
            with zipfile.ZipFile(BytesIO(content)) as archive:
                names = archive.namelist()
                if "[Content_Types].xml" not in names or not any(
                    name.startswith(expected_root) for name in names
                ):
                    raise ValueError("not the declared Office document")
                if archive.testzip() is not None:
                    raise ValueError("corrupt Office document")
        elif media_type in {
            "application/msword",
            "application/vnd.ms-excel",
            "application/vnd.ms-powerpoint",
        }:
            if not content.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
                raise ValueError("not an OLE document")
        elif media_type == "application/zip":
            with zipfile.ZipFile(BytesIO(content)) as archive:
                if archive.testzip() is not None:
                    raise ValueError("corrupt ZIP")

        elif media_type == "application/json":
            json.loads(content.decode("utf-8"))
        elif media_type in {"text/markdown", "text/plain", "text/csv", "text/javascript"}:
            content.decode("utf-8")
        elif media_type == "text/html":
            text = content.decode("utf-8")
            if not re.search(r"<!doctype\s+html|<html\b|<body\b|<head\b", text, re.IGNORECASE):
                raise ValueError("not HTML")
            if size > limits.active_html_max_bytes:
                raise ArtifactValidationError(
                    "active_preview_too_large",
                    f"Active HTML preview is limited to {limits.active_html_max_bytes} bytes.",
                )
            if _loads_externally(text):
                raise ArtifactValidationError(
                    "media_mismatch", "Active HTML must be a self-contained single file."
                )
        # Last, so a container that passed its own checks is still opened: a name says what a
        # file claims to be, and this rule reads what it holds.
        _inspect_container(
            content,
            protected_digests,
            protected_text,
            protected_text_exempt_names=exempt_names,
        )
    except ArtifactValidationError:
        raise
    except (
        OSError,
        RuntimeError,
        UnicodeDecodeError,
        ValueError,
        ET.ParseError,
        Image.DecompressionBombError,
        json.JSONDecodeError,
        ResourceViewError,
    ) as exc:
        raise ArtifactValidationError(
            "media_mismatch", f"Artifact {Path(relative).name} does not match its file extension."
        ) from exc
    resource_id = artifact_resource_id(relative)
    return StagedArtifact(
        relative_path=relative,
        media_type=media_type,
        size_bytes=len(content),
        resource_id=resource_id,
        filename=_safe_filename(relative),
        digest=hashlib.sha256(content).hexdigest(),
        source_digest=source_digest,
        source_size_bytes=source_size_bytes,
        presentation=capability,
        width=width,
        height=height,
        content=content,
    )


def _sanitize_svg(content: bytes) -> bytes:
    """Remove active and external SVG capabilities before static presentation."""
    root = DefusedElementTree.fromstring(content.decode("utf-8"))
    if root.tag.rsplit("}", 1)[-1].casefold() != "svg":
        raise ValueError("not SVG")
    # Animation can set an attribute the loop below removes, such as href, so none stays.
    forbidden = {
        "script",
        "foreignobject",
        "style",
        "animate",
        "animatemotion",
        "animatetransform",
        "set",
    }
    for parent in root.iter():
        for child in list(parent):
            if child.tag.rsplit("}", 1)[-1].casefold() in forbidden:
                parent.remove(child)
        for attribute, value in list(parent.attrib.items()):
            name = attribute.rsplit("}", 1)[-1].casefold()
            normalized = value.strip().casefold()
            if (
                name.startswith("on")
                or name in {"href", "src"}
                and not (normalized.startswith("#") or _SVG_RASTER_DATA_URL.match(normalized))
                or name == "style"
                and ("url(" in normalized or "@import" in normalized)
            ):
                del parent.attrib[attribute]
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def artifact_link(attachment: ArtifactAttachment) -> str:
    """Return the canonical model-facing placement syntax for one attachment."""
    label = _escape_artifact_label(attachment.label)
    uri = artifact_resource_id(attachment.relative_path)
    prefix = "!" if attachment.presentation == "image" else ""
    # A video is deliberately not prefixed here. Inline playback is the Answer's
    # own placement; the framework's trailing affordance for a root the Answer
    # forgot stays a card, so a player never appears in the reading column
    # without the Model asking for one (ADR 0026).
    return f"{prefix}[{label}](artifact:{uri})"


def _escape_artifact_label(label: str) -> str:
    return label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _append_artifact_affordances(answer: str, attachments: Sequence[ArtifactAttachment]) -> str:
    affordances = "\n\n".join(artifact_link(attachment) for attachment in attachments)
    separator = "\n\n" if answer else ""
    appended = f"{answer}{separator}{affordances}"
    targets = {reference.target for reference in markdown_references(appended)}
    expected = {f"artifact:{artifact_resource_id(item.relative_path)}" for item in attachments}
    if expected <= targets:
        return appended
    # An unfinished fence or another enclosing construct can swallow a trailing
    # link. Keep the original source intact and place automatic roots before it.
    return f"{affordances}{separator}{answer}"


def _reference_path(target: str, *, parent: str | None, paths_by_id: Mapping[str, str]) -> str:
    raw = target.partition(":")[2]
    if raw in paths_by_id:
        return paths_by_id[raw]
    return _normalize_reference(raw, parent=parent)


def _normalize_reference(raw: str, *, parent: str | None) -> str:
    value = unquote(raw).strip()
    if not value or "\\" in value or "\x00" in value or "?" in value or "#" in value:
        raise ValueError("invalid Artifact URI")
    value = value.removeprefix("./")
    base = PurePosixPath(parent).parent if parent else PurePosixPath()
    path = base / PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Artifact URI must stay beneath the Artifact root")
    return path.as_posix()


def _cycle_nodes(graph: Mapping[str, Sequence[str]]) -> set[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cyclic: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        if node in visiting:
            index = stack.index(node) if node in stack else 0
            cyclic.update(stack[index:])
            return
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for child in graph.get(node, set()):
            if child in graph:
                visit(child, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node, [])
    return cyclic


def artifact_resource_id(relative: str) -> str:
    """Return the public address of one Artifact root: its path, hashed.

    Deterministic on purpose. The address is what an Answer's own references carry,
    what the browser read surface serves, and what a later Run of the same Session
    names to read the published version again, so it must be computable before the
    publication row exists — which is exactly what lets the writing Tool hand the
    model a handle it can still use in its next turn.
    """
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
    return f"{PUBLISHED_ARTIFACT_HANDLE_PREFIX}{digest[:20]}"


def _unavailable_id(value: str) -> str:
    return f"unavailable-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def _invalid_reference_id(raw: str, *, parent: str | None) -> str:
    identity = raw if parent is None else f"{parent}\0{raw}"
    return _unavailable_id(identity)


def _safe_filename(relative: str) -> str:
    name = PurePosixPath(relative).name.strip().replace("\x00", "")
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return name[:180] or "artifact"


def _safe_issue_text(value: str) -> str:
    # Scan exceptions can include local paths. Keep only the stable policy fact.
    if "working-set" in value:
        return value
    return "The Artifact root contains an unsafe or unreadable entry."


def _unavailable_descriptor(
    resource_id: str, label: str, issue: ArtifactIssue, *, filename: str = "artifact"
) -> Mapping[str, object]:
    return {
        "resource_id": resource_id,
        "media_type": "application/octet-stream",
        "label": label or "Unavailable Artifact",
        "filename": filename,
        "byte_size": 0,
        "digest": "",
        "presentation": "download",
        "status": "unavailable",
        "issue": replace(issue, resource_id=resource_id).as_dict(),
        "artifact_bindings": {},
    }


def _dedupe_issues(issues: Sequence[ArtifactIssue]) -> list[ArtifactIssue]:
    seen: set[tuple[str, str, str | None]] = set()
    result: list[ArtifactIssue] = []
    for issue in issues:
        key = (issue.kind, issue.description, issue.resource_id)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def _reject_special(path: Path) -> None:
    if path.is_symlink():
        raise PublicationScanError("Artifact root contains a symlink")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise PublicationScanError("Artifact root contains an unreadable entry") from exc
    if not stat.S_ISREG(mode):
        raise PublicationScanError("Artifact root contains a special file")


__all__ = [
    "artifact_read_call",
    "artifact_resource_id",
    "ArtifactAttachment",
    "ArtifactIssue",
    "ArtifactIssueKind",
    "PublicationLimits",
    "PublicationPlan",
    "StagedArtifact",
    "ArtifactValidationError",
    "artifact_link",
    "is_empty_answer",
    "prepare_artifact_attachment",
    "run_artifact_check",
    "validate_publication",
]
