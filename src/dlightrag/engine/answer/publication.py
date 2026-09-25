# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Validate and settle attached Agent Workspace Artifacts.

Agent paths are request-local input. This module is the only publication boundary
that may read them: structured attachments authorize roots, safe links discover
dependencies and place outputs. Settlement records document-scoped bindings to
stable resource ids without rewriting model-authored Markdown or HTML.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import xml.etree.ElementTree as ET
import zipfile
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote

import pypdfium2 as pdfium
from defusedxml import ElementTree as DefusedElementTree
from PIL import Image

from dlightrag.engine.answer.citations.contracts import SourceReference
from dlightrag.engine.answer.citations.finalization import finalize_answer
from dlightrag.engine.answer.citations.projection import link_public_citations
from dlightrag.engine.answer.markdown import answer_markdown
from dlightrag.engine.answer.reference import (
    classify_target,
    html_references,
    inline_references,
    markdown_references,
)
from dlightrag.engine.answer.resources.converters import is_convertible
from dlightrag.engine.answer.resources.models import PUBLISHED_ARTIFACT_HANDLE_PREFIX
from dlightrag.engine.rag.retrieval import RetrievalContexts

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
]

_HTML_EXTERNAL_RESOURCE = re.compile(
    r"<(?:script|img|audio|video|source|iframe)\b[^>]*\bsrc\s*=\s*[\"'](?!data:|blob:|artifact:)",
    re.IGNORECASE,
)
_HTML_STYLESHEET = re.compile(
    r"<link\b(?=[^>]*\brel\s*=\s*[\"'][^\"']*stylesheet)(?=[^>]*\bhref\s*=\s*[\"'](?!data:))",
    re.IGNORECASE,
)
_CSS_EXTERNAL_URL = re.compile(r"url\(\s*[\"']?(?!data:|blob:)[^)]+\)", re.IGNORECASE)
_CSS_EXTERNAL_IMPORT = re.compile(r"@import\s+(?!url\(\s*[\"']?data:)", re.IGNORECASE)
_SVG_RASTER_DATA_URL = re.compile(r"^data:image/(?:gif|jpeg|png|webp)(?:;[^,]*)?,", re.IGNORECASE)
_MEDIA_BY_EXTENSION: dict[str, tuple[str, PresentationCapability]] = {
    ".md": ("text/markdown", "markdown"),
    ".html": ("text/html", "html"),
    ".htm": ("text/html", "html"),
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

# Content identification for video containers. The instance carries a model, so
# it is built once per process and only when a video Artifact is actually
# validated.
_MEDIA_IDENTIFIER: Any = None
_ANSWER_MARKDOWN = answer_markdown()


def _identify_media_type(content: bytes) -> str:
    """Return the media type the bytes themselves look like.

    Video has no standard-library container parser and this deployment ships no
    ffmpeg, so identification is delegated to magika, which already arrives with
    the binary converter dependency. That is a weaker guarantee than the
    structural checks published images, Office documents, and PDFs pass: a
    truncated container whose header is still recognizable is admitted, and a
    viewer of bytes the browser cannot decode sees the element's own error path
    rather than a publication issue. Recorded in
    ``docs/adr/0026-video-artifacts-and-link-cards.md``.
    """
    global _MEDIA_IDENTIFIER
    if _MEDIA_IDENTIFIER is None:
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

    @property
    def repairable(self) -> bool:
        return bool(self.issues)

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
    limits: PublicationLimits | None = None,
) -> ArtifactAttachment:
    """Validate one model-selected root and bind its current content digest."""
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
    staged = _validate_file(relative, target, limits=limits)
    normalized_label = " ".join(label.split())[:200]
    return ArtifactAttachment(
        relative_path=relative,
        label=normalized_label or staged.filename,
        content_digest=staged.source_digest,
        size_bytes=staged.source_size_bytes,
        presentation=staged.presentation,
    )


def artifact_read_call(
    relative_path: str,
    *,
    filename: str | None = None,
    mime_type: str | None = None,
) -> str:
    """Return the agent call that reaches one published Artifact again.

    A product whose type has no conversion route is read by decoding the adopted bytes,
    so the call is ``read(resource_id=…)``. One whose type routes to a converter is not
    converted retrospectively — the earlier Run never recorded that view, and recording
    one here would invent a history it never had — so the call a later turn can actually
    make is ``view(resource_id=…)``. Naming ``read`` for such a product would teach a
    call that refuses, which is why this choice lives beside the address it renders.
    """
    handle = artifact_resource_id(relative_path)
    if is_convertible(filename or relative_path, mime_type):
        return f"view(resource_id={handle!r})"
    return f"read(resource_id={handle!r})"


def validate_publication(
    artifacts_root: Path,
    *,
    answer: str,
    attachments: Sequence[ArtifactAttachment] = (),
    limits: PublicationLimits | None = None,
    contexts: RetrievalContexts | None = None,
) -> PublicationPlan:
    """Authorize roots, validate their closure, and bind actual document references.

    Each document keeps its own target-to-resource map and original source. Links
    place structured attachments or their dependencies; they cannot authorize an
    unattached root. Invalid targets bind to an unavailable descriptor in place.
    """
    limits = limits or PublicationLimits()
    roots: list[str] = []
    attached: dict[str, ArtifactAttachment] = {}
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
            staged = _validate_file(relative, files[relative], limits=limits)
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
        if staged.media_type in {"text/markdown", "text/html"}:
            children = collect_references(
                staged.content.decode("utf-8"),
                parent=relative,
                html=staged.media_type == "text/html",
            )
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
    descriptors = [
        item.descriptor(label=labels.get(item.relative_path, "")) for item in settled_artifacts
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


def _validate_file(relative: str, path: Path, *, limits: PublicationLimits) -> StagedArtifact:
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
            with pdfium.PdfDocument(content) as document:
                if len(document) == 0:
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
            if (
                _HTML_EXTERNAL_RESOURCE.search(text)
                or _HTML_STYLESHEET.search(text)
                or _CSS_EXTERNAL_URL.search(text)
                or _CSS_EXTERNAL_IMPORT.search(text)
            ):
                raise ArtifactValidationError(
                    "media_mismatch", "Active HTML must be a self-contained single file."
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
        pdfium.PdfiumError,
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
    forbidden = {"script", "foreignobject", "style"}
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
    "validate_publication",
]
