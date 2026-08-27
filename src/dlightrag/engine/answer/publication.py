# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Validate and settle explicitly referenced Agent artifacts.

Agent paths are request-local input.  This module is the only boundary that may
read them: successful settlement replaces every relative ``artifact:`` target
with a stable resource id and exposes only sanitized filenames and safe issues.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
import stat
import xml.etree.ElementTree as ET
import zipfile
from collections import deque
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote

import pypdfium2 as pdfium
from defusedxml import ElementTree as DefusedElementTree
from PIL import Image

ArtifactRole = Literal["primary_report", "attachment"]
ArtifactStatus = Literal["available", "unavailable"]
PresentationCapability = Literal["image", "markdown", "html", "pdf", "text", "download"]
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
    "multiple_primary_reports",
    "reference_cycle",
]

PRIMARY_REPORT_NAMES = frozenset({"report.md", "report.html", "report.pdf"})
_ARTIFACT_TARGET = re.compile(
    r"(?P<prefix>!?(?:\[[^\]]*\])\(\s*<?artifact:)(?P<path>[^\s)>]+)(?P<suffix>>?(?:\s+[^)]*)?\))",
    re.IGNORECASE,
)
_HTML_ARTIFACT_TARGET = re.compile(
    r"(?P<prefix>\b(?:href|src)\s*=\s*[\"']artifact:)(?P<path>[^\"']+)(?P<suffix>[\"'])",
    re.IGNORECASE,
)
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
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_SUBSTANTIVE_ELEMENT = re.compile(
    r"<(?:audio|embed|iframe|img|object|source|video)\b[^>]*(?:src|data)\s*="
    r"|<(?:input|button)\b"
    r"|<(?:circle|ellipse|line|path|polygon|polyline|rect|text)\b"
    r"|<(?:link|script)\b[^>]*(?:href|src)\s*=",
    re.IGNORECASE,
)
_HTML_STYLE_BLOCK = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.DOTALL | re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]*>")
_PDF_REPORT_SCAN_PAGES = 256
_PDF_REPORT_TEXT_CHARS_PER_PAGE = 4096

_MEDIA_BY_EXTENSION: dict[str, tuple[str, PresentationCapability]] = {
    ".md": ("text/markdown", "markdown"),
    ".html": ("text/html", "html"),
    ".htm": ("text/html", "html"),
    ".pdf": ("application/pdf", "pdf"),
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
class StagedArtifact:
    """One validated, referenced file ready for durable publication."""

    relative_path: str
    role: ArtifactRole
    media_type: str
    size_bytes: int
    path: Path
    resource_id: str = ""
    filename: str = ""
    digest: str = ""
    presentation: PresentationCapability = "download"
    width: int | None = None
    height: int | None = None
    content: bytes = b""

    @property
    def kind(self) -> Literal["primary_report", "published_artifact"]:
        """Internal run-store reference kind, not a client compatibility field."""
        return "primary_report" if self.role == "primary_report" else "published_artifact"

    def descriptor(self, *, label: str = "") -> dict[str, object]:
        value: dict[str, object] = {
            "resource_id": self.resource_id,
            "role": self.role,
            "media_type": self.media_type,
            "label": label or self.filename,
            "filename": self.filename,
            "byte_size": self.size_bytes,
            "digest": self.digest,
            "presentation": self.presentation,
            "status": "available",
        }
        if self.width is not None and self.height is not None:
            value["width"] = self.width
            value["height"] = self.height
        return value


@dataclass(frozen=True, slots=True)
class PublicationPlan:
    answer: str
    artifacts: tuple[StagedArtifact, ...] = ()
    descriptors: tuple[Mapping[str, object], ...] = ()
    issues: tuple[ArtifactIssue, ...] = ()

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
            "Artifact publication validation failed. This is the one correction pass. ",
            "Use the existing file tools to repair the Artifact root, then return the complete final answer again. ",
            "Reference every user-facing file with Markdown artifact:relative/path links.",
        ]
        lines.extend(f"- {issue.kind}: {issue.description}" for issue in self.issues)
        return "\n".join(lines)


def is_substantive_text(text: str) -> bool:
    """True when text has content beyond an Artifact-only affordance."""
    without_artifacts = _ARTIFACT_TARGET.sub("", text)
    return bool(without_artifacts.strip(" \t\r\n.,;:!?，。；：！？"))


def is_empty_answer(*, answer: str, has_primary_report: bool) -> bool:
    """Fail only when neither visible text nor an available Primary Report exists."""
    return not has_primary_report and not is_substantive_text(answer)


def scan_artifact_directory(
    artifacts_root: Path, *, limits: PublicationLimits | None = None
) -> tuple[StagedArtifact, ...]:
    """Validate regular files under the Artifact root without publishing them."""
    limits = limits or PublicationLimits()
    files = _inventory(artifacts_root, limits=limits)
    staged: list[StagedArtifact] = []
    for relative, path in files.items():
        if relative in PRIMARY_REPORT_NAMES and not _report_has_body(path, limits=limits):
            continue
        staged.append(_validate_file(relative, path, limits=limits))
    return tuple(staged)


def validate_publication(
    artifacts_root: Path,
    *,
    answer: str,
    limits: PublicationLimits | None = None,
) -> PublicationPlan:
    """Resolve explicit Artifact references into one bounded publication plan.

    Invalid references become stable unavailable descriptors so every failed
    placement remains visible in the Answer. Unreferenced files are ignored.
    """
    limits = limits or PublicationLimits()
    try:
        files = _inventory(artifacts_root, limits=limits)
    except PublicationScanError as exc:
        issue = ArtifactIssue("unsafe_file", _safe_issue_text(str(exc)))
        settled, descriptor = _unavailable_answer(answer, issue)
        return PublicationPlan(answer=settled, descriptors=descriptor, issues=(issue,))

    report_names = sorted(name for name in files if name in PRIMARY_REPORT_NAMES)
    files = {
        relative: path
        for relative, path in files.items()
        if relative not in PRIMARY_REPORT_NAMES or _report_has_body(path, limits=limits)
    }
    global_issues: list[ArtifactIssue] = []
    if len(report_names) > 1:
        global_issues.append(
            ArtifactIssue(
                "multiple_primary_reports",
                "The Artifact root may contain only one of report.md, report.html, or report.pdf.",
            )
        )

    queue: deque[tuple[str | None, str, bool]] = deque()
    labels: dict[str, str] = {}
    invalid_references: list[tuple[str, ArtifactIssue]] = []
    for path, label, _image in _references(answer, html=False):
        try:
            normalized = _normalize_reference(path, parent=None)
        except ValueError:
            invalid_references.append(
                (
                    path,
                    ArtifactIssue(
                        "invalid_reference", "An Artifact reference is not a safe relative path."
                    ),
                )
            )
            continue
        labels.setdefault(normalized, label)
        queue.append((None, normalized, False))

    selected: dict[str, StagedArtifact] = {}
    graph: dict[str, set[str]] = {}
    issues_by_path: dict[str, ArtifactIssue] = {}
    total = 0
    while queue:
        parent, relative, _is_image = queue.popleft()
        if parent == relative:
            issues_by_path.setdefault(
                relative,
                ArtifactIssue("reference_cycle", "An Artifact cannot reference itself."),
            )
            continue
        if relative in selected or relative in issues_by_path:
            continue
        if relative not in files:
            issues_by_path[relative] = ArtifactIssue(
                "missing_file", f"Referenced Artifact {Path(relative).name or 'file'} is missing."
            )
            continue
        if len(report_names) > 1 and relative in PRIMARY_REPORT_NAMES:
            issues_by_path[relative] = global_issues[0]
            continue
        try:
            staged = _validate_file(relative, files[relative], limits=limits)
        except _FileValidationError as exc:
            issues_by_path[relative] = ArtifactIssue(exc.kind, exc.description)
            continue
        if len(selected) >= limits.max_artifacts:
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
        selected[relative] = staged
        total += staged.size_bytes
        if staged.role == "primary_report" and staged.media_type in {"text/markdown", "text/html"}:
            text = staged.content.decode("utf-8")
            children: set[str] = set()
            for path, label, image in _references(text, html=staged.media_type == "text/html"):
                try:
                    child = _normalize_reference(path, parent=relative)
                except ValueError:
                    invalid_references.append(
                        (
                            path,
                            ArtifactIssue(
                                "invalid_reference",
                                "A report Artifact reference is not a safe relative path.",
                            ),
                        )
                    )
                    continue
                labels.setdefault(child, label)
                children.add(child)
                queue.append((relative, child, image))
            graph[relative] = children

    cycle_nodes = _cycle_nodes(graph)
    for relative in cycle_nodes:
        selected.pop(relative, None)
        issues_by_path[relative] = ArtifactIssue(
            "reference_cycle", "Artifact references must not contain a cycle."
        )

    path_to_resource = {relative: item.resource_id for relative, item in selected.items()}
    unavailable_ids: dict[str, str] = {
        relative: _unavailable_id(relative) for relative in issues_by_path
    }
    answer_settled = _rewrite_references(
        answer,
        resources={**path_to_resource, **unavailable_ids},
        parent=None,
        html=False,
    )
    settled_artifacts: list[StagedArtifact] = []
    for relative, staged in selected.items():
        content = staged.content
        if staged.role == "primary_report" and staged.media_type in {"text/markdown", "text/html"}:
            content = _rewrite_references(
                content.decode("utf-8"),
                resources={**path_to_resource, **unavailable_ids},
                parent=relative,
                html=staged.media_type == "text/html",
            ).encode("utf-8")
            staged = replace(
                staged,
                content=content,
                size_bytes=len(content),
                digest=hashlib.sha256(content).hexdigest(),
            )
        settled_artifacts.append(staged)

    descriptors: list[Mapping[str, object]] = [
        item.descriptor(label=labels.get(item.relative_path, "")) for item in settled_artifacts
    ]
    issues = list(global_issues)
    for relative, issue in issues_by_path.items():
        resource_id = unavailable_ids[relative]
        issue = replace(issue, resource_id=resource_id)
        issues.append(issue)
        descriptors.append(
            {
                "resource_id": resource_id,
                "role": "primary_report" if relative in PRIMARY_REPORT_NAMES else "attachment",
                "media_type": "application/octet-stream",
                "label": labels.get(relative) or Path(relative).name or "Unavailable Artifact",
                "filename": _safe_filename(relative),
                "byte_size": 0,
                "digest": "",
                "presentation": "download",
                "status": "unavailable",
                "issue": issue.as_dict(),
            }
        )
    for raw, issue in invalid_references:
        resource_id = _unavailable_id(raw)
        safe_issue = replace(issue, resource_id=resource_id)
        issues.append(safe_issue)
        descriptors.append(
            {
                "resource_id": resource_id,
                "role": "attachment",
                "media_type": "application/octet-stream",
                "label": "Unavailable Artifact",
                "filename": "artifact",
                "byte_size": 0,
                "digest": "",
                "presentation": "download",
                "status": "unavailable",
                "issue": safe_issue.as_dict(),
            }
        )
        answer_settled = answer_settled.replace(f"artifact:{raw}", f"artifact:{resource_id}")

    return PublicationPlan(
        answer=answer_settled,
        artifacts=tuple(settled_artifacts),
        descriptors=tuple(descriptors),
        issues=tuple(_dedupe_issues(issues)),
    )


@dataclass(frozen=True, slots=True)
class _FileValidationError(ValueError):
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
        raise _FileValidationError("unsafe_file", "An Artifact could not be read safely.") from exc
    if size > limits.max_file_bytes:
        raise _FileValidationError(
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
        raise _FileValidationError("unsafe_file", "An Artifact could not be read safely.") from exc
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
                raise _FileValidationError(
                    "image_too_large",
                    "Artifact image exceeds the 64-megapixel or 8000-pixel original limit.",
                )
            if (
                width * height > limits.preview_image_max_pixels
                or max(width, height) > limits.preview_image_max_edge
            ):
                capability = "download"
        elif media_type == "image/svg+xml":
            content = _sanitize_svg(content)
        elif media_type == "application/pdf":
            with pdfium.PdfDocument(content) as document:
                if len(document) == 0:
                    raise ValueError("PDF has no pages")
            if relative in PRIMARY_REPORT_NAMES and not _pdf_has_body(content):
                raise ValueError("PDF report is blank")
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
                raise _FileValidationError(
                    "active_preview_too_large",
                    f"Active HTML preview is limited to {limits.active_html_max_bytes} bytes.",
                )
            if (
                _HTML_EXTERNAL_RESOURCE.search(text)
                or _HTML_STYLESHEET.search(text)
                or _CSS_EXTERNAL_URL.search(text)
                or _CSS_EXTERNAL_IMPORT.search(text)
            ):
                raise _FileValidationError(
                    "media_mismatch", "Active HTML must be a self-contained single file."
                )
    except _FileValidationError:
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
        raise _FileValidationError(
            "media_mismatch", f"Artifact {Path(relative).name} does not match its file extension."
        ) from exc
    resource_id = _resource_id(relative)
    return StagedArtifact(
        relative_path=relative,
        role="primary_report" if relative in PRIMARY_REPORT_NAMES else "attachment",
        media_type=media_type,
        size_bytes=len(content),
        path=path,
        resource_id=resource_id,
        filename=_safe_filename(relative),
        digest=hashlib.sha256(content).hexdigest(),
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


def _references(text: str, *, html: bool) -> list[tuple[str, str, bool]]:
    values: list[tuple[str, str, bool]] = []
    pattern = _HTML_ARTIFACT_TARGET if html else _ARTIFACT_TARGET
    for match in pattern.finditer(text):
        raw = match.group("path")
        prefix = match.group("prefix")
        label = ""
        if not html:
            label_match = re.search(r"!?\[([^\]]*)\]", prefix)
            label = label_match.group(1) if label_match else ""
        values.append((raw, label.strip(), prefix.startswith("!")))
    return values


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


def _rewrite_references(
    text: str,
    *,
    resources: Mapping[str, str],
    parent: str | None,
    html: bool,
) -> str:
    pattern = _HTML_ARTIFACT_TARGET if html else _ARTIFACT_TARGET

    def replace_target(match: re.Match[str]) -> str:
        raw = match.group("path")
        try:
            normalized = _normalize_reference(raw, parent=parent)
        except ValueError:
            return match.group(0)
        resource = resources.get(normalized)
        if resource is None:
            return match.group(0)
        return f"{match.group('prefix')}{resource}{match.group('suffix')}"

    return pattern.sub(replace_target, text)


def _cycle_nodes(graph: Mapping[str, set[str]]) -> set[str]:
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


def _resource_id(relative: str) -> str:
    return f"artifact-{hashlib.sha256(relative.encode('utf-8')).hexdigest()[:20]}"


def _unavailable_id(value: str) -> str:
    return f"unavailable-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def _safe_filename(relative: str) -> str:
    name = PurePosixPath(relative).name.strip().replace("\x00", "")
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return name[:180] or "artifact"


def _safe_issue_text(value: str) -> str:
    # Scan exceptions can include local paths. Keep only the stable policy fact.
    if "working-set" in value:
        return value
    return "The Artifact root contains an unsafe or unreadable entry."


def _unavailable_answer(
    answer: str, issue: ArtifactIssue
) -> tuple[str, tuple[Mapping[str, object], ...]]:
    descriptors: list[Mapping[str, object]] = []
    settled = answer
    for raw, label, _image in _references(answer, html=False):
        resource_id = _unavailable_id(raw)
        settled = settled.replace(f"artifact:{raw}", f"artifact:{resource_id}")
        descriptors.append(
            {
                "resource_id": resource_id,
                "role": "attachment",
                "media_type": "application/octet-stream",
                "label": label or "Unavailable Artifact",
                "filename": "artifact",
                "byte_size": 0,
                "digest": "",
                "presentation": "download",
                "status": "unavailable",
                "issue": replace(issue, resource_id=resource_id).as_dict(),
            }
        )
    return settled, tuple(descriptors)


def _dedupe_issues(issues: Sequence[ArtifactIssue]) -> list[ArtifactIssue]:
    seen: set[tuple[str, str, str | None]] = set()
    result: list[ArtifactIssue] = []
    for issue in issues:
        key = (issue.kind, issue.description, issue.resource_id)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def _pdf_has_body(content: bytes) -> bool:
    visual_types = [
        pdfium.raw.FPDF_PAGEOBJ_IMAGE,
        pdfium.raw.FPDF_PAGEOBJ_PATH,
        pdfium.raw.FPDF_PAGEOBJ_SHADING,
    ]
    with pdfium.PdfDocument(content) as document:
        page_count = min(len(document), _PDF_REPORT_SCAN_PAGES)
        for page_index in range(page_count):
            page = document[page_index]
            try:
                text_page = page.get_textpage()
                try:
                    count = min(
                        text_page.count_chars(),
                        _PDF_REPORT_TEXT_CHARS_PER_PAGE,
                    )
                    text = text_page.get_text_range(count=count) if count > 0 else ""
                finally:
                    text_page.close()
                if is_substantive_text(text):
                    return True
                with closing(page.get_objects(filter=visual_types, max_depth=8)) as objects:
                    for page_object in objects:
                        page_object.close()
                        return True
            finally:
                page.close()
    return False


def _report_has_body(path: Path, *, limits: PublicationLimits) -> bool:
    try:
        if path.stat().st_size > limits.max_file_bytes:
            return True  # Let _validate_file report the stable size issue without reading it.
        if path.suffix.lower() == ".pdf":
            content = path.read_bytes()
            try:
                return _pdf_has_body(content)
            except pdfium.PdfiumError, RuntimeError, ValueError:
                return True  # Let _validate_file report malformed PDF media.
        text = path.read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        return True  # Let _validate_file report unreadable or malformed text media.
    if path.suffix.lower() in {".html", ".htm"}:
        text = _HTML_COMMENT.sub(" ", text)
        if _HTML_SUBSTANTIVE_ELEMENT.search(text):
            return True
        text = html_lib.unescape(_HTML_TAG.sub(" ", _HTML_STYLE_BLOCK.sub(" ", text)))
    return is_substantive_text(text)


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
    "PRIMARY_REPORT_NAMES",
    "ArtifactIssue",
    "ArtifactIssueKind",
    "ArtifactRole",
    "PublicationLimits",
    "PublicationPlan",
    "PublicationScanError",
    "StagedArtifact",
    "is_empty_answer",
    "is_substantive_text",
    "scan_artifact_directory",
    "validate_publication",
]
