# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Generic read/write/edit/grep/bash factories over an ExecutionEnvironment."""

from __future__ import annotations

import asyncio
import base64
import codecs
import glob
import hashlib
import json
import os
import re
import shlex
import stat
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlightrag.engine.agent.environment.access import (
    AccessScheduler,
    AgentspaceAccess,
    PathAccess,
)
from dlightrag.engine.agent.environment.child import build_child_environment
from dlightrag.engine.agent.environment.errors import (
    TOOL_RESULT_MAX_BYTES,
    TOOL_RESULT_MAX_LINES,
    TOOL_RESULT_PREVIEW_BYTES,
    AgentspaceQuotaExceeded,
    FullOutputUnavailable,
    PathRejected,
)
from dlightrag.engine.agent.environment.execution import ExecutionEnvironment
from dlightrag.engine.agent.environment.local import ProcessChunk
from dlightrag.engine.agent.environment.text import decode_agentspace_text, encode_agentspace_text
from dlightrag.engine.agent.environment.toolchain import SearchToolchain
from dlightrag.engine.agent.tool_content import (
    ToolResourceAttachmentPart,
    ToolTextPart,
    VisualSource,
)
from dlightrag.engine.agent.tools.contracts import (
    AdmittedBytesReader,
    AgentspaceInventoryFacts,
    AgentspacePathFact,
    AgentTool,
    CommittedOutput,
    EvidenceSourceFact,
    ResourceAttachmentBytes,
    ToolDeclaration,
    ToolEffects,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.agent.tools.editing import (
    EditResolutionError,
    apply_edits,
    edit_summary_lines,
    render_hunks,
    resolve_edits,
)
from dlightrag.engine.agent.tools.listing import directory_page as _directory_page
from dlightrag.engine.agent.tools.listing import escape_path as _escape_path
from dlightrag.engine.agent.tools.output import (
    OutputStage,
    StreamingToolOutput,
    ToolOutputSnapshot,
)
from dlightrag.engine.ai.media import ImagePayloadBudget, decode_image_base64, detect_image_mime

type SpillWriter = Callable[[str], Awaitable[CommittedOutput]]
type OutputStageFactory = Callable[[str], OutputStage]

_PATH_MAX_CHARS = 4096
PATH_MAX_CHARS = _PATH_MAX_CHARS
"""The longest workspace path any tool accepts.

Public because a second tool that names a workspace-relative destination — the Skill loader's
``write_to`` — must refuse what this module refuses, and two private copies of one bound are two
bounds the day either of them moves.
"""
_CURSOR_MAX_CHARS = 8192
_VIOLATION_PATH_PREVIEW_BYTES = 256


@dataclass(frozen=True, slots=True)
class PreparedImageAttachment:
    """One provider-safe derivative of a verified source image."""

    data: bytes
    media_type: str
    transformed: bool


type ImagePreparer = Callable[[bytes, str], PreparedImageAttachment | None]


class HttpReadOptions(BaseModel):
    """Representation headers allowed on the first direct URL acquisition."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_agent: str | None = Field(default=None, min_length=1, max_length=256)
    accept: str | None = Field(default=None, min_length=1, max_length=512)
    accept_language: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def _single_line(self) -> HttpReadOptions:
        if any("\r" in value or "\n" in value for value in self.model_dump().values() if value):
            raise ValueError("HTTP representation headers must be single-line values")
        return self


class ReadArgs(BaseModel):
    """``read``: one workspace path. A Resource is ``read_resource``'s, so no argument here
    depends on which kind of target the call names."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str = Field(max_length=_PATH_MAX_CHARS, description="Workspace-relative path to read.")
    offset: int | None = Field(
        default=None,
        ge=1,
        description="1-based line offset in a file.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        description="Maximum lines (file) or entries (directory) to return.",
    )
    cursor: str | None = Field(
        default=None,
        max_length=_CURSOR_MAX_CHARS,
        description="Continuation cursor a directory page printed.",
    )

    @model_validator(mode="after")
    def _one_continuation(self) -> ReadArgs:
        if self.cursor is not None and self.offset is not None:
            raise ValueError("read path cursor and offset are mutually exclusive")
        return self


class ReadResourceArgs(BaseModel):
    """``read_resource``: one Resource, named by its resource_id or by a public url."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    resource_id: str | None = Field(default=None, description="Opaque durable resource id.")
    url: str | None = Field(default=None, description="Anonymous public HTTP(S) URL to read.")
    http: HttpReadOptions | None = None
    focus: str | None = Field(
        default=None,
        min_length=1,
        description="Optional relevance focus within the Resource.",
    )
    cursor: str | None = Field(
        default=None,
        max_length=_CURSOR_MAX_CHARS,
        description="Continuation cursor a page of this resource_id printed.",
    )

    @model_validator(mode="after")
    def _exactly_one_target(self) -> ReadResourceArgs:
        if (self.resource_id is None) == (self.url is None):
            raise ValueError("read_resource requires exactly one of resource_id or url")
        if self.url is not None and self.cursor is not None:
            raise ValueError("continue a URL read with its returned resource_id")
        if self.url is None and self.http is not None:
            raise ValueError("read_resource http options are available only for url")
        return self


class RenderedReadResourceArgs(ReadResourceArgs):
    """``read_resource`` for a Host with an Agent Browser: a page can also be read as it renders."""

    rendered: bool = Field(
        default=False,
        description=(
            "Read the page as the Agent Browser renders it, after its scripts run. "
            "url or a Web resource_id only."
        ),
    )

    @model_validator(mode="after")
    def _rendered_targets(self) -> RenderedReadResourceArgs:
        if self.rendered and self.http is not None:
            raise ValueError(
                "read_resource http options apply to direct acquisition, not rendered=true"
            )
        return self


@dataclass(frozen=True, slots=True)
class ResourceReadRequest:
    resource_id: str | None
    url: str | None
    focus: str | None
    cursor: str | None
    user_agent: str | None = None
    accept: str | None = None
    accept_language: str | None = None
    rendered: bool = False


type ResourceReader = Callable[[ResourceReadRequest, ToolRuntime], Awaitable[ToolResult]]


class WriteArgs(BaseModel):
    path: str = Field(max_length=_PATH_MAX_CHARS, description="Workspace-relative path to write.")
    content: str = Field(description="Full UTF-8 file contents.")


class EditOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_text: str = Field(
        description="Replacement text; an empty string deletes the region.",
    )
    old_text: str | None = Field(
        default=None,
        min_length=1,
        description="Exact text that occurs exactly once in the original file.",
    )
    symbol: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description=(
            "A named definition in the file's language, replaced with the whole statement "
            "that defines it. In a .py file: a module-level assignment (including one that "
            "unpacks several names), def, async def or class, or a dotted Class.member path."
        ),
    )
    start_marker: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Whole-line marker text that begins the region; its line is kept. Give it with "
            "end_marker."
        ),
    )
    end_marker: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Whole-line marker text that ends the region; its line is kept. Give it with "
            "start_marker."
        ),
    )


class EditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str = Field(max_length=_PATH_MAX_CHARS, description="Workspace-relative path to edit.")
    edits: list[EditOperation] = Field(
        min_length=1,
        description="Non-overlapping replacements, all resolved against the original file.",
    )


class GrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pattern: str = Field(min_length=1, max_length=65_536)
    path: str = Field(
        default=".",
        max_length=_PATH_MAX_CHARS,
        description="Workspace path to search.",
    )
    glob: str | None = Field(default=None, description="Optional glob filter.")
    ignore_case: bool = Field(default=False, description="Case-insensitive matching.")
    literal: bool = Field(
        default=False, description="Treat pattern as a literal string, not a regex."
    )
    context: int | None = Field(
        default=None,
        ge=0,
        le=1000,
        description="Context lines shown around each match.",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=100_000,
        description="Maximum matching lines to return; context lines do not count.",
    )


class FindArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pattern: str = Field(min_length=1, max_length=4096, description="Glob pattern to match.")
    path: str = Field(
        default=".",
        max_length=_PATH_MAX_CHARS,
        description="Workspace subtree to search.",
    )
    limit: int = Field(
        default=1000,
        ge=1,
        le=100_000,
        description="Maximum matches to return.",
    )


class LsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str = Field(
        default=".",
        max_length=_PATH_MAX_CHARS,
        description="Workspace directory to list.",
    )
    limit: int = Field(default=500, ge=1, le=10_000, description="Maximum entries to return.")
    cursor: str | None = Field(
        default=None,
        max_length=_CURSOR_MAX_CHARS,
        description="Continuation cursor from a prior page.",
    )


class BashArgs(BaseModel):
    command: str = Field(description="Bash command to run.")
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        le=2_147_483.647,
        allow_inf_nan=False,
        description=(
            "Seconds after which the command is killed and the call reports a timeout. There is "
            "no default, so set one to bound a command that could hang or run long (network "
            "access, installs, long loops, anything waiting on input); omit it for quick commands."
        ),
    )


def spill_continuation(resource_id: str) -> str:
    """The line that tells the model where the rest of an oversized output is kept."""
    return f"Full output: read_resource(resource_id={resource_id!r}, cursor=...)"


async def preview_or_spill(
    text: str,
    *,
    spill: SpillWriter | None,
    tool: str,
    preview: Literal["head", "tail"] = "head",
) -> tuple[str, CommittedOutput | None]:
    """Return (model text, optional committed-spill receipt)."""
    if within_result_bounds(text):
        return text, None
    if spill is None:
        raise FullOutputUnavailable("oversized tool result has no spill or cursor backing")
    receipt = await spill(text)
    excerpt = _utf8_excerpt(text, preview=preview)
    rendered = (
        f"{tool} output exceeded {TOOL_RESULT_MAX_BYTES} UTF-8 bytes or "
        f"{TOOL_RESULT_MAX_LINES} lines ({len(text.encode('utf-8'))} bytes). "
        f"{spill_continuation(receipt.resource_id)}\n{excerpt}"
    )
    return rendered, receipt


#: What a model needs to know to ask for a Rendered Read, and when not to (ADR 0032).
_RENDERED_READ_GUIDANCE = (
    "With rendered=true, a url or Web resource_id is read as the Agent Browser renders it, "
    "after its scripts run; use it only when a read returned a JavaScript shell (a loading "
    "or enable-JavaScript notice instead of the content). Its cursor continues the rendered text."
)


def read_declaration() -> ToolDeclaration:
    """``read``: a workspace path, the one target it has (ADR 0100)."""
    return ToolDeclaration(
        name="read",
        description=(
            "Read bounded text from one workspace path (use view for image pixels, and "
            "read_resource for a url or resource_id). A file pages by line offset and limit; a "
            "directory by the opaque cursor its page prints. Follow the printed continuation "
            "instead of re-reading the whole file."
        ),
        input_model=ReadArgs,
        replay_policy="replayable",
        read_only=True,
        contract_version=5,
    )


def read_resource_declaration(*, rendered: bool = False) -> ToolDeclaration:
    """``read_resource``: a Resource by resource_id, or a public url that registers one.

    ``rendered`` offers the Agent Browser's rendering (ADR 0032).
    """
    description = (
        "Read bounded text of one Resource: a resource_id registered in this run (a search "
        "hit, an upload, a spilled tool output, a published product) or an anonymous public "
        "HTTP(S) url, which registers the page as a Resource. Text comes in bounded windows "
        "with no line offset or limit: continue with the resource_id and cursor the page "
        "prints, and use focus to ask for the part that matters. Use read for workspace files."
    )
    if rendered:
        description = f"{description} {_RENDERED_READ_GUIDANCE}"
    return ToolDeclaration(
        name="read_resource",
        description=description,
        input_model=RenderedReadResourceArgs if rendered else ReadResourceArgs,
        replay_policy="replayable",
        read_only=True,
        contract_version=1,
    )


def read_tool(
    environment: ExecutionEnvironment | None,
    scheduler: AccessScheduler,
    *,
    spill: SpillWriter | None = None,
) -> AgentTool:
    """Build ``read`` over the Run's workspace."""

    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(ReadArgs, args)
        if environment is None:
            return ToolResult.text("path read requires an execution environment", is_error=True)
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        try:
            path = environment.resolve(args.path)
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        canonical_path = _agentspace_relative_path(environment.root, path)
        await runtime.emit_update(ToolResult.text("", subject=_escape_path(canonical_path)))
        async with scheduler.hold(PathAccess(path=str(path), kind="read")):
            if blocked := agentspace_integrity_refusal(environment):
                return blocked
            kind = environment.stat_kind(path)
            if kind == "directory":
                if args.offset is not None:
                    return ToolResult.text(
                        "read directory pages use cursor, not offset",
                        is_error=True,
                    )
                return _directory_page(
                    environment.list_directory(path),
                    path=canonical_path,
                    cursor=args.cursor,
                    limit=args.limit or 500,
                    tool="read",
                )
            if kind == "missing":
                return ToolResult.text(f"file not found: {_escape_path(args.path)}", is_error=True)
            if args.cursor is not None:
                return ToolResult.text("read path cursor requires a directory", is_error=True)
            raw = environment.read_bytes(path)
            media_type = _sniff_image_media_type(raw)
            if media_type is not None:
                return ToolResult.text(
                    f"Image: {_escape_path(canonical_path)} ({media_type}, {len(raw)} bytes). "
                    f"Use view(path={canonical_path!r}) to see pixels; read returns text only."
                )
            try:
                decoded = decode_agentspace_text(raw)
            except ValueError as exc:
                return ToolResult.text(str(exc), is_error=True)
            body, continuation, remaining = _paginate_lines(
                decoded.text,
                path=canonical_path,
                offset=args.offset,
                limit=args.limit,
                notice=(
                    "[mixed line endings preserved; not normalized]"
                    if decoded.mixed_newlines
                    else ""
                ),
            )
            body, committed = await preview_or_spill(body, spill=spill, tool="read")
            total_lines = len(decoded.text.splitlines())
            first_line = max(0, (args.offset or 1) - 1)
            return ToolResult.text(
                body,
                protected_text=continuation,
                effects=ToolEffects(
                    committed_outputs=((committed,) if committed is not None else ())
                ),
                facts={
                    "path": canonical_path,
                    "lines_read": max(0, total_lines - remaining - first_line),
                    "total_lines": total_lines,
                    # Either the file continues past this page or the page itself had to be
                    # spilled for its size: both mean the text the model holds is a window.
                    "truncated": remaining > 0 or committed is not None,
                },
            )

    return read_declaration().bind(execute)


def read_resource_tool(
    scheduler: AccessScheduler,
    *,
    resource_reader: ResourceReader,
    rendered: bool = False,
) -> AgentTool:
    """Build ``read_resource`` over the Run's resource registry."""

    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(ReadResourceArgs, args)
        subject = args.resource_id or args.url
        if subject:
            # An empty target names nothing, so the row stays verb-only.
            await runtime.emit_update(ToolResult.text("", subject=subject))
        options = args.http or HttpReadOptions()
        async with scheduler.hold(PathAccess(path=subject or "resource", kind="read")):
            return await resource_reader(
                ResourceReadRequest(
                    resource_id=args.resource_id,
                    url=args.url,
                    focus=args.focus,
                    cursor=args.cursor,
                    user_agent=options.user_agent,
                    accept=options.accept,
                    accept_language=options.accept_language,
                    rendered=isinstance(args, RenderedReadResourceArgs) and args.rendered,
                ),
                runtime,
            )

    return read_resource_declaration(rendered=rendered).bind(execute)


class ViewArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str | None = Field(
        default=None,
        min_length=1,
        max_length=_PATH_MAX_CHARS,
        description="Workspace path of a standalone image.",
    )
    resource_id: str | None = Field(default=None, min_length=1, max_length=256)
    url: str | None = Field(default=None, min_length=1, max_length=8192)
    http: HttpReadOptions | None = None
    locator: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="1-based physical PDF page number or embedded-image handle from read_resource.",
    )
    cursor: str | None = Field(default=None, min_length=1, max_length=_CURSOR_MAX_CHARS)

    @model_validator(mode="after")
    def _target(self) -> ViewArgs:
        if sum(value is not None for value in (self.path, self.resource_id, self.url)) != 1:
            raise ValueError("view requires exactly one of path, resource_id, or url")
        if self.locator is not None and self.cursor is not None:
            raise ValueError("locator and cursor are mutually exclusive")
        if self.path is not None and (self.locator is not None or self.cursor is not None):
            raise ValueError("workspace view supports standalone images only")
        if self.url is not None and self.cursor is not None:
            raise ValueError("continue a URL view with its returned resource_id")
        if self.url is None and self.http is not None:
            raise ValueError("view http options are available only for url")
        return self


type ResourceViewer = Callable[[ViewArgs, ToolRuntime, ImagePreparer], Awaitable[ToolResult]]


def view_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="view",
        description="View pixels from exactly one resource registered in this run, anonymous "
        "public URL, or workspace image path. A PDF without a locator returns a bounded "
        "overview; use it to find physical pages, not to transcribe small text. No separate "
        "model is called.",
        input_model=ViewArgs,
        replay_policy="replayable",
        read_only=True,
    )


def view_tool(
    environment: ExecutionEnvironment | None,
    scheduler: AccessScheduler,
    *,
    resource_viewer: ResourceViewer | None = None,
    image_preparer: ImagePreparer | None = None,
) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(ViewArgs, raw)
        prepare = image_preparer or _default_image_preparer
        if args.path is None:
            if resource_viewer is None:
                return ToolResult.text("resource view is unavailable", is_error=True)
            subject = args.resource_id or args.url
            if subject is not None:
                await runtime.emit_update(ToolResult.text("", subject=subject))
            return await resource_viewer(args, runtime, prepare)
        if environment is None:
            return ToolResult.text("path view requires an execution environment", is_error=True)
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        try:
            path = environment.resolve(args.path)
            canonical = _agentspace_relative_path(environment.root, path)
            await runtime.emit_update(ToolResult.text("", subject=_escape_path(canonical)))
            async with scheduler.hold(PathAccess(path=str(path), kind="read")):
                if blocked := agentspace_integrity_refusal(environment):
                    return blocked
                if environment.stat_kind(path) != "file":
                    return ToolResult.text("view requires a regular image file", is_error=True)
                content = environment.read_bytes(path)
                media = _sniff_image_media_type(content)
                if media is None:
                    return ToolResult.text(
                        "view(path) supports verified standalone images only", is_error=True
                    )
                # Preparing spends the Run's shared image budget.
                await runtime.in_source_order()
                prepared = await asyncio.to_thread(prepare, content, canonical)
                if prepared is None:
                    return ToolResult.text(
                        "image cannot fit the remaining model image budget", is_error=True
                    )
                return _image_attachment_result(
                    content, source_media_type=media, prepared=prepared, path=canonical
                )
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)

    return view_declaration().bind(execute)


async def replace_agentspace_file(
    environment: ExecutionEnvironment, scheduler: AccessScheduler, path: Path, data: bytes
) -> AgentspaceInventoryFacts | ToolResult:
    """Replace one file with ``data`` under its path's write hold.

    The latch is checked again once the hold is taken, because a Bash command that held the
    workspace first may have latched it. The result is the inventory fact of what is on disk,
    or the refusal that stopped the write.
    """
    async with scheduler.hold(PathAccess(path=str(path), kind="write")):
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        try:
            environment.write_bytes(path, data)
        except (AgentspaceQuotaExceeded, PathRejected) as exc:
            return ToolResult.text(str(exc), is_error=True)
        return _inventory_facts(environment.root, path)


def write_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="write",
        description="Create or overwrite a UTF-8 workspace file. It replaces the whole file; "
        "the success line reports the UTF-8 byte size.",
        input_model=WriteArgs,
        replay_policy="never",
        contract_version=3,
    )


def write_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(WriteArgs, args)
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        try:
            path = environment.resolve(args.path)
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        canonical = _agentspace_relative_path(environment.root, path)
        await runtime.emit_update(ToolResult.text("", subject=_escape_path(canonical)))
        data = args.content.encode("utf-8")
        inventory = await replace_agentspace_file(environment, scheduler, path, data)
        if isinstance(inventory, ToolResult):
            return inventory
        return ToolResult.text(
            f"wrote {args.path} ({len(data)} bytes)",
            effects=ToolEffects(workspace_inventory=inventory),
        )

    return write_declaration().bind(execute)


class MaterializeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    resource_id: str = Field(
        min_length=1,
        max_length=256,
        description="The resource_id to copy, as read_resource and view take it.",
    )
    path: str = Field(
        min_length=1,
        max_length=_PATH_MAX_CHARS,
        description="Workspace-relative destination path.",
    )


def materialize_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="materialize",
        description="Copy the bytes one resource_id of this run holds into a workspace file, byte "
        "for byte, when a process needs the file itself rather than read_resource's text. It replaces the "
        "whole file at path; the success line reports the media type and byte size. It never "
        "fetches or renders: a resource that holds no bytes yet is refused with the call to make "
        "first.",
        input_model=MaterializeArgs,
        replay_policy="never",
        contract_version=1,
    )


def materialize_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    admitted_bytes_reader: AdmittedBytesReader,
) -> AgentTool:
    """Copy a Resource's admitted bytes into the workspace, accounted exactly as ``write`` is.

    The bytes stay in this process: they are never model context, and the tool admits no Evidence.
    """

    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(MaterializeArgs, args)
        # A refusal of the workspace or of the destination loads and adopts nothing.
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        try:
            path = environment.resolve(args.path)
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        if environment.stat_kind(path) == "directory":
            # The write refuses this too, but only once the bytes are loaded.
            return ToolResult.text("cannot overwrite a directory", is_error=True)
        destination = _escape_path(_agentspace_relative_path(environment.root, path))
        await runtime.emit_update(ToolResult.text("", subject=destination))
        admitted = await admitted_bytes_reader(args.resource_id, runtime)
        if isinstance(admitted, ToolResult):
            return admitted
        inventory = await replace_agentspace_file(environment, scheduler, path, admitted.content)
        if isinstance(inventory, ToolResult):
            return inventory
        return ToolResult.text(
            f"materialized {admitted.resource_id} to {destination} "
            f"({admitted.media_type}, {len(admitted.content)} bytes)",
            effects=ToolEffects(workspace_inventory=inventory),
        )

    return materialize_declaration().bind(execute)


def edit_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="edit",
        description="Replace a region of a workspace file. Each edit names exactly one target: "
        "symbol (a named definition in a .py file), start_marker with end_marker (two whole-line "
        "markers; the lines between them are replaced and both marker lines are kept), or "
        "old_text (exact text, once in the file). Prefer symbol or the markers over old_text "
        "when the region is a named block or a delimited section; use old_text only for a short "
        "unique string. All edits resolve against the original file and apply atomically or none "
        "do; read the file first when a target fails.",
        input_model=EditArgs,
        replay_policy="never",
        contract_version=4,
    )


def edit_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    spill: SpillWriter | None = None,
) -> AgentTool:
    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        edit_args = cast(EditArgs, args)
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        try:
            path = environment.resolve(edit_args.path)
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        canonical = _agentspace_relative_path(environment.root, path)
        await runtime.emit_update(ToolResult.text("", subject=_escape_path(canonical)))
        async with scheduler.hold(PathAccess(path=str(path), kind="readwrite")):
            if blocked := agentspace_integrity_refusal(environment):
                return blocked
            if environment.stat_kind(path) != "file":
                return ToolResult.text(
                    f"file not found: {_escape_path(edit_args.path)}",
                    is_error=True,
                )
            try:
                decoded = decode_agentspace_text(environment.read_bytes(path))
            except ValueError as exc:
                return ToolResult.text(str(exc), is_error=True)
            try:
                resolved = resolve_edits(decoded.text, edit_args.edits, path=edit_args.path)
            except EditResolutionError as exc:
                return ToolResult.text(str(exc), is_error=True)
            updated = apply_edits(decoded.text, resolved)
            try:
                environment.write_bytes(path, encode_agentspace_text(decoded, updated))
            except (AgentspaceQuotaExceeded, PathRejected) as exc:
                return ToolResult.text(str(exc), is_error=True)
            inventory = _inventory_facts(environment.root, path)
        first_change = min(resolved, key=lambda edit: edit.start)
        summary = "\n".join(edit_summary_lines(resolved))
        patch = render_hunks(decoded.text, updated, resolved, path=edit_args.path)
        body = (
            f"edited {_escape_path(edit_args.path)} ({len(resolved)} edits; "
            f"first change line {first_change.first_line})\n{summary}\n{patch}"
        )
        body, committed = await preview_or_spill(body, spill=spill, tool="edit")
        return ToolResult.text(
            body,
            effects=ToolEffects(
                committed_outputs=((committed,) if committed is not None else ()),
                workspace_inventory=inventory,
            ),
        )

    return edit_declaration().bind(execute)


def grep_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="grep",
        description="Search workspace files with ripgrep; hidden files are searched while "
        "ignore rules apply.",
        input_model=GrepArgs,
        replay_policy="replayable",
        read_only=True,
        contract_version=3,
    )


def grep_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    search_toolchain: SearchToolchain,
    output_stage_factory: OutputStageFactory | None = None,
) -> AgentTool:
    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        grep_args = cast(GrepArgs, args)
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        try:
            root = (
                environment.root if grep_args.path == "." else environment.resolve(grep_args.path)
            )
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        target = root.relative_to(environment.root).as_posix() if root != environment.root else "."
        try:
            ripgrep = await search_toolchain.path("rg")
        except RuntimeError as exc:
            return ToolResult.text(str(exc), is_error=True)
        await runtime.emit_update(ToolResult.text("", subject=grep_args.pattern))
        argv = [
            ripgrep,
            "--json",
            "--no-config",
            "--hidden",
            "--no-require-git",
            "--glob",
            "!.git",
            # ripgrep reports files in the order its threads finish; sorted, a replayed
            # call answers the same way and a limit cuts at the same match.
            "--sort",
            "path",
            "--max-count",
            str(grep_args.limit + 1),
        ]
        if grep_args.ignore_case:
            argv.append("--ignore-case")
        if grep_args.literal:
            argv.append("--fixed-strings")
        if grep_args.context is not None:
            argv.extend(["--context", str(grep_args.context)])
        if grep_args.glob:
            argv.extend(["--glob", grep_args.glob])
        argv.extend(["-e", grep_args.pattern, "--", target])
        output = _streaming_output("grep", output_stage_factory)
        collector = _GrepJsonCollector(
            output=output,
            workspace_root=environment.root,
            limit=grep_args.limit,
        )

        async def capture(chunk: ProcessChunk) -> None:
            collector.feed(chunk)

        try:
            async with scheduler.hold(PathAccess(path=str(root), kind="search")):
                if blocked := agentspace_integrity_refusal(environment):
                    output.abort()
                    return blocked
                home, tmp = environment.prepare_process_directories()
                completed = await environment.run(
                    argv,
                    env=build_child_environment(home=home, tmp=tmp),
                    cwd=environment.root,
                    on_output=capture,
                )
            collector.finish()
            if completed.returncode == 1 and collector.matches == 0:
                output.append(ProcessChunk("stdout", b"(no matches)"))
            if collector.truncated:
                output.append(
                    ProcessChunk(
                        "stdout",
                        f"\n[limited to {grep_args.limit} matching lines]".encode(),
                    )
                )
            if collector.parse_errors:
                output.append(
                    ProcessChunk(
                        "stderr",
                        f"\n[ripgrep JSON protocol errors: {collector.parse_errors}]".encode(),
                    )
                )
            final = await output.finish()
        except (OSError, PathRejected, AgentspaceQuotaExceeded) as exc:
            output.abort()
            return ToolResult.text(str(exc), is_error=True)
        except asyncio.CancelledError:
            output.abort()
            raise
        except BaseException:
            output.abort()
            raise
        result = _stream_result("grep", final)
        return ToolResult.text(
            result.text_content,
            details={
                **(result.details or {}),
                "matches": collector.matches,
                "parse_errors": collector.parse_errors,
            },
            protected_text=result.protected_text,
            is_error=completed.returncode not in {0, 1} or collector.parse_errors > 0,
            effects=result.effects,
        )

    return grep_declaration().bind(execute)


def bash_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="bash",
        description="Run a bash command in the workspace. Output streams live and stays "
        "bounded; a timed-out or failing command still returns its partial output, as an "
        "error. Never leave symlinks, FIFOs, sockets, device files, or quota overflow "
        "behind: the workspace latches until external cleanup.",
        input_model=BashArgs,
        replay_policy="never",
        contract_version=3,
    )


_BASH_COMMAND_SEPARATORS = frozenset({";", ";;", "&&", "||", "|", "&", "(", ")", "\n"})
#: Shell words after which the next word starts a command, not an operand.
_BASH_COMMAND_INTRODUCERS = frozenset(
    {"if", "then", "else", "elif", "do", "while", "until", "!", "{", "time"}
)
#: Wrappers that run the word after their own options as the command (cd stays a builtin).
_BASH_COMMAND_PREFIXES = frozenset({"command", "builtin"})
#: Tokens that introduce a redirection; the word after them is not a command or cd target.
_BASH_REDIRECTIONS = frozenset({"<", ">", ">>", "<<", "<<<", "<>", ">&", "<&", ">|", "&>", "&>>"})
_BASH_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _bash_tokens(command: str) -> list[str]:
    """Split a command into shell words and control punctuation, without executing it.

    ``shlex`` in POSIX mode splits quoted words but treats ``;``/``&&`` as ordinary
    characters unless punctuation is asked for. This lexer keeps quotes and escapes,
    emits the operators, and turns an unquoted newline into a command separator so
    ``cd ..`` followed by a newline can never be read as one longer word.
    """
    normalized: list[str] = []
    quote: str | None = None
    escaped = False
    for character in command:
        if escaped:
            normalized.append(character)
            escaped = False
            continue
        if character == "\\" and quote != "'":
            normalized.append(character)
            escaped = True
            continue
        if quote is not None:
            normalized.append(character)
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
            normalized.append(character)
            continue
        if character == "\n":
            normalized.append(" ; ")
            continue
        normalized.append(character)
    lexer = shlex.shlex("".join(normalized), posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        # An unterminated quote or escape is a shell syntax error; it cannot run a cd.
        return []


def _bash_workspace_escape_refusal(environment: ExecutionEnvironment, command: str) -> str | None:
    """Refuse a ``cd``/``pushd`` that would leave the Agent Workspace root.

    This is a behavioural recovery, not a sandbox: it reads the command text only, and
    interpreters/wrappers outside this small set are deliberately out of scope. A
    command that stays in the workspace is untouched; a stale absolute path, a ``..``
    climb or ``cd -`` is refused with the current root named, because the model can then
    retry with a relative path that exists in this Run's workspace.
    """
    tokens = _bash_tokens(command)
    if not tokens:
        return None
    root = Path(environment.root).resolve()
    expects_command = True
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _BASH_COMMAND_SEPARATORS:
            expects_command = True
            index += 1
            continue
        if expects_command and token in _BASH_COMMAND_INTRODUCERS:
            expects_command = True
            index += 1
            continue
        if expects_command and token in _BASH_REDIRECTIONS:
            # ``> file`` before a command: the redirection target is not the command.
            index += 2
            continue
        if expects_command and _BASH_ASSIGNMENT.match(token):
            index += 1
            continue
        if expects_command and token in _BASH_COMMAND_PREFIXES:
            index += 1
            # ``command -p cd``: its own options are skipped, the next word is the command.
            while index < len(tokens) and tokens[index].startswith("-"):
                index += 1
            continue
        if expects_command and token in ("cd", "pushd"):
            cursor = index + 1
            target: str | None = None
            after_double_dash = False
            while cursor < len(tokens) and tokens[cursor] not in _BASH_COMMAND_SEPARATORS:
                candidate = tokens[cursor]
                if candidate in _BASH_REDIRECTIONS:
                    cursor += 2
                    continue
                if candidate == "--" and not after_double_dash:
                    after_double_dash = True
                    cursor += 1
                    continue
                if candidate.startswith("-") and not after_double_dash:
                    # cd/pushd options (-L, -P, -e, or a lone '-' for OLDPWD).
                    if candidate == "-":
                        target = candidate
                        break
                    cursor += 1
                    continue
                target = candidate
                break
            if target == "-":
                return (
                    f"bash refused: `{token} -` returns to OLDPWD, which is not necessarily "
                    f"inside the current Agent Workspace root {root}. Use a relative path "
                    "inside the workspace and run `pwd` to confirm where you are."
                )
            if target not in (None, ""):
                resolved: Path | None = None
                if not target.startswith(("~", "$")):
                    path = Path(target)
                    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
                if resolved is None or not resolved.is_relative_to(root):
                    return (
                        f"bash refused: `{token} {target}` leaves the current Agent "
                        f"Workspace root {root}. Commands run in that root; use a relative "
                        "path inside the workspace and run `pwd` to confirm where you are."
                    )
            expects_command = False
            index += 1
            continue
        if token in ("fi", "done", "esac", "}"):
            expects_command = False
            index += 1
            continue
        expects_command = False
        index += 1
    return None


def bash_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    output_stage_factory: OutputStageFactory | None = None,
) -> AgentTool:
    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(BashArgs, args)
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        if refused := _bash_workspace_escape_refusal(environment, args.command):
            return ToolResult.text(refused, is_error=True)
        await runtime.emit_update(ToolResult.text("", subject=args.command))
        output = _streaming_output("bash", output_stage_factory)
        last_update = 0.0

        async def capture(chunk: ProcessChunk) -> None:
            nonlocal last_update
            snapshot = output.append(chunk)
            now = time.monotonic()
            if now - last_update >= 0.1:
                last_update = now
                await runtime.emit_update(_stream_result("bash", snapshot, transient=True))

        try:
            async with scheduler.hold(AgentspaceAccess()):
                if blocked := agentspace_integrity_refusal(environment):
                    output.abort()
                    return blocked
                try:
                    home, tmp = environment.prepare_process_directories()
                except (OSError, PathRejected, AgentspaceQuotaExceeded) as exc:
                    output.abort()
                    return ToolResult.text(str(exc), is_error=True)
                try:
                    completed = await environment.run(
                        ["/bin/bash", "--noprofile", "--norc", "-c", args.command],
                        env=build_child_environment(home=home, tmp=tmp),
                        cwd=environment.root,
                        timeout_seconds=args.timeout_seconds,
                        on_output=capture,
                    )
                finally:
                    violations = environment.refresh_integrity()
                quota_violation = environment.quota_violation
                inventory = (
                    None if violations or quota_violation else _scan_inventory(environment.root)
                )
            status = "timeout" if completed.timed_out else f"exit {completed.returncode}"
            output.append(ProcessChunk("stdout", f"\n{status}".encode()))
            notices: list[str] = []
            if violations:
                notices.append(
                    "bash left unsafe or unreadable workspace entries: "
                    f"{_render_violations(violations)}; the workspace is latched until "
                    "external cleanup"
                )
            if quota_violation is not None:
                notices.append(
                    f"{quota_violation}; the workspace is latched until external cleanup"
                )
            reserve_bytes, reserve_lines = _process_notice_reserve(notices)
            final = await output.finish(
                reserve_bytes=reserve_bytes,
                reserve_lines=reserve_lines,
            )
        except asyncio.CancelledError:
            output.abort()
            raise
        except BaseException:
            output.abort()
            raise
        streamed = _stream_result("bash", final, notices=notices)
        failed = (
            completed.timed_out
            or completed.returncode != 0
            or bool(violations)
            or quota_violation is not None
        )
        return ToolResult.text(
            streamed.text_content,
            details={
                **(streamed.details or {}),
                "integrity_violations": list(violations),
                "quota_violation": quota_violation,
            },
            protected_text=streamed.protected_text,
            is_error=failed,
            effects=ToolEffects(
                committed_outputs=streamed.effects.committed_outputs,
                workspace_inventory=inventory,
            ),
        )

    return bash_declaration().bind(execute)


def _sniff_image_media_type(data: bytes) -> str | None:
    """Return the original media type for a verified image snapshot, else None."""
    if len(data) > 104_857_600:
        return None
    signatures = (
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
        (b"RIFF", "image/webp"),
    )
    media_type = next((mime for magic, mime in signatures if data.startswith(magic)), None)
    if media_type is None:
        return None
    if media_type == "image/webp" and data[8:12] != b"WEBP":
        return None
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        return media_type
    except Exception:
        return None


def _default_image_preparer(data: bytes, path: str) -> PreparedImageAttachment | None:
    budget = ImagePayloadBudget(
        max_total_bytes=3 * 1024 * 1024,
        max_bytes_per_image=3 * 1024 * 1024,
        max_pixels=40_000_000,
        max_px=1536,
        min_px=256,
        quality=89,
        min_quality=65,
        max_images=1,
    )
    encoded = base64.b64encode(data).decode("ascii")
    bounded = budget.add_base64(encoded, label=path)
    if bounded is None:
        return None
    uri, _byte_count = bounded
    prepared, declared_media_type = decode_image_base64(uri)
    media_type = detect_image_mime(prepared, fallback=declared_media_type)
    return PreparedImageAttachment(
        data=prepared,
        media_type=media_type,
        transformed=prepared != data,
    )


def _image_attachment_result(
    data: bytes,
    *,
    source_media_type: str,
    prepared: PreparedImageAttachment,
    path: str,
) -> ToolResult:
    """Persist the source snapshot but expose only its provider-bounded derivative."""
    source_digest = hashlib.sha256(data).hexdigest()
    prepared_digest = hashlib.sha256(prepared.data).hexdigest()
    occurrence = hashlib.sha256(f"{path}\0{source_digest}".encode()).hexdigest()
    source_resource_id = f"att_{occurrence[:32]}"
    model_resource_id = (
        f"att_model_{hashlib.sha256(f'{occurrence}:{prepared_digest}'.encode()).hexdigest()[:32]}"
        if prepared.transformed
        else source_resource_id
    )
    source = VisualSource(resource_id=source_resource_id, kind="workspace_image", path=path)
    safe_name = path.rsplit("/", 1)[-1] or "image"
    attachment = ToolResourceAttachmentPart(
        resource_id=model_resource_id,
        safe_name=safe_name,
        media_type=prepared.media_type,
        content_digest=prepared_digest,
        size_bytes=len(prepared.data),
        data=prepared.data,
        source=source,
    )
    durable_resources = [
        ResourceAttachmentBytes(
            resource_id=source_resource_id,
            filename=safe_name,
            mime_type=source_media_type,
            source_locator=path,
            content=data,
            source=source,
        )
    ]
    if prepared.transformed:
        # The part names this record, so the settlement checks the part's name against its
        # own: the derivative is the file under the same name, told from the source by its
        # resource id and its locator.
        durable_resources.append(
            ResourceAttachmentBytes(
                resource_id=model_resource_id,
                filename=safe_name,
                mime_type=prepared.media_type,
                source_locator=f"{path}#model-derivative",
                content=prepared.data,
                source=source,
            )
        )
    transformation = "resized/re-encoded derivative" if prepared.transformed else "bounded original"
    return ToolResult(
        parts=(
            ToolTextPart(
                f"image attachment: {_escape_path(path)} ({prepared.media_type}, "
                f"{len(prepared.data)} model bytes, source_resource_id={source_resource_id!r}); "
                f"the model receives a {transformation} and durable evidence retains the source"
            ),
            attachment,
        ),
        effects=ToolEffects(
            attached_resources=tuple(durable_resources),
            evidence_sources=(
                EvidenceSourceFact(
                    resource_id=source_resource_id,
                    source_type="web_attachment",
                    source_uri=source_resource_id,
                    title=safe_name,
                    attributes=(("resource_kind", "workspace_image"),),
                ),
            ),
        ),
    )


def _render_violations(violations: tuple[str, ...]) -> str:
    shown = [
        _utf8_prefix(_escape_path(path), max_bytes=_VIOLATION_PATH_PREVIEW_BYTES)
        for path in violations[:20]
    ]
    if len(violations) > len(shown):
        shown.append(f"… ({len(violations) - len(shown)} more)")
    return ", ".join(shown)


def agentspace_integrity_refusal(environment: ExecutionEnvironment) -> ToolResult | None:
    """The refusal every tool gives while the workspace is latched, or None while it is sound."""
    violations = environment.integrity_violations
    quota_violation = environment.quota_violation
    if not violations and quota_violation is None:
        return None
    reasons: list[str] = []
    if violations:
        reasons.append(f"unsafe or unreadable workspace entries: {_render_violations(violations)}")
    if quota_violation is not None:
        reasons.append(quota_violation)
    return ToolResult.text(
        f"workspace integrity latched: {'; '.join(reasons)}; external cleanup is required",
        is_error=True,
    )


def find_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="find",
        description="Find workspace paths recursively with fd glob semantics, relative to the "
        "requested search root; hidden paths are included, .git and active ignore rules "
        "are respected, and symlinks are not followed.",
        input_model=FindArgs,
        replay_policy="replayable",
        read_only=True,
        contract_version=2,
    )


def find_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    search_toolchain: SearchToolchain,
    spill: SpillWriter | None = None,
) -> AgentTool:
    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        find_args = cast(FindArgs, args)
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        try:
            root = (
                environment.root if find_args.path == "." else environment.resolve(find_args.path)
            )
            if environment.stat_kind(root) != "directory":
                return ToolResult.text(
                    f"find path is not a directory: {_escape_path(find_args.path)}",
                    is_error=True,
                )
            fd = await search_toolchain.path("fd")
        except (PathRejected, OSError, RuntimeError) as exc:
            return ToolResult.text(str(exc), is_error=True)
        await runtime.emit_update(ToolResult.text("", subject=find_args.pattern))

        argv = [
            fd,
            "--color",
            "never",
            "--hidden",
            "--no-require-git",
            "--exclude",
            ".git",
            "--print0",
        ]
        pattern = find_args.pattern
        if "/" in pattern:
            argv.append("--full-path")
            pattern = f"{glob.escape(str(root))}/{pattern.removeprefix('./')}"
        argv.extend(["--glob", "--", pattern, str(root)])
        collector = _NulPathCollector(root=root)
        try:
            async with scheduler.hold(PathAccess(path=str(root), kind="search")):
                if blocked := agentspace_integrity_refusal(environment):
                    return blocked
                home, tmp = environment.prepare_process_directories()
                completed = await environment.run(
                    argv,
                    env=build_child_environment(home=home, tmp=tmp),
                    cwd=root,
                    on_output=collector.feed,
                )
            collector.finish()
        except (OSError, ValueError) as exc:
            return ToolResult.text(str(exc), is_error=True)
        if completed.returncode != 0:
            return ToolResult.text(
                collector.stderr or f"fd exited with status {completed.returncode}",
                is_error=True,
            )

        ordered = sorted(collector.paths, key=lambda value: (value.casefold(), value))
        truncated = len(ordered) > find_args.limit
        shown = ordered[: find_args.limit]
        body = "\n".join(_escape_path(value) for value in shown) or "(no matches)"
        if truncated:
            body += f"\n[limited to {find_args.limit} matches; more matches exist]"
        body, committed = await preview_or_spill(body, spill=spill, tool="find")
        return ToolResult.text(
            body,
            effects=ToolEffects(committed_outputs=((committed,) if committed is not None else ())),
        )

    return find_declaration().bind(execute)


def ls_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="ls",
        description="List one workspace directory: one sorted level, kind/size/name per entry; "
        "symlinks are listed, never followed. Continue large listings with the opaque cursor.",
        input_model=LsArgs,
        replay_policy="replayable",
        read_only=True,
        contract_version=2,
    )


def ls_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        ls_args = cast(LsArgs, args)
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        try:
            root = environment.root if ls_args.path == "." else environment.resolve(ls_args.path)
            if environment.stat_kind(root) != "directory":
                return ToolResult.text(
                    f"ls path is not a directory: {_escape_path(ls_args.path)}",
                    is_error=True,
                )
            relative = _agentspace_relative_path(environment.root, root)
            await runtime.emit_update(ToolResult.text("", subject=_escape_path(relative)))
            async with scheduler.hold(PathAccess(path=str(root), kind="read")):
                if blocked := agentspace_integrity_refusal(environment):
                    return blocked
                entries = environment.list_directory(root)
        except (PathRejected, OSError) as exc:
            return ToolResult.text(str(exc), is_error=True)
        return _directory_page(
            entries,
            path=relative,
            cursor=ls_args.cursor,
            limit=ls_args.limit,
            tool="ls",
        )

    return ls_declaration().bind(execute)


class _GrepJsonCollector:
    """Incrementally normalize ripgrep's stable JSON-lines event protocol."""

    _MAX_EVENT_CHARS = 4 * 1024 * 1024

    def __init__(
        self,
        *,
        output: StreamingToolOutput,
        workspace_root: Path,
        limit: int,
        max_line_chars: int = 2000,
    ) -> None:
        self._output = output
        self._workspace_root = workspace_root
        self._limit = limit
        self._max_line_chars = max_line_chars
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._buffer = ""
        self._dropping_event = False
        self.matches = 0
        self.parse_errors = 0
        self.truncated = False

    def feed(self, chunk: ProcessChunk) -> None:
        if chunk.stream == "stderr":
            self._output.append(chunk)
            return
        try:
            decoded = self._decoder.decode(chunk.data, final=False)
        except UnicodeDecodeError:
            self.parse_errors += 1
            self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
            self._buffer = ""
            self._dropping_event = True
            return
        if self._dropping_event:
            _discarded, separator, decoded = decoded.partition("\n")
            if not separator:
                return
            self._dropping_event = False
        self._buffer += decoded
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if len(line) > self._MAX_EVENT_CHARS:
                self.parse_errors += 1
            else:
                self._consume(line)
        if len(self._buffer) > self._MAX_EVENT_CHARS:
            self.parse_errors += 1
            self._buffer = ""
            self._dropping_event = True

    def finish(self) -> None:
        try:
            self._buffer += self._decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            self.parse_errors += 1
            self._buffer = ""
        if self._buffer:
            if len(self._buffer) > self._MAX_EVENT_CHARS:
                self.parse_errors += 1
            else:
                self._consume(self._buffer)
            self._buffer = ""

    def _consume(self, line: str) -> None:
        try:
            event = json.loads(line)
            event_type = event["type"]
            data = event["data"]
            if not isinstance(data, dict):
                raise TypeError
        except json.JSONDecodeError, KeyError, TypeError:
            self.parse_errors += 1
            return
        if event_type not in {"match", "context"}:
            return
        if event_type == "match":
            if self.matches >= self._limit:
                self.truncated = True
                return
            self.matches += 1
        elif self.matches >= self._limit and self.truncated:
            return
        try:
            path = self._normalize_path(_json_text(data["path"]))
            line_number = int(data["line_number"])
            source = _json_text(data["lines"]).rstrip("\r\n")
        except KeyError, TypeError, ValueError:
            self.parse_errors += 1
            return
        separator = ":" if event_type == "match" else "-"
        for index, source_line in enumerate(source.splitlines() or [""]):
            clipped = source_line
            if len(clipped) > self._max_line_chars:
                clipped = clipped[: self._max_line_chars] + "…[line truncated]"
            rendered = f"{_escape_path(path)}{separator}{line_number + index}{separator}{clipped}\n"
            self._output.append(ProcessChunk("stdout", rendered.encode("utf-8")))

    def _normalize_path(self, value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            try:
                value = path.relative_to(self._workspace_root).as_posix()
            except ValueError as exc:
                raise ValueError("ripgrep returned a path outside the workspace") from exc
        return value.removeprefix("./") or "."


def _json_text(value: object) -> str:
    if not isinstance(value, dict):
        raise TypeError
    text = value.get("text")
    if isinstance(text, str):
        return text
    encoded = value.get("bytes")
    if isinstance(encoded, str):
        return base64.b64decode(encoded, validate=True).decode("utf-8", errors="replace")
    raise TypeError


class _NulPathCollector:
    """Bound and parse fd's unambiguous NUL-delimited output."""

    _MAX_OUTPUT_BYTES = 64 * 1024 * 1024
    _MAX_STDERR_CHARS = 64 * 1024

    def __init__(self, *, root: Path) -> None:
        self._root = root
        self._stdout = bytearray()
        self._total_stdout = 0
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.paths: list[str] = []
        self.stderr = ""

    async def feed(self, chunk: ProcessChunk) -> None:
        if chunk.stream == "stderr":
            remaining = self._MAX_STDERR_CHARS - len(self.stderr)
            if remaining > 0:
                self.stderr += self._stderr_decoder.decode(chunk.data, final=False)[:remaining]
            return
        self._total_stdout += len(chunk.data)
        if self._total_stdout > self._MAX_OUTPUT_BYTES:
            raise ValueError("fd output exceeded its 64 MiB safety limit")
        self._stdout.extend(chunk.data)
        while (separator := self._stdout.find(0)) >= 0:
            raw = bytes(self._stdout[:separator])
            del self._stdout[: separator + 1]
            self.paths.append(self._normalize(raw))

    def finish(self) -> None:
        self.stderr += self._stderr_decoder.decode(b"", final=True)[
            : self._MAX_STDERR_CHARS - len(self.stderr)
        ]
        if self._stdout:
            raise ValueError("fd returned an unterminated path")

    def _normalize(self, raw: bytes) -> str:
        value = os.fsdecode(raw)
        directory_suffix = "/" if value.endswith("/") else ""
        candidate = Path(value.rstrip("/"))
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(self._root)
            except ValueError as exc:
                raise ValueError("fd returned a path outside the search root") from exc
        normalized = candidate.as_posix().removeprefix("./")
        if not normalized or ".." in candidate.parts:
            raise ValueError("fd returned an unsafe path")
        return normalized + directory_suffix


def _streaming_output(
    tool: str,
    factory: OutputStageFactory | None,
) -> StreamingToolOutput:
    return StreamingToolOutput(
        stage=(factory(tool) if factory is not None else None),
        max_bytes=TOOL_RESULT_MAX_BYTES,
        max_lines=TOOL_RESULT_MAX_LINES,
    )


def _process_notice_reserve(notices: Sequence[str]) -> tuple[int, int]:
    return (
        sum(len(f"\n{notice}".encode()) for notice in notices),
        sum(len(notice.splitlines()) or 1 for notice in notices),
    )


def _stream_result(
    tool: str,
    snapshot: ToolOutputSnapshot,
    *,
    transient: bool = False,
    notices: Sequence[str] = (),
) -> ToolResult:
    details: dict[str, object] = {
        "output_bytes": snapshot.total_bytes,
        "output_lines": snapshot.total_lines,
        "spill_state": "committed"
        if snapshot.receipt
        else "staging"
        if snapshot.truncated
        else "none",
    }
    protected = ""
    prefix = ""
    if snapshot.truncated and not transient:
        if snapshot.receipt is None:
            raise FullOutputUnavailable("oversized process output has no durable spill backing")
        protected = spill_continuation(snapshot.receipt.resource_id)
        prefix = f"{tool} output required a bounded continuation. {protected}\n"
    body = _compose_bounded_process_result(
        prefix=prefix,
        tail=snapshot.text,
        notices=notices,
    )
    return ToolResult.text(
        body,
        details=details,
        protected_text=protected,
        effects=ToolEffects(
            committed_outputs=((snapshot.receipt,) if snapshot.receipt is not None else ())
        ),
    )


def _compose_bounded_process_result(
    *,
    prefix: str,
    tail: str,
    notices: Sequence[str],
) -> str:
    tail_lines = tail.splitlines(keepends=True)

    def compose(drop: int) -> str:
        body = prefix + "".join(tail_lines[drop:])
        for notice in notices:
            body = f"{body}\n{notice}" if body else notice
        return body

    candidate = compose(0)
    if within_result_bounds(candidate):
        return candidate
    low = 1
    high = len(tail_lines)
    while low < high:
        middle = (low + high) // 2
        if within_result_bounds(compose(middle)):
            high = middle
        else:
            low = middle + 1
    candidate = compose(low)
    if not within_result_bounds(candidate):
        raise FullOutputUnavailable("process result framing exceeded its bounded reserve")
    return candidate


def within_result_bounds(text: str) -> bool:
    """Whether a tool result's text fits the byte and line bounds a model is shown."""
    return (
        len(text.encode("utf-8")) <= TOOL_RESULT_MAX_BYTES
        and len(text.splitlines()) <= TOOL_RESULT_MAX_LINES
    )


def _utf8_prefix(text: str, *, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "…"


def head_excerpt(text: str) -> str:
    """The leading lines of ``text`` that fit a result's preview."""
    return _utf8_excerpt(text, preview="head")


def _utf8_excerpt(text: str, *, preview: Literal["head", "tail"]) -> str:
    lines = text.splitlines(keepends=True)
    selected = lines if preview == "head" else list(reversed(lines))
    kept: list[str] = []
    size = 0
    for line in selected:
        line_size = len(line.encode("utf-8"))
        if size + line_size > TOOL_RESULT_PREVIEW_BYTES:
            break
        kept.append(line)
        size += line_size
    if preview == "tail":
        kept.reverse()
    return "".join(kept)


def _agentspace_relative_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return relative or "."


def _inventory_facts(root: object, path: object) -> AgentspaceInventoryFacts:
    from pathlib import Path

    file_path = Path(path)  # type: ignore[arg-type]
    root_path = Path(root)  # type: ignore[arg-type]
    data = file_path.read_bytes()
    record = AgentspacePathFact(
        relative_path=str(file_path.relative_to(root_path)),
        entry_type="file",
        size_bytes=len(data),
        mode=file_path.stat().st_mode,
        content_digest=hashlib.sha256(data).hexdigest(),
    )
    return AgentspaceInventoryFacts(upserts=(record,))


def _scan_inventory(root: object) -> AgentspaceInventoryFacts:
    from pathlib import Path

    root_path = Path(root)  # type: ignore[arg-type]
    upserts: list[AgentspacePathFact] = []
    for current, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [name for name in dirnames if not (Path(current) / name).is_symlink()]
        for name in filenames:
            file_path = Path(current) / name
            if file_path.is_symlink():
                continue
            try:
                metadata = file_path.stat()
            except OSError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            upserts.append(
                AgentspacePathFact(
                    relative_path=str(file_path.relative_to(root_path)),
                    entry_type="file",
                    size_bytes=metadata.st_size,
                    mode=metadata.st_mode,
                )
            )
    return AgentspaceInventoryFacts(upserts=tuple(upserts), replace_all=True)


def _paginate_lines(
    text: str,
    *,
    path: str,
    offset: int | None,
    limit: int | None,
    notice: str = "",
) -> tuple[str, str, int]:
    """Return one page whose complete continuation participates in its bounds."""
    lines = text.splitlines()
    start = (offset or 1) - 1
    if offset is not None and start >= len(lines):
        return (
            f"[offset {offset} is past the end of {path!r}: the file has "
            f"{len(lines)} line{'s' if len(lines) != 1 else ''}]",
            "",
            0,
        )
    page_size = min(limit or TOOL_RESULT_MAX_LINES, TOOL_RESULT_MAX_LINES)
    end = min(start + page_size, len(lines))

    def render(candidate_end: int) -> tuple[str, str, int]:
        continuation = ""
        remaining = len(lines) - candidate_end
        chunks = ["\n".join(lines[start:candidate_end])]
        if notice:
            chunks.append(notice)
        if remaining > 0:
            continuation = f"read(path={path!r}, offset={candidate_end + 1})"
            chunks.append(f"[{remaining} more lines; {continuation}]")
        return "\n".join(chunk for chunk in chunks if chunk), continuation, remaining

    rendered = render(end)
    if within_result_bounds(rendered[0]) or end <= start + 1:
        return rendered

    # Find the largest advancing page that leaves room for its complete notice
    # and continuation. The search examines at most the global line ceiling,
    # regardless of a model-supplied limit or file size.
    low = start + 1
    high = end - 1
    best = low
    while low <= high:
        middle = (low + high) // 2
        candidate = render(middle)
        if within_result_bounds(candidate[0]):
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return render(best)


__all__ = [
    "BashArgs",
    "EditArgs",
    "EditOperation",
    "FindArgs",
    "GrepArgs",
    "HttpReadOptions",
    "ImagePreparer",
    "LsArgs",
    "MaterializeArgs",
    "PreparedImageAttachment",
    "ReadArgs",
    "ReadResourceArgs",
    "RenderedReadResourceArgs",
    "ResourceReadRequest",
    "OutputStageFactory",
    "PATH_MAX_CHARS",
    "ResourceReader",
    "SpillWriter",
    "WriteArgs",
    "bash_tool",
    "edit_tool",
    "find_tool",
    "grep_tool",
    "head_excerpt",
    "ls_tool",
    "materialize_tool",
    "preview_or_spill",
    "read_resource_tool",
    "read_tool",
    "replace_agentspace_file",
    "spill_continuation",
    "within_result_bounds",
    "agentspace_integrity_refusal",
    "write_tool",
]
