# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Generic read/write/edit/grep/bash factories over an ExecutionEnvironment."""

from __future__ import annotations

import asyncio
import codecs
import difflib
import fnmatch
import hashlib
import json
import os
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlightrag.agent.environment.access import (
    AccessScheduler,
    PathAccess,
    WorkspaceAccess,
)
from dlightrag.agent.environment.child import build_child_environment
from dlightrag.agent.environment.errors import (
    TOOL_RESULT_MAX_BYTES,
    TOOL_RESULT_MAX_LINES,
    TOOL_RESULT_PREVIEW_BYTES,
    FullOutputUnavailable,
    PathRejected,
    WorkspaceQuotaExceeded,
)
from dlightrag.agent.environment.execution import ExecutionEnvironment
from dlightrag.agent.environment.local import CompletedProcess, DirectoryEntry, ProcessChunk
from dlightrag.agent.environment.text import decode_workspace_text, encode_workspace_text
from dlightrag.agent.tool_content import ToolResourceAttachmentPart, ToolTextPart
from dlightrag.agent.tools.contracts import (
    AgentTool,
    CommittedOutput,
    ResourceAttachmentBytes,
    ToolEffects,
    ToolResult,
    ToolRuntime,
    WorkspaceInventoryFacts,
    WorkspacePathFact,
)
from dlightrag.agent.tools.output import OutputStage, StreamingToolOutput, ToolOutputSnapshot

type ResourceReader = Callable[
    [str, str | None, str | None, ToolRuntime],
    Awaitable[ToolResult],
]
type SpillWriter = Callable[[str], Awaitable[CommittedOutput]]
type OutputStageFactory = Callable[[str], OutputStage]


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str | None = Field(default=None, description="Workspace-relative path to read.")
    resource_id: str | None = Field(default=None, description="Opaque durable resource id.")
    offset: int | None = Field(default=None, ge=1, description="1-based line offset.")
    limit: int | None = Field(default=None, ge=1, description="Maximum lines to return.")
    focus: str | None = Field(
        default=None,
        min_length=1,
        description="Optional relevance focus for a durable resource.",
    )
    cursor: str | None = Field(default=None, description="Continuation cursor.")

    @model_validator(mode="after")
    def _exactly_one_target(self) -> ReadArgs:
        if (self.path is None) == (self.resource_id is None):
            raise ValueError("read requires exactly one of path or resource_id")
        if self.path is not None and self.focus is not None:
            raise ValueError("read focus is available only for resource_id")
        return self


class WriteArgs(BaseModel):
    path: str = Field(description="Workspace-relative path to write.")
    content: str = Field(description="Full UTF-8 file contents.")


class EditOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    old_text: str = Field(min_length=1, description="Unique exact text in the original file.")
    new_text: str = Field(description="Replacement text.")


class EditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str = Field(description="Workspace-relative path to edit.")
    edits: list[EditOperation] = Field(
        min_length=1,
        description="Non-overlapping replacements, all matched against the original file.",
    )


class GrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pattern: str = Field(min_length=1, description="Regex (or literal with literal=true).")
    path: str = Field(default=".", description="Workspace path to search.")
    glob: str | None = Field(default=None, description="Optional glob filter.")
    ignore_case: bool = Field(default=False, description="Case-insensitive matching.")
    literal: bool = Field(
        default=False, description="Treat pattern as a literal string, not a regex."
    )
    context: int | None = Field(
        default=None, ge=0, description="Context lines shown around each match."
    )
    limit: int = Field(default=100, ge=1, description="Maximum matching lines to return.")


class FindArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pattern: str = Field(min_length=1, description="Glob pattern to match.")
    path: str = Field(default=".", description="Workspace subtree to search.")
    limit: int = Field(default=1000, ge=1, description="Maximum matches to return.")


class LsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str = Field(default=".", description="Workspace directory to list.")
    limit: int = Field(default=500, ge=1, description="Maximum entries to return.")


class BashArgs(BaseModel):
    command: str = Field(description="Bash command to run.")
    timeout_seconds: float | None = Field(
        default=None, gt=0, description="Optional process timeout in seconds."
    )


def bound_tool_text(text: str, *, spill: SpillWriter | None) -> str:
    """Apply the unified byte/line guard. Spill if available, else raise."""
    if _within_result_bounds(text):
        return text
    if spill is None:
        raise FullOutputUnavailable("oversized tool result has no spill or cursor backing")
    raise FullOutputUnavailable("spill writer must be awaited by the tool, not bound_tool_text")


async def preview_or_spill(
    text: str,
    *,
    spill: SpillWriter | None,
    tool: str,
    preview: Literal["head", "tail"] = "head",
) -> tuple[str, CommittedOutput | None]:
    """Return (model text, optional committed-spill receipt)."""
    if _within_result_bounds(text):
        return text, None
    if spill is None:
        raise FullOutputUnavailable("oversized tool result has no spill or cursor backing")
    receipt = await spill(text)
    resource_id = receipt.resource_id
    excerpt = _utf8_excerpt(text, preview=preview)
    rendered = (
        f"{tool} output exceeded {TOOL_RESULT_MAX_BYTES} UTF-8 bytes or "
        f"{TOOL_RESULT_MAX_LINES} lines ({len(text.encode('utf-8'))} bytes). "
        f"Full output: read(resource_id={resource_id!r}, cursor=...)\n{excerpt}"
    )
    return rendered, receipt


def path_tools(
    environment: ExecutionEnvironment,
    *,
    scheduler: AccessScheduler,
    ripgrep: str = "rg",
    resource_reader: ResourceReader | None = None,
    spill: SpillWriter | None = None,
    output_stage_factory: OutputStageFactory | None = None,
) -> list[AgentTool]:
    """Return read/write/edit/grep/bash bound to one environment instance."""
    return [
        read_tool(environment, scheduler, resource_reader=resource_reader, spill=spill),
        bash_tool(environment, scheduler, output_stage_factory=output_stage_factory),
        edit_tool(environment, scheduler),
        write_tool(environment, scheduler),
        grep_tool(
            environment,
            scheduler,
            ripgrep=ripgrep,
            output_stage_factory=output_stage_factory,
        ),
        find_tool(environment, scheduler),
        ls_tool(environment, scheduler),
    ]


def read_tool(
    environment: ExecutionEnvironment | None,
    scheduler: AccessScheduler,
    *,
    resource_reader: ResourceReader | None = None,
    spill: SpillWriter | None = None,
) -> AgentTool:
    """Build ``read`` with whichever branches the host actually has."""

    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(ReadArgs, args)
        if args.resource_id is not None:
            if resource_reader is None:
                return ToolResult.text("resource read is not available", is_error=True)
            async with scheduler.hold(PathAccess(path=args.resource_id, kind="read")):
                return await resource_reader(
                    args.resource_id,
                    args.focus,
                    args.cursor,
                    runtime,
                )
        if environment is None or args.path is None:
            return ToolResult.text("path read requires an execution environment", is_error=True)
        if blocked := _integrity_blocked(environment):
            return blocked
        try:
            path = environment.resolve(args.path)
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        async with scheduler.hold(PathAccess(path=str(path), kind="read")):
            kind = environment.stat_kind(path)
            if kind == "directory":
                return ToolResult.text(
                    _render_listing(environment.list_directory(path), args.cursor)
                )
            if kind == "missing":
                return ToolResult.text(f"file not found: {_escape_path(args.path)}", is_error=True)
            raw = environment.read_bytes(path)
            media_type = _sniff_image_media_type(raw)
            if media_type is not None:
                return _image_attachment_result(
                    raw,
                    media_type=media_type,
                    path=args.path,
                )
            try:
                decoded = decode_workspace_text(raw)
            except ValueError as exc:
                return ToolResult.text(str(exc), is_error=True)
            body, continuation, remaining = _paginate_lines(
                decoded.text, path=args.path, offset=args.offset, limit=args.limit
            )
            note = ""
            if decoded.mixed_newlines:
                note = "\n[mixed line endings preserved; not normalized]"
            body, committed = await preview_or_spill(body + note, spill=spill, tool="read")
            if continuation:
                body = f"{body}\n[{remaining} more lines; {continuation}]"
            return ToolResult.text(
                body,
                protected_text=continuation,
                effects=ToolEffects(
                    committed_outputs=((committed,) if committed is not None else ())
                ),
            )

    return AgentTool(
        name="read",
        description="Read a workspace path or a durable resource id.",
        input_model=ReadArgs,
        execute=execute,
        replay_policy="replayable",
        guidance=(
            "read: one of path or resource_id; text pages default to 2000 lines and "
            "carry an offset continuation; follow the printed continuation instead of "
            "re-reading the whole file."
        ),
    )


def write_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = cast(WriteArgs, args)
        if blocked := _integrity_blocked(environment):
            return blocked
        try:
            path = environment.resolve(args.path)
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        async with scheduler.hold(PathAccess(path=str(path), kind="write")):
            try:
                environment.write_bytes(path, args.content.encode("utf-8"))
            except WorkspaceQuotaExceeded as exc:
                return ToolResult.text(str(exc), is_error=True)
            except PathRejected as exc:
                return ToolResult.text(str(exc), is_error=True)
        return ToolResult.text(
            f"wrote {args.path} ({len(args.content.encode('utf-8'))} bytes)",
            effects=ToolEffects(workspace_inventory=_inventory_facts(environment.root, path)),
        )

    return AgentTool(
        name="write",
        description="Create or overwrite a UTF-8 workspace file.",
        input_model=WriteArgs,
        execute=execute,
        replay_policy="never",
        guidance="write: replaces the whole file; the success line reports UTF-8 byte size.",
    )


def edit_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        edit_args = cast(EditArgs, args)
        if blocked := _integrity_blocked(environment):
            return blocked
        try:
            path = environment.resolve(edit_args.path)
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        async with scheduler.hold(PathAccess(path=str(path), kind="readwrite")):
            if environment.stat_kind(path) != "file":
                return ToolResult.text(
                    f"file not found: {_escape_path(edit_args.path)}",
                    is_error=True,
                )
            try:
                decoded = decode_workspace_text(environment.read_bytes(path))
            except ValueError as exc:
                return ToolResult.text(str(exc), is_error=True)
            spans: list[tuple[int, int, str]] = []
            for index, operation in enumerate(edit_args.edits, start=1):
                if operation.old_text == operation.new_text:
                    return ToolResult.text(
                        f"edit {index} rejected: old_text and new_text are identical",
                        is_error=True,
                    )
                count = decoded.text.count(operation.old_text)
                if count != 1:
                    return ToolResult.text(
                        f"edit {index} old_text matches {count} times; each match must be unique",
                        is_error=True,
                    )
                start = decoded.text.index(operation.old_text)
                spans.append((start, start + len(operation.old_text), operation.new_text))
            ordered = sorted(spans)
            if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:], strict=False)):
                return ToolResult.text("edit ranges overlap in the original file", is_error=True)
            updated = decoded.text
            for start, end, replacement in reversed(ordered):
                updated = updated[:start] + replacement + updated[end:]
            try:
                environment.write_bytes(path, encode_workspace_text(decoded, updated))
            except (WorkspaceQuotaExceeded, PathRejected) as exc:
                return ToolResult.text(str(exc), is_error=True)
        patch = "\n".join(
            difflib.unified_diff(
                decoded.text.splitlines(),
                updated.splitlines(),
                fromfile=edit_args.path,
                tofile=edit_args.path,
                lineterm="",
            )
        )
        first_line = decoded.text.count("\n", 0, ordered[0][0]) + 1
        return ToolResult.text(
            f"edited {_escape_path(edit_args.path)} ({len(ordered)} edits; "
            f"first change line {first_line})\n{patch}",
            effects=ToolEffects(workspace_inventory=_inventory_facts(environment.root, path)),
        )

    return AgentTool(
        name="edit",
        description="Replace exact text in a workspace file.",
        input_model=EditArgs,
        execute=execute,
        replay_policy="never",
        guidance=(
            "edit: every old_text must match exactly once in the current file; all edits "
            "apply atomically or none do. Read the file first when a match fails."
        ),
    )


def grep_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    ripgrep: str,
    output_stage_factory: OutputStageFactory | None = None,
) -> AgentTool:
    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        grep_args = cast(GrepArgs, args)
        if blocked := _integrity_blocked(environment):
            return blocked
        try:
            root = (
                environment.root if grep_args.path == "." else environment.resolve(grep_args.path)
            )
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        target = root.relative_to(environment.root).as_posix() if root != environment.root else "."
        argv = [
            ripgrep,
            "--line-number",
            "--no-heading",
            "--hidden",
            "--no-require-git",
            "--glob",
            "!.git",
        ]
        if grep_args.ignore_case:
            argv.append("--ignore-case")
        if grep_args.literal:
            argv.append("--fixed-strings")
        if grep_args.context is not None:
            argv.extend(["--context", str(grep_args.context)])
        if grep_args.glob:
            argv.extend(["--glob", grep_args.glob])
        argv.extend(["-e", grep_args.pattern])
        if root != environment.root:
            argv.append(target)
        output = _streaming_output("grep", output_stage_factory)
        limiter = _GrepLineLimiter(limit=grep_args.limit)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        limit_reached = False
        completed: CompletedProcess | None = None

        async def capture(chunk: ProcessChunk) -> None:
            nonlocal limit_reached
            if limiter.truncated:
                raise _GrepLimitReached()
            kept = limiter.feed(decoder.decode(chunk.data))
            if kept:
                output.append(ProcessChunk("stdout", kept.encode("utf-8")))
            if limiter.truncated:
                # Terminating rg once the limit is reached keeps large trees
                # from being fully scanned; partial context is acceptable.
                raise _GrepLimitReached()

        try:
            async with scheduler.hold(PathAccess(path=str(root), kind="search")):
                home = environment.root / "tmp" / "home"
                tmp = environment.root / "tmp"
                home.mkdir(parents=True, exist_ok=True)
                completed = await environment.run(
                    argv,
                    env=build_child_environment(home=home, tmp=tmp),
                    cwd=environment.root,
                    on_output=capture,
                )
        except _GrepLimitReached:
            limit_reached = True
        returncode = None if completed is None else completed.returncode
        try:
            kept = limiter.feed(decoder.decode(b"", final=True))
            if kept:
                output.append(ProcessChunk("stdout", kept.encode("utf-8")))
            if returncode == 1 and output.snapshot().total_bytes == 0:
                output.append(ProcessChunk("stdout", b"(no matches)"))
            final = await output.finish()
        except asyncio.CancelledError:
            output.abort()
            raise
        except BaseException:
            output.abort()
            raise
        result = _stream_result("grep", final)
        if limiter.truncated:
            marker = f"[limited to {grep_args.limit} matching lines]"
            result = ToolResult.text(
                f"{result.text_content}\n{marker}",
                details=result.details,
                protected_text=result.protected_text,
                is_error=result.is_error,
                effects=result.effects,
            )
        if returncode is not None and returncode not in {0, 1}:
            result = ToolResult.text(
                result.text_content,
                details=result.details,
                protected_text=result.protected_text,
                is_error=True,
                effects=result.effects,
            )
        return result

    return AgentTool(
        name="grep",
        description="Search workspace files with ripgrep.",
        input_model=GrepArgs,
        execute=execute,
        replay_policy="replayable",
        guidance=(
            "grep: regex by default (literal=true for plain text); limit caps matching "
            "lines, not context lines; hidden files are searched while ignore rules apply."
        ),
    )


def bash_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    output_stage_factory: OutputStageFactory | None = None,
) -> AgentTool:
    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(BashArgs, args)
        home = environment.root / "tmp" / "home"
        tmp = environment.root / "tmp"
        home.mkdir(parents=True, exist_ok=True)
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
            async with scheduler.hold(WorkspaceAccess()):
                completed = await environment.run(
                    ["/bin/bash", "-lc", args.command],
                    env=build_child_environment(home=home, tmp=tmp),
                    cwd=environment.root,
                    timeout_seconds=args.timeout_seconds,
                    on_output=capture,
                )
            status = "timeout" if completed.timed_out else f"exit {completed.returncode}"
            output.append(ProcessChunk("stdout", f"\n{status}".encode()))
            violations = environment.refresh_integrity()
            final = await output.finish()
        except asyncio.CancelledError:
            output.abort()
            raise
        except BaseException:
            output.abort()
            raise
        streamed = _stream_result("bash", final)
        failed = completed.timed_out or completed.returncode != 0
        body = streamed.text_content
        if violations:
            failed = True
            listed = ", ".join(_escape_path(path) for path in violations[:20])
            body = (
                f"{body}\nbash left forbidden entries (symlink/FIFO/socket/device): "
                f"{listed}; only bash may remove them"
            )
        return ToolResult.text(
            body,
            details={**(streamed.details or {}), "integrity_violations": list(violations)},
            protected_text=streamed.protected_text,
            is_error=failed,
            effects=ToolEffects(
                committed_outputs=streamed.effects.committed_outputs,
                workspace_inventory=_scan_inventory(environment.root),
            ),
        )

    return AgentTool(
        name="bash",
        description="Run a bash command in the workspace.",
        input_model=BashArgs,
        execute=execute,
        replay_policy="never",
        guidance=(
            "bash: output streams live and stays bounded; timed-out or failing commands "
            "still return partial output as errors. Never leave symlinks, FIFOs, sockets, "
            "or device files behind: the workspace stays blocked until bash removes them."
        ),
    )


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


def _image_attachment_result(
    data: bytes,
    *,
    media_type: str,
    path: str,
) -> ToolResult:
    """Attach one verified original image snapshot to the model-visible result."""
    digest = hashlib.sha256(data).hexdigest()
    resource_id = f"att_{digest[:32]}"
    attachment = ToolResourceAttachmentPart(
        resource_id=resource_id,
        safe_name=path.rsplit("/", 1)[-1] or "image",
        media_type=media_type,
        content_digest=digest,
        size_bytes=len(data),
        data=data,
    )
    return ToolResult(
        parts=(
            ToolTextPart(
                f"image attachment: {_escape_path(path)} ({media_type}, "
                f"{len(data)} bytes, resource_id={resource_id!r}); "
                "the original snapshot is attached to this message"
            ),
            attachment,
        ),
        effects=ToolEffects(
            attached_resources=(
                ResourceAttachmentBytes(
                    resource_id=resource_id,
                    filename=attachment.safe_name,
                    mime_type=media_type,
                    source_locator=path,
                    content=data,
                ),
            )
        ),
    )


def _integrity_blocked(environment: ExecutionEnvironment) -> ToolResult | None:
    violations = environment.integrity_violations
    if not violations:
        return None
    listed = ", ".join(_escape_path(path) for path in violations[:20])
    return ToolResult.text(
        "workspace integrity blocked by forbidden entries left by bash: "
        f"{listed}; remove them with bash before using other tools",
        is_error=True,
    )


def find_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        find_args = cast(FindArgs, args)
        if blocked := _integrity_blocked(environment):
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
            async with scheduler.hold(PathAccess(path=str(root), kind="search")):
                entries = environment.scan_tree(root)
        except (PathRejected, OSError) as exc:
            return ToolResult.text(str(exc), is_error=True)
        prefix = (
            "" if root == environment.root else root.relative_to(environment.root).as_posix() + "/"
        )
        matches: list[str] = []
        for entry in entries:
            relative = entry.relative_path.removeprefix(prefix)
            candidate = relative if "/" in find_args.pattern else relative.rsplit("/", 1)[-1]
            if fnmatch.fnmatchcase(candidate, find_args.pattern):
                matches.append(entry.relative_path)
        matches.sort(key=lambda path: (path.casefold(), path))
        shown = matches[: find_args.limit]
        body = "\n".join(_escape_path(path) for path in shown) or "(no matches)"
        if len(matches) > find_args.limit:
            body += f"\n[limited to {find_args.limit} of {len(matches)} matches]"
        return ToolResult.text(body)

    return AgentTool(
        name="find",
        description="Find workspace paths recursively by glob without following symlinks.",
        input_model=FindArgs,
        execute=execute,
        replay_policy="replayable",
        contract_version=1,
        guidance=(
            "find: glob matched against basenames (or full relative paths when the "
            "pattern contains /); sorted case-insensitively; ignore rules apply."
        ),
    )


def ls_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        ls_args = cast(LsArgs, args)
        if blocked := _integrity_blocked(environment):
            return blocked
        try:
            root = environment.root if ls_args.path == "." else environment.resolve(ls_args.path)
            if environment.stat_kind(root) != "directory":
                return ToolResult.text(
                    f"ls path is not a directory: {_escape_path(ls_args.path)}",
                    is_error=True,
                )
            async with scheduler.hold(PathAccess(path=str(root), kind="read")):
                entries = environment.list_directory(root)
        except (PathRejected, OSError) as exc:
            return ToolResult.text(str(exc), is_error=True)
        shown = entries[: ls_args.limit]
        lines = [f"{entry.kind}\t{entry.size}\t{_escape_path(entry.name)}" for entry in shown]
        if len(entries) > ls_args.limit:
            lines.append(f"[limited to {ls_args.limit} of {len(entries)} entries]")
        return ToolResult.text("\n".join(lines) or "(empty directory)")

    return AgentTool(
        name="ls",
        description="List one workspace directory without following symlinks.",
        input_model=LsArgs,
        execute=execute,
        replay_policy="replayable",
        contract_version=1,
        guidance="ls: one directory level, kind/size/name per entry; symlinks listed, never followed.",
    )


def _escape_path(path: str) -> str:
    return json.dumps(path, ensure_ascii=False)[1:-1]


class _GrepLimitReached(Exception):
    """Raised internally once the match limit is reached so rg is terminated."""


class _GrepLineLimiter:
    """Keep the first N matching lines (plus their context), protect line length."""

    # rg --no-heading emits `path:NUM:content` (or bare `NUM:content` for a
    # single search file) for matches and `path-NUM-content` for context lines.
    _MATCH_LINE_RE = re.compile(r"^(?:.*?:)?\d+:")

    def __init__(self, *, limit: int, max_line_chars: int = 2000) -> None:
        self.limit = limit
        self.max_line_chars = max_line_chars
        self.matches = 0
        self.truncated = False
        self._buffer = ""

    def feed(self, text: str) -> str:
        self._buffer += text
        kept: list[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._keep(line + "\n", kept)
        return "".join(kept)

    def flush(self) -> str:
        if not self._buffer:
            return ""
        kept: list[str] = []
        self._keep(self._buffer, kept)
        self._buffer = ""
        return "".join(kept)

    def _keep(self, raw_line: str, kept: list[str]) -> None:
        body = raw_line.rstrip("\n")
        remainder = raw_line[len(body) :]
        # Matches are counted; context lines and non-rg output pass through
        # uncounted; lines past the limit are dropped and flagged truncated.
        if self._MATCH_LINE_RE.match(body):
            if self.matches >= self.limit:
                self.truncated = True
                return
            self.matches += 1
        kept.append(self._clip(body) + remainder)

    def _clip(self, line: str) -> str:
        if len(line) <= self.max_line_chars:
            return line
        return line[: self.max_line_chars] + "…[line truncated]"


def _streaming_output(
    tool: str,
    factory: OutputStageFactory | None,
) -> StreamingToolOutput:
    return StreamingToolOutput(
        stage=(factory(tool) if factory is not None else None),
        max_bytes=TOOL_RESULT_MAX_BYTES,
        max_lines=TOOL_RESULT_MAX_LINES,
    )


def _stream_result(
    tool: str,
    snapshot: ToolOutputSnapshot,
    *,
    transient: bool = False,
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
    body = snapshot.text
    protected = ""
    if snapshot.truncated and not transient:
        if snapshot.receipt is None:
            raise FullOutputUnavailable("oversized process output has no durable spill backing")
        receipt = snapshot.receipt
        resource_id = receipt.resource_id
        protected = f"Full output: read(resource_id={resource_id!r}, cursor=...)"
        body = (
            f"{tool} output exceeded {TOOL_RESULT_MAX_BYTES} UTF-8 bytes or "
            f"{TOOL_RESULT_MAX_LINES} lines. {protected}\n{body}"
        )
    return ToolResult.text(
        body,
        details=details,
        protected_text=protected,
        effects=ToolEffects(
            committed_outputs=((snapshot.receipt,) if snapshot.receipt is not None else ())
        ),
    )


def _within_result_bounds(text: str) -> bool:
    return (
        len(text.encode("utf-8")) <= TOOL_RESULT_MAX_BYTES
        and len(text.splitlines()) <= TOOL_RESULT_MAX_LINES
    )


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


def _inventory_facts(root: object, path: object) -> WorkspaceInventoryFacts:
    from pathlib import Path

    file_path = Path(path)  # type: ignore[arg-type]
    root_path = Path(root)  # type: ignore[arg-type]
    data = file_path.read_bytes()
    record = WorkspacePathFact(
        relative_path=str(file_path.relative_to(root_path)),
        entry_type="file",
        size_bytes=len(data),
        mode=file_path.stat().st_mode,
        content_digest=hashlib.sha256(data).hexdigest(),
    )
    return WorkspaceInventoryFacts(upserts=(record,))


def _scan_inventory(root: object) -> WorkspaceInventoryFacts:
    from pathlib import Path

    root_path = Path(root)  # type: ignore[arg-type]
    upserts: list[WorkspacePathFact] = []
    for current, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [name for name in dirnames if not (Path(current) / name).is_symlink()]
        for name in filenames:
            file_path = Path(current) / name
            if file_path.is_symlink():
                continue
            try:
                stat = file_path.stat()
            except OSError:
                continue
            upserts.append(
                WorkspacePathFact(
                    relative_path=str(file_path.relative_to(root_path)),
                    entry_type="file",
                    size_bytes=stat.st_size,
                    mode=stat.st_mode,
                )
            )
    return WorkspaceInventoryFacts(upserts=tuple(upserts), replace_all=True)


def _paginate_lines(
    text: str,
    *,
    path: str,
    offset: int | None,
    limit: int | None,
) -> tuple[str, str, int]:
    """Return one bounded page, its continuation call, and remaining lines."""
    lines = text.splitlines()
    start = (offset or 1) - 1
    page_size = limit if limit is not None else TOOL_RESULT_MAX_LINES
    end = min(start + page_size, len(lines))
    body = "\n".join(lines[start:end])
    if end < len(lines):
        return (
            body,
            f"read(path={_escape_path(path)!r}, offset={end + 1})",
            len(lines) - end,
        )
    return body, "", 0


def _render_listing(entries: Sequence[DirectoryEntry], cursor: str | None) -> str:
    entries = list(entries)
    start = int(cursor) if cursor and cursor.isdigit() else 0
    page = entries[start : start + 500]
    lines = [f"{entry.kind}\t{entry.size}\t{_escape_path(entry.name)}" for entry in page]
    if start + 500 < len(entries):
        lines.append(f"[{len(entries) - start - 500} more entries; cursor={start + 500}]")
    return "\n".join(lines) or "(empty directory)"


__all__ = [
    "BashArgs",
    "EditArgs",
    "EditOperation",
    "FindArgs",
    "GrepArgs",
    "LsArgs",
    "ReadArgs",
    "OutputStageFactory",
    "ResourceReader",
    "SpillWriter",
    "WriteArgs",
    "bash_tool",
    "bound_tool_text",
    "edit_tool",
    "find_tool",
    "grep_tool",
    "ls_tool",
    "path_tools",
    "preview_or_spill",
    "read_tool",
    "write_tool",
]
