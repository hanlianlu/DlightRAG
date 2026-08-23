# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Generic read/write/edit/grep/bash factories over an ExecutionEnvironment."""

from __future__ import annotations

import asyncio
import hashlib
import os
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
from dlightrag.agent.environment.local import DirectoryEntry, ProcessChunk
from dlightrag.agent.environment.text import decode_workspace_text, encode_workspace_text
from dlightrag.agent.tools.contracts import (
    AgentTool,
    CommittedOutput,
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


class EditArgs(BaseModel):
    path: str = Field(description="Workspace-relative path to edit.")
    old_string: str = Field(description="Exact text to replace.")
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(default=False, description="Replace every match.")


class GrepArgs(BaseModel):
    pattern: str = Field(description="ripgrep pattern.")
    path: str = Field(default=".", description="Workspace path to search.")
    glob: str | None = Field(default=None, description="Optional glob filter.")


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
        write_tool(environment, scheduler),
        edit_tool(environment, scheduler),
        grep_tool(
            environment,
            scheduler,
            ripgrep=ripgrep,
            output_stage_factory=output_stage_factory,
        ),
        bash_tool(environment, scheduler, output_stage_factory=output_stage_factory),
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
                return ToolResult.text("resource read is not available")
            async with scheduler.hold(PathAccess(path=args.resource_id, kind="read")):
                return await resource_reader(
                    args.resource_id,
                    args.focus,
                    args.cursor,
                    runtime,
                )
        if environment is None or args.path is None:
            return ToolResult.text("path read requires an execution environment")
        path = environment.resolve(args.path)
        async with scheduler.hold(PathAccess(path=str(path), kind="read")):
            kind = environment.stat_kind(path)
            if kind == "directory":
                return ToolResult.text(
                    _render_listing(environment.list_directory(path), args.cursor)
                )
            if kind == "missing":
                return ToolResult.text(f"file not found: {args.path}")
            raw = environment.read_bytes(path)
            try:
                decoded = decode_workspace_text(raw)
            except ValueError as exc:
                return ToolResult.text(str(exc))
            text = _slice_lines(decoded.text, offset=args.offset, limit=args.limit)
            note = ""
            if decoded.mixed_newlines:
                note = "\n[mixed line endings preserved; not normalized]"
            body, committed = await preview_or_spill(text + note, spill=spill, tool="read")
            return ToolResult.text(
                body,
                effects=ToolEffects(
                    committed_outputs=((committed,) if committed is not None else ())
                ),
            )

    return AgentTool(
        name="read",
        description="Read a workspace path or a durable resource id.",
        input_model=ReadArgs,
        execute=execute,
        replay_policy="safe",
    )


def write_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = cast(WriteArgs, args)
        path = environment.resolve(args.path)
        async with scheduler.hold(PathAccess(path=str(path), kind="write")):
            try:
                environment.write_bytes(path, args.content.encode("utf-8"))
            except WorkspaceQuotaExceeded as exc:
                return ToolResult.text(str(exc))
            except PathRejected as exc:
                return ToolResult.text(str(exc))
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
    )


def edit_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = cast(EditArgs, args)
        if args.old_string == args.new_string:
            return ToolResult.text("edit rejected: old_string and new_string are identical")
        path = environment.resolve(args.path)
        async with scheduler.hold(PathAccess(path=str(path), kind="readwrite")):
            if environment.stat_kind(path) != "file":
                return ToolResult.text(f"file not found: {args.path}")
            decoded = decode_workspace_text(environment.read_bytes(path))
            count = decoded.text.count(args.old_string)
            if count == 0:
                return ToolResult.text("old_string not found; re-read the file")
            if count > 1 and not args.replace_all:
                return ToolResult.text(f"old_string matches {count} times; set replace_all=true")
            updated = decoded.text.replace(args.old_string, args.new_string)
            try:
                environment.write_bytes(path, encode_workspace_text(decoded, updated))
            except WorkspaceQuotaExceeded as exc:
                return ToolResult.text(str(exc))
        return ToolResult.text(
            f"edited {args.path} ({count} replacement(s))",
            effects=ToolEffects(workspace_inventory=_inventory_facts(environment.root, path)),
        )

    return AgentTool(
        name="edit",
        description="Replace exact text in a workspace file.",
        input_model=EditArgs,
        execute=execute,
        replay_policy="never",
    )


def grep_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    ripgrep: str,
    output_stage_factory: OutputStageFactory | None = None,
) -> AgentTool:
    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(GrepArgs, args)
        root = environment.resolve(args.path) if args.path != "." else environment.root
        argv = [ripgrep, "--line-number", "--no-heading", "-e", args.pattern]
        if args.glob:
            argv.extend(["--glob", args.glob])
        argv.append(str(root))
        output = _streaming_output("grep", output_stage_factory)

        async def capture(chunk: ProcessChunk) -> None:
            output.append(chunk)

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
            if completed.returncode == 1 and output.snapshot().total_bytes == 0:
                output.append(ProcessChunk("stdout", b"(no matches)"))
            final = await output.finish()
        except asyncio.CancelledError:
            output.abort()
            raise
        except BaseException:
            output.abort()
            raise
        result = _stream_result("grep", final)
        if completed.returncode not in {0, 1}:
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
        replay_policy="safe",
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
            final = await output.finish()
        except asyncio.CancelledError:
            output.abort()
            raise
        except BaseException:
            output.abort()
            raise
        streamed = _stream_result("bash", final)
        failed = completed.timed_out or completed.returncode != 0
        return ToolResult.text(
            streamed.text_content,
            details=streamed.details,
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
    )


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


def _slice_lines(text: str, *, offset: int | None, limit: int | None) -> str:
    if offset is None and limit is None:
        return text
    lines = text.splitlines()
    start = (offset or 1) - 1
    end = start + limit if limit is not None else len(lines)
    return "\n".join(lines[start:end])


def _render_listing(entries: Sequence[DirectoryEntry], cursor: str | None) -> str:
    start = int(cursor) if cursor and cursor.isdigit() else 0
    page = list(entries)[start : start + 200]
    lines = [f"{entry.kind}\t{entry.size}\t{entry.name}" for entry in page]
    if start + 200 < len(list(entries)):
        lines.append(f"cursor={start + 200}")
    return "\n".join(lines) or "(empty directory)"


__all__ = [
    "BashArgs",
    "EditArgs",
    "GrepArgs",
    "ReadArgs",
    "OutputStageFactory",
    "ResourceReader",
    "SpillWriter",
    "WriteArgs",
    "bash_tool",
    "bound_tool_text",
    "edit_tool",
    "grep_tool",
    "path_tools",
    "preview_or_spill",
    "read_tool",
    "write_tool",
]
