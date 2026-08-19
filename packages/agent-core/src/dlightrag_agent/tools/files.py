# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Generic read/write/edit/grep/bash factories over an ExecutionEnvironment."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Literal, cast

from pydantic import BaseModel, Field, model_validator

from dlightrag_agent.environment.access import (
    AccessScheduler,
    AllAccess,
    PathAccess,
)
from dlightrag_agent.environment.child import build_child_environment
from dlightrag_agent.environment.errors import (
    TOOL_RESULT_CHAR_LIMIT,
    TOOL_RESULT_PREVIEW_CHARS,
    FullOutputUnavailable,
    PathRejected,
    WorkspaceQuotaExceeded,
)
from dlightrag_agent.environment.protocol import DirectoryEntry, ExecutionEnvironment
from dlightrag_agent.environment.text import decode_workspace_text, encode_workspace_text
from dlightrag_agent.tools.contracts import AgentTool, ToolResult

type ResourceReader = Callable[[str, str | None], Awaitable[str]]
type SpillWriter = Callable[[str], Awaitable[Mapping[str, object]]]


class ReadArgs(BaseModel):
    path: str | None = Field(default=None, description="Workspace-relative path to read.")
    resource_id: str | None = Field(default=None, description="Opaque durable resource id.")
    offset: int | None = Field(default=None, ge=1, description="1-based line offset.")
    limit: int | None = Field(default=None, ge=1, description="Maximum lines to return.")
    cursor: str | None = Field(default=None, description="Continuation cursor.")

    @model_validator(mode="after")
    def _exactly_one_target(self) -> ReadArgs:
        if (self.path is None) == (self.resource_id is None):
            raise ValueError("read requires exactly one of path or resource_id")
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
    """Apply the 50k/2k generic guard. Spill if a writer exists, else raise."""
    if len(text) <= TOOL_RESULT_CHAR_LIMIT:
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
) -> tuple[str, dict[str, object] | None]:
    """Return (model text, optional committed-spill receipt)."""
    if len(text) <= TOOL_RESULT_CHAR_LIMIT:
        return text, None
    if spill is None:
        raise FullOutputUnavailable("oversized tool result has no spill or cursor backing")
    receipt = dict(await spill(text))
    resource_id = str(receipt["resource_id"])
    excerpt = (
        text[:TOOL_RESULT_PREVIEW_CHARS] if preview == "head" else text[-TOOL_RESULT_PREVIEW_CHARS:]
    )
    rendered = (
        f"{tool} output exceeded {TOOL_RESULT_CHAR_LIMIT} characters "
        f"({len(text)} chars). Full output: read(resource_id={resource_id!r}, cursor=...)\n"
        f"{excerpt}"
    )
    return rendered, {"committed_spill": receipt}


def path_tools(
    environment: ExecutionEnvironment,
    *,
    scheduler: AccessScheduler,
    ripgrep: str = "rg",
    resource_reader: ResourceReader | None = None,
    spill: SpillWriter | None = None,
) -> list[AgentTool]:
    """Return read/write/edit/grep/bash bound to one environment instance."""
    return [
        read_tool(environment, scheduler, resource_reader=resource_reader, spill=spill),
        write_tool(environment, scheduler),
        edit_tool(environment, scheduler),
        grep_tool(environment, scheduler, ripgrep=ripgrep, spill=spill),
        bash_tool(environment, scheduler, spill=spill),
    ]


def read_tool(
    environment: ExecutionEnvironment | None,
    scheduler: AccessScheduler,
    *,
    resource_reader: ResourceReader | None = None,
    spill: SpillWriter | None = None,
) -> AgentTool:
    """Build ``read`` with whichever branches the host actually has."""

    async def execute(args: BaseModel) -> ToolResult:
        args = cast(ReadArgs, args)
        if args.resource_id is not None:
            if resource_reader is None:
                return ToolResult(content="resource read is not available")
            async with scheduler.hold(PathAccess(path=args.resource_id, kind="read")):
                text = await resource_reader(args.resource_id, args.cursor)
            return ToolResult(content=text)
        if environment is None or args.path is None:
            return ToolResult(content="path read requires an execution environment")
        path = environment.resolve(args.path)
        async with scheduler.hold(PathAccess(path=str(path), kind="read")):
            kind = environment.stat_kind(path)
            if kind == "directory":
                return ToolResult(
                    content=_render_listing(environment.list_directory(path), args.cursor)
                )
            if kind == "missing":
                return ToolResult(content=f"file not found: {args.path}")
            raw = environment.read_bytes(path)
            try:
                decoded = decode_workspace_text(raw)
            except ValueError as exc:
                return ToolResult(content=str(exc))
            text = _slice_lines(decoded.text, offset=args.offset, limit=args.limit)
            note = ""
            if decoded.mixed_newlines:
                note = "\n[mixed line endings preserved; not normalized]"
            body, extra = await preview_or_spill(text + note, spill=spill, tool="read")
            return ToolResult(content=body, details=extra)

    return AgentTool(
        name="read",
        description="Read a workspace path or a durable resource id.",
        input_model=ReadArgs,
        execute=execute,
        replay_policy="safe",
    )


def write_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel) -> ToolResult:
        args = cast(WriteArgs, args)
        path = environment.resolve(args.path)
        async with scheduler.hold(PathAccess(path=str(path), kind="write")):
            try:
                environment.write_bytes(path, args.content.encode("utf-8"))
            except WorkspaceQuotaExceeded as exc:
                return ToolResult(content=str(exc))
            except PathRejected as exc:
                return ToolResult(content=str(exc))
        return ToolResult(
            content=f"wrote {args.path} ({len(args.content)} chars)",
            details=_inventory_details(environment.root, path),
        )

    return AgentTool(
        name="write",
        description="Create or overwrite a UTF-8 workspace file.",
        input_model=WriteArgs,
        execute=execute,
        replay_policy="never",
    )


def edit_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel) -> ToolResult:
        args = cast(EditArgs, args)
        if args.old_string == args.new_string:
            return ToolResult(content="edit rejected: old_string and new_string are identical")
        path = environment.resolve(args.path)
        async with scheduler.hold(PathAccess(path=str(path), kind="readwrite")):
            if environment.stat_kind(path) != "file":
                return ToolResult(content=f"file not found: {args.path}")
            decoded = decode_workspace_text(environment.read_bytes(path))
            count = decoded.text.count(args.old_string)
            if count == 0:
                return ToolResult(content="old_string not found; re-read the file")
            if count > 1 and not args.replace_all:
                return ToolResult(content=f"old_string matches {count} times; set replace_all=true")
            updated = decoded.text.replace(args.old_string, args.new_string)
            try:
                environment.write_bytes(path, encode_workspace_text(decoded, updated))
            except WorkspaceQuotaExceeded as exc:
                return ToolResult(content=str(exc))
        return ToolResult(
            content=f"edited {args.path} ({count} replacement(s))",
            details=_inventory_details(environment.root, path),
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
    spill: SpillWriter | None = None,
) -> AgentTool:
    async def execute(args: BaseModel) -> ToolResult:
        args = cast(GrepArgs, args)
        root = environment.resolve(args.path) if args.path != "." else environment.root
        argv = [ripgrep, "--line-number", "--no-heading", "-e", args.pattern]
        if args.glob:
            argv.extend(["--glob", args.glob])
        argv.append(str(root))
        async with scheduler.hold(PathAccess(path=str(root), kind="search")):
            home = environment.root / "tmp" / "home"
            tmp = environment.root / "tmp"
            home.mkdir(parents=True, exist_ok=True)
            completed = await environment.run(
                argv, env=build_child_environment(home=home, tmp=tmp), cwd=environment.root
            )
        text = completed.stdout if completed.returncode in {0, 1} else completed.stderr
        body, extra = await preview_or_spill(text or "(no matches)", spill=spill, tool="grep")
        return ToolResult(content=body, details=extra)

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
    spill: SpillWriter | None = None,
) -> AgentTool:
    async def execute(args: BaseModel) -> ToolResult:
        args = cast(BashArgs, args)
        home = environment.root / "tmp" / "home"
        tmp = environment.root / "tmp"
        home.mkdir(parents=True, exist_ok=True)
        async with scheduler.hold(AllAccess()):
            completed = await environment.run(
                ["/bin/bash", "-lc", args.command],
                env=build_child_environment(home=home, tmp=tmp),
                cwd=environment.root,
                timeout_seconds=args.timeout_seconds,
            )
        body = completed.stdout
        if completed.stderr:
            body = f"{body}\n{completed.stderr}".strip()
        suffix = f"\nexit {completed.returncode}"
        body, extra = await preview_or_spill(
            body + suffix, spill=spill, tool="bash", preview="tail"
        )
        details = _scan_inventory(environment.root)
        if extra:
            details = {**details, **extra}
        return ToolResult(content=body, details=details)

    return AgentTool(
        name="bash",
        description="Run a bash command in the workspace.",
        input_model=BashArgs,
        execute=execute,
        replay_policy="never",
    )


def _inventory_details(root: object, path: object) -> dict[str, object]:
    from pathlib import Path

    file_path = Path(path)  # type: ignore[arg-type]
    root_path = Path(root)  # type: ignore[arg-type]
    data = file_path.read_bytes()
    record = {
        "relative_path": str(file_path.relative_to(root_path)),
        "entry_type": "file",
        "size_bytes": len(data),
        "mode": file_path.stat().st_mode,
        "content_digest": hashlib.sha256(data).hexdigest(),
    }
    return {"workspace_inventory": {"replace_all": False, "upserts": [record], "deletes": []}}


def _scan_inventory(root: object) -> dict[str, object]:
    from pathlib import Path

    root_path = Path(root)  # type: ignore[arg-type]
    upserts: list[dict[str, object]] = []
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
                {
                    "relative_path": str(file_path.relative_to(root_path)),
                    "entry_type": "file",
                    "size_bytes": stat.st_size,
                    "mode": stat.st_mode,
                    "content_digest": None,
                }
            )
    return {"workspace_inventory": {"replace_all": True, "upserts": upserts, "deletes": []}}


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
