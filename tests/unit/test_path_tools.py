# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Path tool behavior: edit matches, directory read, oversized guard, bash env."""

from pathlib import Path

import pytest

from dlightrag.agent.environment import (
    AccessScheduler,
    FullOutputUnavailable,
    LocalExecutionEnvironment,
)
from dlightrag.agent.tools.files import (
    BashArgs,
    EditArgs,
    ReadArgs,
    WriteArgs,
    bash_tool,
    edit_tool,
    preview_or_spill,
    read_tool,
    write_tool,
)


def _env(tmp_path: Path) -> tuple[LocalExecutionEnvironment, AccessScheduler]:
    return LocalExecutionEnvironment(tmp_path), AccessScheduler()


@pytest.mark.asyncio
async def test_edit_zero_multi_and_noop(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    writer = write_tool(env, scheduler)
    editor = edit_tool(env, scheduler)
    await writer.execute(WriteArgs(path="note.txt", content="alpha beta alpha"))
    zero = await editor.execute(EditArgs(path="note.txt", old_string="missing", new_string="x"))
    assert "not found" in zero.content
    multi = await editor.execute(EditArgs(path="note.txt", old_string="alpha", new_string="A"))
    assert "matches 2 times" in multi.content
    noop = await editor.execute(EditArgs(path="note.txt", old_string="alpha", new_string="alpha"))
    assert "identical" in noop.content
    done = await editor.execute(
        EditArgs(path="note.txt", old_string="alpha", new_string="A", replace_all=True)
    )
    assert "2 replacement" in done.content
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "A beta A"


@pytest.mark.asyncio
async def test_directory_read_is_sorted(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    (tmp_path / "z").mkdir()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    reader = read_tool(env, scheduler)
    result = await reader.execute(ReadArgs(path="."))
    assert result.content.index("a.txt") < result.content.index("z")


@pytest.mark.asyncio
async def test_oversized_result_without_spill_raises() -> None:
    with pytest.raises(FullOutputUnavailable):
        await preview_or_spill("x" * 50_001, spill=None, tool="read")


@pytest.mark.asyncio
async def test_oversized_result_with_spill_returns_receipt() -> None:
    async def spill(text: str) -> dict[str, object]:
        return {"resource_id": "spill_1", "content_digest": "a" * 64, "size_bytes": len(text)}

    body, extra = await preview_or_spill("x" * 50_001, spill=spill, tool="grep")
    assert "spill_1" in body
    assert extra is not None
    receipt = extra["committed_spill"]
    assert isinstance(receipt, dict)
    assert receipt["resource_id"] == "spill_1"


@pytest.mark.asyncio
async def test_write_attaches_inventory_details(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    writer = write_tool(env, scheduler)
    result = await writer.execute(WriteArgs(path="notes.md", content="hello"))
    assert result.details is not None
    inventory = result.details["workspace_inventory"]
    assert isinstance(inventory, dict)
    assert inventory["replace_all"] is False
    upserts = inventory["upserts"]
    assert isinstance(upserts, list)
    first = upserts[0]
    assert isinstance(first, dict)
    assert first["relative_path"] == "notes.md"
    assert inventory["upserts"][0]["size_bytes"] == 5


@pytest.mark.asyncio
async def test_grep_uses_argv_not_a_shell(tmp_path: Path) -> None:
    fake = tmp_path / "rg"
    fake.write_text(
        "#!/bin/sh\nprintf '%s\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env, scheduler = _env(tmp_path)
    (tmp_path / "hit.txt").write_text("needle", encoding="utf-8")
    from dlightrag.agent.tools.files import GrepArgs, grep_tool

    tool = grep_tool(env, scheduler, ripgrep=str(fake))
    result = await tool.execute(GrepArgs(pattern="needle"))
    assert "--line-number" in result.content
    assert "needle" in result.content


@pytest.mark.asyncio
async def test_bash_does_not_inherit_seeded_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("DLIGHTRAG_POSTGRES_PASSWORD", "pw")
    env, scheduler = _env(tmp_path)
    tool = bash_tool(env, scheduler)
    result = await tool.execute(
        BashArgs(command="printf '%s' \"$OPENAI_API_KEY$DLIGHTRAG_POSTGRES_PASSWORD\"")
    )
    assert "sk-secret" not in result.content
    assert "pw" not in result.content
