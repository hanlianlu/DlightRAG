# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Minimal behavioural recovery in the file tools: bash cwd fence and past-EOF reads."""

from pathlib import Path

import pytest

from dlightrag.engine.agent.environment.access import (
    AccessScheduler,
)
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.tools.files import BashArgs, ReadArgs, bash_tool, read_tool
from tests.support.tool_helpers import tool_runtime


def _env(root: Path) -> tuple[LocalExecutionEnvironment, AccessScheduler]:
    root.mkdir()
    return LocalExecutionEnvironment(root), AccessScheduler()


@pytest.mark.asyncio
async def test_bash_refuses_a_stale_absolute_cd_and_names_the_current_root(
    tmp_path: Path,
) -> None:
    env, scheduler = _env(tmp_path / "workspace")
    outside = tmp_path / "earlier-round"
    outside.mkdir()

    result = await bash_tool(env, scheduler).execute(
        BashArgs(command=f"cd {outside} && touch escaped.txt"), tool_runtime()
    )

    assert result.is_error is True
    assert str(env.root) in result.text_content
    assert "relative path" in result.text_content
    assert not (outside / "escaped.txt").exists()


@pytest.mark.asyncio
async def test_bash_refuses_pushd_that_climbs_out_of_the_workspace(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path / "workspace")

    result = await bash_tool(env, scheduler).execute(
        BashArgs(command="pushd .. && touch escaped.txt"), tool_runtime()
    )

    assert result.is_error is True
    assert str(env.root) in result.text_content


@pytest.mark.asyncio
async def test_bash_allows_a_relative_cd_inside_the_workspace(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path / "workspace")
    (env.root / "scripts").mkdir()

    result = await bash_tool(env, scheduler).execute(
        BashArgs(command="cd scripts && printf ok > made.txt"), tool_runtime()
    )

    assert result.is_error is False, result.text_content
    assert (env.root / "scripts" / "made.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.asyncio
async def test_read_past_eof_returns_a_notice_with_the_line_count(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path / "workspace")
    (env.root / "note.txt").write_text("one\ntwo\n", encoding="utf-8")

    result = await read_tool(env, scheduler).execute(
        ReadArgs(path="note.txt", offset=10), tool_runtime()
    )

    assert result.is_error is False
    assert "past the end" in result.text_content
    assert "2 lines" in result.text_content
    assert result.text_content != ""


@pytest.mark.parametrize(
    "command",
    [
        "cd ..;touch escaped.txt",
        "cd ..&&touch escaped.txt",
        "if true; then cd ..; touch escaped.txt; fi",
        "cd .. || touch escaped.txt",
        "cd ..\ntouch escaped.txt",
        "command cd ..",
        "FOO=1 cd ..; touch escaped.txt",
        "if cd ..; then touch escaped.txt; fi",
    ],
)
@pytest.mark.asyncio
async def test_bash_fence_sees_command_starts_without_spaces(tmp_path: Path, command: str) -> None:
    env, scheduler = _env(tmp_path / "workspace")
    outside = tmp_path / "earlier-round"
    outside.mkdir()

    result = await bash_tool(env, scheduler).execute(BashArgs(command=command), tool_runtime())

    assert result.is_error is True, command
    assert str(env.root) in result.text_content
    assert not (outside / "escaped.txt").exists()


@pytest.mark.asyncio
async def test_bash_refuses_cd_oldpwd(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path / "workspace")

    result = await bash_tool(env, scheduler).execute(BashArgs(command="cd -"), tool_runtime())

    assert result.is_error is True
    assert "OLDPWD" in result.text_content
    assert str(env.root) in result.text_content


@pytest.mark.asyncio
async def test_bash_fence_allows_the_same_shapes_inside_the_workspace(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path / "workspace")
    (env.root / "scripts").mkdir()

    for command in (
        "cd scripts;printf ok > one.txt",
        "cd scripts&&printf ok > two.txt",
        "if cd scripts; then printf ok > three.txt; fi",
        "FOO=1 cd scripts; printf ok > four.txt",
        "command cd scripts; printf ok > five.txt",
    ):
        result = await bash_tool(env, scheduler).execute(BashArgs(command=command), tool_runtime())
        assert result.is_error is False, (command, result.text_content)

    for name in ("one.txt", "two.txt", "three.txt", "four.txt", "five.txt"):
        assert (env.root / "scripts" / name).is_file(), name


@pytest.mark.asyncio
async def test_bash_fence_does_not_read_quoted_separators_as_commands(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path / "workspace")

    result = await bash_tool(env, scheduler).execute(
        BashArgs(command='printf "%s" "cd ..; touch escaped.txt"'), tool_runtime()
    )

    assert result.is_error is False, result.text_content
    assert "cd ..; touch escaped.txt" in result.text_content
    assert not (tmp_path / "earlier-round" / "escaped.txt").exists()
