# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Path policy, encoding, and atomic writes for LocalExecutionEnvironment."""

from pathlib import Path

import pytest

from dlightrag.agent.environment import LocalExecutionEnvironment, PathRejected
from dlightrag.agent.environment.text import decode_workspace_text, encode_workspace_text


def test_rejects_absolute_parent_and_symlink_escape(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    with pytest.raises(PathRejected):
        env.resolve("/etc/passwd")
    with pytest.raises(PathRejected):
        env.resolve("../secret")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    (tmp_path / "link").symlink_to(outside)
    with pytest.raises(PathRejected):
        env.resolve("link")


def test_write_is_atomic_and_creates_parents(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    target = env.resolve("notes/hello.txt")
    env.write_bytes(target, b"hello")
    assert target.read_text(encoding="utf-8") == "hello"
    leftovers = list(target.parent.glob(".dlightrag-write-*"))
    assert leftovers == []


def test_utf8_and_bom_tagged_utf16_round_trip() -> None:
    utf8 = decode_workspace_text("café\n".encode())
    assert utf8.text == "café\n"
    tagged = decode_workspace_text(b"\xff\xfeh\x00i\x00")
    assert tagged.text == "hi"
    with pytest.raises(ValueError, match="not UTF-8"):
        decode_workspace_text(b"\x80\x81not-utf8")
    crlf = decode_workspace_text(b"a\r\nb\r\n")
    assert crlf.newline == "\r\n"
    assert crlf.text == "a\nb\n"
    assert encode_workspace_text(crlf, "a\nb\n") == b"a\r\nb\r\n"


def test_directory_listing_is_sorted_one_level(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    (tmp_path / "b").mkdir()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    names = [entry.name for entry in env.list_directory(tmp_path)]
    assert names == ["a.txt", "b"]
