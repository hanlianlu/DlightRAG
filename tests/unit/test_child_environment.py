# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Functional child env: PATH works; seeded secrets do not appear."""

from pathlib import Path

from dlightrag.agent.environment import build_child_environment, looks_like_secret_name


def test_child_env_keeps_path_and_drops_seeded_secrets(tmp_path: Path) -> None:
    parent = {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "SSH_AUTH_SOCK": "/tmp/ssh",
        "DLIGHTRAG_STORAGE__POSTGRES__PASSWORD": "secret",
        "POSTGRES_PASSWORD": "secret",
        "OPENAI_API_KEY": "sk-test",
        "AWS_SECRET_ACCESS_KEY": "aws",
        "PYTHONPATH": "/evil",
        "HTTP_PROXY": "http://user:pass@proxy.example:8080",
        "HTTPS_PROXY": "http://proxy.example:8080",
    }
    env = build_child_environment(home=tmp_path / "home", tmp=tmp_path / "tmp", parent=parent)
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == str(tmp_path / "home")
    assert env["TERM"] == "dumb"
    assert "SSH_AUTH_SOCK" not in env
    assert "DLIGHTRAG_STORAGE__POSTGRES__PASSWORD" not in env
    assert "POSTGRES_PASSWORD" not in env
    assert "OPENAI_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "PYTHONPATH" not in env
    assert "HTTP_PROXY" not in env
    assert env["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert looks_like_secret_name("OPENAI_API_KEY")
    assert not looks_like_secret_name("PATH")
