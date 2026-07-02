# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for local Langfuse headless bootstrap."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_script_path = Path(__file__).resolve().parents[2] / "scripts" / "langfuse" / "headless.py"
_spec = importlib.util.spec_from_file_location("langfuse_headless", _script_path)
assert _spec is not None and _spec.loader is not None
_langfuse_headless = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_langfuse_headless)

bootstrap = _langfuse_headless.bootstrap
read_env = _langfuse_headless.read_env


def test_bootstrap_creates_matching_headless_and_dlightrag_env(tmp_path: Path) -> None:
    langfuse_env = tmp_path / "langfuse-local" / ".env"
    dlightrag_env = tmp_path / ".env"

    result = bootstrap(
        langfuse_env=langfuse_env,
        dlightrag_env=dlightrag_env,
        host="http://localhost:3300",
        public_key="pk-test",
        secret_key="sk-test",
    )

    assert result.public_key == "pk-test"
    assert result.langfuse_env == langfuse_env
    assert result.dlightrag_env == dlightrag_env

    langfuse_values = read_env(langfuse_env).values
    assert langfuse_values["LANGFUSE_INIT_ORG_ID"] == "dlight-local-org"
    assert langfuse_values["LANGFUSE_INIT_PROJECT_ID"] == "dlight-local-project"
    assert langfuse_values["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"] == "pk-test"
    assert langfuse_values["LANGFUSE_INIT_PROJECT_SECRET_KEY"] == "sk-test"
    assert langfuse_values["LANGFUSE_INIT_USER_EMAIL"] == "admin@localhost.local"
    assert langfuse_values["LANGFUSE_INIT_USER_PASSWORD"]

    dlightrag_values = read_env(dlightrag_env).values
    assert dlightrag_values["DLIGHTRAG_LANGFUSE_PUBLIC_KEY"] == "pk-test"
    assert dlightrag_values["DLIGHTRAG_LANGFUSE_SECRET_KEY"] == "sk-test"
    assert dlightrag_values["DLIGHTRAG_LANGFUSE_HOST"] == "http://localhost:3300"
    assert dlightrag_values["DLIGHTRAG_LANGFUSE_EXPORT_EXTERNAL_SPANS"] == "false"


def test_bootstrap_reuses_existing_langfuse_init_project_keys(tmp_path: Path) -> None:
    langfuse_env = tmp_path / "langfuse-local" / ".env"
    langfuse_env.parent.mkdir()
    langfuse_env.write_text(
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-existing\n"
        "LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-existing\n",
        encoding="utf-8",
    )
    dlightrag_env = tmp_path / ".env"

    bootstrap(
        langfuse_env=langfuse_env,
        dlightrag_env=dlightrag_env,
        host="http://localhost:3300",
    )

    langfuse_values = read_env(langfuse_env).values
    dlightrag_values = read_env(dlightrag_env).values
    assert langfuse_values["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"] == "pk-existing"
    assert langfuse_values["LANGFUSE_INIT_PROJECT_SECRET_KEY"] == "sk-existing"
    assert dlightrag_values["DLIGHTRAG_LANGFUSE_PUBLIC_KEY"] == "pk-existing"
    assert dlightrag_values["DLIGHTRAG_LANGFUSE_SECRET_KEY"] == "sk-existing"


def test_bootstrap_accepts_headless_keys_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    langfuse_env = tmp_path / "langfuse-local" / ".env"
    dlightrag_env = tmp_path / ".env"
    monkeypatch.setenv("LANGFUSE_INIT_PROJECT_PUBLIC_KEY", "pk-env")
    monkeypatch.setenv("LANGFUSE_INIT_PROJECT_SECRET_KEY", "sk-env")

    bootstrap(
        langfuse_env=langfuse_env,
        dlightrag_env=dlightrag_env,
        host="http://localhost:3300",
    )

    langfuse_values = read_env(langfuse_env).values
    dlightrag_values = read_env(dlightrag_env).values
    assert langfuse_values["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"] == "pk-env"
    assert langfuse_values["LANGFUSE_INIT_PROJECT_SECRET_KEY"] == "sk-env"
    assert dlightrag_values["DLIGHTRAG_LANGFUSE_PUBLIC_KEY"] == "pk-env"
    assert dlightrag_values["DLIGHTRAG_LANGFUSE_SECRET_KEY"] == "sk-env"


def test_bootstrap_preserves_user_choices(tmp_path: Path) -> None:
    langfuse_env = tmp_path / "langfuse-local" / ".env"
    langfuse_env.parent.mkdir()
    langfuse_env.write_text(
        "LANGFUSE_INIT_USER_PASSWORD=already-set\n"
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-existing\n"
        "LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-existing\n",
        encoding="utf-8",
    )
    dlightrag_env = tmp_path / ".env"
    dlightrag_env.write_text(
        "DLIGHTRAG_LANGFUSE_EXPORT_EXTERNAL_SPANS=true\n",
        encoding="utf-8",
    )

    bootstrap(
        langfuse_env=langfuse_env,
        dlightrag_env=dlightrag_env,
        host="http://localhost:3300",
    )

    langfuse_values = read_env(langfuse_env).values
    dlightrag_values = read_env(dlightrag_env).values
    assert langfuse_values["LANGFUSE_INIT_USER_PASSWORD"] == "already-set"
    assert dlightrag_values["DLIGHTRAG_LANGFUSE_EXPORT_EXTERNAL_SPANS"] == "true"
