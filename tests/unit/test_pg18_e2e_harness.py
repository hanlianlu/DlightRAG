# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in PG18 E2E smoke harness."""

from __future__ import annotations

from tests.e2e.pg18_harness import (
    REQUIRED_EXTENSIONS,
    e2e_enabled,
    missing_preload_libraries,
    pg_conn_kwargs_from_env,
)


def test_pg18_harness_is_opt_in() -> None:
    assert e2e_enabled({}) is False
    assert e2e_enabled({"DLIGHTRAG_RUN_E2E_PG18": "0"}) is False
    assert e2e_enabled({"DLIGHTRAG_RUN_E2E_PG18": "true"}) is True


def test_pg18_harness_connection_env_prefers_e2e_namespace() -> None:
    env = {
        "DLIGHTRAG_POSTGRES_HOST": "primary",
        "DLIGHTRAG_POSTGRES_PORT": "5432",
        "DLIGHTRAG_E2E_POSTGRES_HOST": "localhost",
        "DLIGHTRAG_E2E_POSTGRES_PORT": "55432",
        "DLIGHTRAG_E2E_POSTGRES_USER": "e2e_user",
        "DLIGHTRAG_E2E_POSTGRES_PASSWORD": "e2e_pass",
        "DLIGHTRAG_E2E_POSTGRES_DATABASE": "e2e_db",
    }

    assert pg_conn_kwargs_from_env(env) == {
        "host": "localhost",
        "port": 55432,
        "user": "e2e_user",
        "password": "e2e_pass",
        "database": "e2e_db",
    }


def test_pg18_harness_tracks_required_postgres_extensions() -> None:
    assert REQUIRED_EXTENSIONS == ("vector", "age", "pg_textsearch", "pg_jieba")


def test_pg18_harness_validates_preloaded_libraries() -> None:
    assert missing_preload_libraries("age, pg_textsearch, pg_jieba") == []
    assert missing_preload_libraries("pg_stat_statements") == [
        "age",
        "pg_textsearch",
        "pg_jieba",
    ]
