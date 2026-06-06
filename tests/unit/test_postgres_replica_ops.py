# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests that PostgreSQL replication stays outside DlightRAG app routing."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_makefile_does_not_expose_app_level_replica_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "postgres-replica-" not in makefile
    assert "--profile replica" not in makefile
    assert "replication-role.sh" not in makefile


def test_compose_does_not_define_app_level_replica_profile() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "postgres-replica:" not in compose
    assert 'profiles: ["replica"]' not in compose
    assert "image: dlightrag-postgres:pg18" in compose
    assert "context: postgres" in compose
    assert "shared_preload_libraries=age,pg_textsearch,pg_jieba" in compose


def test_compose_sets_long_term_postgres_shm_size() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "shm_size: ${DLIGHTRAG_POSTGRES_SHM_SIZE:-8gb}" in compose
    assert compose.count("shm_size:") == 1


def test_repo_does_not_ship_app_level_replica_scripts() -> None:
    scripts = [
        ROOT / "scripts/postgres/replica-env.sh",
        ROOT / "scripts/postgres/replication-role.sh",
        ROOT / "scripts/postgres/replica-entrypoint.sh",
        ROOT / "scripts/postgres/replica-smoke.sh",
        ROOT / "scripts/postgres/replica-reset.sh",
    ]

    assert all(not path.exists() for path in scripts)
