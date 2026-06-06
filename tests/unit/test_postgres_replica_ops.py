# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for optional PostgreSQL 18 read-replica operations."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_makefile_exposes_replica_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "\npostgres-replica-prepare:\n\tscripts/postgres/replication-role.sh\n" in makefile
    assert (
        "\npostgres-replica-start: postgres-replica-prepare\n"
        "\tdocker compose --profile replica up -d postgres-replica\n"
    ) in makefile
    assert "\npostgres-replica-smoke:\n\tscripts/postgres/replica-smoke.sh\n" in makefile
    assert "\npostgres-replica-reset:\n\tscripts/postgres/replica-reset.sh\n" in makefile


def test_compose_has_pg18_replica_profile() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "postgres-replica:" in compose
    assert 'profiles: ["replica"]' in compose
    assert "image: dlightrag-postgres:pg18" in compose
    assert "context: postgres" in compose
    assert "shared_preload_libraries=age,pg_textsearch" in compose
    assert "wal_level=replica" in compose
    assert "max_replication_slots=10" in compose


def test_compose_sets_long_term_postgres_shm_size() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "shm_size: ${DLIGHTRAG_POSTGRES_SHM_SIZE:-8gb}" in compose
    assert "shm_size: 8gb" in compose
    assert compose.count("shm_size:") == 2


def test_postgres_replica_scripts_are_syntax_valid() -> None:
    scripts = [
        ROOT / "scripts/postgres/replica-env.sh",
        ROOT / "scripts/postgres/replication-role.sh",
        ROOT / "scripts/postgres/replica-entrypoint.sh",
        ROOT / "scripts/postgres/replica-smoke.sh",
        ROOT / "scripts/postgres/replica-reset.sh",
    ]

    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)


def test_replication_role_exports_values_for_python_bridge() -> None:
    script = (ROOT / "scripts/postgres/replication-role.sh").read_text(encoding="utf-8")

    assert "export DLIGHTRAG_POSTGRES_REPLICATION_USER" in script
    assert "export DLIGHTRAG_POSTGRES_REPLICATION_PASSWORD" in script
