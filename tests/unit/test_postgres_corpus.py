# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Focused behavior for the PostgreSQL corpus composition adapter."""

import logging

import pytest

from dlightrag.adapters.postgres.corpus import PGCorpusCoordination
from dlightrag.config import DlightragConfig


class _Connection:
    def __init__(self, *, max_connections: str) -> None:
        self._max_connections = max_connections

    async def fetchval(self, query: str) -> str:
        assert query == "SHOW max_connections"
        return self._max_connections


async def test_connection_budget_warning_is_owned_by_coordination(
    test_config: DlightragConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    coordination = PGCorpusCoordination(
        connection_kwargs=test_config.pg_connection_kwargs(),
        workspace=test_config.workspace,
        reader=False,
        require_halfvec=False,
        required_extensions=(),
        lightrag_pool_max_size=16,
        domain_pool_max_size=10,
        acquire_timeout=test_config.postgres_acquire_timeout,
    )

    with caplog.at_level(logging.WARNING, logger="dlightrag.adapters.postgres.corpus"):
        await coordination._log_connection_budget(_Connection(max_connections="50"))

    assert "PostgreSQL connection budget is tight" in caplog.text
    assert "estimated_pool_connections=52" in caplog.text
