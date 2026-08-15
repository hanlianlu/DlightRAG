# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Contract tests for durable Answer run storage that need no database."""

import re

import pytest

from dlightrag.storage.answer_runs import (
    ANSWER_RUN_HEARTBEAT_SECONDS,
    ANSWER_RUN_LEASE_SECONDS,
    ANSWER_RUN_MIGRATION_SCOPE,
    ANSWER_RUN_MIGRATIONS,
    MAX_CONSECUTIVE_RECOVERIES,
    RUN_ABANDONED_ERROR_KIND,
    RUN_RETENTION_SECONDS,
    PGAnswerRunStore,
)


def _all_statements() -> str:
    return "\n".join(
        statement for migration in ANSWER_RUN_MIGRATIONS for statement in migration.statements
    )


class TestMigrationDeclaration:
    def test_scope_and_versions_are_unique_and_ordered(self) -> None:
        versions = [migration.version for migration in ANSWER_RUN_MIGRATIONS]
        assert ANSWER_RUN_MIGRATION_SCOPE == "answer_runs"
        assert versions == sorted(versions)
        assert len(set(versions)) == len(versions)

    def test_declares_exactly_the_four_contract_tables(self) -> None:
        created = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", _all_statements()))
        assert created == {
            "dlightrag_answer_runs",
            "dlightrag_answer_run_events",
            "dlightrag_answer_artifacts",
            "dlightrag_answer_run_artifacts",
        }

    def test_every_statement_is_idempotent(self) -> None:
        for migration in ANSWER_RUN_MIGRATIONS:
            for statement in migration.statements:
                assert "IF NOT EXISTS" in statement, statement

    def test_does_not_touch_ingest_job_or_web_conversation_schemas(self) -> None:
        statements = _all_statements()
        assert "dlightrag_ingest_jobs" not in statements
        assert "web_conversation" not in statements


class TestFixedRuntimeBounds:
    def test_recovery_bound_and_error_kind_match_the_contract(self) -> None:
        assert MAX_CONSECUTIVE_RECOVERIES == 4
        assert RUN_ABANDONED_ERROR_KIND == "run_abandoned"

    def test_retention_is_thirty_days(self) -> None:
        assert RUN_RETENTION_SECONDS == 30 * 24 * 3600

    def test_workers_heartbeat_well_inside_their_lease(self) -> None:
        assert 0 < ANSWER_RUN_HEARTBEAT_SECONDS <= ANSWER_RUN_LEASE_SECONDS // 2


class TestCreationValidation:
    """Input rejected before any connection is acquired needs no database."""

    async def test_rejects_a_blank_owner(self) -> None:
        with pytest.raises(ValueError):
            await PGAnswerRunStore().create_run(
                owner_id="   ",
                request={"query": "a"},
                idempotency_fingerprint="test-fingerprint",
            )

    async def test_rejects_a_request_that_is_not_json(self) -> None:
        with pytest.raises(TypeError):
            await PGAnswerRunStore().create_run(
                owner_id="owner",
                request={"q": object()},
                idempotency_fingerprint="test-fingerprint",
            )
