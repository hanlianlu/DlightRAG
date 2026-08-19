# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Contract tests for durable M3 Answer run storage that need no database."""

import re

import pytest

from dlightrag.adapters.postgres.answer_runs import (
    ANSWER_RUN_MIGRATION_SCOPE,
    ANSWER_RUN_MIGRATIONS,
    PGAnswerRunStore,
)
from dlightrag.runtime import (
    ANSWER_RUN_LEASE_SECONDS,
    MAX_RECLAIMS_WITHOUT_PROGRESS,
    RUN_ABANDONED_ERROR_KIND,
    RUN_HEARTBEAT_SECONDS,
    RUN_RETENTION_SECONDS,
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

    def test_declares_the_m3_tables_and_additive_workspace_tables(self) -> None:
        created = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", _all_statements()))
        assert created == {
            "dlightrag_answer_runs",
            "dlightrag_answer_run_events",
            "dlightrag_agent_sessions",
            "dlightrag_agent_session_entries",
            "dlightrag_agent_context_projections",
            "dlightrag_agent_effects",
            "dlightrag_answer_run_stages",
            "dlightrag_answer_evidence",
            "dlightrag_answer_resources",
            "dlightrag_blobs",
            "dlightrag_blob_chunks",
            "dlightrag_answer_run_artifacts",
            "dlightrag_answer_workspace_inventory",
            "dlightrag_answer_committed_spills",
            "dlightrag_answer_run_routing",
        }

    def test_no_checkpoint_or_single_row_artifact_columns_remain(self) -> None:
        statements = _all_statements()
        assert "checkpoint_json" not in statements
        assert "completed_turns" not in statements
        assert "recovery_count" not in statements
        assert "dlightrag_answer_artifacts" not in statements

    def test_publication_kinds_are_declared(self) -> None:
        from dlightrag.adapters.postgres.answer_runs import _M5_PUBLICATION_DDL

        statements = _all_statements() + "\n".join(_M5_PUBLICATION_DDL)
        assert "primary_report" in statements
        assert "published_artifact" in statements

    def test_run_artifacts_reference_blobs_not_a_content_table(self) -> None:
        statements = _all_statements()
        assert "REFERENCES dlightrag_blobs (owner_id, digest)" in statements

    def test_create_table_statements_are_idempotent(self) -> None:
        for migration in ANSWER_RUN_MIGRATIONS:
            for statement in migration.statements:
                if statement.lstrip().startswith("CREATE TABLE"):
                    assert "IF NOT EXISTS" in statement, statement
                elif statement.lstrip().startswith("CREATE INDEX"):
                    assert "IF NOT EXISTS" in statement, statement

    def test_does_not_touch_ingest_job_or_web_conversation_schemas(self) -> None:
        statements = _all_statements()
        assert "dlightrag_ingest_jobs" not in statements
        assert "web_conversation" not in statements


class TestFixedRuntimeBounds:
    def test_reclaim_bound_and_error_kind_match_the_contract(self) -> None:
        assert MAX_RECLAIMS_WITHOUT_PROGRESS == 4
        assert RUN_ABANDONED_ERROR_KIND == "run_abandoned"

    def test_retention_is_thirty_days(self) -> None:
        assert RUN_RETENTION_SECONDS == 30 * 24 * 3600

    def test_workers_heartbeat_well_inside_their_lease(self) -> None:
        assert 0 < RUN_HEARTBEAT_SECONDS <= ANSWER_RUN_LEASE_SECONDS // 2


class TestCreationValidation:
    """Input rejected before any connection is acquired needs no database."""

    async def test_rejects_a_blank_owner(self) -> None:
        with pytest.raises(ValueError):
            await PGAnswerRunStore().create_run(
                owner_id="   ",
                prepared_input={"query": "a"},
                idempotency_fingerprint="test-fingerprint",
            )

    async def test_rejects_a_prepared_input_that_is_not_json(self) -> None:
        with pytest.raises(TypeError):
            await PGAnswerRunStore().create_run(
                owner_id="owner",
                prepared_input={"q": object()},
                idempotency_fingerprint="test-fingerprint",
            )

    async def test_rejects_a_fetched_resource_reference_at_creation(self) -> None:
        from dlightrag.runtime import PendingArtifactReference

        with pytest.raises(ValueError):
            await PGAnswerRunStore().create_run_in(
                object(),
                owner_id="owner",
                request={"query": "a"},
                idempotency_fingerprint="fp",
                references=(
                    PendingArtifactReference(
                        resource_id="r",
                        reference_kind="fetched_resource",
                        ordinal=0,
                        digest="a" * 64,
                        filename="f",
                        mime_type="text/plain",
                    ),
                ),
            )
