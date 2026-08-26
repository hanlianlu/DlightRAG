# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Contract tests for durable Answer run storage that need no database."""

import datetime
import re
from dataclasses import replace

import pytest

from dlightrag.adapters.postgres.answer_runs import (
    ANSWER_RUN_MIGRATION_SCOPE,
    ANSWER_RUN_MIGRATIONS,
    PGAnswerRunStore,
)
from dlightrag.runtime import (
    ANSWER_RUN_LEASE_SECONDS,
    DEFAULT_RUN_RETENTION_SECONDS,
    MAX_RECLAIMS_WITHOUT_PROGRESS,
    RUN_ABANDONED_ERROR_KIND,
    RUN_HEARTBEAT_SECONDS,
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

    def test_declares_the_final_answer_and_session_tables(self) -> None:
        created = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", _all_statements()))
        assert created == {
            "dlightrag_answer_runs",
            "dlightrag_answer_run_events",
            "dlightrag_agent_sessions",
            "dlightrag_agent_session_entries",
            "dlightrag_agent_session_registers",
            "dlightrag_answer_run_stages",
            "dlightrag_answer_evidence",
            "dlightrag_answer_resources",
            "dlightrag_blobs",
            "dlightrag_blob_chunks",
            "dlightrag_answer_run_artifacts",
            "dlightrag_answer_workspace_inventory",
            "dlightrag_answer_committed_spills",
            "dlightrag_answer_run_routing",
            "dlightrag_answer_child_sessions",
            "dlightrag_agent_controls",
            "dlightrag_answer_memory_settings",
        }

    def test_no_checkpoint_or_single_row_artifact_columns_remain(self) -> None:
        statements = _all_statements()
        assert "checkpoint_json" not in statements
        assert "completed_turns" not in statements
        assert "recovery_count" not in statements
        assert "dlightrag_answer_artifacts" not in statements

    def test_final_baseline_has_publication_kinds_and_no_compatibility_alters(self) -> None:
        statements = _all_statements()
        assert "primary_report" in statements
        assert "published_artifact" in statements
        assert "ALTER TABLE" not in statements

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

    def test_retention_floor_is_365_days(self) -> None:
        assert DEFAULT_RUN_RETENTION_SECONDS == 365 * 24 * 3600

    def test_workers_heartbeat_well_inside_their_lease(self) -> None:
        assert 0 < RUN_HEARTBEAT_SECONDS <= ANSWER_RUN_LEASE_SECONDS // 2

    def test_accepted_input_envelope_keeps_continuation_context_not_model_facts(self) -> None:
        from dlightrag.runtime.records import accepted_input_envelope

        envelope = accepted_input_envelope(
            {
                "query": "why",
                "workspaces": ["alpha", "beta"],
                "mode": "research",
                "attachments": [{"ordinal": 1, "digest": "d" * 64}],
                "history": [{"role": "user", "content": "secret"}],
                "pinned_models": [{"role": "query"}],
                "resource_manifest": [],
            }
        )

        assert envelope == {
            "query": "why",
            "workspaces": ["alpha", "beta"],
            "history": [{"role": "user", "content": "secret"}],
            "episodic_summary": "",
            "top_k": None,
            "chunk_top_k": None,
            "filters": None,
            "semantic_highlights": False,
            "mode": "research",
            "links": [],
            "attachments": [{"ordinal": 1, "digest": "d" * 64}],
            "history_attachments": [],
            "agent_session_id": "",
            "agent_lane_id": "main",
            "source_lane_id": None,
        }
        assert "pinned_models" not in envelope
        assert "resource_manifest" not in envelope

    def test_request_input_prefers_the_accepted_envelope(self) -> None:
        from dlightrag.runtime import AnswerRunRecord

        record = AnswerRunRecord(
            owner_id="owner-1",
            run_id="00000000-0000-0000-0000-000000000001",
            idempotency_key=None,
            prepared_input={"query": "execution copy"},
            accepted_input={"query": "envelope copy"},
            status="succeeded",
            phase=None,
            stop_reason=None,
            cancel_requested_at=None,
            lease_owner=None,
            lease_expires_at=None,
            fencing_epoch=0,
            durable_progress_version=0,
            last_reclaim_progress_version=0,
            reclaims_without_progress=0,
            next_event_sequence=1,
            events_trimmed_at=None,
            result=None,
            error_kind=None,
            error_message=None,
            created_at=datetime.datetime(2026, 8, 12, tzinfo=datetime.UTC),
            updated_at=datetime.datetime(2026, 8, 12, tzinfo=datetime.UTC),
            started_at=None,
            finished_at=None,
        )

        assert record.request_input()["query"] == "envelope copy"

        cleared = replace(record, accepted_input=None)
        assert cleared.request_input()["query"] == "execution copy"

        both_cleared = replace(cleared, prepared_input=None)
        assert both_cleared.request_input() == {}


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
