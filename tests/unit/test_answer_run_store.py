# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Contract tests for durable Answer run storage that need no database."""

import datetime
from dataclasses import replace

import pytest

from dlightrag.adapters.postgres.runtime.run_store import (
    RUN_MIGRATION_SCOPE,
    RUN_MIGRATIONS,
    PGRunStore,
)
from dlightrag.engine.runtime.coordinator import RUN_HEARTBEAT_SECONDS
from dlightrag.engine.runtime.policy import (
    DEFAULT_RUN_RETENTION_SECONDS,
    MAX_RECLAIMS_WITHOUT_PROGRESS,
    RUN_ABANDONED_ERROR_KIND,
    RUN_LEASE_SECONDS,
)
from dlightrag.engine.runtime.records import (
    PreparedRunEnvelope,
    RunAccessScope,
)


def _all_statements() -> str:
    return "\n".join(
        statement for migration in RUN_MIGRATIONS for statement in migration.statements
    )


class TestMigrationDeclaration:
    def test_scope_and_versions_are_unique_and_ordered(self) -> None:
        versions = [migration.version for migration in RUN_MIGRATIONS]
        assert RUN_MIGRATION_SCOPE == "runs"
        assert versions[0] == "run_runtime_v1"
        assert len(set(versions)) == len(versions)

    def test_run_artifacts_reference_blobs_not_a_content_table(self) -> None:
        statements = _all_statements()
        assert "REFERENCES dlightrag_blobs (owner_id, digest)" in statements

    def test_create_table_statements_are_idempotent(self) -> None:
        for migration in RUN_MIGRATIONS:
            for statement in migration.statements:
                if statement.lstrip().startswith("CREATE TABLE"):
                    assert "IF NOT EXISTS" in statement, statement
                elif statement.lstrip().startswith("CREATE INDEX"):
                    assert "IF NOT EXISTS" in statement, statement


class TestFixedRuntimeBounds:
    def test_reclaim_bound_and_error_kind_match_the_contract(self) -> None:
        assert MAX_RECLAIMS_WITHOUT_PROGRESS == 4
        assert RUN_ABANDONED_ERROR_KIND == "run_abandoned"

    def test_retention_floor_is_365_days(self) -> None:
        assert DEFAULT_RUN_RETENTION_SECONDS == 365 * 24 * 3600

    def test_workers_heartbeat_well_inside_their_lease(self) -> None:
        assert 0 < RUN_HEARTBEAT_SECONDS <= RUN_LEASE_SECONDS // 2

    def test_accepted_input_envelope_keeps_continuation_context_not_model_facts(self) -> None:
        from dlightrag.engine.answer.runs.envelope import accepted_input_envelope

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
            "federated_rerank": False,
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
        from dlightrag.engine.runtime.records import (
            RunAccessScope,
            RunRecord,
        )

        record = RunRecord(
            run_id="00000000-0000-0000-0000-000000000001",
            run_kind="answer",
            lane="query",
            submitted_by="owner-1",
            access_scope=RunAccessScope(kind="owner", scope_id="owner-1"),
            submission_key=None or "00000000-0000-0000-0000-000000000001",
            request_fingerprint="test-fingerprint",
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

    @staticmethod
    def _envelope(*, owner: str = "owner", payload: object = None) -> PreparedRunEnvelope:
        prepared = {"query": "a"} if payload is None else payload
        return PreparedRunEnvelope(
            run_kind="answer",
            lane="query",
            submitted_by=owner,
            access_scope=RunAccessScope(kind="owner", scope_id=owner),
            submission_key="submission-1",
            request_fingerprint="test-fingerprint",
            payload=prepared,  # type: ignore[arg-type]
            accepted_input={"query": "a"},
            retention_seconds=DEFAULT_RUN_RETENTION_SECONDS,
        )

    async def test_rejects_a_blank_owner(self) -> None:
        with pytest.raises(ValueError):
            await PGRunStore().accept_run(
                envelope=self._envelope(owner="   "),
                run_id="00000000-0000-0000-0000-000000000001",
            )

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("submission_key", "", "submission_key"),
            ("request_fingerprint", "", "request_fingerprint"),
            ("retention_seconds", 0, "retention_seconds"),
        ],
    )
    async def test_rejects_invalid_required_envelope_fields(
        self, field: str, value: object, message: str
    ) -> None:
        envelope = replace(self._envelope(), **{field: value})
        with pytest.raises(ValueError, match=message):
            await PGRunStore().accept_run(
                envelope=envelope,
                run_id="00000000-0000-0000-0000-000000000001",
            )

    async def test_rejects_supersession_on_a_non_mutation_envelope(self) -> None:
        envelope = replace(
            self._envelope(),
            supersedes_run_id="00000000-0000-0000-0000-000000000002",
        )

        with pytest.raises(ValueError, match="only Corpus Mutation"):
            await PGRunStore().accept_run(
                envelope=envelope,
                run_id="00000000-0000-0000-0000-000000000001",
            )

    async def test_rejects_a_prepared_input_that_is_not_json(self) -> None:
        with pytest.raises(TypeError):
            await PGRunStore().accept_run(
                envelope=self._envelope(payload={"q": object()}),
                run_id="00000000-0000-0000-0000-000000000001",
            )

    async def test_rejects_a_fetched_resource_reference_at_creation(self) -> None:
        from dlightrag.engine.runtime.records import PendingArtifactReference

        with pytest.raises(ValueError):
            await PGRunStore().accept_run(
                envelope=self._envelope(),
                run_id="00000000-0000-0000-0000-000000000001",
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
