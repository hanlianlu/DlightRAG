# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner-scoped Memory Record schema. Writers land in Task 3."""

from dlightrag.adapters.postgres._migrations import TableRequirement

_CREATE_MEMORY_RECORDS = """
CREATE TABLE IF NOT EXISTS dlightrag_answer_memory_records (
    owner_id       TEXT             NOT NULL,
    memory_id      UUID             NOT NULL,
    kind           TEXT             NOT NULL,
    body           TEXT             NOT NULL,
    confidence     DOUBLE PRECISION NOT NULL,
    run_id         UUID             NOT NULL,
    session_id     UUID             NOT NULL,
    status         TEXT             NOT NULL,
    supersedes_id  UUID,
    created_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (owner_id, memory_id),
    CONSTRAINT dlightrag_answer_memory_records_kind_check
        CHECK (kind IN ('preference', 'fact')),
    CONSTRAINT dlightrag_answer_memory_records_status_check
        CHECK (status IN ('active', 'superseded', 'forgotten')),
    CONSTRAINT dlightrag_answer_memory_records_body_check
        CHECK (char_length(body) BETWEEN 1 AND 500),
    CONSTRAINT dlightrag_answer_memory_records_confidence_check
        CHECK (confidence > 0 AND confidence <= 1)
)
"""

_CREATE_MEMORY_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_memory_records_recall "
    "ON dlightrag_answer_memory_records (owner_id, status, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_dlightrag_answer_memory_records_purge "
    "ON dlightrag_answer_memory_records (status, updated_at) "
    "WHERE status = 'superseded'",
)

MEMORY_DDL = (_CREATE_MEMORY_RECORDS, *_CREATE_MEMORY_INDEXES)

MEMORY_SCHEMA_TABLE = TableRequirement(
    name="dlightrag_answer_memory_records",
    columns=(
        "owner_id",
        "memory_id",
        "kind",
        "body",
        "confidence",
        "run_id",
        "session_id",
        "status",
        "supersedes_id",
        "created_at",
        "updated_at",
    ),
    primary_key=("owner_id", "memory_id"),
    checks=(
        "dlightrag_answer_memory_records_kind_check",
        "dlightrag_answer_memory_records_status_check",
        "dlightrag_answer_memory_records_body_check",
        "dlightrag_answer_memory_records_confidence_check",
    ),
    indexes=(
        "idx_dlightrag_answer_memory_records_recall",
        "idx_dlightrag_answer_memory_records_purge",
    ),
)

__all__ = ["MEMORY_DDL", "MEMORY_SCHEMA_TABLE"]
