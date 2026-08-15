# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared builders for durable Answer run state in Web tests."""

import datetime
from typing import Any

from dlightrag.storage.answer_runs import AnswerRunRecord, AnswerRunStatus
from dlightrag.storage.web_conversations import LinkedTurn
from dlightrag.web.conversation_models import ConversationSummary
from dlightrag.web.conversations import WebAnswerSubmission

NOW = datetime.datetime(2026, 8, 12, tzinfo=datetime.UTC)
RUN_ID = "019893f4-0000-7000-8000-000000000001"
TURN_ID = "00000000-0000-0000-0000-000000000010"
SUBMISSION_ID = "00000000-0000-0000-0000-0000000000aa"


def run_request(**overrides: Any) -> dict[str, Any]:
    return {
        "query": "What changed?",
        "workspaces": ["default"],
        "history": [],
        "top_k": None,
        "chunk_top_k": None,
        "filters": None,
        "semantic_highlights": True,
        "links": [],
        "attachments": [],
        "history_attachments": [],
        "pinned_models": [
            {
                "role": "query",
                "fingerprint": {
                    "provider": "openai",
                    "model": "test-model",
                    "endpoint_fingerprint": None,
                },
                "profile": {
                    "context_window_tokens": 1_000_000,
                    "max_input_tokens": None,
                    "max_output_tokens": 128_000,
                    "supports_images": True,
                    "supports_tools": True,
                    "supports_reasoning": True,
                },
            }
        ],
        "context_policy_revision": "m1-v1",
        "model_catalog_revision": "2026-08-14",
        "idempotency_fingerprint": "test-public-request-hash",
        "image_descriptions": [],
        **overrides,
    }


def answer_run(
    *,
    status: AnswerRunStatus = "queued",
    request: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    cancel_requested_at: datetime.datetime | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
    events_trimmed_at: datetime.datetime | None = None,
    run_id: str = RUN_ID,
    owner_id: str = "owner-1",
) -> AnswerRunRecord:
    terminal = status in ("succeeded", "failed", "cancelled")
    return AnswerRunRecord(
        owner_id=owner_id,
        run_id=run_id,
        idempotency_key=SUBMISSION_ID,
        request=request if request is not None else run_request(),
        status=status,
        phase=None,
        stop_reason=None,
        completed_turns=0,
        cancel_requested_at=cancel_requested_at,
        lease_owner=None,
        lease_expires_at=None,
        fencing_epoch=0,
        recovery_count=0,
        next_event_sequence=1,
        events_trimmed_at=events_trimmed_at,
        result=result,
        error_kind=error_kind,
        error_message=error_message,
        created_at=NOW,
        updated_at=NOW,
        started_at=None,
        finished_at=NOW if terminal else None,
    )


def linked_turn(run: AnswerRunRecord | None = None, *, turn_number: int = 1) -> LinkedTurn:
    return LinkedTurn(
        turn_id=TURN_ID,
        turn_number=turn_number,
        submission_id=SUBMISSION_ID,
        created_at=NOW,
        run=run if run is not None else answer_run(),
    )


def conversation_summary(conversation_id: str) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=conversation_id,
        title=None,
        created_at=NOW,
        updated_at=NOW,
    )


def web_answer_submission(
    *,
    conversation_id: str,
    run: AnswerRunRecord | None = None,
) -> WebAnswerSubmission:
    return WebAnswerSubmission(
        run=run if run is not None else answer_run(),
        turn_id=TURN_ID,
        turn_number=1,
        conversation=conversation_summary(conversation_id),
    )


def stored_result(answer: str = "Revenue increased [1].") -> dict[str, Any]:
    return {
        "answer": answer,
        "contexts": {"chunks": []},
        "sources": [
            {
                "id": "1",
                "title": "Report",
                "type": "document",
                "source_uri": "local://report.pdf",
                "workspace": "default",
                "document_id": "report",
                "chunks": [],
            }
        ],
        "answer_images": [],
        "trace": {},
        "image_descriptions": [],
    }
