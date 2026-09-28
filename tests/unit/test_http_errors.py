# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One HTTP projection for every typed failure family."""

import pytest
from dlightrag_memory.errors import MemoryUnavailableError, MemoryWriteRejectedError
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from dlightrag.adapters.http.errors import error_type_for_status, install_error_handlers
from dlightrag.application.access import AccessDeniedError
from dlightrag.application.errors import (
    ApplicationClosedError,
    CorpusUnavailableError,
    StorageSchemaError,
    WorkspaceWriteFencedError,
)
from dlightrag.application.memory import MemoryDisabledError
from dlightrag.application.runs import IdempotencyKeyConflict, RunAdmissionLimitExceededError


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, "validation"),
        (401, "auth"),
        (403, "auth"),
        (404, "not_found"),
        (409, "conflict"),
        (412, "conflict"),
        (413, "validation"),
        (416, "validation"),
        (422, "validation"),
        (429, "unavailable"),
        (500, "internal"),
        (503, "unavailable"),
    ],
)
def test_statuses_classify_into_the_public_vocabulary(status: int, error_type: str) -> None:
    assert error_type_for_status(status) == error_type


def _client(failure: BaseException) -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/fail")
    async def fail() -> None:
        raise failure

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("failure", "status", "body"),
    [
        (
            IdempotencyKeyConflict(),
            409,
            {
                "detail": "Idempotency key was reused with a different request",
                "error_type": "conflict",
            },
        ),
        (
            RunAdmissionLimitExceededError(),
            503,
            {
                "detail": "Deployment-wide nonterminal admission limit reached",
                "error_type": "unavailable",
            },
        ),
        (
            ApplicationClosedError(),
            503,
            {"detail": "Application is shutting down", "error_type": "unavailable"},
        ),
        (
            # Also an Engine TransientDependencyError: the Application family still wins.
            CorpusUnavailableError(),
            503,
            {"detail": "Corpus storage is temporarily unavailable", "error_type": "unavailable"},
        ),
        (
            StorageSchemaError("column gone: secret detail"),
            503,
            {
                "detail": "Durable storage is unavailable on this deployment",
                "error_type": "unavailable",
            },
        ),
        (
            HTTPException(404, "Run not found"),
            404,
            {"detail": "Run not found", "error_type": "not_found"},
        ),
        (
            HTTPException(409, {"kind": "submission_conflict", "message": "Used"}),
            409,
            {"kind": "submission_conflict", "message": "Used"},
        ),
    ],
)
def test_each_family_answers_one_way(failure: BaseException, status: int, body: object) -> None:
    response = _client(failure).get("/fail")

    assert response.status_code == status
    assert response.json() == body


def test_routing_failures_use_the_same_envelope() -> None:
    response = _client(RuntimeError("unused")).get("/no-such-route")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found", "error_type": "not_found"}


def test_only_an_access_denial_is_forbidden() -> None:
    denied = _client(AccessDeniedError("Access denied for action=workspace.query")).get("/fail")
    # An OS permission failure is a server fault: no 403, and no path in the body.
    os_failure = _client(PermissionError(13, "Permission denied", "/srv/corpus/a.pdf")).get("/fail")

    assert denied.status_code == 403
    assert denied.json() == {
        "detail": "Access denied for action=workspace.query",
        "error_type": "auth",
    }
    assert os_failure.status_code == 500
    assert "/srv/corpus" not in os_failure.text


@pytest.mark.parametrize(
    ("failure", "status", "error_type"),
    [
        (MemoryDisabledError(), 409, "conflict"),
        (MemoryUnavailableError(), 403, "auth"),
        (MemoryWriteRejectedError("Memory idempotency key was reused."), 409, "conflict"),
    ],
    ids=["disabled", "unavailable", "write-rejected"],
)
def test_memory_refusals_answer_with_their_public_message(
    failure: MemoryDisabledError | MemoryUnavailableError | MemoryWriteRejectedError,
    status: int,
    error_type: str,
) -> None:
    response = _client(failure).get("/fail")

    assert response.status_code == status
    assert response.json() == {"detail": failure.public_message, "error_type": error_type}


def test_a_write_fence_says_when_to_retry() -> None:
    response = _client(WorkspaceWriteFencedError(workspace="finance", retry_after_seconds=7.2)).get(
        "/fail"
    )

    assert response.status_code == 409
    assert response.headers["retry-after"] == "8"
    assert response.json()["error_type"] == "conflict"


def test_request_validation_names_fields_without_echoing_values() -> None:
    """FastAPI's default 422 returns each submitted value; this one never does."""
    from pydantic import BaseModel, field_validator

    class Body(BaseModel):
        workspace: str
        top_k: int

        @field_validator("workspace")
        @classmethod
        def _no_secrets(cls, value: str) -> str:
            raise ValueError("workspace is not allowed here")

    app = FastAPI()
    install_error_handlers(app)

    @app.post("/submit")
    async def submit(body: Body) -> None:  # noqa: ARG001
        return None

    response = TestClient(app).post(
        "/submit", json={"workspace": "SECRET-VALUE", "top_k": "SECRET-NUMBER"}
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error_type"] == "validation"
    assert "SECRET" not in response.text
    assert "body.workspace: workspace is not allowed here" in body["detail"]
    assert "body.top_k:" in body["detail"]
    assert "Value error," not in body["detail"]
