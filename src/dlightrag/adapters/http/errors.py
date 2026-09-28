# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""HTTP projection of typed failures, installed once per app.

Routes raise typed Application errors and let these handlers answer. A route
catches only what it translates differently: a browser command envelope, or a
refusal it deliberately masks as not found.
"""

import logging
import math
from collections.abc import Mapping

from dlightrag_memory.errors import MemoryUnavailableError, MemoryWriteRejectedError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException

from dlightrag.adapters.http.rest.models import ErrorDetail
from dlightrag.application.access import AccessDeniedError
from dlightrag.application.corpus_admin import MetadataValidationError
from dlightrag.application.errors import (
    ApplicationConflictError,
    ApplicationInputError,
    ApplicationUnavailableError,
    RunSchemaError,
    StorageSchemaError,
    WorkspaceWriteFencedError,
)
from dlightrag.application.model_catalogue import ModelCatalogueSchemaError
from dlightrag.application.retrieval import RetrievalInputError
from dlightrag.application.web_conversations import WebConversationSchemaError
from dlightrag.engine.answer.errors import AnswerInputError, InvalidToolConfigurationError

logger = logging.getLogger(__name__)

_SCHEMA_ERRORS = (
    StorageSchemaError,
    RunSchemaError,
    WebConversationSchemaError,
    ModelCatalogueSchemaError,
)


def error_type_for_status(status: int) -> str:
    """Classify an HTTP status into the public ``error_type`` vocabulary."""
    if status in {401, 403}:
        return "auth"
    if status in {404, 410}:
        return "not_found"
    if status in {409, 412}:
        return "conflict"
    if status in {429, 503}:
        return "unavailable"
    if 400 <= status < 500:
        return "validation"
    return "internal"


def invalid_fields(exc: ValidationError) -> str:
    """Name each invalid field and why, without echoing the submitted value."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    )


def error_response(
    status: int,
    detail: str,
    *,
    error_type: str | None = None,
    error_kind: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Answer ``{detail, error_type, error_kind?}``, classified by status unless given."""
    body = ErrorDetail(
        detail=detail,
        error_type=error_type or error_type_for_status(status),
        error_kind=error_kind,
    )
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    """Map every typed failure family once, for every route of ``app``."""

    @app.exception_handler(HTTPException)
    async def http_exception(
        request: Request,  # noqa: ARG001
        exc: HTTPException,
    ) -> JSONResponse:
        """Wrap HTTP errors, routing's own included, keeping browser command envelopes."""
        if (
            isinstance(exc.detail, dict)
            and isinstance(exc.detail.get("kind"), str)
            and isinstance(exc.detail.get("message"), str)
        ):
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail,
                headers=exc.headers,
            )
        return error_response(exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(ApplicationUnavailableError)
    async def unavailable(
        request: Request,
        exc: ApplicationUnavailableError,
    ) -> JSONResponse:
        # A translated outage keeps its cause's traceback; a plain refusal (a read-only
        # replica, the admission limit) is expected and needs only one line.
        logger.warning(
            "%s %s is unavailable: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=exc if exc.__cause__ is not None else None,
        )
        return error_response(503, str(exc))

    @app.exception_handler(ApplicationConflictError)
    async def conflict(
        request: Request,  # noqa: ARG001
        exc: ApplicationConflictError,
    ) -> JSONResponse:
        return error_response(409, str(exc))

    @app.exception_handler(ApplicationInputError)
    async def invalid_input(
        request: Request,  # noqa: ARG001
        exc: ApplicationInputError,
    ) -> JSONResponse:
        return error_response(422, str(exc))

    @app.exception_handler(WorkspaceWriteFencedError)
    async def write_fenced(
        request: Request,  # noqa: ARG001
        exc: WorkspaceWriteFencedError,
    ) -> JSONResponse:
        """A promotion fence clears; Retry-After rounds up so an early retry still meets it."""
        retry_after = max(1, math.ceil(exc.retry_after_seconds))
        return error_response(409, str(exc), headers={"Retry-After": str(retry_after)})

    @app.exception_handler(RetrievalInputError)
    async def retrieval_input(
        request: Request,  # noqa: ARG001
        exc: RetrievalInputError,
    ) -> JSONResponse:
        return error_response(422, str(exc))

    @app.exception_handler(AccessDeniedError)
    async def access_denied(
        request: Request,  # noqa: ARG001
        exc: AccessDeniedError,
    ) -> JSONResponse:
        """An access-control denial; an OS permission failure stays an internal error."""
        return error_response(403, str(exc))

    @app.exception_handler(AnswerInputError)
    async def answer_input(
        request: Request,  # noqa: ARG001
        exc: AnswerInputError,
    ) -> JSONResponse:
        """Answer input rejection -> 422 with a stable error kind."""
        return error_response(422, str(exc), error_kind=exc.error_kind)

    @app.exception_handler(InvalidToolConfigurationError)
    async def invalid_tool_configuration(
        request: Request,  # noqa: ARG001
        exc: InvalidToolConfigurationError,
    ) -> JSONResponse:
        """Server tool-composition failure -> 500; the colliding names stay in the log."""
        logger.error("Answer tool composition is invalid", exc_info=exc)
        return error_response(
            500,
            exc.public_message,
            error_type="configuration",
            error_kind=exc.error_kind,
        )

    @app.exception_handler(MemoryUnavailableError)
    async def memory_unavailable(
        request: Request,  # noqa: ARG001
        exc: MemoryUnavailableError,
    ) -> JSONResponse:
        """Profile Memory is not offered to this caller's authentication."""
        return error_response(403, exc.public_message)

    @app.exception_handler(MemoryWriteRejectedError)
    async def memory_write_rejected(
        request: Request,  # noqa: ARG001
        exc: MemoryWriteRejectedError,
    ) -> JSONResponse:
        return error_response(409, exc.public_message)

    @app.exception_handler(MetadataValidationError)
    async def metadata_validation(
        request: Request,  # noqa: ARG001
        exc: MetadataValidationError,
    ) -> JSONResponse:
        """Metadata is validated below the request model, so it needs its own mapping."""
        return error_response(400, str(exc))

    async def schema_incompatible(
        request: Request,  # noqa: ARG001
        exc: Exception,
    ) -> JSONResponse:
        """An incompatible schema is an operator fault; callers see no schema detail."""
        logger.error("Durable schema is incompatible with this revision", exc_info=exc)
        return error_response(503, "Durable storage is unavailable on this deployment")

    for schema_error in _SCHEMA_ERRORS:
        app.add_exception_handler(schema_error, schema_incompatible)


__all__ = ["error_response", "error_type_for_status", "install_error_handlers", "invalid_fields"]
