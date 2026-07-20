# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Normalize JSON and multipart Web answer requests."""

import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from fastapi import HTTPException, Request
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from dlightrag.web.attachment_models import ValidatedWebDocument, validate_web_documents
from dlightrag.web.requests import WebAnswerRequest, read_limited_answer_body


@dataclass(frozen=True, slots=True)
class ParsedWebAnswerRequest:
    query: str
    images: list[str]
    workspaces: list[str] | None
    conversation_id: UUID
    submission_id: UUID
    documents: tuple[ValidatedWebDocument, ...]


def _json_list(value: Any, *, field: str) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    raise HTTPException(status_code=422, detail=f"Invalid {field}")


async def parse_web_answer_request(
    request: Request,
    *,
    max_images: int,
    max_image_upload_bytes: int,
) -> ParsedWebAnswerRequest:
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" not in content_type:
        try:
            raw_body = await read_limited_answer_body(
                request,
                max_images=max_images,
                max_upload_bytes=max_image_upload_bytes,
            )
            body = WebAnswerRequest.model_validate_json(raw_body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return ParsedWebAnswerRequest(
            query=body.query,
            images=body.images,
            workspaces=body.workspaces,
            conversation_id=body.conversation_id,
            submission_id=body.submission_id,
            documents=(),
        )

    form = await request.form()
    files = form.getlist("documents")
    document_inputs: list[tuple[str, str | None, bytes]] = []
    for item in files:
        if not isinstance(item, UploadFile) or not item.filename:
            continue
        payload = await item.read()
        document_inputs.append((item.filename, item.content_type, payload))
    try:
        documents = validate_web_documents(document_inputs)
        images = [str(item) for item in _json_list(form.get("images"), field="images")]
        workspaces_raw = _json_list(form.get("workspaces"), field="workspaces")
        body = WebAnswerRequest(
            query=str(form.get("query") or ""),
            images=images,
            workspaces=[str(item) for item in workspaces_raw] or None,
            conversation_id=UUID(str(form.get("conversation_id"))),
            submission_id=UUID(str(form.get("submission_id"))),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ParsedWebAnswerRequest(
        query=body.query,
        images=body.images,
        workspaces=body.workspaces,
        conversation_id=body.conversation_id,
        submission_id=body.submission_id,
        documents=documents,
    )


__all__ = ["ParsedWebAnswerRequest", "parse_web_answer_request"]
