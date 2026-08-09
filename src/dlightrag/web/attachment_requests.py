# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Normalize JSON and multipart Web answer requests."""

import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from fastapi import HTTPException, Request
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from dlightrag.web.attachment_models import (
    MAX_ATTACHMENT_BYTES,
    ValidatedWebAttachment,
    validate_web_attachments,
)
from dlightrag.web.requests import WebAnswerRequest

# Bound the multipart parse *before* buffering any bodies so a client cannot
# push Starlette's default 1000 parts into memory/disk ahead of the attachment
# caps. Allow one extra file part beyond the cap so an over-limit request
# surfaces a precise 413 here instead of Starlette's generic 400. ``max_fields``
# covers the handful of non-file form fields (query/workspaces/conversation_id/
# submission_id) plus slack.
_MAX_FORM_FIELDS = 16


@dataclass(frozen=True, slots=True)
class ParsedWebAnswerRequest:
    query: str
    workspaces: list[str] | None
    conversation_id: UUID
    submission_id: UUID
    attachments: tuple[ValidatedWebAttachment, ...]


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
    max_attachments: int,
    image_max_bytes: int,
    image_max_pixels: int,
    image_max_count: int | None = None,
) -> ParsedWebAnswerRequest:
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" not in content_type:
        try:
            body = WebAnswerRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return ParsedWebAnswerRequest(
            query=body.query,
            workspaces=body.workspaces,
            conversation_id=body.conversation_id,
            submission_id=body.submission_id,
            attachments=(),
        )

    form = await request.form(
        max_files=max_attachments + 1,
        max_fields=_MAX_FORM_FIELDS,
        max_part_size=MAX_ATTACHMENT_BYTES,
    )
    files = form.getlist("attachments")
    attachment_parts = [item for item in files if isinstance(item, UploadFile) and item.filename]
    if len(attachment_parts) > max_attachments:
        raise HTTPException(
            status_code=413,
            detail=f"Web answer accepts at most {max_attachments} attachments per message",
        )
    attachment_inputs: list[tuple[str, str | None, bytes]] = []
    for item in attachment_parts:
        payload = await item.read()
        attachment_inputs.append((str(item.filename), item.content_type, payload))
    try:
        attachments = validate_web_attachments(
            attachment_inputs,
            max_attachments=max_attachments,
            image_max_bytes=image_max_bytes,
            image_max_pixels=image_max_pixels,
            image_max_count=image_max_count,
        )
        workspaces_raw = _json_list(form.get("workspaces"), field="workspaces")
        body = WebAnswerRequest(
            query=str(form.get("query") or ""),
            workspaces=[str(item) for item in workspaces_raw] or None,
            conversation_id=UUID(str(form.get("conversation_id"))),
            submission_id=UUID(str(form.get("submission_id"))),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ParsedWebAnswerRequest(
        query=body.query,
        workspaces=body.workspaces,
        conversation_id=body.conversation_id,
        submission_id=body.submission_id,
        attachments=attachments,
    )


__all__ = ["ParsedWebAnswerRequest", "parse_web_answer_request"]
