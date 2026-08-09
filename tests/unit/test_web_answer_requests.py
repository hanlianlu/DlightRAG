# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Web answer JSON and multipart request normalization."""

import io
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from PIL import Image

from dlightrag.web.attachment_requests import parse_web_answer_request

_IMAGE_MAX_BYTES = 15 * 1024 * 1024


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_parse_json_web_answer_request_carries_no_attachments() -> None:
    app = FastAPI()

    @app.post("/probe")
    async def probe(request: Request):
        body = await parse_web_answer_request(
            request,
            max_attachments=6,
            image_max_bytes=_IMAGE_MAX_BYTES,
            image_max_pixels=40_000_000,
        )
        return {
            "query": body.query,
            "attachments": len(body.attachments),
            "workspaces": body.workspaces,
        }

    payload = {
        "query": "hello",
        "workspaces": ["default"],
        "conversation_id": str(uuid4()),
        "submission_id": str(uuid4()),
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/probe", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "query": "hello",
        "attachments": 0,
        "workspaces": ["default"],
    }


@pytest.mark.asyncio
async def test_parse_multipart_web_answer_request_reads_ordered_attachments() -> None:
    app = FastAPI()

    @app.post("/probe")
    async def probe(request: Request):
        body = await parse_web_answer_request(
            request,
            max_attachments=6,
            image_max_bytes=_IMAGE_MAX_BYTES,
            image_max_pixels=40_000_000,
        )
        return {
            "query": body.query,
            "attachments": [
                {"filename": item.filename, "kind": item.kind} for item in body.attachments
            ],
        }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/probe",
            data={
                "query": "compare this",
                "workspaces": '["default"]',
                "conversation_id": str(uuid4()),
                "submission_id": str(uuid4()),
            },
            files=[
                ("attachments", ("chart.png", _png_bytes(), "image/png")),
                ("attachments", ("report.pdf", b"%PDF-test", "application/pdf")),
            ],
        )

    assert response.status_code == 200
    assert response.json() == {
        "query": "compare this",
        "attachments": [
            {"filename": "chart.png", "kind": "image"},
            {"filename": "report.pdf", "kind": "document"},
        ],
    }
