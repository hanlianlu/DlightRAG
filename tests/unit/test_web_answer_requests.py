# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Web answer JSON and multipart request normalization."""

from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from dlightrag.web.attachment_requests import parse_web_answer_request


@pytest.mark.asyncio
async def test_parse_json_web_answer_request_preserves_existing_shape() -> None:
    app = FastAPI()

    @app.post("/probe")
    async def probe(request: Request):
        body = await parse_web_answer_request(request, max_images=3, max_image_upload_bytes=1000)
        return {
            "query": body.query,
            "images": body.images,
            "documents": len(body.documents),
            "workspaces": body.workspaces,
        }

    payload = {
        "query": "hello",
        "images": ["data:image/png;base64,abcd"],
        "workspaces": ["default"],
        "conversation_id": str(uuid4()),
        "submission_id": str(uuid4()),
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/probe", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "query": "hello",
        "images": ["data:image/png;base64,abcd"],
        "documents": 0,
        "workspaces": ["default"],
    }


@pytest.mark.asyncio
async def test_parse_multipart_web_answer_request_reads_documents() -> None:
    app = FastAPI()

    @app.post("/probe")
    async def probe(request: Request):
        body = await parse_web_answer_request(request, max_images=3, max_image_upload_bytes=1000)
        return {
            "query": body.query,
            "images": body.images,
            "documents": [doc.filename for doc in body.documents],
        }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/probe",
            data={
                "query": "compare this",
                "images": "[]",
                "workspaces": '["default"]',
                "conversation_id": str(uuid4()),
                "submission_id": str(uuid4()),
            },
            files={"documents": ("report.pdf", b"%PDF-test", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "query": "compare this",
        "images": [],
        "documents": ["report.pdf"],
    }
