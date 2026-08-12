# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Web answer JSON and multipart request normalization."""

import io
import threading
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from PIL import Image

from dlightrag.core.answer.capability import AnswerImageCapability
from dlightrag.web.attachment_requests import parse_web_answer_request

_IMAGE_MAX_BYTES = 15 * 1024 * 1024


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _supported_capability() -> AnswerImageCapability:
    return AnswerImageCapability(
        status="supported",
        configured_ceiling=8,
        effective_max_images=8,
        provider="test",
        base_url=None,
        model="vision-test",
        failure_kind=None,
    )


def _multipart_payload(
    *,
    boundary: str,
    files: list[tuple[str, bytes, str]],
) -> bytes:
    parts: list[bytes] = []
    fields = {
        "query": "inspect",
        "conversation_id": str(uuid4()),
        "submission_id": str(uuid4()),
    }
    for name, value in fields.items():
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            ).encode()
        )
    for filename, payload, content_type in files:
        parts.append(
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="attachments"; '
                f'filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
            + payload
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts)


async def _chunked(payload: bytes) -> AsyncIterator[bytes]:
    midpoint = max(1, len(payload) // 2)
    yield payload[:midpoint]
    yield payload[midpoint:]


@pytest.mark.asyncio
async def test_parse_json_web_answer_request_carries_no_attachments() -> None:
    app = FastAPI()

    @app.post("/probe")
    async def probe(request: Request):
        body = await parse_web_answer_request(
            request,
            max_attachments=6,
            max_attachment_bytes=_IMAGE_MAX_BYTES,
            max_total_attachment_bytes=128 * 1024 * 1024,
            image_max_pixels=40_000_000,
            answer_image_capability=_supported_capability(),
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
            max_attachment_bytes=_IMAGE_MAX_BYTES,
            max_total_attachment_bytes=128 * 1024 * 1024,
            image_max_pixels=40_000_000,
            answer_image_capability=_supported_capability(),
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


@pytest.mark.asyncio
async def test_parse_multipart_validates_attachments_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.web import attachment_requests

    app = FastAPI()
    loop_thread = threading.get_ident()
    validation_threads: list[int] = []
    validate = attachment_requests.validate_web_attachments

    def capture_validation(*args, **kwargs):
        validation_threads.append(threading.get_ident())
        return validate(*args, **kwargs)

    monkeypatch.setattr(attachment_requests, "validate_web_attachments", capture_validation)

    @app.post("/probe")
    async def probe(request: Request):
        body = await parse_web_answer_request(
            request,
            max_attachments=6,
            max_attachment_bytes=_IMAGE_MAX_BYTES,
            max_total_attachment_bytes=128 * 1024 * 1024,
            image_max_pixels=40_000_000,
            answer_image_capability=_supported_capability(),
        )
        return {"attachments": len(body.attachments)}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/probe",
            data={
                "query": "inspect",
                "conversation_id": str(uuid4()),
                "submission_id": str(uuid4()),
            },
            files=[("attachments", ("chart.png", _png_bytes(), "image/png"))],
        )

    assert response.status_code == 200
    assert validation_threads
    assert all(thread_id != loop_thread for thread_id in validation_threads)


@pytest.mark.asyncio
async def test_parse_chunked_multipart_rejects_running_total_limit() -> None:
    app = FastAPI()

    @app.post("/probe")
    async def probe(request: Request):
        await parse_web_answer_request(
            request,
            max_attachments=6,
            max_attachment_bytes=64,
            max_total_attachment_bytes=6,
            image_max_pixels=40_000_000,
            answer_image_capability=_supported_capability(),
        )
        return {"ok": True}

    boundary = "dlightrag-boundary"
    payload = _multipart_payload(
        boundary=boundary,
        files=[("one.txt", b"1234", "text/plain"), ("two.txt", b"5678", "text/plain")],
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/probe",
            content=_chunked(payload),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Attachments exceed the total size limit"


@pytest.mark.asyncio
async def test_parse_chunked_multipart_rejects_oversized_attachment() -> None:
    app = FastAPI()

    @app.post("/probe")
    async def probe(request: Request):
        await parse_web_answer_request(
            request,
            max_attachments=6,
            max_attachment_bytes=64,
            max_total_attachment_bytes=128,
            image_max_pixels=40_000_000,
            answer_image_capability=_supported_capability(),
        )
        return {"ok": True}

    boundary = "dlightrag-boundary"
    payload = _multipart_payload(
        boundary=boundary,
        files=[("note.txt", b"x" * 65, "text/plain")],
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/probe",
            content=_chunked(payload),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    assert response.status_code == 413
