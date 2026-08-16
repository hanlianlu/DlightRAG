# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for API and web visual image routes."""

from dataclasses import dataclass
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dlightrag.access import AccessAction, UserContext
from dlightrag.api.auth import get_current_user
from dlightrag.api.routes.images import router as api_images_router
from dlightrag.web.routes.images import router as web_images_router


@dataclass(frozen=True)
class _Asset:
    data: bytes
    media_type: str


def _api_client(manager: object) -> AsyncClient:
    app = FastAPI()
    app.state.manager = manager
    app.include_router(api_images_router)
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="test", auth_mode="none"
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _web_client(manager: object, access_control: object | None = None) -> AsyncClient:
    app = FastAPI()
    app.state.manager = manager
    if access_control is not None:
        app.state.access_control = access_control
    app.include_router(web_images_router)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_api_image_route_serves_asset() -> None:
    manager = AsyncMock()
    manager.aget_visual_asset.return_value = _Asset(data=b"png", media_type="image/png")

    async with _api_client(manager) as client:
        response = await client.get("/images/default/chunk_1?size=thumb")

    assert response.status_code == 200
    assert response.content == b"png"
    assert response.headers["content-type"] == "image/png"
    manager.aget_visual_asset.assert_awaited_once_with("default", "chunk_1", size="thumb")


async def test_api_image_route_returns_404_for_missing_asset() -> None:
    manager = AsyncMock()
    manager.aget_visual_asset.return_value = None

    async with _api_client(manager) as client:
        response = await client.get("/images/default/missing")

    assert response.status_code == 404


async def test_web_image_route_serves_same_origin_asset() -> None:
    manager = AsyncMock()
    manager.aget_visual_asset.return_value = _Asset(data=b"jpeg", media_type="image/jpeg")

    async with _web_client(manager) as client:
        response = await client.get("/images/default/chunk_1?size=full")

    assert response.status_code == 200
    assert response.content == b"jpeg"
    assert response.headers["content-type"] == "image/jpeg"
    manager.aget_visual_asset.assert_awaited_once_with("default", "chunk_1", size="full")


async def test_web_image_route_canonicalizes_scope_before_access_and_read() -> None:
    manager = AsyncMock()
    manager.aget_visual_asset.return_value = _Asset(data=b"jpeg", media_type="image/jpeg")
    access_control = AsyncMock()

    async with _web_client(manager, access_control) as client:
        response = await client.get("/images/Finance%20Reports/chunk_1?size=full")

    assert response.status_code == 200
    access_control.check.assert_awaited_once_with(
        None,
        AccessAction.WORKSPACE_READ_VISUAL_ASSET,
        workspace="finance_reports",
    )
    manager.aget_visual_asset.assert_awaited_once_with(
        "finance_reports",
        "chunk_1",
        size="full",
    )
