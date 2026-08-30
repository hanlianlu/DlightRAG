# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""HTTP transport projections for the application-owned model catalogue."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from dlightrag.adapters.http.rest.routes import model_catalogue as routes
from dlightrag.application.access import UserContext
from dlightrag.application.model_catalogue import (
    ModelCatalogueEntryView,
    ModelCatalogueReadOnlyError,
    ModelCatalogueRevisionConflict,
    ModelCatalogueUnavailableError,
    ModelCatalogueView,
)

_REVISION = "sha256:" + "1" * 64
_NEXT_REVISION = "sha256:" + "2" * 64


def _view(revision: str = _REVISION) -> ModelCatalogueView:
    return ModelCatalogueView(
        revision=revision,
        models=(
            ModelCatalogueEntryView(
                provider="openai",
                model="test-model",
                base_url=None,
                profile={
                    "context_window_tokens": 100_000,
                    "max_input_tokens": None,
                    "max_output_tokens": 10_000,
                    "supports_images": False,
                    "reasoning": None,
                },
                source="builtin",
            ),
        ),
    )


def _client(catalogue: object) -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    app.state.application = SimpleNamespace(model_catalogue=catalogue)
    app.dependency_overrides[routes.get_current_user] = lambda: UserContext(
        user_id="admin", auth_mode="none"
    )
    return TestClient(app)


def _payload() -> dict[str, object]:
    return {
        "provider": "openai",
        "model": "test-model",
        "base_url": None,
        "profile": {
            "context_window_tokens": 100_000,
            "max_input_tokens": None,
            "max_output_tokens": 10_000,
            "supports_images": False,
            "reasoning": None,
        },
    }


def test_get_returns_effective_catalogue_with_http_etag() -> None:
    catalogue = SimpleNamespace(read=lambda: _view())

    response = _client(catalogue).get("/models/catalogue")

    assert response.status_code == 200
    assert response.headers["etag"] == f'"{_REVISION}"'
    assert response.json()["models"][0]["source"] == "builtin"


def test_get_maps_unsynchronized_catalogue_to_service_unavailable() -> None:
    def read() -> ModelCatalogueView:
        raise ModelCatalogueUnavailableError("not ready")

    catalogue = SimpleNamespace(read=read)

    response = _client(catalogue).get("/models/catalogue")

    assert response.status_code == 503


def test_put_forwards_normalized_if_match_and_authenticated_actor(monkeypatch) -> None:
    catalogue = SimpleNamespace(upsert=AsyncMock(return_value=_view(_NEXT_REVISION)))
    monkeypatch.setattr(routes, "enforce_access", AsyncMock())

    response = _client(catalogue).put(
        "/models/catalogue",
        headers={"If-Match": f'W/"{_REVISION}"'},
        json=_payload(),
    )

    assert response.status_code == 200
    assert response.headers["etag"] == f'"{_NEXT_REVISION}"'
    catalogue.upsert.assert_awaited_once_with(
        _payload(), expected_revision=_REVISION, actor="admin"
    )


def test_put_maps_read_only_deployment_to_forbidden(monkeypatch) -> None:
    catalogue = SimpleNamespace(
        upsert=AsyncMock(side_effect=ModelCatalogueReadOnlyError("read-only"))
    )
    monkeypatch.setattr(routes, "enforce_access", AsyncMock())

    response = _client(catalogue).put(
        "/models/catalogue",
        headers={"If-Match": _REVISION},
        json=_payload(),
    )

    assert response.status_code == 403


def test_delete_forwards_endpoint_identity(monkeypatch) -> None:
    catalogue = SimpleNamespace(remove=AsyncMock(return_value=_view(_NEXT_REVISION)))
    monkeypatch.setattr(routes, "enforce_access", AsyncMock())

    response = _client(catalogue).delete(
        "/models/catalogue",
        headers={"If-Match": _REVISION},
        params={"provider": " OpenAI ", "model": " test-model "},
    )

    assert response.status_code == 200
    catalogue.remove.assert_awaited_once_with(
        provider=" OpenAI ",
        model=" test-model ",
        base_url=None,
        expected_revision=_REVISION,
        actor="admin",
    )


def test_stale_put_maps_to_precondition_failed_with_current_etag(monkeypatch) -> None:
    catalogue = SimpleNamespace(
        upsert=AsyncMock(side_effect=ModelCatalogueRevisionConflict(_NEXT_REVISION))
    )
    monkeypatch.setattr(routes, "enforce_access", AsyncMock())

    response = _client(catalogue).put(
        "/models/catalogue",
        headers={"If-Match": _REVISION},
        json=_payload(),
    )

    assert response.status_code == 412
    assert response.headers["etag"] == f'"{_NEXT_REVISION}"'


def test_rest_and_browser_routers_publish_the_same_catalogue_surface() -> None:
    from dlightrag.adapters.http.browser.routes import model_catalogue as browser_routes

    rest = {
        (route.path, next(iter(route.methods or ())))
        for route in routes.router.routes
        if isinstance(route, APIRoute)
    }
    browser = {
        (route.path, next(iter(route.methods or ())))
        for route in browser_routes.router.routes
        if isinstance(route, APIRoute)
    }

    assert (
        rest
        == browser
        == {
            ("/models/catalogue", "GET"),
            ("/models/catalogue", "PUT"),
            ("/models/catalogue", "DELETE"),
        }
    )
