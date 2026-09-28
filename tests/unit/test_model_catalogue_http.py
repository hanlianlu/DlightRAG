# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""HTTP transport projections for the application-owned model catalogue."""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from dlightrag.adapters.http.errors import install_error_handlers
from dlightrag.adapters.http.rest.routes import model_catalogue as routes
from dlightrag.application.access import UserContext
from dlightrag.application.config import DlightragConfig
from dlightrag.application.model_catalogue import (
    ModelCatalogueEntryView,
    ModelCatalogueReadOnlyError,
    ModelCatalogueRevisionConflict,
    ModelCatalogueUnavailableError,
    ModelCatalogueView,
)
from tests.support.application_double import application_double

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


@pytest.fixture
def application(test_config: DlightragConfig) -> Any:
    return application_double(test_config)


def _client(application: Any) -> TestClient:
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(routes.router)
    app.state.application = application
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


def test_get_returns_effective_catalogue_with_http_etag(application: Any) -> None:
    application.model_catalogue.read.return_value = _view()

    response = _client(application).get("/models/catalogue")

    assert response.status_code == 200
    assert response.headers["etag"] == f'"{_REVISION}"'
    assert response.json()["models"][0]["source"] == "builtin"


def test_get_maps_unsynchronized_catalogue_to_service_unavailable(application: Any) -> None:
    application.model_catalogue.read.side_effect = ModelCatalogueUnavailableError("not ready")

    response = _client(application).get("/models/catalogue")

    assert response.status_code == 503
    assert response.json() == {"detail": "not ready", "error_type": "unavailable"}


def test_put_forwards_normalized_if_match_and_authenticated_actor(
    application: Any, monkeypatch
) -> None:
    application.model_catalogue.upsert.return_value = _view(_NEXT_REVISION)
    monkeypatch.setattr(routes, "enforce_access", AsyncMock())

    response = _client(application).put(
        "/models/catalogue",
        headers={"If-Match": f'W/"{_REVISION}"'},
        json=_payload(),
    )

    assert response.status_code == 200
    assert response.headers["etag"] == f'"{_NEXT_REVISION}"'
    application.model_catalogue.upsert.assert_awaited_once_with(
        _payload(), expected_revision=_REVISION, actor="admin"
    )


def test_put_on_a_read_only_replica_names_the_remedy(application: Any, monkeypatch) -> None:
    application.model_catalogue.upsert.side_effect = ModelCatalogueReadOnlyError()
    monkeypatch.setattr(routes, "enforce_access", AsyncMock())

    response = _client(application).put(
        "/models/catalogue",
        headers={"If-Match": _REVISION},
        json=_payload(),
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "This deployment is a read-only replica: it cannot change the model catalogue. "
            "Send the change to a writer."
        ),
        "error_type": "unavailable",
    }


def test_delete_forwards_endpoint_identity(application: Any, monkeypatch) -> None:
    application.model_catalogue.remove.return_value = _view(_NEXT_REVISION)
    monkeypatch.setattr(routes, "enforce_access", AsyncMock())

    response = _client(application).delete(
        "/models/catalogue",
        headers={"If-Match": _REVISION},
        params={"provider": " OpenAI ", "model": " test-model "},
    )

    assert response.status_code == 200
    application.model_catalogue.remove.assert_awaited_once_with(
        provider=" OpenAI ",
        model=" test-model ",
        base_url=None,
        expected_revision=_REVISION,
        actor="admin",
    )


def test_stale_put_maps_to_precondition_failed_with_current_etag(
    application: Any, monkeypatch
) -> None:
    application.model_catalogue.upsert.side_effect = ModelCatalogueRevisionConflict(_NEXT_REVISION)
    monkeypatch.setattr(routes, "enforce_access", AsyncMock())

    response = _client(application).put(
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
