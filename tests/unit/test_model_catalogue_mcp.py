# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP catalogue tools are thin adapters over ModelCatalogueAdmin."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from dlightrag.adapters.mcp import server as mcp_server
from dlightrag.adapters.mcp.tools.model_catalogue import (
    get_model_catalogue_tool,
    upsert_model_catalogue_entry_tool,
)
from dlightrag.application.model_catalogue import ModelCatalogueEntryView, ModelCatalogueView

_REVISION = "sha256:" + "1" * 64


def _view() -> ModelCatalogueView:
    return ModelCatalogueView(
        revision=_REVISION,
        models=(
            ModelCatalogueEntryView(
                provider="openai",
                model="m",
                base_url=None,
                profile={
                    "context_window_tokens": 100_000,
                    "max_input_tokens": None,
                    "max_output_tokens": 10_000,
                    "supports_images": False,
                    "reasoning": None,
                },
                source="overlay",
            ),
        ),
    )


async def test_get_model_catalogue_projects_application_view(monkeypatch) -> None:
    catalogue = SimpleNamespace(read=lambda: _view())
    monkeypatch.setattr(
        mcp_server,
        "_ensure_application",
        AsyncMock(return_value=SimpleNamespace(model_catalogue=catalogue)),
    )

    result = await get_model_catalogue_tool()

    assert result["revision"] == _REVISION
    assert result["models"][0]["source"] == "overlay"


async def test_upsert_model_catalogue_enforces_admin_and_forwards_revision(monkeypatch) -> None:
    catalogue = SimpleNamespace(upsert=AsyncMock(return_value=_view()))
    application = SimpleNamespace(model_catalogue=catalogue)
    enforce = AsyncMock()
    monkeypatch.setattr(mcp_server, "_ensure_application", AsyncMock(return_value=application))
    monkeypatch.setattr(mcp_server, "_enforce_access", enforce)
    entry = {
        "provider": "openai",
        "model": "m",
        "base_url": None,
        "profile": {
            "context_window_tokens": 100_000,
            "max_input_tokens": None,
            "max_output_tokens": 10_000,
            "supports_images": False,
            "reasoning": None,
        },
    }

    result = await upsert_model_catalogue_entry_tool(entry, _REVISION)

    enforce.assert_awaited_once()
    catalogue.upsert.assert_awaited_once()
    assert catalogue.upsert.await_args.kwargs["expected_revision"] == _REVISION
    assert result["revision"] == _REVISION
