# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The deployment's default workspace is normalized once, in configuration."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dlightrag.adapters.http.browser.routes.workspaces import _default_workspace
from dlightrag.application.config.sections import DeploymentSettings


def test_the_configured_display_name_yields_one_canonical_id() -> None:
    settings = DeploymentSettings(workspace="Finance Team")

    assert settings.workspace == "Finance Team"
    assert settings.workspace_id == "finance_team"


@pytest.mark.parametrize("name", ["", "   ", "x" * 65])
def test_a_default_that_names_no_workspace_is_rejected_at_load(name: str) -> None:
    with pytest.raises(ValidationError, match="canonical workspace id required"):
        DeploymentSettings(workspace=name)


def _request(workspace: str) -> Any:
    config = SimpleNamespace(deployment=DeploymentSettings(workspace=workspace))
    application = SimpleNamespace(config=config)
    return cast(
        Any, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(application=application)))
    )


def test_the_web_fallback_prefers_the_configured_default_over_a_literal() -> None:
    request = _request("Finance")

    assert _default_workspace(request, ["default", "finance"]) == "finance"
    assert _default_workspace(request, ["default", "legal"]) == "default"
    assert _default_workspace(request, []) == ""
