# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The deployment's default workspace is normalized once, in configuration."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dlightrag.adapters.http.browser.routes.workspaces import _default_workspace
from dlightrag.application.config import DlightragConfig
from dlightrag.application.config.sections import DeploymentSettings
from tests.support.application_double import application_double


def test_the_configured_display_name_yields_one_canonical_id() -> None:
    settings = DeploymentSettings(workspace="Finance Team")

    assert settings.workspace == "Finance Team"
    assert settings.workspace_id == "finance_team"
    assert DeploymentSettings(workspace="2026 Q3").workspace_id == "_2026_q3"


# A leading digit gains a prefix, so 64 characters that start with one are too long.
@pytest.mark.parametrize("name", ["", "   ", "x" * 65, "1" + "x" * 63])
def test_a_default_that_names_no_workspace_is_rejected_at_load(name: str) -> None:
    with pytest.raises(ValidationError, match="canonical workspace id required"):
        DeploymentSettings(workspace=name)


def _request(config: DlightragConfig, workspace: str) -> Any:
    application = application_double(
        config.model_copy(update={"deployment": DeploymentSettings(workspace=workspace)})
    )
    return cast(
        Any, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(application=application)))
    )


def test_the_web_fallback_prefers_the_configured_default_over_a_literal(
    test_config: DlightragConfig,
) -> None:
    request = _request(test_config, "Finance")

    assert _default_workspace(request, ["default", "finance"]) == "finance"
    assert _default_workspace(request, ["default", "legal"]) == "default"
    assert _default_workspace(request, []) == ""
