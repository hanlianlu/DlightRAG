# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Access-control policy tests."""

from __future__ import annotations

import pytest

from dlightrag.access_control import AccessDeniedError, access_control_from_config
from dlightrag.api.auth import UserContext
from dlightrag.config import AccessControlConfig, AccessControlRuleConfig, DlightragConfig


async def test_allow_all_access_control_is_default(test_config: DlightragConfig) -> None:
    access_control = access_control_from_config(test_config)

    await access_control.check(
        UserContext(user_id="anonymous", auth_mode="none"),
        "workspace.delete",
        workspace="finance",
    )


async def test_jwt_claims_access_control_matches_claim_workspace_and_action(
    test_config: DlightragConfig,
) -> None:
    test_config.auth_mode = "jwt"
    test_config.jwt_verification_key = "test-key"
    test_config.access_control = AccessControlConfig(
        mode="jwt_claims",
        rules=[
            AccessControlRuleConfig(
                claim="groups",
                value="finance-rag-readers",
                workspaces=["finance"],
                actions=["workspace.query", "workspace.list_files"],
            )
        ],
    )
    access_control = access_control_from_config(test_config)
    user = UserContext(
        user_id="alice",
        auth_mode="jwt",
        claims={"groups": ["finance-rag-readers"]},
    )

    await access_control.check(user, "workspace.query", workspace="finance")
    assert await access_control.filter_workspaces(
        user, "workspace.query", ["finance", "legal"]
    ) == ["finance"]

    with pytest.raises(AccessDeniedError):
        await access_control.check(user, "workspace.reset", workspace="finance")
