# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Access-control policy tests."""

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


def _preset_access_control(preset: str, test_config: DlightragConfig):
    test_config.auth_mode = "jwt"
    test_config.jwt_verification_key = "test-key"
    test_config.access_control = AccessControlConfig(
        mode="jwt_claims",
        rules=[
            AccessControlRuleConfig(
                claim="roles",
                value=f"finance.{preset}",
                workspaces=["finance"],
                actions=[preset],
            )
        ],
    )
    user = UserContext(user_id="alice", auth_mode="jwt", claims={"roles": [f"finance.{preset}"]})
    return access_control_from_config(test_config), user


async def test_reader_preset_allows_reads_and_denies_writes(
    test_config: DlightragConfig,
) -> None:
    access_control, user = _preset_access_control("reader", test_config)

    await access_control.check(user, "workspace.query", workspace="finance")
    await access_control.check(user, "workspace.read_metadata", workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, "workspace.ingest", workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, "workspace.delete", workspace="finance")


async def test_editor_preset_allows_ingest_and_job_read_but_not_workspace_admin(
    test_config: DlightragConfig,
) -> None:
    access_control, user = _preset_access_control("editor", test_config)

    await access_control.check(user, "workspace.query", workspace="finance")
    await access_control.check(user, "workspace.ingest", workspace="finance")
    await access_control.check(user, "workspace.delete_files", workspace="finance")
    await access_control.check(user, "job.read", workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, "workspace.delete", workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, "workspace.reset", workspace="finance")


async def test_admin_preset_allows_every_action(test_config: DlightragConfig) -> None:
    access_control, user = _preset_access_control("admin", test_config)

    for action in (
        "workspace.query",
        "workspace.ingest",
        "workspace.create",
        "workspace.delete",
        "workspace.reset",
        "job.read",
    ):
        await access_control.check(user, action, workspace="finance")
