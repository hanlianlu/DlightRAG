# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for the transport-neutral Access module."""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

import pytest

from dlightrag.application.access.control import (
    ACTION_PRESETS,
    AccessAction,
    AccessDeniedError,
    access_control_from_settings,
    corpus_mutation_access_action,
    unknown_action_patterns,
)
from dlightrag.application.access.principal import (
    DEPLOYMENT_OWNER_ID,
    UserContext,
    owner_id_from_user,
)
from dlightrag.application.access.workspaces import (
    AccessGate,
    WorkspaceRecord,
)
from dlightrag.application.config.sections import (
    AccessControlConfig,
    AccessControlRuleConfig,
    DlightragConfig,
)
from dlightrag.application.settings.projections import access_settings
from tests.support.config_helpers import mutate_config, replace_config


class _Creators:
    """The registry's creator column, as the policy reads it."""

    def __init__(self, creators: Mapping[str, str] | None = None) -> None:
        self._creators = dict(creators or {})

    async def workspace_creators(self, workspaces: Sequence[str]) -> dict[str, str]:
        return {name: self._creators[name] for name in workspaces if name in self._creators}


class _WorkspaceCatalog:
    async def alist_workspace_records(self) -> list[WorkspaceRecord]:
        return [
            {"workspace": "finance", "display_name": "Finance"},
            {"workspace": "legal", "display_name": "Legal"},
        ]


class _FinanceOnlyAccess:
    async def check(self, subject: Any, action: str, *, workspace: str | None = None) -> None:
        raise AssertionError("all-workspace expansion must use the filtered catalog")

    async def filter_workspaces(
        self,
        subject: Any,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]:
        assert action == AccessAction.WORKSPACE_QUERY
        return [workspace for workspace in workspaces if workspace == "finance"]

    async def filter_run_submitters(
        self, subject: Any, *, workspace: str, submitters: AbstractSet[str]
    ) -> set[str]:
        return set()


async def test_all_workspaces_expands_only_to_authorized_catalog_entries() -> None:
    gate = AccessGate(
        _FinanceOnlyAccess(),
        UserContext(user_id="alice", auth_mode="jwt"),
    )
    resolved = await gate.resolve_query_workspaces(
        _WorkspaceCatalog(),
        default_workspace="legal",
        workspaces=None,
        all_workspaces=True,
    )

    assert resolved == ["finance"]


async def test_allow_all_access_control_is_default(test_config: DlightragConfig) -> None:
    access_control = access_control_from_settings(
        access_settings(test_config), creators=_Creators()
    )

    await access_control.check(
        UserContext(user_id="anonymous", auth_mode="none"),
        AccessAction.WORKSPACE_RESET,
        workspace="finance",
    )


async def test_jwt_claims_access_control_matches_claim_workspace_and_action(
    test_config: DlightragConfig,
) -> None:
    mutate_config(test_config, "access.auth_mode", "jwt")
    mutate_config(test_config, "access.jwt_verification_key", "test-key")
    test_config = replace_config(
        test_config,
        "access.control",
        AccessControlConfig(
            rules=[
                AccessControlRuleConfig(
                    claim="groups",
                    value="finance-rag-readers",
                    workspaces=["Finance Reports"],
                    actions=["workspace.query", "workspace.list_files"],
                )
            ],
        ),
    )
    access_control = access_control_from_settings(
        access_settings(test_config), creators=_Creators()
    )
    user = UserContext(
        user_id="alice",
        auth_mode="jwt",
        claims={"groups": ["finance-rag-readers"]},
    )

    await access_control.check(user, AccessAction.WORKSPACE_QUERY, workspace="finance_reports")
    assert await access_control.filter_workspaces(
        user,
        AccessAction.WORKSPACE_QUERY,
        ["finance_reports", "legal"],
    ) == ["finance_reports"]

    with pytest.raises(AccessDeniedError):
        await access_control.check(user, AccessAction.WORKSPACE_RESET, workspace="finance_reports")


_TEAM = "https://team.cloudflareaccess.com"

# Read-only is no preset: a deployment that wants it names the read Actions.
_READ_ONLY_ACTIONS = [
    AccessAction.WORKSPACE_QUERY,
    AccessAction.WORKSPACE_LIST_FILES,
    AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
    AccessAction.WORKSPACE_READ_METADATA,
    AccessAction.WORKSPACE_READ_VISUAL_ASSET,
]


def _person(subject: str, email: str) -> UserContext:
    return UserContext(
        user_id=subject,
        auth_mode="jwt",
        claims={"iss": _TEAM, "sub": subject, "email": email},
    )


async def test_people_hold_the_workspaces_they_create_and_see_no_one_elses(
    test_config: DlightragConfig,
) -> None:
    """The admin holds everything; everyone else reads the default and owns their own."""
    mutate_config(test_config, "access.auth_mode", "jwt")
    mutate_config(test_config, "access.jwt_verification_key", "test-key")
    test_config = replace_config(
        test_config,
        "access.control",
        AccessControlConfig(
            rules=[
                AccessControlRuleConfig(
                    claim="email", value="admin@example.com", workspaces=["*"], actions=["admin"]
                ),
                AccessControlRuleConfig(
                    claim="iss",
                    value=_TEAM,
                    workspaces=["default"],
                    actions=_READ_ONLY_ACTIONS,
                ),
                AccessControlRuleConfig(
                    claim="iss",
                    value=_TEAM,
                    workspaces=["*"],
                    actions=[AccessAction.WORKSPACE_CREATE],
                ),
            ],
        ),
    )
    admin = _person("admin", "admin@example.com")
    alice = _person("alice", "alice@example.com")
    bob = _person("bob", "bob@example.com")
    access = access_control_from_settings(
        access_settings(test_config),
        creators=_Creators(
            {"alice_notes": owner_id_from_user(alice), "bob_notes": owner_id_from_user(bob)}
        ),
    )
    catalog = ["default", "alice_notes", "bob_notes"]

    assert await access.filter_workspaces(alice, AccessAction.WORKSPACE_QUERY, catalog) == [
        "default",
        "alice_notes",
    ]
    assert await access.filter_workspaces(admin, AccessAction.WORKSPACE_QUERY, catalog) == catalog
    for action in (
        AccessAction.WORKSPACE_INGEST,
        AccessAction.WORKSPACE_RESET,
        AccessAction.WORKSPACE_DELETE,
    ):
        await access.check(alice, action, workspace="alice_notes")
        await access.check(admin, action, workspace="bob_notes")
        for refused in ("default", "bob_notes"):
            with pytest.raises(AccessDeniedError):
                await access.check(alice, action, workspace=refused)
    await access.check(alice, AccessAction.WORKSPACE_CREATE, workspace="alice_drafts")
    # Operator facts and deployment-wide changes stay with the rules.
    with pytest.raises(AccessDeniedError):
        await access.check(alice, AccessAction.WORKSPACE_STORAGE_STATUS, workspace="alice_notes")
    with pytest.raises(AccessDeniedError):
        await access.check(alice, AccessAction.MODEL_CATALOGUE_WRITE)
    await access.check(admin, AccessAction.MODEL_CATALOGUE_WRITE)
    with pytest.raises(AccessDeniedError):
        await access.check(alice, AccessAction.RUNTIME_POLICY_MANAGE, workspace="alice_notes")


async def test_a_run_shows_to_its_submitter_and_never_to_the_next_holder_of_its_name(
    test_config: DlightragConfig,
) -> None:
    """A deleted workspace keeps showing its runs to whoever submitted them, and to no one
    who later creates the same name; a workspace without a creator shows its rule holders
    every run. Seeing a run never lets anyone change it without current access."""
    mutate_config(test_config, "access.auth_mode", "jwt")
    mutate_config(test_config, "access.jwt_verification_key", "test-key")
    test_config = replace_config(
        test_config,
        "access.control",
        AccessControlConfig(
            rules=[
                AccessControlRuleConfig(
                    claim="iss",
                    value=_TEAM,
                    workspaces=["default"],
                    actions=_READ_ONLY_ACTIONS,
                )
            ],
        ),
    )
    alice = _person("alice", "alice@example.com")
    bob = _person("bob", "bob@example.com")
    alice_id, bob_id = owner_id_from_user(alice), owner_id_from_user(bob)
    settings = access_settings(test_config)
    # Alice deleted "notes": its registry row, and so its creator, are gone.
    gone = access_control_from_settings(settings, creators=_Creators())
    # Bob then created "notes" again.
    recreated = access_control_from_settings(settings, creators=_Creators({"notes": bob_id}))
    both = {alice_id, bob_id}

    assert await gone.filter_run_submitters(alice, workspace="notes", submitters=both) == {alice_id}
    assert await recreated.filter_run_submitters(bob, workspace="notes", submitters=both) == {
        bob_id
    }
    assert await recreated.filter_run_submitters(
        bob, workspace="default", submitters={"admin"}
    ) == {"admin"}

    # Alice still sees her old run, but cannot resume it into Bob's "notes";
    # Bob holds "notes" but never sees, so never resumes, Alice's run.
    ingest = corpus_mutation_access_action("ingest")
    await AccessGate(recreated, alice).check_run(
        workspace="notes", submitted_by=alice_id, change=None
    )
    for gate in (AccessGate(recreated, alice), AccessGate(recreated, bob)):
        with pytest.raises(AccessDeniedError):
            await gate.check_run(workspace="notes", submitted_by=alice_id, change=ingest)
    await AccessGate(recreated, bob).check_run(
        workspace="notes", submitted_by=bob_id, change=ingest
    )


def _rule_access_control(role: str, actions: list[str], test_config: DlightragConfig):
    """One rule matching ``roles: [finance.<role>]`` and granting the Actions it names."""
    mutate_config(test_config, "access.auth_mode", "jwt")
    mutate_config(test_config, "access.jwt_verification_key", "test-key")
    test_config = replace_config(
        test_config,
        "access.control",
        AccessControlConfig(
            rules=[
                AccessControlRuleConfig(
                    claim="roles",
                    value=f"finance.{role}",
                    workspaces=["finance"],
                    actions=actions,
                )
            ],
        ),
    )
    user = UserContext(
        user_id="alice",
        auth_mode="jwt",
        claims={"roles": [f"finance.{role}"]},
    )
    return access_control_from_settings(access_settings(test_config), creators=_Creators()), user


async def test_naming_the_read_actions_allows_reads_and_denies_writes(
    test_config: DlightragConfig,
) -> None:
    access_control, user = _rule_access_control("read-only", _READ_ONLY_ACTIONS, test_config)

    await access_control.check(user, AccessAction.WORKSPACE_QUERY, workspace="finance")
    await access_control.check(user, AccessAction.WORKSPACE_READ_METADATA, workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, AccessAction.WORKSPACE_INGEST, workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, AccessAction.WORKSPACE_RESET, workspace="finance")


async def test_common_user_preset_allows_corpus_edits_but_not_workspace_admin(
    test_config: DlightragConfig,
) -> None:
    access_control, user = _rule_access_control("common_user", ["common_user"], test_config)

    await access_control.check(user, AccessAction.WORKSPACE_QUERY, workspace="finance")
    await access_control.check(user, AccessAction.WORKSPACE_INGEST, workspace="finance")
    await access_control.check(user, AccessAction.WORKSPACE_DELETE_FILES, workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, AccessAction.WORKSPACE_RESET, workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, AccessAction.MODEL_CATALOGUE_WRITE)


async def test_admin_preset_allows_every_action(test_config: DlightragConfig) -> None:
    access_control, user = _rule_access_control("admin", ["admin"], test_config)

    for action in (
        AccessAction.WORKSPACE_QUERY,
        AccessAction.WORKSPACE_INGEST,
        AccessAction.WORKSPACE_CREATE,
        AccessAction.WORKSPACE_RESET,
        AccessAction.WORKSPACE_DELETE,
        AccessAction.MODEL_CATALOGUE_WRITE,
        AccessAction.RUNTIME_POLICY_MANAGE,
    ):
        await access_control.check(user, action, workspace="finance")


async def test_only_admin_or_a_rule_naming_the_action_manages_runtime_policies(
    test_config: DlightragConfig,
) -> None:
    """A policy lists the people it targets, so no preset below `admin` holds it, and it is
    deployment-wide: a rule that names it over some workspaces does not reach it."""
    mutate_config(test_config, "access.auth_mode", "jwt")
    mutate_config(test_config, "access.jwt_verification_key", "test-key")
    mutate_config(
        test_config,
        "access.control",
        AccessControlConfig(
            rules=[
                AccessControlRuleConfig(
                    claim="roles", value=role, workspaces=workspaces, actions=actions
                )
                for role, workspaces, actions in (
                    ("read-only", ["*"], _READ_ONLY_ACTIONS),
                    ("common_user", ["*"], ["common_user"]),
                    ("scoped-admin", ["finance"], ["admin"]),
                    ("scoped-manager", ["finance"], [AccessAction.RUNTIME_POLICY_MANAGE]),
                    ("admin", ["*"], ["admin"]),
                    ("manager", ["*"], [AccessAction.RUNTIME_POLICY_MANAGE]),
                )
            ],
        ),
    )
    access = access_control_from_settings(access_settings(test_config), creators=_Creators())

    for role, holds in (
        ("read-only", False),
        ("common_user", False),
        ("scoped-admin", False),
        ("scoped-manager", False),
        ("admin", True),
        ("manager", True),
    ):
        person = UserContext(user_id=role, auth_mode="jwt", claims={"roles": [role]})
        if holds:
            await access.check(person, AccessAction.RUNTIME_POLICY_MANAGE)
        else:
            with pytest.raises(AccessDeniedError):
                await access.check(person, AccessAction.RUNTIME_POLICY_MANAGE)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("ingest", AccessAction.WORKSPACE_INGEST),
        ("replace", AccessAction.WORKSPACE_INGEST),
        ("retry", AccessAction.WORKSPACE_INGEST),
        ("delete", AccessAction.WORKSPACE_DELETE_FILES),
        ("reset", AccessAction.WORKSPACE_RESET),
        ("delete_workspace", AccessAction.WORKSPACE_DELETE),
        ("unknown", AccessAction.WORKSPACE_DELETE),
    ],
)
def test_corpus_mutation_action_mapping_fails_closed(mutation: str, expected: str) -> None:
    assert corpus_mutation_access_action(mutation) == expected


async def test_workspace_wildcard_rule_matches_any_canonical_workspace(
    test_config: DlightragConfig,
) -> None:
    test_config = replace_config(
        test_config,
        "access.control",
        AccessControlConfig(
            rules=[
                AccessControlRuleConfig(
                    claim="roles",
                    value="reader",
                    workspaces=["*"],
                    actions=[AccessAction.WORKSPACE_QUERY],
                )
            ],
        ),
    )
    access_control = access_control_from_settings(
        access_settings(test_config), creators=_Creators()
    )
    user = UserContext(user_id="alice", auth_mode="jwt", claims={"roles": ["reader"]})

    await access_control.check(user, AccessAction.WORKSPACE_QUERY, workspace="finance_reports")
    await access_control.check(user, AccessAction.WORKSPACE_QUERY, workspace="legal")


def test_jwt_principal_uses_trust_domain_and_subject() -> None:
    alice = UserContext(
        user_id="alice",
        auth_mode="jwt",
        claims={"iss": "https://issuer.example"},
    )
    bob = UserContext(
        user_id="bob",
        auth_mode="jwt",
        claims={"iss": "https://issuer.example"},
    )

    assert owner_id_from_user(alice) == owner_id_from_user(alice)
    assert owner_id_from_user(alice) != owner_id_from_user(bob)


def test_simple_principals_are_the_deployment_owner() -> None:
    first = UserContext(user_id="header-a", auth_mode="simple")
    second = UserContext(user_id="header-b", auth_mode="simple")

    assert owner_id_from_user(first) == owner_id_from_user(second) == DEPLOYMENT_OWNER_ID


def test_direct_in_process_calls_share_the_deployment_owner() -> None:
    assert owner_id_from_user(None) == DEPLOYMENT_OWNER_ID
    assert owner_id_from_user(UserContext(user_id="anonymous", auth_mode="none")) == (
        DEPLOYMENT_OWNER_ID
    )


def test_two_presets_state_a_place_and_not_a_role() -> None:
    """`common_user` reads a corpus and contributes to it; `admin` is every Action."""
    assert set(ACTION_PRESETS) == {"common_user", "admin"}
    assert ACTION_PRESETS["admin"] == ("*",)
    common_user = set(ACTION_PRESETS["common_user"])
    assert common_user == {
        AccessAction.WORKSPACE_QUERY,
        AccessAction.WORKSPACE_LIST_FILES,
        AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
        AccessAction.WORKSPACE_READ_METADATA,
        AccessAction.WORKSPACE_READ_VISUAL_ASSET,
        AccessAction.WORKSPACE_INGEST,
        AccessAction.WORKSPACE_UPDATE_METADATA,
        AccessAction.WORKSPACE_DELETE_FILES,
    }
    # Operator-facing and deployment-wide Actions stay with `admin`, never a common user.
    for operator_only in (
        AccessAction.WORKSPACE_CREATE,
        AccessAction.WORKSPACE_RESET,
        AccessAction.WORKSPACE_DELETE,
        AccessAction.WORKSPACE_STORAGE_STATUS,
        AccessAction.MODEL_CATALOGUE_WRITE,
        AccessAction.RUNTIME_POLICY_MANAGE,
    ):
        assert operator_only not in common_user


@pytest.mark.parametrize(
    "pattern",
    ["reader", "editor", "commonuser", "commom_user", "workpsace.query", "", "workspace."],
)
def test_a_pattern_that_can_never_match_a_product_action_is_unknown(pattern: str) -> None:
    """These grant nothing, so Access configuration refuses them instead of loading them."""
    assert unknown_action_patterns([pattern]) == (pattern,)


@pytest.mark.parametrize(
    "pattern",
    ["admin", "common_user", "*", AccessAction.WORKSPACE_QUERY, "workspace.*"],
)
def test_a_preset_an_action_and_an_action_namespace_are_known(pattern: str) -> None:
    assert unknown_action_patterns([pattern]) == ()
