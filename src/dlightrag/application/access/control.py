# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authorization policy for DlightRAG product resources."""

from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from dlightrag.application.access.principal import Principal, owner_id_from_user

type AccessSubject = Principal | None


@dataclass(frozen=True, slots=True)
class AccessRule:
    claim: str
    value: str
    workspaces: tuple[str, ...]
    actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AccessSettings:
    # No rules: every authenticated caller holds everything.
    rules: tuple[AccessRule, ...] = ()


class AccessAction:
    WORKSPACE_QUERY = "workspace.query"
    WORKSPACE_INGEST = "workspace.ingest"
    WORKSPACE_LIST_FILES = "workspace.list_files"
    WORKSPACE_DELETE_FILES = "workspace.delete_files"
    WORKSPACE_DOWNLOAD_SOURCE = "workspace.download_source"
    WORKSPACE_READ_METADATA = "workspace.read_metadata"
    WORKSPACE_UPDATE_METADATA = "workspace.update_metadata"
    WORKSPACE_READ_VISUAL_ASSET = "workspace.read_visual_asset"
    WORKSPACE_CREATE = "workspace.create"
    WORKSPACE_RESET = "workspace.reset"
    WORKSPACE_DELETE = "workspace.delete"
    # Storage/promotion facts are operator-facing: only the admin preset (and
    # explicitly granted rules) carry this action; ordinary readers/editors
    # never see tier, promotion state, or retry details.
    WORKSPACE_STORAGE_STATUS = "workspace.storage_status"
    MODEL_CATALOGUE_WRITE = "model_catalogue.write"
    # Reading and changing the Runtime Policies. Deployment-wide like the catalogue, so a rule
    # needs every workspace, and in no preset: a policy names the people it targets.
    RUNTIME_POLICY_MANAGE = "runtime_policy.manage"


# What a caller may do with a corpus it can reach, and nothing that changes it.
_READ_ACTIONS: tuple[str, ...] = (
    AccessAction.WORKSPACE_QUERY,
    AccessAction.WORKSPACE_LIST_FILES,
    AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
    AccessAction.WORKSPACE_READ_METADATA,
    AccessAction.WORKSPACE_READ_VISUAL_ASSET,
)
# An ordinary caller reads a corpus and contributes to it: ingestion, metadata
# updates, and file deletion. No preset is read-only; a deployment that wants a
# read-only grant names the read Actions themselves, which every rule may do.
_COMMON_USER_ACTIONS: tuple[str, ...] = (
    *_READ_ACTIONS,
    AccessAction.WORKSPACE_INGEST,
    AccessAction.WORKSPACE_UPDATE_METADATA,
    AccessAction.WORKSPACE_DELETE_FILES,
)
# A workspace's creator holds everything a common user does on it, and may reset
# or delete it. Operator-facing storage status stays with deployment rules.
CREATOR_ACTIONS: tuple[str, ...] = (
    *_COMMON_USER_ACTIONS,
    AccessAction.WORKSPACE_RESET,
    AccessAction.WORKSPACE_DELETE,
)
# The presets a rule may name. Two, so a rule states a place in the deployment
# rather than a role: an administrator, or an ordinary caller.
ACTION_PRESETS: dict[str, tuple[str, ...]] = {
    "common_user": _COMMON_USER_ACTIONS,
    "admin": ("*",),
}
_ACTION_NAMES: frozenset[str] = frozenset(
    value
    for name, value in vars(AccessAction).items()
    if not name.startswith("_") and isinstance(value, str)
)


def unknown_action_patterns(patterns: Sequence[str]) -> tuple[str, ...]:
    """The patterns among these that could never match a product Action.

    A rule matches nothing through such a pattern, so a deployment that wrote one
    would deny the callers it meant to allow and say nothing about it. Access
    configuration refuses them instead of loading them.
    """
    return tuple(pattern for pattern in patterns if not _action_pattern_is_known(pattern))


def _action_pattern_is_known(pattern: str) -> bool:
    """Whether this pattern names a preset, an Action, or an Action namespace."""
    if pattern == "*" or pattern in ACTION_PRESETS or pattern in _ACTION_NAMES:
        return True
    return pattern.endswith(".*") and any(name.startswith(pattern[:-1]) for name in _ACTION_NAMES)


# Keyed by Corpus Mutation action; a test keeps it in step with that action list,
# which this package may not import.
CORPUS_MUTATION_ACCESS_ACTIONS: Mapping[str, str] = MappingProxyType(
    {
        "ingest": AccessAction.WORKSPACE_INGEST,
        "replace": AccessAction.WORKSPACE_INGEST,
        "retry": AccessAction.WORKSPACE_INGEST,
        "delete": AccessAction.WORKSPACE_DELETE_FILES,
        "reset": AccessAction.WORKSPACE_RESET,
        "delete_workspace": AccessAction.WORKSPACE_DELETE,
    }
)


def corpus_mutation_access_action(action: object) -> str:
    """Map one mutation's accepted action to its authorization boundary.

    Unknown values fail closed to the strongest corpus mutation permission.
    """
    return CORPUS_MUTATION_ACCESS_ACTIONS.get(str(action or ""), AccessAction.WORKSPACE_DELETE)


class AccessDeniedError(PermissionError):
    """Raised when an authenticated user is not authorized for a resource."""


class WorkspaceCreators(Protocol):
    """The owner that created each of these workspaces, where one did."""

    async def workspace_creators(self, workspaces: Sequence[str]) -> Mapping[str, str]: ...


class AccessControl(Protocol):
    async def check(
        self,
        subject: AccessSubject,
        action: str,
        *,
        workspace: str | None = None,
    ) -> None: ...

    async def filter_workspaces(
        self,
        subject: AccessSubject,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]: ...

    async def filter_run_submitters(
        self,
        subject: AccessSubject,
        *,
        workspace: str,
        submitters: Set[str],
    ) -> set[str]:
        """The submitters among these whose Runs on ``workspace`` the subject may see."""
        ...


class AllowAllAccessControl:
    async def check(
        self,
        subject: AccessSubject,
        action: str,
        *,
        workspace: str | None = None,
    ) -> None:
        return None

    async def filter_workspaces(
        self,
        subject: AccessSubject,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]:
        return list(workspaces)

    async def filter_run_submitters(
        self,
        subject: AccessSubject,
        *,
        workspace: str,
        submitters: Set[str],
    ) -> set[str]:
        return set(submitters)


class JwtClaimsAccessControl:
    """Deployment rules over JWT claims; each workspace's creator also holds it."""

    def __init__(self, settings: AccessSettings, creators: WorkspaceCreators) -> None:
        self._rules = settings.rules
        self._creators = creators

    async def check(
        self,
        subject: AccessSubject,
        action: str,
        *,
        workspace: str | None = None,
    ) -> None:
        if workspace is None:
            allowed = self._ruled(subject, action, None)
        else:
            allowed = bool(await self.filter_workspaces(subject, action, [workspace]))
        if allowed:
            return
        target = f" workspace={workspace}" if workspace else ""
        raise AccessDeniedError(f"Access denied for action={action}{target}")

    async def filter_workspaces(
        self,
        subject: AccessSubject,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]:
        ruled = {workspace for workspace in workspaces if self._ruled(subject, action, workspace)}
        created = await self._created(
            subject, action, [workspace for workspace in workspaces if workspace not in ruled]
        )
        return [workspace for workspace in workspaces if workspace in ruled | created]

    async def filter_run_submitters(
        self,
        subject: AccessSubject,
        *,
        workspace: str,
        submitters: Set[str],
    ) -> set[str]:
        """A Run shows to its submitter; others see it by listing its workspace's files.

        A workspace its creator holds shows others only the Runs that creator
        submitted, so a deleted workspace's history never passes to whoever
        creates the same name next.
        """
        own = (
            {owner_id_from_user(subject)} & submitters
            if subject is not None and subject.auth_mode == "jwt"
            else set()
        )
        others = submitters - own
        if not others or not await self.filter_workspaces(
            subject, AccessAction.WORKSPACE_LIST_FILES, [workspace]
        ):
            return own
        creator = (await self._creators.workspace_creators([workspace])).get(workspace)
        return own | (others if creator is None else others & {creator})

    def _ruled(self, subject: AccessSubject, action: str, workspace: str | None) -> bool:
        if subject is None or subject.auth_mode != "jwt":
            return False
        return any(
            claim_matches(subject.claims, rule.claim, rule.value)
            and _action_matches(rule.actions, action)
            and _workspace_matches(rule.workspaces, workspace)
            for rule in self._rules
        )

    async def _created(
        self,
        subject: AccessSubject,
        action: str,
        workspaces: Sequence[str],
    ) -> set[str]:
        """The workspaces among these that this subject created, for a creator action."""
        if (
            not workspaces
            or subject is None
            or subject.auth_mode != "jwt"
            or action not in CREATOR_ACTIONS
        ):
            return set()
        creator = owner_id_from_user(subject)
        creators = await self._creators.workspace_creators(workspaces)
        return {workspace for workspace in workspaces if creators.get(workspace) == creator}


def access_control_from_settings(
    settings: AccessSettings,
    *,
    creators: WorkspaceCreators,
) -> AccessControl:
    if settings.rules:
        return JwtClaimsAccessControl(settings, creators)
    return AllowAllAccessControl()


def claim_matches(claims: Mapping[str, object], claim_name: str, expected: str) -> bool:
    """Whether a verified claim holds the expected value.

    A claim of one value holds it when they are equal, and a claim of several (a list of roles
    or groups) holds it when it is one of them. Access Rules and Runtime Policies both name the
    callers they apply to this way.
    """
    raw = claims.get(claim_name)
    if isinstance(raw, str):
        return raw == expected
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, Mapping)):
        return expected in {str(value) for value in raw}
    return str(raw) == expected if raw is not None else False


def _action_matches(patterns: Sequence[str], action: str) -> bool:
    return any(_pattern_allows_action(pattern, action) for pattern in patterns)


def _pattern_allows_action(pattern: str, action: str) -> bool:
    preset = ACTION_PRESETS.get(pattern)
    if preset is not None:
        return any(_pattern_allows_action(entry, action) for entry in preset)
    return (
        pattern == "*"
        or pattern == action
        or (pattern.endswith(".*") and action.startswith(pattern[:-1]))
    )


def _workspace_matches(patterns: Sequence[str], workspace: str | None) -> bool:
    return any(pattern == "*" or pattern == workspace for pattern in patterns)


__all__ = [
    "ACTION_PRESETS",
    "AccessRule",
    "AccessSettings",
    "AccessAction",
    "AccessControl",
    "AccessDeniedError",
    "AccessSubject",
    "AllowAllAccessControl",
    "JwtClaimsAccessControl",
    "WorkspaceCreators",
    "access_control_from_settings",
    "claim_matches",
    "unknown_action_patterns",
]
