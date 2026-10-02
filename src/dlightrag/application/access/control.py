# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authorization policy for DlightRAG product resources."""

from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

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
    mode: Literal["allow_all", "jwt_claims"] = "allow_all"
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


_READER_ACTIONS: tuple[str, ...] = (
    AccessAction.WORKSPACE_QUERY,
    AccessAction.WORKSPACE_LIST_FILES,
    AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
    AccessAction.WORKSPACE_READ_METADATA,
    AccessAction.WORKSPACE_READ_VISUAL_ASSET,
)
_EDITOR_ACTIONS: tuple[str, ...] = (
    *_READER_ACTIONS,
    AccessAction.WORKSPACE_INGEST,
    AccessAction.WORKSPACE_UPDATE_METADATA,
    AccessAction.WORKSPACE_DELETE_FILES,
)
# A workspace's creator holds everything an editor does on it, and may reset or
# delete it. Operator-facing storage status stays with deployment rules.
CREATOR_ACTIONS: tuple[str, ...] = (
    *_EDITOR_ACTIONS,
    AccessAction.WORKSPACE_RESET,
    AccessAction.WORKSPACE_DELETE,
)
ACTION_PRESETS: dict[str, tuple[str, ...]] = {
    "reader": _READER_ACTIONS,
    "editor": _EDITOR_ACTIONS,
    "admin": ("*",),
}


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
            _claim_matches(subject.claims, rule.claim, rule.value)
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
    if settings.mode == "jwt_claims":
        return JwtClaimsAccessControl(settings, creators)
    return AllowAllAccessControl()


def _claim_matches(claims: Mapping[str, object], claim_name: str, expected: str) -> bool:
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
]
