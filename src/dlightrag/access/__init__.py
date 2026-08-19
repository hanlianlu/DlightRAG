# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral principals, authorization, and workspace scope."""

from dlightrag.access.authentication import (
    AuthenticationError,
    AuthenticationErrorKind,
    AuthenticationSettings,
    authenticate_bearer_token,
)
from dlightrag.access.control import (
    ACTION_PRESETS,
    AccessAction,
    AccessControl,
    AccessDeniedError,
    AccessRule,
    AccessSettings,
    AccessSubject,
    AllowAllAccessControl,
    JwtClaimsAccessControl,
    Principal,
    access_control_from_settings,
)
from dlightrag.access.principal import (
    DEPLOYMENT_OWNER_ID,
    SIMPLE_OWNER_ID,
    UserContext,
    auth_mode_for_owner,
    owner_id_from_principal,
    owner_id_from_user,
)
from dlightrag.access.scope import RequestScope, current_request_scope, request_scope_context
from dlightrag.access.workspaces import (
    AccessGate,
    NoQueryableWorkspacesError,
    WorkspaceCatalog,
    WorkspaceRecord,
    WorkspaceSelectionConflictError,
    resolve_query_workspaces,
    validate_query_workspace_selection,
)

__all__ = [
    "ACTION_PRESETS",
    "AuthenticationError",
    "AuthenticationErrorKind",
    "AuthenticationSettings",
    "AccessAction",
    "AccessControl",
    "AccessDeniedError",
    "AccessGate",
    "AccessRule",
    "AccessSettings",
    "AccessSubject",
    "AllowAllAccessControl",
    "JwtClaimsAccessControl",
    "DEPLOYMENT_OWNER_ID",
    "SIMPLE_OWNER_ID",
    "NoQueryableWorkspacesError",
    "Principal",
    "RequestScope",
    "UserContext",
    "WorkspaceCatalog",
    "WorkspaceRecord",
    "WorkspaceSelectionConflictError",
    "access_control_from_settings",
    "authenticate_bearer_token",
    "auth_mode_for_owner",
    "current_request_scope",
    "owner_id_from_principal",
    "owner_id_from_user",
    "request_scope_context",
    "resolve_query_workspaces",
    "validate_query_workspace_selection",
]
