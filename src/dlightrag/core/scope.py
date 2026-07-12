# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request and session scoping primitives."""

import contextvars
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from dlightrag.utils import normalize_workspace


def _workspace_tuple(workspaces: Iterable[str] | None) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for workspace in workspaces or ():
        normalized = normalize_workspace(workspace)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(sorted(result))


@dataclass(frozen=True)
class RequestScope:
    """Identity and workspace scope for one external request.

    This is intentionally not an authorization model. It carries authenticated
    identity and authorized workspace context across REST, Web, and MCP adapters.
    """

    user_id: str = "anonymous"
    auth_mode: str = "none"
    workspaces: tuple[str, ...] = ()
    claims: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def anonymous(cls) -> RequestScope:
        return cls()

    @classmethod
    def from_user(cls, user: Any | None) -> RequestScope:
        if user is None:
            return cls.anonymous()
        return cls(
            user_id=str(getattr(user, "user_id", "") or "anonymous"),
            auth_mode=str(getattr(user, "auth_mode", "") or "none"),
            claims=dict(getattr(user, "claims", {}) or {}),
        )

    def for_workspaces(self, workspaces: Iterable[str] | None) -> RequestScope:
        return RequestScope(
            user_id=self.user_id,
            auth_mode=self.auth_mode,
            workspaces=_workspace_tuple(workspaces),
            claims=dict(self.claims),
        )


_CURRENT_SCOPE: contextvars.ContextVar[RequestScope | None] = contextvars.ContextVar(
    "dlightrag_request_scope",
    default=None,
)


def current_request_scope() -> RequestScope:
    """Return the current contextvar-backed request scope."""
    return _CURRENT_SCOPE.get() or RequestScope.anonymous()


@contextmanager
def request_scope_context(scope: RequestScope) -> Iterator[None]:
    """Temporarily set the contextvar-backed request scope."""
    token = _CURRENT_SCOPE.set(scope)
    try:
        yield
    finally:
        _CURRENT_SCOPE.reset(token)


__all__ = ["RequestScope", "current_request_scope", "request_scope_context"]
