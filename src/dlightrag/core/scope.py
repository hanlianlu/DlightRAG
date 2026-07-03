# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request and session scoping primitives."""

from __future__ import annotations

import contextvars
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from dlightrag.utils import normalize_workspace


def _clean_part(value: str) -> str:
    """Keep scoped keys unambiguous without changing their human-readable shape."""
    return value.replace("\\", "\\\\").replace(":", "\\:")


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

    This is intentionally not an authorization model. It only gives internal
    mutable request state, such as session images and checkpoints, a stable
    namespace across REST, Web, and MCP adapters.
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

    def session_key(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        workspace_part = ",".join(self.workspaces) if self.workspaces else "*"
        return ":".join(
            (
                _clean_part(self.auth_mode),
                _clean_part(self.user_id),
                _clean_part(workspace_part),
                _clean_part(session_id),
            )
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
