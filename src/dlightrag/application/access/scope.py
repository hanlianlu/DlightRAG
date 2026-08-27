# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-local authenticated principal for MCP middleware."""

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestScope:
    """Identity projected from one MCP request into shared ACL checks."""

    user_id: str = "anonymous"
    auth_mode: str = "none"
    claims: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def anonymous(cls) -> RequestScope:
        return cls()


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
