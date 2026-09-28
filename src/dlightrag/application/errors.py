# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Errors owned by the Application lifecycle."""

import math

from dlightrag.engine.dependencies import TransientDependencyError
from dlightrag.engine.runtime.errors import RunSchemaError


class ApplicationError(Exception):
    """A typed Application outcome whose message is safe to show the caller.

    Transports map each family once rather than per route: an unavailable
    outcome may succeed later, a conflict names durable state the caller can
    observe or change. A subclass belongs to exactly one family; the base is
    never raised. Anything else that escapes a use case is internal.
    """


class ApplicationUnavailableError(ApplicationError, RuntimeError):
    """The Application cannot take this request now; the same request may succeed later."""


class ApplicationConflictError(ApplicationError, RuntimeError):
    """The request conflicts with durable state the caller can observe or change."""


class ApplicationClosedError(ApplicationUnavailableError):
    """Raised when a closed Application is asked for one of its services."""

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or "Application is shutting down")


class CorpusUnavailableError(TransientDependencyError, ApplicationUnavailableError):
    """An Application use case cannot currently reach corpus state."""

    def __init__(self, detail: str | None = None) -> None:
        super().__init__("corpus_storage", detail or "Corpus storage is temporarily unavailable")


class StorageSchemaError(RuntimeError):
    """Durable storage schema is incompatible with this revision."""


class WorkspaceWriteFencedError(ApplicationConflictError):
    """A workspace write was refused while its promotion fence is active.

    Retryable: transports surface HTTP 409 with a ``Retry-After`` header.
    """

    def __init__(self, *, workspace: str, retry_after_seconds: float) -> None:
        self.workspace = workspace
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Workspace '{workspace}' is being promoted to dedicated storage; "
            f"retry after {int(math.ceil(retry_after_seconds))} seconds"
        )


__all__ = [
    "ApplicationClosedError",
    "ApplicationConflictError",
    "ApplicationError",
    "ApplicationUnavailableError",
    "CorpusUnavailableError",
    "RunSchemaError",
    "StorageSchemaError",
    "WorkspaceWriteFencedError",
]
