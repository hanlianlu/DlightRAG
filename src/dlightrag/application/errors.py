# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Errors owned by the Application lifecycle."""

import math

from dlightrag.engine.dependencies import DependencyComponent, TransientDependencyError
from dlightrag.engine.runtime.errors import RunSchemaError


class ApplicationError(Exception):
    """A typed Application outcome whose message is safe to show the caller.

    Transports map each family once rather than per route: an unavailable
    outcome may succeed later, a conflict names durable state the caller can
    observe or change, and an input error names what the caller must change.
    A subclass belongs to exactly one family; the base is never raised.
    Anything else that escapes a use case is internal.
    """


class ApplicationUnavailableError(ApplicationError, RuntimeError):
    """The Application cannot take this request now; the same request may succeed later."""


class ApplicationConflictError(ApplicationError, RuntimeError):
    """The request conflicts with durable state the caller can observe or change."""


class ApplicationInputError(ApplicationError, ValueError):
    """The request itself is invalid; the caller must change it before retrying."""


class ApplicationNotFoundError(ApplicationError, LookupError):
    """The named thing does not exist, or no longer does."""


class ApplicationClosedError(ApplicationUnavailableError):
    """Raised when a closed Application is asked for one of its services."""

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or "Application is shutting down")


_UNAVAILABLE_DETAILS: dict[DependencyComponent, str] = {
    "corpus_storage": "Corpus storage is temporarily unavailable",
    "parser": "The document parser is temporarily unavailable",
    "providers": "The model provider is temporarily unavailable",
}


class DependencyUnavailableError(TransientDependencyError, ApplicationUnavailableError):
    """An Application use case cannot currently reach a dependency it needs."""

    def __init__(self, component: DependencyComponent, detail: str | None = None) -> None:
        super().__init__(component, detail or _UNAVAILABLE_DETAILS[component])


class CorpusUnavailableError(DependencyUnavailableError):
    """An Application use case cannot currently reach corpus state."""

    def __init__(self, detail: str | None = None) -> None:
        super().__init__("corpus_storage", detail)


def dependency_unavailable(component: DependencyComponent) -> DependencyUnavailableError:
    """The Application error for a dependency that is briefly out; its text names it."""
    if component == "corpus_storage":
        return CorpusUnavailableError()
    return DependencyUnavailableError(component)


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
    "ApplicationInputError",
    "ApplicationNotFoundError",
    "ApplicationUnavailableError",
    "CorpusUnavailableError",
    "DependencyUnavailableError",
    "RunSchemaError",
    "StorageSchemaError",
    "WorkspaceWriteFencedError",
    "dependency_unavailable",
]
