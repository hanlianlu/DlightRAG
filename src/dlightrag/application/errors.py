# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Errors owned by the Application lifecycle."""

from dlightrag.engine.runtime import RunSchemaError


class ApplicationClosedError(RuntimeError):
    """Raised when a closed Application is asked for one of its services."""

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or "Application is shutting down"
        super().__init__(self.detail)


class CorpusUnavailableError(RuntimeError):
    """An Application use case cannot currently reach corpus state."""


class StorageSchemaError(RuntimeError):
    """Durable storage schema is incompatible with this revision."""


__all__ = [
    "ApplicationClosedError",
    "CorpusUnavailableError",
    "RunSchemaError",
    "StorageSchemaError",
]
