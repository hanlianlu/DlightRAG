# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Public exceptions one Memory operation may raise."""


class MemoryWriteRejectedError(Exception):
    """A named Memory Write failed the closed checklist."""

    error_kind = "memory_write_rejected"

    def __init__(self, public_message: str) -> None:
        super().__init__(public_message)
        self.public_message = public_message


class MemoryUnavailableError(Exception):
    """This principal cannot write or auto-recall Memory Records."""

    error_kind = "memory_unavailable"

    def __init__(self) -> None:
        super().__init__("Long-term memory requires a JWT owner.")
        self.public_message = "Long-term memory requires a JWT owner."


__all__ = ["MemoryUnavailableError", "MemoryWriteRejectedError"]
