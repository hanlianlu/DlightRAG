# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The passwords an Agent Session filled into pages, and what stands in for them (ADR 0034).

It depends on nothing else of the Agent Browser, so the port, the Agent Accounts and the Agent
Mailbox can each use it without importing one another.
"""

from __future__ import annotations

from pydantic import SecretStr

#: What stands in for a filled password in every text a page or a mail yields.
PASSWORD_MASK = "********"  # noqa: S105 - what replaces a password, not a password


class FilledPasswords:
    """The passwords DlightRAG filled into one Agent Session's pages in this Run (ADR 0034).

    Every text a page or a mail yields passes through it before the tool sees it. A generated
    password is made of letters, digits and ``-._``, which no encoding changes, and begins and
    ends with a letter or a digit, which a browser's naming of a download leaves alone, so the
    one spelling it has is all there is to find. A page that rewrites a value on purpose is not
    found.
    """

    def __init__(self) -> None:
        self._values: set[str] = set()

    def add(self, password: SecretStr) -> None:
        self._values.add(password.get_secret_value())

    def __bool__(self) -> bool:
        return bool(self._values)

    def found_in(self, text: str) -> bool:
        return any(value in text for value in self._values)

    def redact(self, text: str) -> str:
        """``text`` with every filled password replaced by the mask."""
        for value in self._values:
            text = text.replace(value, PASSWORD_MASK)
        return text

    def redact_bytes(self, data: bytes) -> bytes:
        """The same over UTF-8, for a page's serialized HTML and a downloaded file."""
        for value in self._values:
            data = data.replace(value.encode(), PASSWORD_MASK.encode())
        return data

    def __repr__(self) -> str:
        return f"FilledPasswords({len(self._values)} filled)"


__all__ = ["PASSWORD_MASK", "FilledPasswords"]
