# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Describe a caller's invalid fields the same way on every transport."""

from pydantic import ValidationError


def invalid_fields(exc: ValidationError, *, within: str | None = None) -> str:
    """Name each invalid field and why, without echoing the submitted value.

    ``within`` prefixes locations when the model validates one argument, such as
    a filter, rather than the whole request.
    """
    described: list[str] = []
    for error in exc.errors(include_input=False, include_url=False):
        location = [within] if within else []
        location.extend(str(part) for part in error["loc"])
        described.append(f"{'.'.join(location) or 'request'}: {error['msg']}")
    return "; ".join(described)


__all__ = ["invalid_fields"]
