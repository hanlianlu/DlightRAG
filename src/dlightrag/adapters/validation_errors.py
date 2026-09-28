# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Describe a caller's invalid fields the same way on every transport."""

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

_VALUE_ERROR_PREFIX = "Value error, "


def describe_invalid_fields(
    errors: Iterable[Mapping[str, Any]], *, within: str | None = None
) -> str:
    """Name each error's location and reason; submitted values are never read.

    ``within`` prefixes locations when the model validates one argument, such as
    a filter, rather than the whole request.
    """
    described: list[str] = []
    for error in errors:
        location = [within] if within else []
        location.extend(str(part) for part in error.get("loc", ()))
        reason = str(error.get("msg", "invalid value"))
        if error.get("type") == "value_error":
            # A validator's own message already says what is wrong.
            reason = reason.removeprefix(_VALUE_ERROR_PREFIX)
        described.append(f"{'.'.join(location) or 'request'}: {reason}")
    return "; ".join(described)


def invalid_fields(exc: ValidationError, *, within: str | None = None) -> str:
    """Name each invalid field and why, without echoing the submitted value."""
    return describe_invalid_fields(
        exc.errors(include_input=False, include_url=False), within=within
    )


__all__ = ["describe_invalid_fields", "invalid_fields"]
