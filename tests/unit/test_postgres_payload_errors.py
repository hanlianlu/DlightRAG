# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Payload-representability classification at the PostgreSQL adapter edge."""

import asyncpg
import pytest

from dlightrag.adapters.postgres.core._errors import (
    guard_payload,
    is_unrepresentable_payload,
)
from dlightrag.engine.agent.session.repository import UnrepresentablePayloadError


def test_nul_refusals_are_classified_as_payload_problems() -> None:
    assert is_unrepresentable_payload(
        asyncpg.exceptions.UntranslatableCharacterError("unsupported Unicode escape sequence")
    )
    assert is_unrepresentable_payload(
        asyncpg.exceptions.CharacterNotInRepertoireError(
            'invalid byte sequence for encoding "UTF8": 0x00'
        )
    )
    assert not is_unrepresentable_payload(asyncpg.exceptions.UniqueViolationError("duplicate"))
    assert not is_unrepresentable_payload(ValueError("not a database error"))


@pytest.mark.asyncio
async def test_guard_payload_names_the_refused_payload_without_echoing_it() -> None:
    async def refuse() -> None:
        raise asyncpg.exceptions.UntranslatableCharacterError("unsupported Unicode escape sequence")

    with pytest.raises(UnrepresentablePayloadError) as failure:
        await guard_payload(refuse(), surface="Session Register")

    assert "Session Register" in failure.value.detail
    assert "\\u0000" not in str(failure.value)


@pytest.mark.asyncio
async def test_guard_payload_passes_other_failures_through() -> None:
    async def refuse() -> None:
        raise asyncpg.exceptions.UniqueViolationError("duplicate key value")

    with pytest.raises(asyncpg.exceptions.UniqueViolationError):
        await guard_payload(refuse(), surface="Session Entry")
