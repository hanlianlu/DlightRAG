# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The bounded backoff every transient PostgreSQL retry shares."""

from unittest.mock import AsyncMock, patch

import pytest

from dlightrag.adapters.postgres.core._retry import retry_transient


class _Transient(Exception):
    pass


async def _retry(operation: AsyncMock, *, attempts: int) -> object:
    return await retry_transient(
        operation,
        is_transient=lambda exc: isinstance(exc, _Transient),
        attempts=attempts,
        backoff=1.0,
        backoff_max=3.0,
        subject="test store",
    )


async def test_a_transient_failure_is_retried_with_a_capped_doubling_pause() -> None:
    operation = AsyncMock(side_effect=[_Transient(), _Transient(), _Transient(), "ok"])

    with patch("asyncio.sleep", new=AsyncMock()) as sleep:
        assert await _retry(operation, attempts=4) == "ok"

    assert [call.args[0] for call in sleep.await_args_list] == [1.0, 2.0, 3.0]


async def test_the_last_attempts_failure_propagates() -> None:
    operation = AsyncMock(side_effect=_Transient("down"))

    with patch("asyncio.sleep", new=AsyncMock()) as sleep, pytest.raises(_Transient):
        await _retry(operation, attempts=3)

    assert operation.await_count == 3
    assert sleep.await_count == 2


async def test_a_failure_that_is_not_transient_is_not_retried() -> None:
    operation = AsyncMock(side_effect=ValueError("bad statement"))

    with patch("asyncio.sleep", new=AsyncMock()) as sleep, pytest.raises(ValueError):
        await _retry(operation, attempts=5)

    assert operation.await_count == 1
    sleep.assert_not_awaited()
