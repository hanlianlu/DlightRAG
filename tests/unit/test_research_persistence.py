# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Parent-fenced callbacks preserve operation outcomes and reject miswiring."""

from functools import partial
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from dlightrag.engine.answer.research.runtime import _check_child_write
from dlightrag.engine.runtime.coordinator import LeaseLostError


@pytest.mark.parametrize("result", [None, 0, 1, True, {"outcome": "accepted"}])
async def test_child_write_preserves_non_rejection_outcomes(result: Any) -> None:
    method = AsyncMock(return_value=result)
    write = _check_child_write(partial(method, worker_id="worker", fencing_epoch=3))

    assert await write(child_session_id="child") is result
    method.assert_awaited_once_with(child_session_id="child", worker_id="worker", fencing_epoch=3)


@pytest.mark.parametrize("false_is_lease_loss", [True, False])
async def test_false_retains_the_operations_lease_policy(false_is_lease_loss: bool) -> None:
    write = _check_child_write(
        partial(AsyncMock(return_value=False), worker_id="worker", fencing_epoch=3),
        false_is_lease_loss=false_is_lease_loss,
    )
    if false_is_lease_loss:
        with pytest.raises(LeaseLostError):
            await write()
    else:
        # Finish and guidance expiry must inspect the winning durable result.
        assert await write() is False


@pytest.mark.parametrize("override", [{"worker_id": "other"}, {"fencing_epoch": 4}])
async def test_unchecked_payload_cannot_replace_the_bound_parent_fence(override: dict) -> None:
    method = AsyncMock(return_value=True)
    write = _check_child_write(partial(method, worker_id="worker", fencing_epoch=3))

    with pytest.raises(TypeError, match="parent Run fence"):
        await write(**override)
    method.assert_not_awaited()


if TYPE_CHECKING:
    # These deliberate errors must keep failing Pyright. An unnecessary ignore
    # is itself an error, so restoring Any or **kwargs erasure breaks this gate.
    from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
    from dlightrag.engine.answer.research.persistence import (
        ClaimChild,
        RenewChild,
        ResearchRunStore,
    )

    store: ResearchRunStore = PGRunStore()
    bound = _check_child_write(
        partial(store.heartbeat_child_session, worker_id="worker", fencing_epoch=1)
    )
    renew: RenewChild = bound
    wrong_binding: ClaimChild = bound  # pyright: ignore[reportAssignmentType]
    missing_store: ResearchRunStore = object()  # pyright: ignore[reportAssignmentType]

    async def _callback_contract_probe() -> None:
        await bound(owner_id="o", run_id="r", child_session_id="c", child_fencing_epoch=1)
        await bound(owner_id="o", run_id="r", child_session_id="c")  # pyright: ignore[reportCallIssue]
        await bound(owner_id="o", run_id="r", child_session_id="c", child_fencing_epoch="bad")  # pyright: ignore[reportArgumentType]
        await renew(child_session_id="c", child_fencing_epoch=1)  # pyright: ignore[reportCallIssue]
