# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""An Answer is accepted under the Runtime Policies as they stand for its caller, once.

The Answer service is the real one over the real policy service, whose store keeps its documents
in memory, and the run store records what acceptance handed it. So what these tests observe is what
a Run is pinned to, what the store is asked to admit it under, and that nothing is admitted or
replayed on a policy that cannot be read.
"""

from typing import Any

import pytest

from dlightrag.application.policies.service import (
    ACTIVE_ANSWER_RUNS,
    AGENT_EFFORT_CEILING,
    PolicyUnreadableError,
    RuntimePolicies,
    StoredPolicy,
)
from dlightrag.engine.runtime.records import RunCreation
from tests.support.callers import ANONYMOUS, signed_in
from tests.support.policy_store import InMemoryPolicyStore
from tests.unit.test_answer_service import _OWNER, _record, _request, _service, _Store

_INTERN = signed_in("intern@example.com")
_LEAD = signed_in("lead@example.com")
_LOW_FOR_THE_INTERN = {
    "default": "max",
    "rules": [{"claim": "email", "value": "intern@example.com", "then": "low"}],
}
_ONE_AT_A_TIME_FOR_THE_INTERN = {
    "default": None,
    "rules": [{"claim": "email", "value": "intern@example.com", "then": 1}],
}
_THREE_EXCEPT_THE_LEAD = {
    "default": 3,
    "rules": [{"claim": "email", "value": "lead@example.com", "then": None}],
}


async def _policies(**documents: dict[str, Any]) -> tuple[RuntimePolicies, InMemoryPolicyStore]:
    store = InMemoryPolicyStore()
    policies = RuntimePolicies(store=store)
    for key, document in documents.items():
        await policies.write(key, document, "admin")
    return policies, store


@pytest.mark.parametrize(
    ("caller", "ceiling"), [(_INTERN, "low"), (_LEAD, "max"), (ANONYMOUS, "max")]
)
async def test_a_run_is_pinned_to_the_ceiling_its_caller_has_and_keeps_what_they_chose(
    caller, ceiling
) -> None:
    policies, _ = await _policies(**{AGENT_EFFORT_CEILING: _LOW_FOR_THE_INTERN})
    store = _Store()

    await _service(store=store, policies=policies).create(
        request=_request(mode="research", effort="max"), owner_id=_OWNER, caller=caller
    )

    prepared = store.created[0]["prepared_input"]
    assert (prepared["effort"], prepared["effort_ceiling"]) == ("max", ceiling)


async def test_the_ceiling_is_no_part_of_what_a_caller_asked() -> None:
    """A policy that changes between a submission and its retry must not turn the retry into a
    conflict, so the fingerprint a key is compared by is the request's, never the ceiling's."""
    policies, _ = await _policies(**{AGENT_EFFORT_CEILING: {"default": "max"}})
    store = _Store()
    service = _service(store=store, policies=policies)

    await service.create(
        request=_request(effort="high"), owner_id=_OWNER, caller=_LEAD, idempotency_key="k"
    )
    await policies.write(AGENT_EFFORT_CEILING, {"default": "low"}, "admin")
    await service.create(
        request=_request(effort="high"), owner_id=_OWNER, caller=_LEAD, idempotency_key="k"
    )

    first, second = store.created
    assert (
        first["prepared_input"]["effort_ceiling"],
        second["prepared_input"]["effort_ceiling"],
    ) == (
        "max",
        "low",
    )
    assert first["idempotency_fingerprint"] == second["idempotency_fingerprint"]


async def test_the_active_run_limit_is_a_condition_of_the_acceptance_and_is_not_stored() -> None:
    policies, _ = await _policies(**{ACTIVE_ANSWER_RUNS: _THREE_EXCEPT_THE_LEAD})
    store = _Store()
    service = _service(store=store, policies=policies)

    for caller in (_INTERN, _LEAD):
        await service.create(request=_request(), owner_id=_OWNER, caller=caller)

    limited, unlimited = store.created
    assert (limited["envelope"].active_run_limit, unlimited["envelope"].active_run_limit) == (
        3,
        None,
    )
    assert all("active_run_limit" not in row["prepared_input"] for row in store.created)
    assert all("active_run_limit" not in row["envelope"].accepted_input for row in store.created)


async def test_an_unlimited_deployment_asks_the_store_for_no_limit() -> None:
    store = _Store()

    await _service(store=store).create(request=_request(), owner_id=_OWNER, caller=ANONYMOUS)

    assert store.created[0]["envelope"].active_run_limit is None


class _CountingStore(InMemoryPolicyStore):
    """A policy store that says how often it was read."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def read_runtime_policy(self, key: str) -> StoredPolicy | None:
        self.reads += 1
        return await super().read_runtime_policy(key)


async def test_a_replay_is_answered_without_reading_a_policy() -> None:
    """A retried key returns its Run whatever the policies have become since: no limit refuses
    it and no unreadable document can."""
    policy_store = _CountingStore()
    accepted = RunCreation(run=_record(run_id="accepted"), replayed=True)
    store = _Store(replay=accepted)
    service = _service(store=store, policies=RuntimePolicies(store=policy_store))

    creation = await service.create(
        request=_request(), owner_id=_OWNER, caller=_INTERN, idempotency_key="k"
    )

    assert creation.replayed and creation.run.run_id == "accepted"
    assert policy_store.reads == 0
    assert store.created == []


@pytest.mark.parametrize("key", [AGENT_EFFORT_CEILING, ACTIVE_ANSWER_RUNS])
async def test_a_submission_is_refused_while_a_policy_cannot_be_read(key: str) -> None:
    policy_store = InMemoryPolicyStore()
    policy_store.rows[key] = StoredPolicy(
        value={"default": "everything"},
        updated_at=_record().created_at,
        updated_by="admin",
    )
    store = _Store()
    service = _service(store=store, policies=RuntimePolicies(store=policy_store))

    with pytest.raises(PolicyUnreadableError) as refused:
        await service.create(request=_request(), owner_id=_OWNER, caller=_LEAD)

    assert key in str(refused.value)
    assert "everything" not in str(refused.value), "a document is named by where it is wrong only"
    assert store.created == []


@pytest.mark.parametrize("continuation", ["follow_up", "fork"])
async def test_a_continuation_is_accepted_under_the_policies_of_whoever_continues(
    continuation: str,
) -> None:
    policies, _ = await _policies(
        **{
            AGENT_EFFORT_CEILING: _LOW_FOR_THE_INTERN,
            ACTIVE_ANSWER_RUNS: _ONE_AT_A_TIME_FOR_THE_INTERN,
        }
    )
    parent = _record(
        status="succeeded",
        result={"answer": "the parent's answer"},
        accepted_input={
            "query": "the parent's question",
            "workspaces": ["finance"],
            "mode": "auto",
            "agent_session_id": "0199a0a0-0000-7000-8000-000000000099",
            "agent_lane_id": "main",
        },
    )
    store = _Store(run=parent)
    service = _service(store=store, policies=policies)

    for caller in (_INTERN, _LEAD):
        await getattr(service, continuation)(
            owner_id=_OWNER,
            caller=caller,
            run_id="run-1",
            query="and then?",
            authorized_workspaces=("finance",),
        )

    intern, lead = store.created
    assert intern["prepared_input"]["effort_ceiling"] == "low"
    assert intern["envelope"].active_run_limit == 1
    assert lead["prepared_input"]["effort_ceiling"] == "max"
    assert lead["envelope"].active_run_limit is None
