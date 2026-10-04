# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Run is pinned to whether its Agent may register, over a real owner switch and Run store.

Acceptance reads the owner's switch once, under the deployment's allowance, and stores it with the
Run and with the Run's plan. Execution derives the Run's tools from what was stored, so nothing
the owner switches afterwards changes a Run that was already accepted.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dlightrag.adapters.postgres.answer.agent_accounts import (
    PGAgentAccountSettingsStore,
    PGAgentAccountStore,
)
from dlightrag.adapters.postgres.runtime import PGRunBlobStore
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.application.agent_accounts import AgentAccounts
from dlightrag.application.answer_runs import AnswerRequest, AnswerService
from dlightrag.engine.agent.environment import AccessScheduler
from dlightrag.engine.agent.session.plan import AgentToolPlan
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.answer.agent_browser import (
    MAY_REGISTER_PIN,
    AgentAccountsBinding,
    AgentBrowserBinding,
    run_agent_accounts,
)
from dlightrag.engine.answer.execution.input import AnswerRunInput
from dlightrag.engine.answer.tools.browser import browser_tool
from dlightrag.engine.credential_cipher import CredentialCipher
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.integration.test_answer_run_api_pg import (
    _Capabilities,
    _CapabilityView,
    _fingerprint,
    _Resources,
    _Retrieval,
    _StoreScheduler,
)
from tests.support.agent_browser import FakeProvider, browser_settings, inert_browser_host
from tests.support.pg import skip_without_postgres
from tests.unit.test_answer_executor import _executor

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
async def _postgres() -> None:
    await skip_without_postgres()


@dataclass
class Deployment:
    """Run acceptance and execution over a scratch database, for a deployment that may or may
    not allow the Agent to register."""

    runs: PGRunStore
    accounts: AgentAccounts
    binding: AgentAccountsBinding
    service: AnswerService

    async def accept(self, owner: str) -> dict[str, Any]:
        """Accept a Research Run for ``owner`` and read back what was stored for it."""
        created = await self.service.create(
            request=AnswerRequest(query="Find it", workspaces=("default",), mode="research"),
            owner_id=owner,
        )
        record = await self.runs.get_run(owner_id=owner, run_id=created.run.run_id)
        assert record is not None and record.prepared_input is not None
        return dict(record.prepared_input)

    def planned(self, prepared: dict[str, Any]) -> AgentToolPlan:
        """The browser tool as the Run's accepted plan holds it."""
        plan = AnswerRunInput.from_prepared_input(prepared).agent_run_plan
        assert plan is not None
        return next(tool for tool in plan.tools if tool.name == "browser")

    def executed(self, owner: str, prepared: dict[str, Any]) -> AgentToolPlan:
        """The browser tool as a Run with this stored input executes with it."""
        host = replace(
            inert_browser_host(),
            accounts=run_agent_accounts(self.binding, owner_id=owner, prepared_input=prepared),
        )
        tool = browser_tool(
            host,
            environment=None,
            scheduler=AccessScheduler(),
            spill=None,
            image_preparer=None,
            child=False,
        )
        return AgentToolPlan.from_tool(tool)


def actions(plan: AgentToolPlan) -> set[str]:
    return set(plan.definition["parameters"]["properties"]["action"]["enum"])


async def deployment(allowed: bool) -> AsyncIterator[Deployment]:
    async with isolated_run_runtime("registration_pin") as (runs, pool):
        accounts_store = PGAgentAccountStore(pool=pool)
        binding = AgentAccountsBinding(
            accounts_store, CredentialCipher(None), registration_allowed=allowed
        )
        accounts = AgentAccounts(
            store=accounts_store,
            settings_store=PGAgentAccountSettingsStore(pool=pool),
            available=True,
            registration_allowed=allowed,
        )
        # Acceptance plans with what this deployment's executor composes, as production wires it.
        executor = _executor(
            execution_environment="disabled",
            browser=AgentBrowserBinding(FakeProvider(), browser_settings(), binding),
        )
        service = AnswerService(
            store=runs,
            blob_store=PGRunBlobStore(pool=pool),
            coordinator=cast(Any, _StoreScheduler(runs)),
            retrieval=cast(Any, _Retrieval()),
            capabilities=cast(Any, _Capabilities()),
            capability_view=cast(Any, _CapabilityView()),
            models=cast(
                Any,
                SimpleNamespace(
                    query_image_describer=lambda: None,
                    model_settings=lambda role: ModelSettings(model="test"),
                ),
            ),
            resources=cast(Any, _Resources()),
            model_invocation_fingerprint_for_role=_fingerprint,
            research_tool_declarations=executor.research_tool_declarations,
            agent_may_register=accounts.may_register,
            child_roster_cursor_secret=b"registration-pin-child-roster-test",
        )
        yield Deployment(runs, accounts, binding, service)


@pytest.fixture
async def allowing() -> AsyncIterator[Deployment]:
    async for web in deployment(True):
        yield web


@pytest.fixture
async def forbidding() -> AsyncIterator[Deployment]:
    async for web in deployment(False):
        yield web


async def test_a_run_keeps_the_switch_it_was_accepted_with_whatever_the_owner_switches_next(
    allowing: Deployment,
) -> None:
    first = await allowing.accept("alice")
    await allowing.accounts.set_sign_ups(owner_id="alice", enabled=False)
    second = await allowing.accept("alice")
    await allowing.accounts.set_sign_ups(owner_id="alice", enabled=True)

    # An owner who has not chosen has sign-ups on, and one who turned them off before
    # acceptance gets a plan without register, though login stays.
    assert (first[MAY_REGISTER_PIN], second[MAY_REGISTER_PIN]) == (True, False)
    assert actions(allowing.planned(first)) >= {"register", "login"}
    assert "register" not in actions(allowing.planned(second))
    assert "login" in actions(allowing.planned(second))
    # Switching after acceptance changes neither Run: each executes with the tool it was planned,
    # the one accepted with the switch on after it was turned off, and the other after it was on.
    assert allowing.executed("alice", first) == allowing.planned(first)
    assert allowing.executed("alice", second) == allowing.planned(second)
    # The switch is each owner's own.
    other = await allowing.accept("bob")
    assert other[MAY_REGISTER_PIN] is True


async def test_a_deployment_that_does_not_allow_registration_pins_every_run_to_login_alone(
    forbidding: Deployment,
) -> None:
    accepted = await forbidding.accept("alice")

    assert accepted[MAY_REGISTER_PIN] is False
    assert "register" not in actions(forbidding.planned(accepted))
    assert "login" in actions(forbidding.planned(accepted))
    assert forbidding.executed("alice", accepted) == forbidding.planned(accepted)
    # The owner's own switch is theirs to set, and the deployment is still the ceiling.
    await forbidding.accounts.set_sign_ups(owner_id="alice", enabled=True)
    assert (await forbidding.accept("alice"))[MAY_REGISTER_PIN] is False
