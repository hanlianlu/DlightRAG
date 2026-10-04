# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Research Run's browser may register only if the Run was accepted able to, and its deployment
still allows it (ADR 0034).

Each test accepts a Run for an owner and executes it with a real ``AnswerExecutor`` until its
model is first called, so what is observed is the tools the model is offered, or that the Run is
refused before it reaches the model.
"""

from pathlib import Path
from typing import Any

import pytest

from dlightrag.engine.answer.agent_browser import MAY_REGISTER_PIN, AgentBrowserBinding
from dlightrag.engine.answer.execution import AnswerExecutor
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from tests.support.agent_browser import FakeProvider, browser_settings, idle_accounts_binding
from tests.support.research_run import ResearchRig, research_rig
from tests.unit.test_answer_executor import _executor
from tests.unit.test_answer_service import _fingerprint, _request, _Retrieval, _service, _Store

_OWNER = "owner-1"


def _deployment(rig: ResearchRig, *, allowed: bool) -> AnswerExecutor:
    """The executor of a deployment with an Agent Browser that does or does not allow the Agent
    to register."""
    return _executor(
        **rig.collaborators,
        model_invocation_fingerprint_for_role=_fingerprint,
        browser=AgentBrowserBinding(
            FakeProvider(), browser_settings(), idle_accounts_binding(registration_allowed=allowed)
        ),
    )


async def _accepted(deployment: AnswerExecutor, *, registers: bool) -> dict[str, Any]:
    """What is stored for a Research Run that ``deployment`` accepts for the owner, whose
    permission to register is ``registers``; another owner's is the opposite."""

    async def may_register(*, owner_id: str) -> bool:
        return registers if owner_id == _OWNER else not registers

    store = _Store()
    service = _service(
        store=store,
        agent_may_register=may_register,
        research_tool_declarations=deployment.research_tool_declarations,
    )
    await service.create(request=_request(mode="research"), owner_id=_OWNER)
    return store.created[0]["prepared_input"]


@pytest.mark.parametrize(
    ("registers", "allowed"),
    [(True, True), (False, True), (False, False)],
    ids=["pinned-on", "pinned-off", "not-allowed"],
)
async def test_a_research_run_is_offered_register_exactly_as_it_was_accepted_able_to(
    registers: bool, allowed: bool, tmp_path: Path
) -> None:
    async with ResourceRegistry() as registry:
        rig = research_rig(tmp_path=tmp_path, planning=_Retrieval(), registry=registry)
        deployment = _deployment(rig, allowed=allowed)

        await rig.run(deployment, await _accepted(deployment, registers=registers))

    assert len(rig.offered) == 1, "the Run never reached its model"
    browser = rig.offered[0]["browser"]
    actions = set(browser.parameters["properties"]["action"]["enum"])
    # Login is every Run's, and register only the Run that was accepted able to.
    assert ("register" in actions, "login" in actions) == (registers, True)
    # A Run that cannot register is told, in what the model is sent, not to sign up by hand.
    assert ("sign-up form" in browser.description) == (not registers)


async def test_a_run_stored_with_no_pin_cannot_register_so_one_planned_to_is_refused(
    tmp_path: Path,
) -> None:
    async with ResourceRegistry() as registry:
        rig = research_rig(tmp_path=tmp_path, planning=_Retrieval(), registry=registry)
        deployment = _deployment(rig, allowed=True)
        accepted = await _accepted(deployment, registers=True)
        del accepted[MAY_REGISTER_PIN]

        await rig.run(deployment, accepted)

    assert rig.offered == []
    assert rig.failure is not None and rig.failure.kind == "incompatible_answer_run"


async def test_a_run_accepted_to_register_is_refused_once_its_deployment_withdraws_the_allowance(
    tmp_path: Path,
) -> None:
    async with ResourceRegistry() as registry:
        rig = research_rig(tmp_path=tmp_path, planning=_Retrieval(), registry=registry)
        accepted = await _accepted(_deployment(rig, allowed=True), registers=True)

        await rig.run(_deployment(rig, allowed=False), accepted)

    # Refused as incompatible, as any Run whose tools changed since it was accepted, and never
    # offered the model.
    assert rig.offered == []
    assert rig.failure is not None and rig.failure.kind == "incompatible_answer_run"
