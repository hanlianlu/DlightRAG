# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Research Run's browser may register only if the Run was accepted able to (ADR 0034)."""

from pathlib import Path
from typing import Any

import pytest

from dlightrag.engine.answer.agent_browser import AgentBrowserBinding
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from tests.support.agent_browser import FakeProvider, browser_settings, idle_accounts_binding
from tests.support.research_run import research_rig
from tests.unit.test_answer_executor import _executor
from tests.unit.test_answer_service import _fingerprint, _request, _Retrieval, _service, _Store


@pytest.mark.parametrize("registers", [True, False])
async def test_a_research_run_executes_with_the_browser_it_was_accepted_with(
    registers: bool, tmp_path: Path
) -> None:
    """The tools the model is offered are the ones the Run's accepted plan holds, so a Run
    accepted without registration is never offered it, and one accepted with it is."""
    async with ResourceRegistry() as registry:
        rig = research_rig(tmp_path=tmp_path, planning=_Retrieval(), registry=registry)
        executor = _executor(
            **rig.collaborators,
            model_invocation_fingerprint_for_role=_fingerprint,
            browser=AgentBrowserBinding(
                FakeProvider(), browser_settings(), idle_accounts_binding()
            ),
        )

        async def registration(**_: Any) -> bool:
            return registers

        store = _Store()
        service = _service(
            store=store,
            agent_registration=registration,
            research_tool_declarations=executor.research_tool_declarations,
        )
        await service.create(request=_request(mode="research"), owner_id="owner-1")

        await rig.run(executor, store.created[0]["prepared_input"])

    assert len(rig.offered) == 1, "the Run never reached its model"
    browser = rig.offered[0]["browser"]
    actions = set(browser.parameters["properties"]["action"]["enum"])
    assert ("register" in actions, "login" in actions) == (registers, True)
