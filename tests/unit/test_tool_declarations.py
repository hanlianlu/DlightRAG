# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Declaration acceptance and runtime binding share one immutable tool contract."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from dlightrag.application.connections.models import CatalogueTool
from dlightrag.application.connections.service import Connections
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.agent.session.plan import AgentRunPlan, AgentToolPlan
from dlightrag.engine.agent.skills import SkillsBundleFactory
from dlightrag.engine.agent.tools import AgentTool, ToolDeclaration, ToolResult, ToolRuntime
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.tools.composition import (
    compose_research_tools,
    research_tool_declarations,
)
from dlightrag.engine.answer.tools.memory import MemoryHost
from dlightrag.engine.answer.tools.subagents import (
    SubagentHost,
    child_guidance_declarations,
    subagent_declarations,
)


class Arguments(BaseModel):
    query: str


async def test_binding_preserves_the_plan_and_adds_real_execution() -> None:
    declaration = ToolDeclaration(
        "lookup",
        "Look up one fact.",
        Arguments,
        guidance="Ask one question.",
        replay_policy="replayable",
        contract_version=7,
    )
    execute = AsyncMock(return_value=ToolResult.text("found"))
    assert not hasattr(declaration, "execute")

    bound = declaration.bind(execute)
    assert isinstance(bound, AgentTool)
    assert AgentToolPlan.from_tool(declaration) == AgentToolPlan.from_tool(bound)
    declared_plan = AgentRunPlan.from_tools(
        [declaration], model_role="query", context_policy_revision="test-policy"
    )
    bound_plan = AgentRunPlan.from_tools(
        [bound], model_role="query", context_policy_revision="test-policy"
    )
    assert declared_plan.canonical_json() == bound_plan.canonical_json()
    assert declared_plan.digest == bound_plan.digest
    assert declared_plan.tool_schema_tokens == bound_plan.tool_schema_tokens
    execute.assert_not_called()

    arguments = declaration.input_model.model_validate({"query": "fact"})
    runtime = ToolRuntime("call", "lookup", IntentId.new(), "run", AsyncMock())
    assert (await bound.execute(arguments, runtime)).text_content == "found"
    execute.assert_awaited_once_with(arguments, runtime)


@pytest.mark.parametrize(
    ("paths", "web", "memory", "skills", "child", "narrow"),
    [
        (False, False, False, False, False, None),
        (False, True, True, True, False, None),
        (True, False, False, False, False, None),
        (True, True, True, True, False, None),
        (False, True, True, True, True, None),
        (True, True, True, True, True, None),
        (True, True, True, True, True, ("read", "search_web")),
    ],
)
def test_research_acceptance_and_execution_use_identical_declarations(
    tmp_path: Path,
    paths: bool,
    web: bool,
    memory: bool,
    skills: bool,
    child: bool,
    narrow: tuple[str, ...] | None,
) -> None:
    factory = SkillsBundleFactory(global_root=tmp_path / "global", owner_root=tmp_path / "owners")
    model_guidance = "Configured model roles."
    connection = ToolDeclaration("mcp_test", "Remote tool.", Arguments)
    declared = research_tool_declarations(
        web_search=web,
        resource_read=True,
        resource_view=True,
        environment=paths,
        artifact_publication=paths,
        memory=memory,
        skills=factory.declarations(child=child) if skills else (),
        subagents=(
            child_guidance_declarations()
            if child
            else subagent_declarations(model_guidance=model_guidance)
        ),
        injected=[connection],
        child=child,
        tool_names=narrow,
    )
    bound = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=AsyncMock(),
        search_web=AsyncMock() if web else None,
        register_web_source=None,
        resource_reader=AsyncMock(),
        resource_viewer=AsyncMock(),
        environment=LocalExecutionEnvironment(tmp_path) if paths else None,
        artifacts_root=tmp_path / "artifacts" if paths else None,
        subagent_host=SubagentHost(model_guidance=model_guidance),
        memory_host=MemoryHost() if memory else None,
        skill_tools=factory("real-owner").tools(child=child) if skills else [],
        injected_tools=[connection.bind(AsyncMock())],
        child=child,
        tool_names=narrow,
    )
    assert [AgentToolPlan.from_tool(tool) for tool in bound] == [
        AgentToolPlan.from_tool(tool) for tool in declared
    ]
    assert all(type(tool) is ToolDeclaration for tool in declared)
    assert all(isinstance(tool, AgentTool) for tool in bound)
    if child:
        assert "attach_artifact" not in {tool.name for tool in declared}
        assert "remember" not in {tool.name for tool in declared}
        assert "ask_parent" in {tool.name for tool in declared}


def test_skill_declarations_need_no_owner_or_catalogue(tmp_path: Path, monkeypatch) -> None:
    from dlightrag.engine.agent.skills import SkillCatalog

    def forbid_discovery(**_kwargs):
        raise AssertionError("declaration must not discover files")

    monkeypatch.setattr(SkillCatalog, "discover", forbid_discovery)
    factory = SkillsBundleFactory(global_root=tmp_path / "global", owner_root=tmp_path / "owners")
    assert [tool.name for tool in factory.declarations(child=False)] == [
        "load_skill",
        "publish_skill",
        "delete_skill",
    ]
    assert [tool.name for tool in factory.declarations(child=True)] == ["load_skill"]
    assert list(tmp_path.iterdir()) == []


async def test_connection_acceptance_returns_only_the_stored_declaration() -> None:
    from dlightrag.engine.answer.execution.connection_binding import RunConnectionBinding

    schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    catalogue = CatalogueTool("lookup", "mcp_lookup", "Remote lookup.", schema)
    binding = RunConnectionBinding("owner", "a" * 32, 1, 1, "b" * 64)
    store = AsyncMock()
    store.research_catalogues.return_value = [(binding, (catalogue,))]
    mcp = AsyncMock()
    connections = Connections(store=cast(Any, store), mcp=cast(Any, mcp))

    accepted = await connections.bind_research(owner_id="owner", auth_mode="jwt")

    assert accepted.bindings == (binding,)
    assert len(accepted.tools) == 1
    declaration = accepted.tools[0]
    assert type(declaration) is ToolDeclaration
    assert declaration.definition.parameters == schema
    assert not hasattr(declaration, "execute")
    assert declaration.input_model.model_validate({"query": "fact"}).model_dump() == {
        "query": "fact"
    }
    with pytest.raises(ValueError):
        declaration.input_model.model_validate({"query": 3})
    assert mcp.mock_calls == []
