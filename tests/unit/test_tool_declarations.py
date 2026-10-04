# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Declaration acceptance and runtime binding share one immutable tool contract."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from dlightrag.application.connections.models import CatalogueTool
from dlightrag.application.connections.service import Connections
from dlightrag.engine.agent.environment import SearchToolchain
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.agent.session.plan import AgentRunPlan, AgentToolPlan
from dlightrag.engine.agent.skills import SkillsBundleFactory
from dlightrag.engine.agent.tools import AgentTool, ToolDeclaration, ToolResult, ToolRuntime
from dlightrag.engine.agent.tools.files import ls_declaration, materialize_declaration
from dlightrag.engine.answer.continuation_handles import SESSION_NOTE_DIRECTORY
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.execution.connection_binding import is_connection_tool
from dlightrag.engine.answer.resources.models import PUBLISHED_ARTIFACT_HANDLE_PREFIX
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
from dlightrag.engine.credential_cipher import CredentialCipher
from tests.support.agent_browser import inert_browser_host


class Arguments(BaseModel):
    query: str


async def test_binding_preserves_the_plan_and_adds_real_execution() -> None:
    declaration = ToolDeclaration(
        "lookup",
        "Look up one fact.",
        Arguments,
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
@pytest.mark.parametrize("agent_browser", [False, True])
def test_research_acceptance_and_execution_use_identical_declarations(
    tmp_path: Path,
    paths: bool,
    web: bool,
    memory: bool,
    skills: bool,
    child: bool,
    narrow: tuple[str, ...] | None,
    agent_browser: bool,
) -> None:
    factory = SkillsBundleFactory(global_root=tmp_path / "global", owner_root=tmp_path / "owners")
    model_guidance = "Configured model roles."
    connection = ToolDeclaration("mcp_test", "Remote tool.", Arguments)
    declared = research_tool_declarations(
        web_search=web,
        resource_read=True,
        agent_browser=agent_browser,
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
        browser=inert_browser_host() if agent_browser else None,
        resource_viewer=AsyncMock(),
        admitted_bytes_reader=AsyncMock(),
        environment=LocalExecutionEnvironment(tmp_path) if paths else None,
        artifacts_root=tmp_path / "artifacts" if paths else None,
        subagent_host=SubagentHost(model_guidance=model_guidance),
        memory_host=MemoryHost() if memory else None,
        skill_tools=factory("real-owner").tools(child=child) if skills else [],
        injected_tools=[connection.bind(AsyncMock())],
        child=child,
        tool_names=narrow,
        search_toolchain=SearchToolchain(),
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
    # Copying a Resource into the workspace needs one, and every Session may ask for it.
    assert ("materialize" in {tool.name for tool in declared}) == (paths and narrow is None)
    # The browser is offered to every Session of a Run that has one, and to no other, and
    # it can send workspace files to a page only where there is a workspace.
    browsers = [tool for tool in declared if tool.name == "browser"]
    assert bool(browsers) == (agent_browser and (narrow is None))
    for browser in browsers:
        actions = browser.definition.parameters["properties"]["action"]["enum"]
        assert ("upload" in actions) == paths


def test_the_agent_browser_is_part_of_the_plan_a_run_is_pinned_to() -> None:
    """Acceptance pins the tools, so a Run accepted with a browser executes with one."""
    plain = research_tool_declarations(resource_read=True)
    browsing = research_tool_declarations(resource_read=True, agent_browser=True)
    plans = [
        AgentRunPlan.from_tools(tools, model_role="query", context_policy_revision="test-policy")
        for tools in (plain, browsing)
    ]

    assert plans[0].digest != plans[1].digest
    read = {tool.name: tool for tool in browsing}["read"]
    assert "rendered" in read.definition.parameters["properties"]
    assert "rendered=true" in read.description
    # A Host with no Agent Browser, like a Fast Run, is offered nothing to ask for.
    assert (
        "rendered"
        not in {tool.name: tool for tool in plain}["read"].definition.parameters["properties"]
    )


@pytest.mark.parametrize("child", [False, True])
def test_no_built_in_tool_is_named_like_a_connection_tool(tmp_path: Path, child: bool) -> None:
    """The Connection prefix is Connections' alone, so their tools never meet a built-in name."""
    factory = SkillsBundleFactory(global_root=tmp_path / "global", owner_root=tmp_path / "owners")
    declared = research_tool_declarations(
        web_search=True,
        resource_read=True,
        agent_browser=True,
        resource_view=True,
        environment=True,
        artifact_publication=True,
        memory=True,
        skills=factory.declarations(child=child),
        subagents=(
            child_guidance_declarations()
            if child
            else subagent_declarations(model_guidance="Configured model roles.")
        ),
        child=child,
    )
    assert [tool.name for tool in declared if is_connection_tool(tool.name)] == []


def test_only_tools_that_change_nothing_outside_their_run_are_read_only(tmp_path: Path) -> None:
    """Read-only calls run beside each other; every other call is a barrier.

    A Connection tool stays sequential whatever its server claims: its effects are
    the remote account's, and nothing here can know them.
    """
    factory = SkillsBundleFactory(global_root=tmp_path / "global", owner_root=tmp_path / "owners")
    declared = research_tool_declarations(
        web_search=True,
        resource_read=True,
        agent_browser=True,
        resource_view=True,
        environment=True,
        artifact_publication=True,
        subagents=subagent_declarations(model_guidance="Configured model roles."),
        memory=True,
        skills=factory.declarations(child=False),
        injected=[ToolDeclaration("mcp__test__lookup", "Remote tool.", Arguments)],
    )

    assert {tool.name for tool in declared if tool.read_only} == {
        "search_knowledge_base",
        "search_web",
        "read",
        "view",
        "ls",
        "grep",
        "find",
    }
    assert {
        "browser",
        "bash",
        "write",
        "materialize",
        "edit",
        "attach_artifact",
        "remember",
        "mcp__test__lookup",
    } <= {tool.name for tool in declared if not tool.read_only}


def test_workspace_tools_state_what_a_run_workspace_holds() -> None:
    """All 24 `ls`, `find`, and `grep` calls of 34 live Research Runs met an empty
    workspace, some hunting knowledge-base documents as files, and follow-ups looked
    for an earlier Artifact at its old path. The shell and the listing tool say what it
    holds, once each, since the model reads every description; the tools themselves stay
    product-neutral."""
    declared = {
        tool.name: tool
        for tool in research_tool_declarations(
            resource_read=True, environment=True, artifact_publication=True
        )
    }

    for name in ("find", "grep"):
        assert "starts with only" not in declared[name].description
    for name in ("bash", "ls"):
        description = declared[name].description
        assert (
            f"starts with only `{SESSION_NOTE_DIRECTORY}/` from earlier Runs of this "
            "conversation" in description
        )
        assert "`tmp/` is scratch for this Run alone" in description
        assert "Knowledge-base documents are never files in it" in description
        assert "a Resource becomes one only when materialize copies its resource_id" in description
        assert "fetched pages" not in description, "what copies is materialize's to list"
    assert SESSION_NOTE_DIRECTORY not in ls_declaration().description


def test_materialize_states_what_copies_and_that_the_copy_is_not_the_source() -> None:
    """The model reads what a Run's Resources can become in the workspace on the tool that
    copies them, once; the tool itself stays product-neutral."""
    declared = {
        tool.name: tool for tool in research_tool_declarations(resource_read=True, environment=True)
    }

    description = declared["materialize"].description
    assert "Agent Browser downloads and captures" in description
    assert "an earlier turn's Resources and Artifacts all copy" in description
    assert "cite what read returns from it, never the copy" in description
    assert "Agent Browser" not in materialize_declaration().description


def test_a_workspace_with_no_resource_reading_has_nothing_to_copy() -> None:
    """A copy takes the bytes of a Resource, and a Run that cannot read Resources holds none."""
    declared = {tool.name for tool in research_tool_declarations(environment=True)}

    assert {"read", "bash", "write"} <= declared
    assert "materialize" not in declared


def test_read_states_what_a_url_and_an_earlier_artifact_return() -> None:
    publishing = {
        tool.name: tool
        for tool in research_tool_declarations(
            resource_read=True, environment=True, artifact_publication=True
        )
    }
    read = publishing["read"].description

    assert "A url read returns the page's full content" in read
    assert (
        "An Artifact an earlier Run published is reopened by its resource_id, the "
        f"`{PUBLISHED_ARTIFACT_HANDLE_PREFIX}…` id in its `artifact:` link." in read
    )
    # A Run that cannot publish has no earlier Artifact to reopen.
    without_publication = {
        tool.name: tool for tool in research_tool_declarations(resource_read=True)
    }
    assert "reopened by its resource_id" not in without_publication["read"].description


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

    schema = {
        "type": "object",
        "title": "Remote lookup arguments",
        "properties": {"query": {"type": "string", "title": "What to look up"}},
        "required": ["query"],
    }
    catalogue = CatalogueTool("lookup", "Remote lookup.", schema, local_name="mcp__Remote__lookup")
    binding = RunConnectionBinding("owner", "a" * 32, 1, 1, "b" * 64)
    store = AsyncMock()
    store.research_catalogues.return_value = [(binding, (catalogue,))]
    mcp = AsyncMock()
    connections = Connections(
        store=cast(Any, store), mcp=cast(Any, mcp), cipher=CredentialCipher(None)
    )

    accepted = await connections.bind_research(owner_id="owner")

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
