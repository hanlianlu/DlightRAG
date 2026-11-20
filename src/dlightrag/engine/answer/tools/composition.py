# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The peer tools one research run offers, composed per run and never globally."""

import hashlib
from collections.abc import Callable, Collection, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel

from dlightrag.engine.agent.environment.access import (
    AccessScheduler,
)
from dlightrag.engine.agent.environment.errors import FullOutputUnavailable
from dlightrag.engine.agent.environment.execution import ExecutionEnvironment
from dlightrag.engine.agent.environment.toolchain import SearchToolchain
from dlightrag.engine.agent.sealed_artifact import SealedArtifact
from dlightrag.engine.agent.skill_delivery import DeliveryLedger
from dlightrag.engine.agent.skill_protection import ProtectedTextIndex
from dlightrag.engine.agent.tool_content import ToolTextPart, tool_content_attachments
from dlightrag.engine.agent.tools.contracts import (
    AdmittedBytesReader,
    AgentTool,
    ToolDeclaration,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.agent.tools.files import (
    ImagePreparer,
    ResourceViewer,
    SpillWriter,
    bash_declaration,
    bash_tool,
    edit_declaration,
    edit_tool,
    find_declaration,
    find_tool,
    grep_declaration,
    grep_tool,
    ls_declaration,
    ls_tool,
    materialize_declaration,
    materialize_tool,
    preview_or_spill,
    read_declaration,
    read_tool,
    spill_continuation,
    view_declaration,
    view_tool,
    write_declaration,
    write_tool,
)
from dlightrag.engine.agent.tools.registry import DuplicateToolError, ToolRegistry
from dlightrag.engine.answer.agent_browser.accounts import (
    has_mailbox,
)
from dlightrag.engine.answer.citations.utils import ATTACHMENT_WORKSPACE, WEB_SEARCH_WORKSPACE
from dlightrag.engine.answer.continuation_handles import SESSION_NOTE_DIRECTORY
from dlightrag.engine.answer.errors import (
    InvalidToolConfigurationError,
)
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.publication import PublicationLimits
from dlightrag.engine.answer.resources.models import PUBLISHED_ARTIFACT_HANDLE_PREFIX
from dlightrag.engine.answer.tools.artifacts import (
    attach_artifact_declaration,
    attach_artifact_tool,
)
from dlightrag.engine.answer.tools.browser import (
    BrowserToolHost,
    browser_declaration,
    browser_tool,
)
from dlightrag.engine.answer.tools.goals import (
    GOAL_TOOL_NAMES,
    GoalHost,
    goal_tool_declarations,
    goal_tools,
)
from dlightrag.engine.answer.tools.mcp_resources import (
    McpResourceHost,
    mcp_resource_declarations,
    mcp_resource_tools,
)
from dlightrag.engine.answer.tools.memory import (
    MEMORY_FORGET,
    MEMORY_LOOKUP,
    MEMORY_MUTATION_TOOLS,
    MEMORY_REMEMBER,
    MEMORY_UNDO,
    MemoryHost,
    forget_declaration,
    forget_tool,
    lookup_declaration,
    lookup_tool,
    remember_declaration,
    remember_tool,
    undo_declaration,
    undo_tool,
)
from dlightrag.engine.answer.tools.search import (
    KnowledgeRetrieval,
    RegisterWebSource,
    WebSearch,
    knowledge_base_search_declaration,
    knowledge_base_search_tool,
    web_search_declaration,
    web_search_tool,
)
from dlightrag.engine.answer.tools.subagents import (
    SubagentHost,
    child_guidance_declarations,
    child_guidance_tools,
    subagent_declarations,
    subagent_tools,
)

#: What a Run's workspace holds, stated once on the shell and once on the listing tool.
#: The Session owns one workspace that each Run of this conversation continues (S1.1); a
#: Resource becomes a file only when the Agent asks for a copy (ADR 0033), and a
#: knowledge-base document is reached by search instead.
_WORKSPACE_FACT = (
    "Each Run's workspace is the Session's workspace: the files earlier Runs of this "
    f"conversation left are already there, `{SESSION_NOTE_DIRECTORY}/` is its memory, and "
    "`tmp/` is scratch that is not carried to the next Run. Knowledge-base documents "
    "are never files in it; a Resource becomes one only when materialize copies its resource_id."
)

#: What materialize copies and what the copy is; the tool itself stays product-neutral. The
#: shell and the listing tool say only that a copy is the way in, so this is the one list.
_MATERIALIZE_FACT = (
    "Uploads, fetched pages and files, Agent Browser downloads and captures, and an earlier "
    "turn's Resources and Artifacts all copy. The Resource stays the source: cite what read "
    "returns from it, never the copy."
)

#: A later Run of the Session adopts a published Artifact by the handle its link carries.
_EARLIER_ARTIFACT_FACT = (
    "An Artifact an earlier Run published is reopened by its resource_id, the "
    f"`{PUBLISHED_ARTIFACT_HANDLE_PREFIX}…` id in its `artifact:` link."
)


def _stating[T: ToolDeclaration](tool: T, fact: str) -> T:
    """Append one product fact to a product-neutral tool's description."""
    return replace(tool, description=f"{tool.description} {fact}")


def _reading[T: ToolDeclaration](read: T, *, earlier_artifacts: bool) -> T:
    """Only a resource-reading ``read`` in a Run that publishes can reopen an Artifact."""
    return _stating(read, _EARLIER_ARTIFACT_FACT) if earlier_artifacts else read


def _recording_execution(tool: AgentTool, ledger: DeliveryLedger | None) -> AgentTool:
    """Record one execution Tool's command for the delivery preview, and run it unchanged.

    The wrapper adds no behaviour: the Tool's declaration, result, effects and replay policy are
    the Tool's own. Only the command line reaches the ledger, because a command that names a
    declared verifier is the one in-loop fact a preview can state about it (ADR 0058).
    """
    if ledger is None:
        return tool

    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        result = await tool.execute(raw, runtime)
        command = getattr(raw, "command", None)
        ledger.record_execution(
            tool_name=tool.name,
            command=command if isinstance(command, str) else None,
            failed=result.is_error,
        )
        return result

    return replace(tool, execute=execute)


class ResearchToolDeclarations(Protocol):
    """Request-specific declaration projection over configured Research capabilities."""

    def __call__(
        self,
        *,
        web_search: bool,
        memory: bool,
        agent_may_register: bool,
        model_guidance: str,
        injected: Sequence[ToolDeclaration],
        mcp_resources: bool = False,
    ) -> tuple[ToolDeclaration, ...]: ...


def research_tool_declarations(
    *,
    web_search: bool = False,
    resource_read: bool = False,
    agent_browser: bool = False,
    agent_may_register: bool = False,
    agent_mailbox: bool = False,
    resource_view: bool = False,
    environment: bool = False,
    artifact_publication: bool = False,
    subagents: Sequence[ToolDeclaration] = (),
    memory: bool = False,
    goals: bool = False,
    skills: Sequence[ToolDeclaration] = (),
    injected: Sequence[ToolDeclaration] = (),
    mcp_resources: bool = False,
    child: bool = False,
    tool_names: tuple[str, ...] | None = None,
) -> tuple[ToolDeclaration, ...]:
    """Own Research tool membership and narrowing using only declaration facts."""
    declarations = [knowledge_base_search_declaration()]
    if web_search:
        declarations.append(web_search_declaration())
    if resource_read:
        declarations.append(
            _reading(
                read_declaration(public_url=True, rendered=agent_browser),
                earlier_artifacts=artifact_publication,
            )
        )
    if resource_view or environment:
        declarations.append(view_declaration())
    if agent_browser:
        declarations.append(
            browser_declaration(
                upload=environment, may_register=agent_may_register, mailbox=agent_mailbox
            )
        )
    declarations.extend(injected)
    if mcp_resources:
        # The three shared tools are how a Run reads an external server's documents on
        # demand; they are pinned declarations like every other tool.
        declarations.extend(mcp_resource_declarations())
    if environment:
        if not resource_read:
            declarations.append(read_declaration(public_url=False))
        declarations.extend(
            (
                _stating(bash_declaration(), _WORKSPACE_FACT),
                edit_declaration(),
                write_declaration(),
            )
        )
        if resource_read:
            # Only a Run that can read Resources has any to copy.
            declarations.append(_stating(materialize_declaration(), _MATERIALIZE_FACT))
        declarations.extend(
            (
                grep_declaration(),
                find_declaration(),
                _stating(ls_declaration(), _WORKSPACE_FACT),
            )
        )
        if artifact_publication and not child:
            declarations.append(attach_artifact_declaration())
    declarations.extend(subagents)
    if memory:
        declarations.extend(
            (
                remember_declaration(),
                forget_declaration(),
                undo_declaration(),
                lookup_declaration(),
            )
        )
    if goals and not child:
        declarations.extend(goal_tool_declarations())
    declarations.extend(skills)
    try:
        registry = ToolRegistry(declarations)
        selected_names = tool_names
        if selected_names is not None:
            # A caller's list narrows what the Run composed; it never fails the spawn. A
            # name this Run does not offer is not there to keep, and one a Child may
            # never hold is dropped by the table below.
            selected_names = tuple(name for name in selected_names if name in registry)
            if child and "ask_parent" in registry:
                selected_names = tuple(dict.fromkeys((*selected_names, "ask_parent")))
        return registry.resolve(selected_names, exclude=CHILD_FORBIDDEN_TOOLS if child else ())
    except DuplicateToolError as exc:
        raise InvalidToolConfigurationError(exc.names) from exc


def compose_research_tools(
    *,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
    retrieve_knowledge_base: KnowledgeRetrieval,
    search_web: WebSearch | None,
    injected_tools: list[AgentTool],
    register_web_source: RegisterWebSource | None,
    resource_reader: Any | None = None,
    browser: BrowserToolHost | None = None,
    resource_viewer: ResourceViewer | None = None,
    admitted_bytes_reader: AdmittedBytesReader | None = None,
    protected_digests: Collection[str] = frozenset(),
    protected_index: Callable[[], ProtectedTextIndex] | None = None,
    environment: ExecutionEnvironment | None = None,
    scheduler: AccessScheduler | None = None,
    spill: Any | None = None,
    output_stage_factory: Any | None = None,
    artifacts_root: Path | None = None,
    publication_limits: PublicationLimits | None = None,
    search_toolchain: SearchToolchain,
    image_preparer: ImagePreparer | None = None,
    subagent_host: SubagentHost | None = None,
    memory_host: MemoryHost | None = None,
    goal_host: GoalHost | None = None,
    mcp_resource_host: McpResourceHost | None = None,
    skill_tools: list[AgentTool] | None = None,
    delivery_ledger: DeliveryLedger | None = None,
    sealed_artifacts: Sequence[SealedArtifact] = (),
    child: bool = False,
    tool_names: tuple[str, ...] | None = None,
) -> list[AgentTool]:
    """Bind the selected declarations to real run capabilities only.

    ``delivery_ledger`` is the current attempt's record: the execution Tool reports its command
    line into it, and the attach Tool reads it to preview the reader's record. ``None`` composes
    exactly the tools it composed before the preview existed.
    """
    access = scheduler or AccessScheduler()
    child_declarations: tuple[ToolDeclaration, ...] = ()
    if subagent_host is not None:
        child_declarations = (
            child_guidance_declarations()
            if child
            else subagent_declarations(model_guidance=subagent_host.model_guidance)
        )
    accounts = None if browser is None else browser.accounts
    declarations = research_tool_declarations(
        web_search=search_web is not None,
        resource_read=resource_reader is not None,
        agent_browser=browser is not None,
        agent_may_register=accounts is not None and accounts.may_register,
        agent_mailbox=has_mailbox(accounts),
        resource_view=resource_viewer is not None,
        environment=environment is not None,
        artifact_publication=artifacts_root is not None,
        subagents=child_declarations,
        memory=memory_host is not None,
        goals=goal_host is not None,
        skills=skill_tools or (),
        injected=injected_tools,
        mcp_resources=mcp_resource_host is not None,
        child=child,
        tool_names=tool_names,
    )
    bindings: dict[str, Callable[[], AgentTool]] = {
        "search_knowledge_base": lambda: knowledge_base_search_tool(
            retrieve=retrieve_knowledge_base, evidence=evidence, trace=trace
        ),
        "search_web": lambda: web_search_tool(
            search=cast(WebSearch, search_web),
            evidence=evidence,
            trace=trace,
            register_web_source=register_web_source,
            # A Run that can drive a browser can still search when every hosted provider is out,
            # so the failure it reads says so.
            browser_available=browser is not None,
        ),
        "read": lambda: _ledger_backed(
            _reading(
                read_tool(
                    environment,
                    access,
                    resource_reader=resource_reader,
                    spill=spill,
                    rendered=browser is not None,
                ),
                earlier_artifacts=resource_reader is not None and artifacts_root is not None,
            ),
            evidence,
        ),
        "view": lambda: _ledger_backed(
            view_tool(
                environment,
                access,
                resource_viewer=resource_viewer,
                image_preparer=image_preparer,
            ),
            evidence,
        ),
        "browser": lambda: _ledger_backed(
            browser_tool(
                cast(BrowserToolHost, browser),
                environment=environment,
                scheduler=access,
                spill=spill,
                image_preparer=image_preparer,
                child=child,
            ),
            evidence,
        ),
        "bash": lambda: _recording_execution(
            _stating(
                bash_tool(
                    cast(ExecutionEnvironment, environment),
                    access,
                    output_stage_factory=output_stage_factory,
                ),
                _WORKSPACE_FACT,
            ),
            delivery_ledger,
        ),
        "edit": lambda: edit_tool(cast(ExecutionEnvironment, environment), access, spill=spill),
        "write": lambda: write_tool(cast(ExecutionEnvironment, environment), access),
        "materialize": lambda: _stating(
            materialize_tool(
                cast(ExecutionEnvironment, environment),
                access,
                admitted_bytes_reader=cast(AdmittedBytesReader, admitted_bytes_reader),
            ),
            _MATERIALIZE_FACT,
        ),
        "grep": lambda: grep_tool(
            cast(ExecutionEnvironment, environment),
            access,
            search_toolchain=search_toolchain,
            output_stage_factory=output_stage_factory,
        ),
        "find": lambda: find_tool(
            cast(ExecutionEnvironment, environment),
            access,
            search_toolchain=search_toolchain,
            spill=spill,
        ),
        "ls": lambda: _stating(
            ls_tool(cast(ExecutionEnvironment, environment), access), _WORKSPACE_FACT
        ),
        "attach_artifact": lambda: attach_artifact_tool(
            cast(Path, artifacts_root),
            scheduler=access,
            limits=publication_limits or PublicationLimits(),
            environment=environment,
            # An earlier Artifact is reached the way materialize reaches it.
            admitted_bytes_reader=admitted_bytes_reader if resource_reader is not None else None,
            # A Skill's own documents are its author's, never this Run's deliverable.
            protected_digests=protected_digests,
            # And a deliverable that reproduces their words is refused too (ADR 0061).
            protected_index=protected_index,
            # The record the reader will see, previewed while the Agent can still correct it.
            delivery_ledger=delivery_ledger,
            # The one Artifact name a built-in Skill lets claim sealed, and the gate run on it.
            sealed_artifacts=sealed_artifacts,
        ),
        MEMORY_REMEMBER: lambda: remember_tool(host=cast(MemoryHost, memory_host)),
        MEMORY_FORGET: lambda: forget_tool(host=cast(MemoryHost, memory_host)),
        MEMORY_UNDO: lambda: undo_tool(host=cast(MemoryHost, memory_host)),
        MEMORY_LOOKUP: lambda: lookup_tool(host=cast(MemoryHost, memory_host)),
    }
    bound_children = ()
    if subagent_host is not None and child_declarations:
        bound_children = (
            child_guidance_tools(host=subagent_host)
            if child
            else subagent_tools(host=subagent_host)
        )
    for tool in (
        *bound_children,
        *(skill_tools or ()),
        *(goal_tools(host=goal_host) if goal_host else ()),
    ):
        bindings[tool.name] = lambda tool=tool: tool
    if mcp_resource_host is not None:
        for tool in mcp_resource_tools(mcp_resource_host):
            bindings[tool.name] = lambda tool=tool: tool
    for tool in injected_tools:
        bindings[tool.name] = lambda tool=tool: _bounded_injected_result(tool, spill)
    return [bindings[declaration.name]() for declaration in declarations]


def _bounded_injected_result(tool: AgentTool, spill: SpillWriter | None) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        result = await tool.execute(raw, runtime)
        # Existing cursor/spill results already own their continuation. Runtime
        # fitting retains protected text and attachments without duplicating it.
        if result.protected_text or result.effects.committed_outputs:
            return result
        try:
            text, receipt = await preview_or_spill(result.text_content, spill=spill, tool=tool.name)
        except FullOutputUnavailable, OSError:
            return replace(
                result,
                parts=(
                    ToolTextPart(
                        f"Tool {tool.name} completed but its full output is unavailable. Do not retry automatically; identify the unavailable part in the final Answer."
                    ),
                    *tool_content_attachments(result.parts),
                ),
                is_error=True,
            )
        if receipt is None:
            return result
        return replace(
            result,
            parts=(ToolTextPart(text), *tool_content_attachments(result.parts)),
            protected_text=spill_continuation(receipt.resource_id),
            effects=replace(
                result.effects, committed_outputs=(*result.effects.committed_outputs, receipt)
            ),
        )

    return replace(tool, execute=execute)


def _ledger_backed(tool: AgentTool, evidence: EvidenceLedger) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        result = await tool.execute(raw, runtime)
        rows = _resource_rows(tool.name, result)
        if rows:
            await runtime.in_source_order()
            evidence.add_rows(rows)
            await evidence.aflush_images()
        return result

    return replace(tool, execute=execute)


def _resource_rows(tool_name: str, result: ToolResult) -> list[dict[str, Any]]:
    if not result.effects.evidence_sources or not result.text_content.strip():
        return []
    source = result.effects.evidence_sources[0]
    resource_id = source.resource_id
    source_type = source.source_type
    source_uri = source.source_uri
    metadata = {
        "source_type": source_type,
        "source_uri": source_uri,
        "source_download_locator": source_uri,
        "title": source.title,
        **dict(source.attributes),
    }
    evidence_key = result.text_content
    if tool_name == "read":
        content, marker, cursor = evidence_key.rpartition("\n[more text available; cursor=")
        if marker and cursor.endswith("]"):
            evidence_key = content
    identity = hashlib.sha256(f"{tool_name}\0{evidence_key}".encode()).hexdigest()[:16]
    row = {
        "chunk_id": f"{resource_id}::{tool_name}::{identity}",
        "reference_id": resource_id,
        "full_doc_id": resource_id,
        "file_path": str(metadata.get("title") or resource_id),
        "content": evidence_key if tool_name == "read" else result.text_content,
        "page_number": None,
        "_workspace": WEB_SEARCH_WORKSPACE if source_type == "web_search" else ATTACHMENT_WORKSPACE,
        "_evidence_key": f"{tool_name}:{identity}",
        # This Tool's own result already carries the row's body — the excerpt's text
        # for `read`, its pixels for `view` — so the ledger labels it where it stands
        # rather than carrying the same passage twice in one request. The Tool knows
        # this; no caller has to guess it from a substring.
        "_carried_by_tool": True,
        "metadata": metadata,
    }
    if tool_name != "view":
        return [row]
    rows = []
    for attachment in tool_content_attachments(result.parts):
        source = attachment.source
        if source is None:
            raise ValueError("visual evidence requires exact source provenance")
        rows.append(
            {
                **row,
                "chunk_id": f"{source.resource_id}::view::{attachment.resource_id}",
                "_evidence_key": attachment.resource_id,
                # The pixels' exact origin is the row's bookkeeping, not the document's.
                "_visual_source": asdict(source),
                "_content_digest": attachment.content_digest,
                "content": f"Viewed pixels: {attachment.safe_name}",
                "page_number": source.page,
            }
        )
    return rows


#: Tools a Child Session never holds, because they spend the Run's authority rather than
#: its capability (ADR 0025). Splitting and the roster belong to the Run that owns its
#: shape, durable owner memory and publication belong to the Run that owns the answer,
#: and an explicit ``tools`` request can narrow a Child but never restore one of these.
CHILD_FORBIDDEN_TOOLS = frozenset(
    {
        "spawn_agent",
        "subagent_status",
        "wait_subagent",
        "read_subagent",
        "cancel_subagent",
        "steer_subagent",
        "continue_subagent",
        "reply_subagent",
        *MEMORY_MUTATION_TOOLS,
        *GOAL_TOOL_NAMES,
        "attach_artifact",
        "publish_skill",
        "delete_skill",
        "set_skill_enabled",
    }
)

__all__ = [
    "CHILD_FORBIDDEN_TOOLS",
    "ResearchToolDeclarations",
    "compose_research_tools",
    "research_tool_declarations",
]
