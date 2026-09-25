# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The peer tools one research run offers, composed per run and never globally."""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel

from dlightrag.engine.agent.environment import AccessScheduler
from dlightrag.engine.agent.environment.errors import FullOutputUnavailable
from dlightrag.engine.agent.environment.execution import ExecutionEnvironment
from dlightrag.engine.agent.environment.toolchain import SearchToolchain
from dlightrag.engine.agent.tool_content import ToolTextPart, tool_content_attachments
from dlightrag.engine.agent.tools import AgentTool, ToolDeclaration, ToolResult, ToolRuntime
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
    preview_or_spill,
    read_declaration,
    read_tool,
    view_declaration,
    view_tool,
    write_declaration,
    write_tool,
)
from dlightrag.engine.agent.tools.registry import DuplicateToolError, ToolRegistry
from dlightrag.engine.answer.errors import (
    ChildToolNarrowingError,
    InvalidToolConfigurationError,
)
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.publication import PublicationLimits
from dlightrag.engine.answer.tools.artifacts import (
    attach_artifact_declaration,
    attach_artifact_tool,
)
from dlightrag.engine.answer.tools.memory import (
    MemoryHost,
    forget_declaration,
    forget_tool,
    recall_memory_declaration,
    recall_memory_tool,
    remember_declaration,
    remember_tool,
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


class ResearchToolDeclarations(Protocol):
    """Request-specific declaration projection over configured Research capabilities."""

    def __call__(
        self,
        *,
        web_search: bool,
        memory: bool,
        model_guidance: str,
        injected: Sequence[ToolDeclaration],
    ) -> tuple[ToolDeclaration, ...]: ...


def research_tool_declarations(
    *,
    web_search: bool = False,
    resource_read: bool = False,
    resource_view: bool = False,
    environment: bool = False,
    artifact_publication: bool = False,
    subagents: Sequence[ToolDeclaration] = (),
    memory: bool = False,
    skills: Sequence[ToolDeclaration] = (),
    injected: Sequence[ToolDeclaration] = (),
    child: bool = False,
    tool_names: tuple[str, ...] | None = None,
) -> tuple[ToolDeclaration, ...]:
    """Own Research tool membership and narrowing using only declaration facts."""
    declarations = [knowledge_base_search_declaration()]
    if web_search:
        declarations.append(web_search_declaration())
    if resource_read:
        declarations.append(read_declaration(public_url=True))
    if resource_view or environment:
        declarations.append(view_declaration())
    declarations.extend(injected)
    if environment:
        if not resource_read:
            declarations.append(read_declaration(public_url=False))
        declarations.extend(
            (
                bash_declaration(),
                edit_declaration(),
                write_declaration(),
                grep_declaration(),
                find_declaration(),
                ls_declaration(),
            )
        )
        if artifact_publication and not child:
            declarations.append(attach_artifact_declaration())
    declarations.extend(subagents)
    if memory:
        declarations.extend(
            (remember_declaration(), forget_declaration(), recall_memory_declaration())
        )
    declarations.extend(skills)
    try:
        registry = ToolRegistry(declarations)
        if tool_names is not None:
            offered = {
                tool.name
                for tool in registry.resolve(None, exclude=CHILD_FORBIDDEN_TOOLS if child else ())
            }
            withheld = sorted(set(tool_names) & CHILD_FORBIDDEN_TOOLS) if child else []
            if withheld:
                raise ChildToolNarrowingError(tuple(withheld), reason="a Child Session never holds")
            unknown = sorted(set(tool_names) - offered)
            if unknown:
                raise ChildToolNarrowingError(tuple(unknown), reason="this Run offers no such Tool")
        selected_names = tool_names
        if child and selected_names is not None and "ask_parent" in registry:
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
    resource_viewer: ResourceViewer | None = None,
    environment: ExecutionEnvironment | None = None,
    scheduler: AccessScheduler | None = None,
    spill: Any | None = None,
    output_stage_factory: Any | None = None,
    artifacts_root: Path | None = None,
    publication_limits: PublicationLimits | None = None,
    ripgrep: str = "rg",
    search_toolchain: SearchToolchain | None = None,
    image_preparer: ImagePreparer | None = None,
    subagent_host: SubagentHost | None = None,
    memory_host: MemoryHost | None = None,
    skill_tools: list[AgentTool] | None = None,
    child: bool = False,
    tool_names: tuple[str, ...] | None = None,
) -> list[AgentTool]:
    """Bind the selected declarations to real run capabilities only."""
    access = scheduler or AccessScheduler()
    child_declarations: tuple[ToolDeclaration, ...] = ()
    if subagent_host is not None:
        if child:
            if subagent_host.async_lifecycle and subagent_host.interactive_controls:
                child_declarations = child_guidance_declarations()
        else:
            child_declarations = subagent_declarations(
                model_guidance=subagent_host.model_guidance,
                async_lifecycle=subagent_host.async_lifecycle,
                interactive_controls=subagent_host.interactive_controls,
            )
    declarations = research_tool_declarations(
        web_search=search_web is not None,
        resource_read=resource_reader is not None,
        resource_view=resource_viewer is not None,
        environment=environment is not None,
        artifact_publication=artifacts_root is not None,
        subagents=child_declarations,
        memory=memory_host is not None,
        skills=skill_tools or (),
        injected=injected_tools,
        child=child,
        tool_names=tool_names,
    )
    toolchain = search_toolchain or SearchToolchain(ripgrep=ripgrep)
    bindings: dict[str, Callable[[], AgentTool]] = {
        "search_knowledge_base": lambda: knowledge_base_search_tool(
            retrieve=retrieve_knowledge_base, evidence=evidence, trace=trace
        ),
        "search_web": lambda: web_search_tool(
            search=cast(WebSearch, search_web),
            evidence=evidence,
            trace=trace,
            register_web_source=register_web_source,
        ),
        "read": lambda: _ledger_backed(
            read_tool(
                environment,
                access,
                resource_reader=resource_reader,
                spill=spill,
                image_preparer=image_preparer,
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
        "bash": lambda: bash_tool(
            cast(ExecutionEnvironment, environment),
            access,
            output_stage_factory=output_stage_factory,
        ),
        "edit": lambda: edit_tool(cast(ExecutionEnvironment, environment), access, spill=spill),
        "write": lambda: write_tool(cast(ExecutionEnvironment, environment), access),
        "grep": lambda: grep_tool(
            cast(ExecutionEnvironment, environment),
            access,
            search_toolchain=toolchain,
            output_stage_factory=output_stage_factory,
        ),
        "find": lambda: find_tool(
            cast(ExecutionEnvironment, environment), access, search_toolchain=toolchain, spill=spill
        ),
        "ls": lambda: ls_tool(cast(ExecutionEnvironment, environment), access),
        "attach_artifact": lambda: attach_artifact_tool(
            cast(Path, artifacts_root),
            scheduler=access,
            limits=publication_limits or PublicationLimits(),
        ),
        "remember": lambda: remember_tool(host=cast(MemoryHost, memory_host)),
        "forget": lambda: forget_tool(host=cast(MemoryHost, memory_host)),
        "recall_memory": lambda: recall_memory_tool(host=cast(MemoryHost, memory_host)),
    }
    bound_children = ()
    if subagent_host is not None and child_declarations:
        bound_children = (
            child_guidance_tools(host=subagent_host)
            if child
            else subagent_tools(host=subagent_host)
        )
    for tool in (*bound_children, *(skill_tools or ())):
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
            protected_text=f"Full output: read(resource_id={receipt.resource_id!r}, cursor=...)",
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
        "_workspace": "__web_search__" if source_type == "web_search" else "__attachment__",
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
                "content": f"Viewed pixels: {attachment.safe_name}",
                "page_number": source.page,
                "metadata": {
                    **metadata,
                    "visual_source": asdict(source),
                    "content_digest": attachment.content_digest,
                },
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
        "cancel_subagent",
        "steer_subagent",
        "continue_subagent",
        "reply_subagent",
        "remember",
        "forget",
        "attach_artifact",
        "publish_skill",
        "delete_skill",
    }
)

__all__ = [
    "CHILD_FORBIDDEN_TOOLS",
    "ResearchToolDeclarations",
    "compose_research_tools",
    "research_tool_declarations",
]
