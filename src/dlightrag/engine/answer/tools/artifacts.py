# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Model-facing attachment of completed Agent Workspace artifacts."""

from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.engine.agent.environment.access import AccessScheduler, WorkspaceAccess
from dlightrag.engine.agent.tools import AgentTool, ToolDeclaration, ToolResult, ToolRuntime
from dlightrag.engine.answer.publication import (
    ArtifactValidationError,
    PublicationLimits,
    artifact_link,
    artifact_read_call,
    artifact_resource_id,
    prepare_artifact_attachment,
    run_artifact_check,
)


class AttachArtifactArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str = Field(
        min_length=1,
        max_length=500,
        description="Path relative to the artifacts/ directory.",
    )
    label: str | None = Field(
        default=None,
        max_length=200,
        description="Optional user-facing label; defaults to the safe filename.",
    )


def attach_artifact_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="attach_artifact",
        description="Attach one completed artifacts/ file as a root user deliverable, after its "
        "final write or edit. Files that an attached Markdown or HTML root links to are "
        "included automatically, so attach only the root. Place the returned Artifact link "
        "where it helps in the Answer; the Host adds an attached root at the end if the "
        "Answer omits it.",
        input_model=AttachArtifactArgs,
        replay_policy="replayable",
        contract_version=1,
    )


def attach_artifact_tool(
    artifacts_root: Path,
    *,
    scheduler: AccessScheduler,
    limits: PublicationLimits,
) -> AgentTool:
    """Build the optional parent-Research publication-intent tool."""

    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        attachment_args = cast(AttachArtifactArgs, args)
        await runtime.emit_update(ToolResult.text("", subject=attachment_args.path))
        async with scheduler.hold(WorkspaceAccess()):
            try:
                # The check scans the Agent Workspace and decodes the file.
                attachment = await run_artifact_check(
                    prepare_artifact_attachment,
                    artifacts_root,
                    path=attachment_args.path,
                    label=attachment_args.label or "",
                    limits=limits,
                )
            except ArtifactValidationError as exc:
                return ToolResult.text(
                    f"{exc.kind}: {exc.description}",
                    is_error=True,
                )
        link = artifact_link(attachment)
        resource_id = artifact_resource_id(attachment.relative_path)
        read_call = artifact_read_call(attachment.relative_path)
        return ToolResult.text(
            f"attached {attachment.relative_path} ({attachment.size_bytes} bytes); "
            f"place it with {link}; a later turn continues from this published version "
            f"with {read_call}",
            details={
                "artifact_attachment": {
                    "relative_path": attachment.relative_path,
                    "label": attachment.label,
                    "content_digest": attachment.content_digest,
                    "size_bytes": attachment.size_bytes,
                    "presentation": attachment.presentation,
                    "resource_id": resource_id,
                }
            },
        )

    return attach_artifact_declaration().bind(execute)


__all__ = ["AttachArtifactArgs", "attach_artifact_tool"]
