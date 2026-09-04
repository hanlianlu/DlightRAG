# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Model-facing attachment of completed Agent Workspace artifacts."""

from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.engine.agent.environment.access import AccessScheduler, WorkspaceAccess
from dlightrag.engine.agent.tools import AgentTool, ToolResult, ToolRuntime
from dlightrag.engine.answer.publication import (
    ArtifactValidationError,
    PublicationLimits,
    artifact_link,
    prepare_artifact_attachment,
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


def attach_artifact_tool(
    artifacts_root: Path,
    *,
    scheduler: AccessScheduler,
    limits: PublicationLimits,
) -> AgentTool:
    """Build the optional parent-Research publication-intent tool."""

    async def execute(args: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        attachment_args = cast(AttachArtifactArgs, args)
        async with scheduler.hold(WorkspaceAccess()):
            try:
                attachment = prepare_artifact_attachment(
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
        return ToolResult.text(
            f"attached {attachment.relative_path} ({attachment.size_bytes} bytes); "
            f"place it with {link}",
            details={
                "artifact_attachment": {
                    "relative_path": attachment.relative_path,
                    "label": attachment.label,
                    "content_digest": attachment.content_digest,
                    "size_bytes": attachment.size_bytes,
                    "presentation": attachment.presentation,
                }
            },
        )

    return AgentTool(
        name="attach_artifact",
        description=(
            "Attach one completed artifacts/ file as a root user deliverable. Call only "
            "after its final write or edit; linked dependencies are included automatically."
        ),
        input_model=AttachArtifactArgs,
        execute=execute,
        replay_policy="replayable",
        contract_version=1,
        guidance=(
            "attach_artifact: path is relative to artifacts/; attach only root deliverables "
            "after their final modification. The returned artifact link controls placement; "
            "the Host places an attached root automatically if the final answer omits it."
        ),
    )


__all__ = ["AttachArtifactArgs", "attach_artifact_tool"]
