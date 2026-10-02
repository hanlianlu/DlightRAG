# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared browser workspace contracts for bootstrap and catalog pages."""

from collections.abc import Sequence
from typing import get_args

from dlightrag.application.access import AccessGate, WorkspaceRecord, corpus_mutation_access_action
from dlightrag.application.corpus_admin import CorpusMutationAction
from dlightrag.engine.answer.client_contracts import ClientContractModel

_MUTATIONS: tuple[CorpusMutationAction, ...] = get_args(CorpusMutationAction.__value__)


class WebBootstrapWorkspace(ClientContractModel):
    workspace: str
    display_name: str
    embedding_model: str
    # The Corpus Mutations this caller may request here, so the Web offers only those.
    changes: list[CorpusMutationAction]


async def project_workspace_records(
    gate: AccessGate,
    records: Sequence[WorkspaceRecord],
    *,
    default_workspace: str,
) -> list[WebBootstrapWorkspace]:
    """Project authorized catalog rows with the Corpus Mutations the caller may request.

    No one deletes the deployment's default workspace, so it never offers that.
    """
    workspaces = [str(record["workspace"]) for record in records]
    permitted = {
        action: await gate.authorized_workspace_ids(action, workspaces)
        for action in {corpus_mutation_access_action(mutation) for mutation in _MUTATIONS}
    }
    return [
        WebBootstrapWorkspace(
            workspace=workspace,
            display_name=str(record.get("display_name") or workspace),
            embedding_model=str(record.get("embedding_model") or ""),
            changes=[
                mutation
                for mutation in _MUTATIONS
                if workspace in permitted[corpus_mutation_access_action(mutation)]
                and not (mutation == "delete_workspace" and workspace == default_workspace)
            ],
        )
        for workspace, record in zip(workspaces, records, strict=True)
    ]


__all__ = [
    "WebBootstrapWorkspace",
    "project_workspace_records",
]
