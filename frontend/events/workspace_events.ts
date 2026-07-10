// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import type { WorkspaceRecord } from './bus';

/**
 * Raw detail of the server's workspace `HX-Trigger` events (and the `data-all`
 * bootstrap attribute). The wire shape is snake_case because it comes straight
 * from the Python backend — this is the ONE place that is allowed to know that.
 */
export interface WorkspaceEventDetail {
  workspace?: string;
  display_name?: string;
  embedding_model?: string;
  next_workspace?: string;
}

/** Unwrap an htmx `HX-Trigger` CustomEvent into its raw workspace detail. */
export function readWorkspaceDetail(event: Event): WorkspaceEventDetail {
  const detail = (event as CustomEvent<WorkspaceEventDetail & { value?: WorkspaceEventDetail }>)
    .detail;
  return detail?.value ?? detail ?? {};
}

/**
 * Normalize a raw (snake_case) workspace detail into the canonical camelCase
 * {@link WorkspaceRecord}. Returns null when the detail carries no workspace id.
 */
export function toWorkspaceRecord(detail: WorkspaceEventDetail): WorkspaceRecord | null {
  if (!detail.workspace) return null;
  return {
    workspace: detail.workspace,
    displayName: detail.display_name || detail.workspace,
    embeddingModel: detail.embedding_model || '',
  };
}
