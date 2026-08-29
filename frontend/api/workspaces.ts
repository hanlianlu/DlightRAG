// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {csrfHeaders} from './csrf';

export interface CreatedWorkspace {
  workspace: string;
  display_name: string;
}

export interface DeletedWorkspace {
  workspace: string;
  next_workspace: string;
}

export interface WorkspacePageItem {
  workspace: string;
  display_name: string;
  embedding_model: string;
}

export interface WorkspacePage {
  workspaces: WorkspacePageItem[];
  next_cursor: string | null;
}

export class WorkspaceApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = 'WorkspaceApiError';
  }
}

export async function getWorkspacesPage(
  cursor: string | null,
  signal?: AbortSignal,
): Promise<WorkspacePage> {
  const query = cursor === null ? '' : `?cursor=${encodeURIComponent(cursor)}`;
  const response = await fetch(`/web/api/workspaces${query}`, {signal});
  if (!response.ok) {
    throw new WorkspaceApiError(response.status, 'Failed to load workspaces');
  }
  const payload = await response.json() as {
    workspaces: WorkspacePageItem[];
    next_cursor?: string | null;
  };
  return {workspaces: payload.workspaces ?? [], next_cursor: payload.next_cursor ?? null};
}

async function post<T>(
  path: string,
  body: Record<string, string>,
  fallback: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: csrfHeaders('application/x-www-form-urlencoded'),
    body: new URLSearchParams(body).toString(),
    signal,
  });
  if (!response.ok) {
    // The route answers with {"error": ...}; fall back when it cannot.
    const detail = await response.json().catch(() => null) as {error?: unknown} | null;
    const message = typeof detail?.error === 'string' ? detail.error : fallback;
    throw new WorkspaceApiError(response.status, message);
  }
  return await response.json() as T;
}

export function createWorkspaceRequest(
  name: string,
  signal?: AbortSignal,
): Promise<CreatedWorkspace> {
  return post<CreatedWorkspace>(
    '/web/api/workspaces/create',
    {workspace_name: name},
    'Failed to create workspace',
    signal,
  );
}

export function deleteWorkspaceRequest(
  name: string,
  signal?: AbortSignal,
): Promise<DeletedWorkspace> {
  return post<DeletedWorkspace>(
    '/web/api/workspaces/delete',
    {workspace_name: name, confirm_name: name},
    'Could not delete workspace.',
    signal,
  );
}
