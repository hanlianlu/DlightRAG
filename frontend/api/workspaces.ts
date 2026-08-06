// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export interface CreatedWorkspace {
  workspace: string;
  display_name: string;
}

export interface DeletedWorkspace {
  workspace: string;
  next_workspace: string;
}

export class WorkspaceApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = 'WorkspaceApiError';
  }
}

async function post<T>(path: string, body: Record<string, string>, fallback: string): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams(body).toString(),
  });
  if (!response.ok) {
    // The route answers with {"error": ...}; fall back when it cannot.
    const detail = await response.json().catch(() => null) as {error?: unknown} | null;
    const message = typeof detail?.error === 'string' ? detail.error : fallback;
    throw new WorkspaceApiError(response.status, message);
  }
  return await response.json() as T;
}

export function createWorkspaceRequest(name: string): Promise<CreatedWorkspace> {
  return post<CreatedWorkspace>(
    '/web/workspaces/create',
    {workspace_name: name},
    'Failed to create workspace',
  );
}

export function deleteWorkspaceRequest(name: string): Promise<DeletedWorkspace> {
  return post<DeletedWorkspace>(
    '/web/workspaces/delete',
    {workspace_name: name, confirm_name: name},
    'Could not delete workspace.',
  );
}
