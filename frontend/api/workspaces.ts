// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import {corpusRunReceipt, type WebCorpusRunReceipt} from './corpus-runs.ts';
import {csrfHeaders} from './csrf.ts';
import {parseWire} from './wire.ts';

/** The Corpus Mutations there are; a workspace lists the ones its caller may request. */
export const WORKSPACE_CHANGES = [
  'ingest', 'replace', 'delete', 'retry', 'reset', 'delete_workspace',
] as const;
export type WorkspaceChange = (typeof WORKSPACE_CHANGES)[number];

export const workspacePageItem = v.pipe(
  v.object({
    workspace: v.string(),
    display_name: v.string(),
    embedding_model: v.string(),
    changes: v.array(v.picklist(WORKSPACE_CHANGES)),
  }),
  v.transform((w) => ({
    workspace: w.workspace,
    displayName: w.display_name,
    embeddingModel: w.embedding_model,
    changes: w.changes as readonly WorkspaceChange[],
  })),
);
export type WorkspacePageItem = v.InferOutput<typeof workspacePageItem>;

const workspacePage = v.pipe(
  v.object({
    workspaces: v.optional(v.array(workspacePageItem)),
    next_cursor: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({workspaces: w.workspaces ?? [], nextCursor: w.next_cursor ?? null})),
);
export type WorkspacePage = v.InferOutput<typeof workspacePage>;

export async function getWorkspacesPage(
  cursor: string | null,
  signal?: AbortSignal,
): Promise<WorkspacePage> {
  const query = cursor === null ? '' : `?cursor=${encodeURIComponent(cursor)}`;
  const response = await fetch(`/web/api/workspaces${query}`, {signal});
  return parseWire(response, workspacePage);
}

async function post<Input, Output>(
  path: string,
  body: Record<string, string>,
  schema: v.GenericSchema<Input, Output>,
  signal?: AbortSignal,
): Promise<Output> {
  const response = await fetch(path, {
    method: 'POST',
    headers: csrfHeaders('application/x-www-form-urlencoded'),
    body: new URLSearchParams(body).toString(),
    signal,
  });
  return parseWire(response, schema);
}

export function createWorkspaceRequest(
  name: string,
  signal?: AbortSignal,
): Promise<WorkspacePageItem> {
  return post(
    '/web/api/workspaces/create',
    {workspace_name: name},
    workspacePageItem,
    signal,
  );
}

export function resetWorkspaceRequest(
  name: string,
  signal?: AbortSignal,
): Promise<WebCorpusRunReceipt> {
  return post(
    '/web/api/workspaces/reset',
    {workspace_name: name, confirm_name: name},
    corpusRunReceipt,
    signal,
  );
}

export function deleteWorkspaceRequest(
  name: string,
  signal?: AbortSignal,
): Promise<WebCorpusRunReceipt> {
  return post(
    '/web/api/workspaces/delete',
    {workspace_name: name, confirm_name: name},
    corpusRunReceipt,
    signal,
  );
}
