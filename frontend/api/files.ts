// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {parseWire} from './wire.ts';

const webFileItem = v.pipe(
  v.object({file_name: v.string(), file_path: v.string()}),
  v.transform((w) => ({fileName: w.file_name, filePath: w.file_path})),
);
export type WebFileItem = v.InferOutput<typeof webFileItem>;

const webIngestStatus = v.pipe(
  v.object({
    busy: v.boolean(),
    message: v.string(),
    progress_percent: v.nullable(v.number()),
    current_batch: v.nullable(v.number()),
    total_batches: v.nullable(v.number()),
    documents: v.nullable(v.number()),
    pending_enqueues: v.number(),
  }),
  v.transform((w) => ({
    busy: w.busy,
    message: w.message,
    progressPercent: w.progress_percent,
    currentBatch: w.current_batch,
    totalBatches: w.total_batches,
    documents: w.documents,
    pendingEnqueues: w.pending_enqueues,
  })),
);
export type WebIngestStatus = v.InferOutput<typeof webIngestStatus>;

const webFilePanelSnapshot = v.pipe(
  v.object({
    workspace: v.string(),
    files: v.array(webFileItem),
    ingest: webIngestStatus,
    next_cursor: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({
    workspace: w.workspace,
    files: w.files,
    ingest: w.ingest,
    nextCursor: w.next_cursor ?? null,
  })),
);
export type WebFilePanelSnapshot = v.InferOutput<typeof webFilePanelSnapshot>;

const webUploadReceipt = v.pipe(
  v.object({
    workspace: v.string(),
    file_count: v.number(),
    queued: v.boolean(),
    ingest: webIngestStatus,
  }),
  v.transform((w) => ({
    workspace: w.workspace,
    fileCount: w.file_count,
    queued: w.queued,
    ingest: w.ingest,
  })),
);
export type WebUploadReceipt = v.InferOutput<typeof webUploadReceipt>;

export type FailedRecoveryStatus = 'queued' | 'running' | 'succeeded' | 'partial' | 'failed';

const webFailedFileItem = v.pipe(
  v.object({
    document_id: v.string(),
    file_name: v.string(),
    error: v.string(),
    updated_at: v.string(),
  }),
  v.transform((w) => ({
    documentId: w.document_id,
    fileName: w.file_name,
    error: w.error,
    updatedAt: w.updated_at,
  })),
);
export type WebFailedFileItem = v.InferOutput<typeof webFailedFileItem>;

const webFailedRecoveryJob = v.pipe(
  v.object({
    job_id: v.string(),
    workspace: v.string(),
    status: v.picklist(['queued', 'running', 'succeeded', 'partial', 'failed']),
    retried: v.number(),
    succeeded: v.number(),
    failed: v.number(),
  }),
  v.transform((w) => ({
    jobId: w.job_id,
    workspace: w.workspace,
    status: w.status,
    retried: w.retried,
    succeeded: w.succeeded,
    failed: w.failed,
  })),
);
export type WebFailedRecoveryJob = v.InferOutput<typeof webFailedRecoveryJob>;

const webFailedFilesPage = v.pipe(
  v.object({
    workspace: v.string(),
    failed: v.array(webFailedFileItem),
    next_cursor: v.optional(v.nullable(v.string())),
    active_recovery: v.nullable(webFailedRecoveryJob),
  }),
  v.transform((w) => ({
    workspace: w.workspace,
    failed: w.failed,
    nextCursor: w.next_cursor ?? null,
    activeRecovery: w.active_recovery,
  })),
);
export type WebFailedFilesPage = v.InferOutput<typeof webFailedFilesPage>;

export class FilesApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'FilesApiError';
    this.status = status;
  }
}

function url(path: string, workspace: string): string {
  const target = new URL(path, window.location.origin);
  if (workspace) target.searchParams.set('workspace', workspace);
  return target.pathname + target.search;
}

async function json<Input, Output>(
  response: Response,
  schema: v.GenericSchema<Input, Output>,
  fallback: string,
): Promise<Output> {
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as {
      detail?: unknown;
      error?: unknown;
    } | null;
    const detail = typeof payload?.detail === 'string'
      ? payload.detail
      : typeof payload?.error === 'string'
        ? payload.error
        : fallback;
    throw new FilesApiError(response.status, detail);
  }
  try {
    return v.parse(schema, await response.json());
  } catch {
    throw new FilesApiError(response.status, fallback);
  }
}

export async function getFilePanel(
  workspace: string,
  cursor: string | null = null,
  signal?: AbortSignal,
): Promise<WebFilePanelSnapshot> {
  const target = new URL(url('/web/api/files', workspace), window.location.origin);
  if (cursor !== null) target.searchParams.set('cursor', cursor);
  const response = await fetch(target.pathname + target.search, {signal});
  return json(response, webFilePanelSnapshot, 'Failed to load files');
}

export async function getIngestStatus(
  workspace: string,
  signal?: AbortSignal,
): Promise<WebIngestStatus> {
  const response = await fetch(url('/web/api/ingest-status', workspace), {signal});
  return json(response, webIngestStatus, 'Failed to read ingest status');
}

export async function getFailedFiles(
  workspace: string,
  cursor: string | null = null,
  signal?: AbortSignal,
): Promise<WebFailedFilesPage> {
  const target = new URL(url('/web/api/files/failed', workspace), window.location.origin);
  if (cursor !== null) target.searchParams.set('cursor', cursor);
  const response = await fetch(target.pathname + target.search, {signal});
  return json(response, webFailedFilesPage, 'Failed to load documents needing attention');
}

export async function startFailedFileRetry(
  workspace: string,
  signal?: AbortSignal,
): Promise<WebFailedRecoveryJob> {
  const response = await fetch(url('/web/api/files/retry', workspace), {
    method: 'POST',
    headers: csrfHeaders(),
    signal,
  });
  return json(response, webFailedRecoveryJob, 'Document recovery could not be started');
}

export async function getFailedFileRetryStatus(
  workspace: string,
  jobId: string,
  signal?: AbortSignal,
): Promise<WebFailedRecoveryJob> {
  const path = `/web/api/files/retry/${encodeURIComponent(jobId)}`;
  const response = await fetch(url(path, workspace), {signal});
  return json(response, webFailedRecoveryJob, 'Failed to read document recovery status');
}

export async function uploadFileBatch(
  workspace: string,
  files: readonly File[],
  signal?: AbortSignal,
): Promise<WebUploadReceipt> {
  const body = new FormData();
  body.append('workspace', workspace);
  for (const file of files) {
    const relative = file as File & {_relativePath?: string; webkitRelativePath?: string};
    body.append('files', file, relative._relativePath || relative.webkitRelativePath || file.name);
  }
  const response = await fetch('/web/api/files/upload', {
    method: 'POST',
    headers: csrfHeaders(),
    body,
    signal,
  });
  return json(response, webUploadReceipt, 'Upload failed');
}

export async function deleteFileRequest(
  workspace: string,
  filePath: string,
  signal?: AbortSignal,
): Promise<WebFilePanelSnapshot> {
  const target = new URL('/web/api/files', window.location.origin);
  target.searchParams.set('workspace', workspace);
  target.searchParams.set('file_path', filePath);
  const response = await fetch(target.pathname + target.search, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  return json(response, webFilePanelSnapshot, 'Deletion failed');
}
