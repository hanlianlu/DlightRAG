// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {csrfHeaders} from './csrf.ts';

export interface WebFileItem {
  file_name: string;
  file_path: string;
}

export interface WebIngestStatus {
  busy: boolean;
  message: string;
  progress_percent: number | null;
  current_batch: number | null;
  total_batches: number | null;
  documents: number | null;
  pending_enqueues: number;
}

export interface WebFilePanelSnapshot {
  workspace: string;
  files: WebFileItem[];
  ingest: WebIngestStatus;
  next_cursor: string | null;
}

export interface WebUploadReceipt {
  workspace: string;
  file_count: number;
  queued: boolean;
  ingest: WebIngestStatus;
}

export type FailedRecoveryStatus = 'queued' | 'running' | 'succeeded' | 'partial' | 'failed';

export interface WebFailedFileItem {
  document_id: string;
  file_name: string;
  error: string;
  updated_at: string;
}

export interface WebFailedRecoveryJob {
  job_id: string;
  workspace: string;
  status: FailedRecoveryStatus;
  retried: number;
  succeeded: number;
  failed: number;
}

export interface WebFailedFilesPage {
  workspace: string;
  failed: WebFailedFileItem[];
  next_cursor: string | null;
  active_recovery: WebFailedRecoveryJob | null;
}

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

async function json<T>(response: Response, fallback: string): Promise<T> {
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
    return await response.json() as T;
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
  const snapshot = await json<Omit<WebFilePanelSnapshot, 'next_cursor'> & {
    next_cursor?: string | null;
  }>(response, 'Failed to load files');
  return {...snapshot, next_cursor: snapshot.next_cursor ?? null};
}

export async function getIngestStatus(
  workspace: string,
  signal?: AbortSignal,
): Promise<WebIngestStatus> {
  const response = await fetch(url('/web/api/ingest-status', workspace), {signal});
  return json(response, 'Failed to read ingest status');
}

export async function getFailedFiles(
  workspace: string,
  cursor: string | null = null,
  signal?: AbortSignal,
): Promise<WebFailedFilesPage> {
  const target = new URL(url('/web/api/files/failed', workspace), window.location.origin);
  if (cursor !== null) target.searchParams.set('cursor', cursor);
  const response = await fetch(target.pathname + target.search, {signal});
  return json(response, 'Failed to load documents needing attention');
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
  return json(response, 'Document recovery could not be started');
}

export async function getFailedFileRetryStatus(
  workspace: string,
  jobId: string,
  signal?: AbortSignal,
): Promise<WebFailedRecoveryJob> {
  const path = `/web/api/files/retry/${encodeURIComponent(jobId)}`;
  const response = await fetch(url(path, workspace), {signal});
  return json(response, 'Failed to read document recovery status');
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
  return json(response, 'Upload failed');
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
  const snapshot = await json<Omit<WebFilePanelSnapshot, 'next_cursor'> & {
    next_cursor?: string | null;
  }>(response, 'Deletion failed');
  return {...snapshot, next_cursor: snapshot.next_cursor ?? null};
}
