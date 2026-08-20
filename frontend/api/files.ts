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
}

export interface WebUploadReceipt {
  workspace: string;
  file_count: number;
  queued: boolean;
  ingest: WebIngestStatus;
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
  signal?: AbortSignal,
): Promise<WebFilePanelSnapshot> {
  const response = await fetch(url('/web/api/files', workspace), {signal});
  return json(response, 'Failed to load files');
}

export async function getIngestStatus(
  workspace: string,
  signal?: AbortSignal,
): Promise<WebIngestStatus> {
  const response = await fetch(url('/web/api/ingest-status', workspace), {signal});
  return json(response, 'Failed to read ingest status');
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
  return json(response, 'Deletion failed');
}
