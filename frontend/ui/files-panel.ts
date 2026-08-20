// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  FilesApiError,
  deleteFileRequest,
  getFilePanel,
  getIngestStatus,
  uploadFileBatch,
  type WebFilePanelSnapshot,
  type WebIngestStatus,
} from '../api/files.ts';
import {isAbortError} from '../lib/errors.ts';
import {LightElement, StoreController} from '../lib/lit_host.ts';
import {ingestStore} from '../stores/ingestStore.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import {bus} from '../events/bus.ts';
import {openPanel} from './panel.ts';
import {withRelativePath} from './folder-upload.ts';
import {showToast} from './toast.ts';
import './ingest_target.ts';

const POLL_INTERVAL_MS = 2000;
let activeFileMutations = 0;

export function hasActiveFileMutation(): boolean {
  return activeFileMutations > 0;
}

function beginFileMutation(): void {
  activeFileMutations += 1;
}

function finishFileMutation(): void {
  activeFileMutations = Math.max(0, activeFileMutations - 1);
}

function isFilesPanelActive(): boolean {
  const panel = document.getElementById('panel');
  return Boolean(panel?.classList.contains('open') && panel.dataset.panelKind === 'files');
}

function uploadLabel(files: readonly File[], label?: string | null): string {
  if (label) return label;
  return files.length === 1 ? files[0].name : `${files.length} files`;
}

export class FilePanel extends LightElement {
  static properties = {
    snapshot: {state: true},
    loading: {state: true},
    error: {state: true},
    uploading: {state: true},
    acceptedFiles: {state: true},
  };

  declare snapshot: WebFilePanelSnapshot | null;
  declare loading: boolean;
  declare error: string | null;
  declare uploading: boolean;
  declare acceptedFiles: number;

  #workspace = '';
  #request: AbortController | null = null;
  #pollController: AbortController | null = null;
  #pollTimer: number | null = null;

  constructor() {
    super();
    this.snapshot = null;
    this.loading = true;
    this.error = null;
    this.uploading = false;
    this.acceptedFiles = 0;
    this.#workspace = ingestStore.workspace;
    new StoreController(this, ingestStore);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    queueMicrotask(() => { void this.reload(); });
  }

  override disconnectedCallback(): void {
    this.pause();
    super.disconnectedCallback();
  }

  protected override updated(): void {
    const workspace = ingestStore.workspace;
    if (workspace !== this.#workspace && this.isConnected) {
      this.#workspace = workspace;
      void this.reload();
    }
    this.querySelectorAll<HTMLElement>('.progress-bar-fill[data-pct]').forEach((fill) => {
      const value = Number(fill.dataset.pct);
      fill.style.width = `${Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0))}%`;
    });
  }

  async reload(showLoading = true): Promise<void> {
    const workspace = ingestStore.workspace;
    this.#workspace = workspace;
    this.uploading = false;
    const controller = this.#startRequest();
    this.#stopPolling();
    if (showLoading) this.loading = true;
    this.error = null;
    try {
      const snapshot = await getFilePanel(workspace, controller.signal);
      if (!this.#isCurrent(controller, workspace)) return;
      this.snapshot = snapshot;
      this.acceptedFiles = 0;
      if (snapshot.ingest.busy) this.#schedulePoll(workspace);
    } catch (error) {
      if (isAbortError(error) || !this.#isCurrent(controller, workspace)) return;
      this.error = error instanceof FilesApiError ? error.message : 'Failed to load files.';
    } finally {
      if (this.#request === controller) {
        this.#request = null;
        this.loading = false;
      }
    }
  }

  async upload(files: readonly File[], label?: string | null): Promise<void> {
    if (files.length === 0) return;
    const workspace = ingestStore.workspace;
    this.#workspace = workspace;
    const controller = this.#startRequest();
    this.#stopPolling();
    beginFileMutation();
    this.uploading = true;
    this.error = null;
    const name = uploadLabel(files, label);
    showToast(`Uploading ${name}...`);
    try {
      const receipt = await uploadFileBatch(workspace, files, controller.signal);
      if (!this.#isCurrent(controller, workspace)) return;
      this.snapshot = {
        workspace,
        files: this.snapshot?.workspace === workspace ? this.snapshot.files : [],
        ingest: receipt.ingest,
      };
      this.acceptedFiles = receipt.file_count;
      showToast('Files received — processing in background', 3000);
      this.#schedulePoll(workspace);
    } catch (error) {
      if (isAbortError(error) || !this.#isCurrent(controller, workspace)) return;
      const message = error instanceof FilesApiError ? error.message : 'Upload failed.';
      this.error = message;
      showToast(message, 5000);
    } finally {
      finishFileMutation();
      if (this.#request === controller) {
        this.#request = null;
        this.uploading = false;
      }
    }
  }

  pause(): void {
    this.#request?.abort();
    this.#request = null;
    this.uploading = false;
    this.#stopPolling();
  }

  async #deleteFile(filePath: string): Promise<void> {
    if (!filePath) return;
    const filename = filePath.split('/').pop() || filePath;
    if (!window.confirm(`Delete ${filename}?`)) return;
    const workspace = ingestStore.workspace;
    this.#stopPolling();
    const controller = this.#startRequest();
    beginFileMutation();
    this.error = null;
    try {
      const snapshot = await deleteFileRequest(workspace, filePath, controller.signal);
      if (!this.#isCurrent(controller, workspace)) return;
      this.snapshot = snapshot;
      showToast('File deleted.', 3000);
      if (snapshot.ingest.busy) this.#schedulePoll(workspace);
    } catch (error) {
      if (isAbortError(error) || !this.#isCurrent(controller, workspace)) return;
      const message = error instanceof FilesApiError ? error.message : 'Deletion failed.';
      this.error = message;
      showToast(message, 5000);
    } finally {
      finishFileMutation();
      if (this.#request === controller) this.#request = null;
    }
  }

  async #poll(workspace: string): Promise<void> {
    const controller = new AbortController();
    this.#pollController = controller;
    try {
      const status = await getIngestStatus(workspace, controller.signal);
      if (workspace !== ingestStore.workspace || !isFilesPanelActive()) return;
      if (!status.busy) {
        await this.reload(false);
        return;
      }
      this.#setIngestStatus(workspace, status);
      this.#schedulePoll(workspace);
    } catch (error) {
      if (isAbortError(error)) return;
      if (workspace === ingestStore.workspace && isFilesPanelActive()) {
        this.#schedulePoll(workspace);
      }
    } finally {
      if (this.#pollController === controller) this.#pollController = null;
    }
  }

  #setIngestStatus(workspace: string, ingest: WebIngestStatus): void {
    this.snapshot = {
      workspace,
      files: this.snapshot?.workspace === workspace ? this.snapshot.files : [],
      ingest,
    };
  }

  #schedulePoll(workspace: string): void {
    this.#stopPolling();
    this.#pollTimer = window.setTimeout(() => {
      this.#pollTimer = null;
      void this.#poll(workspace);
    }, POLL_INTERVAL_MS);
  }

  #stopPolling(): void {
    if (this.#pollTimer !== null) window.clearTimeout(this.#pollTimer);
    this.#pollTimer = null;
    this.#pollController?.abort();
    this.#pollController = null;
  }

  #startRequest(): AbortController {
    this.#request?.abort();
    const controller = new AbortController();
    this.#request = controller;
    return controller;
  }

  #isCurrent(controller: AbortController, workspace: string): boolean {
    return this.#request === controller && ingestStore.workspace === workspace;
  }

  #chooseFiles(): void {
    this.querySelector<HTMLInputElement>('#file-input')?.click();
  }

  #fileInputChanged(event: Event): void {
    const input = event.currentTarget as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    input.value = '';
    if (files.length > 0) void this.upload(files);
  }

  #progress(status: WebIngestStatus): TemplateResult | typeof nothing {
    if (!status.busy) return nothing;
    return html`
      <div id="ingest-progress">
        <div class="file-status">
          <div class="spinner"></div>
          <span>${status.message || 'Ingesting...'}</span>
        </div>
        ${status.progress_percent === null ? nothing : html`
          <div class="progress-bar-track" role="progressbar"
               aria-valuenow=${String(status.progress_percent)} aria-valuemin="0"
               aria-valuemax="100" aria-label="Ingest progress">
            <div class="progress-bar-fill" data-pct=${String(status.progress_percent)}></div>
          </div>
          <div class="progress-label">
            batch ${status.current_batch}/${status.total_batches} · ${status.documents} doc(s)
          </div>
        `}
        ${status.pending_enqueues > 0 ? html`
          <div class="ingest-queue-notice">
            ${status.pending_enqueues} upload(s) queued — will process after current batch
          </div>
        ` : nothing}
      </div>
    `;
  }

  protected override render(): TemplateResult {
    const snapshot = this.snapshot;
    const files = snapshot?.files ?? [];
    return html`
      ${snapshot ? this.#progress(snapshot.ingest) : nothing}
      ${this.error ? html`<div class="file-error" role="alert">${this.error}</div>` : nothing}
      <div class="upload-zone${this.uploading ? ' is-uploading' : ''}" id="upload-zone">
        <button type="button" class="upload-zone-file-action" aria-label="Choose files"
                @click=${() => { this.#chooseFiles(); }}>
          <span class="upload-text">Drop files or folders, or click to choose files</span>
        </button>
        <button type="button" class="upload-folder-action"
                @click=${() => {
                  this.dispatchEvent(new CustomEvent('files-folder-request', {bubbles: true}));
                }}>Choose folder</button>
        <input class="hidden" type="file" id="file-input" name="files" multiple
               @change=${(event: Event) => { this.#fileInputChanged(event); }}>
        <div id="upload-spinner" class="file-status">Uploading...</div>
      </div>
      ${this.loading ? html`
        <div class="file-status file-status--loading"><div class="spinner"></div><span>Loading files...</span></div>
      ` : nothing}
      ${!this.loading ? repeat(
        files,
        (file) => file.file_path,
        (file) => html`
          <div class="file-item">
            <span class="file-name" title=${file.file_path}>${file.file_name}</span>
            <button class="file-delete" type="button" aria-label=${`Delete ${file.file_name}`}
                    @click=${() => { void this.#deleteFile(file.file_path); }}>
              <svg class="file-delete-icon" width="14" height="14" viewBox="0 0 24 24"
                   fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
        `,
      ) : nothing}
      ${!this.loading && !this.error && files.length === 0 && !snapshot?.ingest.busy ? html`
        <div class="empty-state">No files ingested in workspace “${this.#workspace}”.</div>
      ` : nothing}
      ${this.acceptedFiles > 0 && snapshot?.ingest.busy ? html`
        <div class="ingest-queue-notice ingest-queue-notice--inline">
          ${this.acceptedFiles} new file(s) accepted for ingest
        </div>
      ` : nothing}
    `;
  }
}

customElements.define('file-panel', FilePanel);

declare global {
  interface HTMLElementTagNameMap {
    'file-panel': FilePanel;
  }
}

function filePanel(): FilePanel | null {
  return document.querySelector('file-panel');
}

function ensureFilePanel(): FilePanel | null {
  const content = document.getElementById('panel-content');
  if (!content) return null;
  const existing = content.querySelector('file-panel');
  if (existing) return existing;
  const panel = document.createElement('file-panel');
  content.replaceChildren(panel);
  return panel;
}

export async function refreshFilePanel(): Promise<void> {
  await ensureFilePanel()?.reload();
}

export async function uploadFilesToWorkspace(
  files: readonly File[],
  label?: string | null,
): Promise<void> {
  if (!isFilesPanelActive()) openFilesPanel();
  const panel = ensureFilePanel();
  if (!panel) return;
  await panel.updateComplete;
  await panel.upload(files, label);
}

function handleFolderInputChange(input: HTMLInputElement): void {
  const fileList = input.files;
  if (!fileList || fileList.length === 0) return;
  const rawFiles = Array.from(fileList);
  let folderName: string | null = null;
  const files = rawFiles.map(function(file) {
    const path = file.webkitRelativePath || file.name;
    const relativeFile = withRelativePath(file, path);
    if (!folderName && file.webkitRelativePath) folderName = file.webkitRelativePath.split('/')[0];
    return relativeFile;
  });
  void uploadFilesToWorkspace(files, folderName);
  input.value = '';
}

export function openFilesPanel(): void {
  ingestStore.resetToPrimary();
  openPanel('FILES');
  const panel = ensureFilePanel();
  // A newly connected element owns its first load; a reused one refreshes.
  if (panel?.snapshot || panel?.error) void panel.reload();
}

export function setupFilesPanel(): void {
  document.getElementById('files-btn')?.addEventListener('click', function(event) {
    event.preventDefault();
    openFilesPanel();
  });

  const folderInput = document.getElementById('folder-input') as HTMLInputElement | null;
  folderInput?.addEventListener('change', function() { handleFolderInputChange(folderInput); });
  document.addEventListener('files-folder-request', function() { folderInput?.click(); });

  bus.on('workspaceCreated', ({workspace}) => {
    ingestStore.set(workspace);
  });
  bus.on('workspaceDeleted', ({nextWorkspace}) => {
    ingestStore.set(nextWorkspace || workspaceStore.primary);
  });

  document.body.addEventListener('panelOpening', function(event) {
    const title = (event as CustomEvent<{title?: string}>).detail?.title;
    if (title && title !== 'FILES') filePanel()?.pause();
  });
  document.body.addEventListener('panelClosed', function() { filePanel()?.pause(); });
}
