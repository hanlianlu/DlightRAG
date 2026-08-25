// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
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
import {withRelativePath} from './folder-upload.ts';
import {showToast} from './toast.ts';

const POLL_INTERVAL_MS = 2000;

function uploadLabel(files: readonly File[], label?: string | null): string {
  if (label) return label;
  return files.length === 1 ? files[0].name : `${files.length} files`;
}

/** File-management content, async work, and upload intent owned by the Inspector. */
export class DlInspectorFiles extends LightElement {
  static properties = {
    active: {attribute: false},
    snapshot: {state: true},
    loading: {state: true},
    error: {state: true},
    uploading: {state: true},
    acceptedFiles: {state: true},
  };

  declare active: boolean;
  declare snapshot: WebFilePanelSnapshot | null;
  declare loading: boolean;
  declare error: string | null;
  declare uploading: boolean;
  declare acceptedFiles: number;

  #workspace = '';
  #request: AbortController | null = null;
  #pollController: AbortController | null = null;
  #pollTimer: number | null = null;
  #activeMutations = 0;
  #releaseWorkspaceEvents: (() => void)[] = [];

  constructor() {
    super();
    this.active = false;
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
    this.#releaseWorkspaceEvents = [
      bus.on('workspaceCreated', ({workspace}) => { ingestStore.set(workspace); }),
      bus.on('workspaceDeleted', ({nextWorkspace}) => {
        ingestStore.set(nextWorkspace || workspaceStore.primary);
      }),
    ];
    if (this.active) queueMicrotask(() => { void this.reload(); });
  }

  override disconnectedCallback(): void {
    this.pause();
    for (const release of this.#releaseWorkspaceEvents) release();
    this.#releaseWorkspaceEvents = [];
    super.disconnectedCallback();
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('active')) {
      if (this.active) void this.reload();
      else this.pause();
    }
    const workspace = ingestStore.workspace;
    if (this.active && workspace !== this.#workspace && this.isConnected) {
      this.#workspace = workspace;
      void this.reload();
    }
    this.querySelectorAll<HTMLElement>('.progress-bar-fill[data-pct]').forEach((fill) => {
      const value = Number(fill.dataset.pct);
      fill.style.width = `${Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0))}%`;
    });
  }

  get hasActiveMutation(): boolean {
    return this.#activeMutations > 0;
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
    this.#beginMutation();
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
      this.#finishMutation();
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
    this.#beginMutation();
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
      this.#finishMutation();
      if (this.#request === controller) this.#request = null;
    }
  }

  async #poll(workspace: string): Promise<void> {
    const controller = new AbortController();
    this.#pollController = controller;
    try {
      const status = await getIngestStatus(workspace, controller.signal);
      if (workspace !== ingestStore.workspace || !this.active || !this.isConnected) return;
      if (!status.busy) {
        await this.reload(false);
        return;
      }
      this.#setIngestStatus(workspace, status);
      this.#schedulePoll(workspace);
    } catch (error) {
      if (isAbortError(error)) return;
      if (workspace === ingestStore.workspace && this.active && this.isConnected) {
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

  #beginMutation(): void {
    this.#activeMutations += 1;
  }

  #finishMutation(): void {
    this.#activeMutations = Math.max(0, this.#activeMutations - 1);
  }

  #chooseFiles(): void {
    this.querySelector<HTMLInputElement>('#file-input')?.click();
  }

  #chooseFolder(): void {
    this.querySelector<HTMLInputElement>('#folder-input')?.click();
  }

  #fileInputChanged(event: Event): void {
    const input = event.currentTarget as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    input.value = '';
    if (files.length > 0) void this.upload(files);
  }

  #folderInputChanged(event: Event): void {
    const input = event.currentTarget as HTMLInputElement;
    const rawFiles = Array.from(input.files ?? []);
    input.value = '';
    if (rawFiles.length === 0) return;
    let folderName: string | null = null;
    const files = rawFiles.map((file) => {
      const path = file.webkitRelativePath || file.name;
      if (!folderName && file.webkitRelativePath) folderName = path.split('/')[0];
      return withRelativePath(file, path);
    });
    void this.upload(files, folderName);
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
                @click=${() => { this.#chooseFolder(); }}>Choose folder</button>
        <input class="hidden" type="file" id="file-input" name="files" multiple
               @change=${(event: Event) => { this.#fileInputChanged(event); }}>
        <input class="hidden" type="file" id="folder-input" webkitdirectory directory multiple
               @change=${(event: Event) => { this.#folderInputChanged(event); }}>
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

customElements.define('dl-inspector-files', DlInspectorFiles);

declare global {
  interface HTMLElementTagNameMap {
    'dl-inspector-files': DlInspectorFiles;
  }
}
