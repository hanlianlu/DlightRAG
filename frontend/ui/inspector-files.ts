// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges } from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  deleteFileRequest,
  FilesApiError,
  getFilePanel,
  getIngestStatus,
  uploadFileBatch,
  type WebFilePanelSnapshot,
  type WebIngestStatus,
} from '../api/files.ts';
import {icon} from '../design-system/index.ts';
import {isAbortError} from '../lib/errors.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {type AppHandles, productionHandles } from '../stores/app-handles.ts';
import {withRelativePath} from './folder-upload.ts';
import {modalResult} from './modal.ts';
import {requestToast} from './toast-request.ts';
import './failed-file-recovery.ts';
import fileStyles from '../styles/inspector-files.module.css';
import type {DlFailedFileRecovery} from './failed-file-recovery.ts';
import {InspectorFilesSession} from './inspector-files-session.ts';
import type {ToastRequestDetail} from './toast.ts';

function uploadLabel(files: readonly File[], label?: string | null): string {
  if (label) return label;
  return files.length === 1
    ? files[0].name
    : msg(str`${files.length} files`, {id: 'inspectorFiles.nFiles'});
}

/** File-management content, async work, and upload intent owned by the Inspector. */
export class DlInspectorFiles extends LightElement {
  static properties = {
    handles: {attribute: false},
    active: {attribute: false},
    snapshot: {state: true},
    loading: {state: true},
    error: {state: true},
    uploading: {state: true},
    acceptedFiles: {state: true},
    filesLoadMoreState: {state: true},
  };

  declare handles: AppHandles;
  declare active: boolean;
  declare snapshot: WebFilePanelSnapshot | null;
  declare loading: boolean;
  declare error: string | null;
  declare uploading: boolean;
  declare acceptedFiles: number;
  declare filesLoadMoreState: 'idle' | 'loading' | 'error';

  #workspace = '';
  readonly #session = new InspectorFilesSession();
  #olderFilesFlight: Promise<void> | null = null;
  #olderFilesAnnouncement = '';
  #restoreOlderFocus = false;
  #deleteTrigger: HTMLElement | null = null;
  #releaseWorkspaceEvents: (() => void)[] = [];

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.handles = productionHandles();
    this.active = false;
    this.snapshot = null;
    this.loading = true;
    this.error = null;
    this.uploading = false;
    this.acceptedFiles = 0;
    this.filesLoadMoreState = 'idle';
    this.#workspace = this.handles.ingest.workspace;
    /** Store reads: this.handles.ingest.workspace. */
    new StoreController(this, this.handles.ingest);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    if (this.active) queueMicrotask(() => { void this.reload(); });
  }

  override disconnectedCallback(): void {
    this.pause();
    super.disconnectedCallback();
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('active')) {
      if (this.active) void this.reload();
      else this.pause();
    }
    const workspace = this.handles.ingest.workspace;
    if (this.active && workspace !== this.#workspace && this.isConnected) {
      this.#workspace = workspace;
      void this.reload();
    }
    this.querySelectorAll<HTMLElement>('[data-pct]').forEach((fill) => {
      const value = Number(fill.dataset.pct);
      fill.style.width = `${Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0))}%`;
    });
  }

  get hasActiveMutation(): boolean {
    return this.#session.mutating;
  }

  async reload(showLoading = true): Promise<void> {
    const workspace = this.handles.ingest.workspace;
    this.#invalidateOlderFiles();
    if (this.snapshot !== null && this.snapshot.workspace !== workspace) {
      this.snapshot = null;
      this.acceptedFiles = 0;
    }
    this.#workspace = workspace;
    this.uploading = false;
    const controller = this.#startRequest();
    this.#stopPolling();
    if (showLoading) this.loading = true;
    this.error = null;
    try {
      const snapshot = await getFilePanel(workspace, null, controller.signal);
      if (!this.#isCurrent(controller, workspace)) return;
      this.snapshot = snapshot;
      this.acceptedFiles = 0;
      if (snapshot.ingest.busy) this.#schedulePoll(workspace);
    } catch (error) {
      if (isAbortError(error) || !this.#isCurrent(controller, workspace)) return;
      this.error = error instanceof FilesApiError
        ? error.message
        : msg('Failed to load files.', {id: 'inspectorFiles.loadFailed'});
    } finally {
      if (this.#session.finishRequest(controller)) this.loading = false;
    }
  }

  loadOlderFiles(): Promise<void> {
    if (this.#olderFilesFlight !== null) return this.#olderFilesFlight;
    const workspace = this.handles.ingest.workspace;
    const cursor = this.snapshot?.workspace === workspace
      ? this.snapshot.nextCursor
      : null;
    if (!cursor || this.loading || this.#session.requestBusy || !this.active) {
      return Promise.resolve();
    }
    const flight = this.#loadOlderFilesPage(
      workspace,
      cursor,
      this.#session.olderGeneration,
    );
    this.#olderFilesFlight = flight;
    void flight.finally(() => {
      if (this.#olderFilesFlight === flight) this.#olderFilesFlight = null;
    });
    return flight;
  }

  async #loadOlderFilesPage(
    workspace: string,
    cursor: string,
    generation: number,
  ): Promise<void> {
    const controller = this.#session.startOlder();
    this.filesLoadMoreState = 'loading';
    this.#olderFilesAnnouncement = msg('Loading older files…', {id: 'inspectorFiles.loadingOlder'});
    try {
      const older = await getFilePanel(workspace, cursor, controller.signal);
      const current = this.snapshot;
      if (
        !this.#session.isOlderCurrent(controller, generation)
        || workspace !== this.handles.ingest.workspace
        || current?.workspace !== workspace
        || current.nextCursor !== cursor
      ) {
        if (this.#session.isOlderCurrent(controller, generation)) {
          this.filesLoadMoreState = 'idle';
          this.#olderFilesAnnouncement = '';
        }
        return;
      }
      if (older.workspace !== workspace) {
        throw new Error('older file page changed workspace identity');
      }
      const paths = new Set(current.files.map((file) => file.filePath));
      const appended = older.files.filter((file) => {
        if (paths.has(file.filePath)) return false;
        paths.add(file.filePath);
        return true;
      });
      this.snapshot = {
        ...current,
        files: [...current.files, ...appended],
        nextCursor: older.nextCursor,
      };
      this.filesLoadMoreState = 'idle';
      this.#olderFilesAnnouncement = appended.length === 1
        ? msg('Loaded 1 older file.', {id: 'inspectorFiles.loadedOneOlder'})
        : msg(str`Loaded ${appended.length} older files.`, {id: 'inspectorFiles.loadedOlder'});
      if (older.nextCursor === null && this.#restoreOlderFocus) {
        this.#restoreOlderFocus = false;
        await this.updateComplete;
        this.querySelector<HTMLElement>('#file-list')?.focus({preventScroll: true});
      }
    } catch (error) {
      if (
        isAbortError(error)
        || !this.#session.isOlderCurrent(controller, generation)
      ) return;
      this.filesLoadMoreState = 'error';
      this.#olderFilesAnnouncement = msg('Older files could not be loaded.', {id: 'inspectorFiles.olderFilesFailed'});
    } finally {
      this.#session.finishOlder(controller);
    }
  }

  async upload(files: readonly File[], label?: string | null): Promise<void> {
    if (files.length === 0) return;
    const workspace = this.handles.ingest.workspace;
    this.#invalidateOlderFiles();
    this.#workspace = workspace;
    const controller = this.#startRequest();
    this.#stopPolling();
    this.#beginMutation();
    this.uploading = true;
    this.error = null;
    const name = uploadLabel(files, label);
    requestToast(this, {message: msg(str`Uploading ${name}...`, {id: 'inspectorFiles.uploadingToast'})});
    try {
      const receipt = await uploadFileBatch(workspace, files, controller.signal);
      if (!this.#isCurrent(controller, workspace)) return;
      this.snapshot = {
        workspace,
        files: this.snapshot?.workspace === workspace ? this.snapshot.files : [],
        ingest: receipt.ingest,
        nextCursor: this.snapshot?.workspace === workspace
          ? this.snapshot.nextCursor
          : null,
      };
      this.acceptedFiles = receipt.fileCount;
      requestToast(this, {
        message: msg('Files received — processing in background', {id: 'inspectorFiles.filesReceived'}),
        duration: 3000,
      });
      this.#schedulePoll(workspace);
    } catch (error) {
      if (isAbortError(error) || !this.#isCurrent(controller, workspace)) return;
      const message = error instanceof FilesApiError
        ? error.message
        : msg('Upload failed.', {id: 'inspectorFiles.uploadFailed'});
      this.error = message;
      requestToast(this, {message, duration: 3000});
    } finally {
      this.#finishMutation();
      if (this.#session.finishRequest(controller)) {
        this.uploading = false;
        this.loading = false;
      }
    }
  }

  pause(): void {
    this.#invalidateOlderFiles();
    this.#session.pause();
    this.uploading = false;
  }

  async #deleteFile(filePath: string): Promise<void> {
    if (!filePath) return;
    const filename = filePath.split('/').pop() || filePath;
    const dialog = this.querySelector<HTMLDialogElement>('#delete-file-dialog');
    const message = this.querySelector<HTMLElement>('#delete-file-message');
    if (!dialog || !message) return;
    message.textContent = msg(
      str`${filename} will be permanently removed from this workspace.`,
      {id: 'inspectorFiles.deleteNotice'},
    );
    if (await modalResult(this, dialog, () => this.#restoreDeleteTrigger()) !== 'confirm') return;
    const workspace = this.handles.ingest.workspace;
    this.#invalidateOlderFiles();
    this.#stopPolling();
    const controller = this.#startRequest();
    this.#beginMutation();
    this.error = null;
    try {
      const snapshot = await deleteFileRequest(workspace, filePath, controller.signal);
      if (!this.#isCurrent(controller, workspace)) return;
      this.snapshot = snapshot;
      requestToast(this, {
        message: msg('File deleted.', {id: 'inspectorFiles.fileDeleted'}),
        duration: 3000,
      });
      if (snapshot.ingest.busy) this.#schedulePoll(workspace);
    } catch (error) {
      if (isAbortError(error) || !this.#isCurrent(controller, workspace)) return;
      const message = error instanceof FilesApiError
        ? error.message
        : msg('Deletion failed.', {id: 'inspectorFiles.deletionFailed'});
      this.error = message;
      requestToast(this, {message, duration: 3000});
    } finally {
      this.#finishMutation();
      if (this.#session.finishRequest(controller)) this.loading = false;
    }
  }

  async #poll(workspace: string): Promise<void> {
    const controller = this.#session.startPollRequest();
    try {
      const status = await getIngestStatus(workspace, controller.signal);
      if (workspace !== this.handles.ingest.workspace || !this.active || !this.isConnected) return;
      if (!status.busy) {
        await this.reload(false);
        const recovery = this.querySelector<DlFailedFileRecovery>('dl-failed-file-recovery');
        await recovery?.refresh(false);
        return;
      }
      this.#setIngestStatus(workspace, status);
      this.#schedulePoll(workspace);
    } catch (error) {
      if (isAbortError(error)) return;
      if (workspace === this.handles.ingest.workspace && this.active && this.isConnected) {
        this.#schedulePoll(workspace);
      }
    } finally {
      this.#session.finishPollRequest(controller);
    }
  }

  #setIngestStatus(workspace: string, ingest: WebIngestStatus): void {
    this.snapshot = {
      workspace,
      files: this.snapshot?.workspace === workspace ? this.snapshot.files : [],
      ingest,
      nextCursor: this.snapshot?.workspace === workspace
        ? this.snapshot.nextCursor
        : null,
    };
  }

  #schedulePoll(workspace: string): void {
    this.#session.schedulePoll(workspace, (next) => { void this.#poll(next); });
  }

  #invalidateOlderFiles(): void {
    this.#session.invalidateOlder();
    this.#olderFilesFlight = null;
    this.filesLoadMoreState = 'idle';
    this.#olderFilesAnnouncement = '';
    this.#restoreOlderFocus = false;
  }

  #stopPolling(): void {
    this.#session.stopPolling();
  }

  #startRequest(): AbortController {
    return this.#session.startRequest();
  }

  #isCurrent(controller: AbortController, workspace: string): boolean {
    return this.#session.isCurrent(controller, workspace, this.handles.ingest.workspace);
  }

  #beginMutation(): void {
    this.#session.beginMutation();
  }

  #finishMutation(): void {
    this.#session.finishMutation();
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

  #loadOlderFiles = (event: Event): void => {
    const button = event.currentTarget as HTMLButtonElement;
    this.#restoreOlderFocus = document.activeElement === button;
    void this.loadOlderFiles();
  };


  #restoreDeleteTrigger(): void {
    const trigger = this.#deleteTrigger;
    this.#deleteTrigger = null;
    if (trigger?.isConnected) trigger.focus();
  }

  #deleteDialog(): TemplateResult {
    return html`
      <dialog id="delete-file-dialog" class="confirm-dialog"
              aria-labelledby="delete-file-title" aria-describedby="delete-file-message">
        <form method="dialog">
          <h2 id="delete-file-title">${msg('Delete file', {id: 'inspectorFiles.deleteTitle'})}</h2>
          <p id="delete-file-message"></p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'inspectorFiles.cancel'})}</button>
            <button type="submit" value="confirm" class="dl-dialog-danger">${msg('Delete', {id: 'inspectorFiles.delete'})}</button>
          </div>
        </form>
      </dialog>
    `;
  }

  #progress(status: WebIngestStatus): TemplateResult | typeof nothing {
    if (!status.busy) return nothing;
    return html`
      <div id="ingest-progress">
        <div class=${fileStyles['file-status']}>
          <div class=${fileStyles['spinner']}></div>
          <span>${status.message || msg('Ingesting...', {id: 'inspectorFiles.ingesting'})}</span>
        </div>
        ${status.progressPercent === null ? nothing : html`
          <div class=${fileStyles['progress-bar-track']} role="progressbar"
               aria-valuenow=${String(status.progressPercent)} aria-valuemin="0"
               aria-valuemax="100"
               aria-label=${msg('Ingest progress', {id: 'inspectorFiles.ingestProgressAria'})}>
            <div class=${fileStyles['progress-bar-fill']} data-pct=${String(status.progressPercent)}></div>
          </div>
          <div class=${fileStyles['progress-label']}>
            ${msg(str`batch ${status.currentBatch}/${status.totalBatches} · ${status.documents} doc(s)`, {id: 'inspectorFiles.batchProgress'})}
          </div>
        `}
        ${status.pendingEnqueues > 0 ? html`
          <div class=${fileStyles['ingest-queue-notice']}>
            ${msg(str`${status.pendingEnqueues} upload(s) queued — will process after current batch`, {id: 'inspectorFiles.queueNotice'})}
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
      <div class=${`${fileStyles['upload-zone']}${this.uploading ? ` ${fileStyles['is-uploading']}` : ''}`} id="upload-zone">
        <button type="button" class=${fileStyles['upload-zone-file-action']}
                data-upload-file-action
                aria-label=${msg('Choose files', {id: 'inspectorFiles.chooseFilesAria'})}
                @click=${() => { this.#chooseFiles(); }}>
          <span class=${fileStyles['upload-text']}>${msg('Drop files or folders, or click to choose files', {id: 'inspectorFiles.dropHint'})}</span>
        </button>
        <button type="button" class=${fileStyles['upload-folder-action']}
                @click=${() => { this.#chooseFolder(); }}>${msg('Choose folder', {id: 'inspectorFiles.chooseFolder'})}</button>
        <input class="hidden" type="file" id="file-input" name="files" multiple
               @change=${(event: Event) => { this.#fileInputChanged(event); }}>
        <input class="hidden" type="file" id="folder-input" webkitdirectory directory multiple
               @change=${(event: Event) => { this.#folderInputChanged(event); }}>
        <div id="upload-spinner" class=${fileStyles['file-status']}>${msg('Uploading...', {id: 'inspectorFiles.uploadingStatus'})}</div>
      </div>
      <dl-failed-file-recovery
        .workspace=${this.handles.ingest.workspace}
        .active=${this.active}
        @dl-failed-file-recovery-complete=${() => { void this.reload(false); }}
      ></dl-failed-file-recovery>
      ${this.loading ? html`
        <div class=${fileStyles['file-status']}><div class=${fileStyles['spinner']}></div><span>${msg('Loading files...', {id: 'inspectorFiles.loadingFiles'})}</span></div>
      ` : nothing}
      ${!this.loading ? html`
        <div id="file-list" role="list" aria-label=${msg('Processed files', {id: 'inspectorFiles.processedFilesAria'})} tabindex="-1">
          ${repeat(
            files,
            (file) => file.filePath,
            (file) => html`
              <div class=${fileStyles['file-item']} role="listitem" data-file-item>
                <span class=${fileStyles['file-name']} title=${file.filePath}>${file.fileName}</span>
                <button class=${fileStyles['file-delete']} type="button" data-file-delete
                        aria-label=${msg(str`Delete ${file.fileName}`, {id: 'inspectorFiles.deleteFileAria'})}
                        @click=${(event: Event) => {
                          this.#deleteTrigger = event.currentTarget as HTMLElement;
                          void this.#deleteFile(file.filePath);
                        }}>
                  ${icon('close', {size: 'sm', className: fileStyles['file-delete-icon']})}
                </button>
              </div>
            `,
          )}
        </div>
        ${snapshot?.nextCursor ? html`
          <div class=${fileStyles['file-page-control']}>
            <button type="button" data-load-older-files
                    aria-busy=${this.filesLoadMoreState === 'loading' ? 'true' : 'false'}
                    ?disabled=${this.filesLoadMoreState === 'loading'}
                    @click=${this.#loadOlderFiles}>
              ${this.filesLoadMoreState === 'error'
                ? msg('Retry loading older files', {id: 'inspectorFiles.retryLoadOlder'})
                : msg('Load older files', {id: 'inspectorFiles.loadOlder'})}
            </button>
          </div>
        ` : nothing}
        <span class="sr-only" data-older-files-status role="status" aria-live="polite">
          ${this.#olderFilesAnnouncement}
        </span>
      ` : nothing}
      ${!this.loading && !this.error && files.length === 0 && !snapshot?.ingest.busy ? html`
        <div class="empty-state">${msg(str`No files ingested in workspace “${this.#workspace}”.`, {id: 'inspectorFiles.emptyState'})}</div>
      ` : nothing}
      ${this.acceptedFiles > 0 && snapshot?.ingest.busy ? html`
        <div class=${`${fileStyles['ingest-queue-notice']} ${fileStyles['ingest-queue-notice--inline']}`}>
          ${msg(str`${this.acceptedFiles} new file(s) accepted for ingest`, {id: 'inspectorFiles.acceptedForIngest'})}
        </div>
      ` : nothing}
      ${this.#deleteDialog()}
    `;
  }
}

customElements.define('dl-inspector-files', DlInspectorFiles);

declare global {
  interface HTMLElementTagNameMap {
    'dl-inspector-files': DlInspectorFiles;
  }
}
