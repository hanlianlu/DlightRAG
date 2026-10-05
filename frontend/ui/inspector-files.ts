// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  deleteFileRequest,
  getFilePanel,
  uploadFileBatch,
  type WebFilePanelSnapshot,
} from '../api/files.ts';
import {icon} from '../design-system/index.ts';
import {CorpusRunTracker, type TrackedCorpusRun} from '../lib/corpus-run-tracker.ts';
import {ApiError} from '../api/wire.ts';
import {apiErrorMessage, authRefusalMessage, isAbortError} from '../lib/errors.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {KeysetPager, type PageLoadState} from '../lib/paged.ts';
import {type AppHandles, productionHandles } from '../stores/app-handles.ts';
import {withRelativePath} from './folder-upload.ts';
import {modalResult, publishModalState, showOwnedModal} from './modal.ts';
import {deleteWorkspaceRequest, resetWorkspaceRequest} from '../api/workspaces.ts';
import {requestToast} from './toast-request.ts';
import {corpusRepairNotice, resumeRepairLabel, resumeRepairResult} from './corpus-repair.ts';
import {loadOlderControl} from './load-older.ts';
import './failed-file-recovery.ts';
import fileStyles from '../styles/inspector-files.module.css';
import type {DlFailedFileRecovery} from './failed-file-recovery.ts';
import {InspectorFilesSession} from './inspector-files-session.ts';

/** One destructive Workspace action awaiting typed confirmation. */
interface WorkspaceActionIntent {
  kind: 'reset' | 'delete';
  workspace: string;
}

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
    actionIntent: {state: true},
    actionPending: {state: true},
    actionConfirmed: {state: true},
  };

  declare handles: AppHandles;
  declare active: boolean;
  declare snapshot: WebFilePanelSnapshot | null;
  declare loading: boolean;
  declare error: string | null;
  declare uploading: boolean;
  declare acceptedFiles: number;

  declare actionIntent: WorkspaceActionIntent | null;
  declare actionPending: boolean;
  declare actionConfirmed: boolean;
  #actionReturnFocus: HTMLElement | null = null;
  /** The accepted Workspace Delete Run this panel is still tracking. */
  #deleteRunId: string | null = null;

  #workspace = '';
  #requestGeneration = 0;
  /** A list refresh asked for while a mutation held the request slot. */
  #reloadDeferred = false;
  readonly #session = new InspectorFilesSession();
  readonly #tracker = new CorpusRunTracker({
    onChange: () => { this.requestUpdate(); },
    onSettled: (run) => { void this.#mutationSettled(run); },
    onLost: (error) => { void this.#mutationLost(error); },
  });
  readonly #olderFiles = new KeysetPager<WebFilePanelSnapshot>(async (cursor, signal) => {
    const workspace = this.#workspace;
    const page = await getFilePanel(workspace, cursor, signal);
    // A page of another Workspace must never join this list.
    if (page.workspace !== workspace) throw new Error('older file page changed workspace identity');
    return page;
  }, () => { this.requestUpdate(); });
  #appendedFiles = 0;
  #restoreOlderFocus = false;
  #deleteTrigger: HTMLElement | null = null;

  constructor() {
    super();
    this.handles = productionHandles();
    this.active = false;
    this.snapshot = null;
    this.loading = true;
    this.error = null;
    this.uploading = false;
    this.acceptedFiles = 0;
    this.actionIntent = null;
    this.actionPending = false;
    this.actionConfirmed = false;
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
      void this.reload();
    }
  }

  /** How the latest older-files page load stands. */
  get filesLoadMoreState(): PageLoadState {
    return this.#olderFiles.state;
  }

  get hasActiveMutation(): boolean {
    return this.#session.mutating || this.#tracker.resuming;
  }

  /** The accepted Corpus Mutation this panel follows, kept after it settles. */
  get mutationRun(): TrackedCorpusRun | null {
    return this.#tracker.run;
  }

  get #deletingWorkspace(): boolean {
    return this.#deleteRunId !== null
      && this.mutationRun?.runId === this.#deleteRunId
      && this.#tracker.active;
  }

  async reload(showLoading = true): Promise<void> {
    const workspace = this.handles.ingest.workspace;
    this.#reloadDeferred = false;
    this.#invalidateOlderFiles();
    if (workspace !== this.#workspace) {
      // Hide the old Workspace before any new-Workspace I/O. A failed load must
      // never leave actionable rows from the previously selected Workspace.
      this.querySelector<HTMLDialogElement>('#workspace-action-dialog')?.close();
      this.snapshot = null;
      this.#olderFiles.reset(null);
      this.acceptedFiles = 0;
      this.#tracker.clear();
      this.#deleteRunId = null;
    }
    this.#workspace = workspace;
    this.uploading = false;
    const {controller, generation} = this.#startRequest();
    if (showLoading) this.loading = true;
    this.error = null;
    try {
      const snapshot = await getFilePanel(workspace, null, controller.signal);
      if (!this.#isCurrent(controller, workspace, generation)) return;
      if (snapshot.workspace !== workspace) {
        throw new Error('file panel response changed workspace identity');
      }
      this.snapshot = snapshot;
      this.#olderFiles.reset(snapshot.nextCursor);
      if (!this.#tracker.active) this.acceptedFiles = 0;
    } catch (error) {
      if (
        isAbortError(error)
        || !this.#isCurrent(controller, workspace, generation)
      ) return;
      // Keep the workspace transition fail closed even when a transport ignores
      // AbortSignal and resolves an invalidated request later.
      if (this.snapshot?.workspace !== workspace) {
        this.snapshot = null;
        this.#olderFiles.reset(null);
      }
      this.error = apiErrorMessage(
        error,
        msg('Failed to load files.', {id: 'inspectorFiles.loadFailed'}),
      );
    } finally {
      if (this.#session.finishRequest(controller)) {
        this.loading = false;
        if (this.active && this.isConnected) this.#tracker.wake();
      }
    }
  }

  loadOlderFiles(): Promise<void> {
    if (
      this.loading || this.#session.requestBusy || !this.active
      || this.snapshot?.workspace !== this.handles.ingest.workspace
    ) return Promise.resolve();
    return this.#olderFiles.loadNext((older) => {
      const current = this.snapshot;
      if (!current) return;
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
      this.#appendedFiles = appended.length;
      if (older.nextCursor === null && this.#restoreOlderFocus) {
        this.#restoreOlderFocus = false;
        void this.updateComplete.then(() => {
          this.querySelector<HTMLElement>('#file-list')?.focus({preventScroll: true});
        });
      }
    });
  }

  async upload(files: readonly File[], label?: string | null): Promise<void> {
    if (files.length === 0) return;
    if (this.#deletingWorkspace) {
      requestToast(this, {
        message: msg('This workspace is being deleted.', {id: 'inspectorFiles.uploadWhileDeleting'}),
      });
      return;
    }
    const workspace = this.handles.ingest.workspace;
    if (!this.handles.workspaces.changes(workspace).includes('ingest')) {
      requestToast(this, {message: authRefusalMessage(403)});
      return;
    }
    this.#invalidateOlderFiles();
    this.#workspace = workspace;
    const {controller, generation} = this.#startRequest();
    this.#beginMutation();
    // A followed Run settling now would reload the list and abort this request.
    this.#tracker.pause();
    let followed = false;
    this.uploading = true;
    this.error = null;
    const name = uploadLabel(files, label);
    requestToast(this, {message: msg(str`Uploading ${name}...`, {id: 'inspectorFiles.uploadingToast'})});
    try {
      const receipt = await uploadFileBatch(workspace, files, controller.signal);
      if (!this.#isCurrent(controller, workspace, generation)) return;
      this.acceptedFiles = receipt.fileCount ?? files.length;
      requestToast(this, {
        message: msg('Files received — Corpus update accepted', {id: 'inspectorFiles.filesReceived'}),
      });
      this.#tracker.follow(receipt);
      followed = true;
    } catch (error) {
      if (
        isAbortError(error)
        || !this.#isCurrent(controller, workspace, generation)
      ) return;
      const message = apiErrorMessage(
        error,
        msg('Upload failed.', {id: 'inspectorFiles.uploadFailed'}),
      );
      this.error = message;
      requestToast(this, {message});
    } finally {
      this.#finishMutation();
      if (this.#session.finishRequest(controller)) {
        this.uploading = false;
        this.loading = false;
      }
      if (!followed) this.#resumeFollowing();
      this.#reloadIfDeferred();
    }
  }

  pause(): void {
    this.querySelector<HTMLDialogElement>('#workspace-action-dialog')?.close();
    this.#actionClosed();
    this.#invalidateOlderFiles();
    this.#session.pause();
    this.#tracker.pause();
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
    if (await modalResult(this, dialog, () => this.#restoreDeleteTrigger(), this.lifetime) !== 'confirm') return;
    const workspace = this.handles.ingest.workspace;
    this.#invalidateOlderFiles();
    const {controller, generation} = this.#startRequest();
    this.#beginMutation();
    this.#tracker.pause();
    let followed = false;
    this.error = null;
    try {
      const receipt = await deleteFileRequest(workspace, filePath, controller.signal);
      if (!this.#isCurrent(controller, workspace, generation)) return;
      requestToast(this, {
        message: msg('File deletion accepted.', {id: 'inspectorFiles.fileDeleted'}),
      });
      this.#tracker.follow(receipt);
      followed = true;
    } catch (error) {
      if (
        isAbortError(error)
        || !this.#isCurrent(controller, workspace, generation)
      ) return;
      const message = apiErrorMessage(
        error,
        msg('Deletion failed.', {id: 'inspectorFiles.deletionFailed'}),
      );
      this.error = message;
      requestToast(this, {message});
    } finally {
      this.#finishMutation();
      if (this.#session.finishRequest(controller)) this.loading = false;
      if (!followed) this.#resumeFollowing();
      this.#reloadIfDeferred();
    }
  }

  async #mutationSettled(run: TrackedCorpusRun): Promise<void> {
    this.acceptedFiles = 0;
    if (run.runId === this.#deleteRunId) {
      this.#deleteRunId = null;
      await this.#settleWorkspaceDelete(run.workspace, run.status === 'succeeded');
      return;
    }
    requestToast(this, {
      message: run.status === 'succeeded'
        ? msg('Corpus update finished.', {id: 'inspectorFiles.corpusUpdateFinished'})
        : msg('Corpus update did not finish.', {id: 'inspectorFiles.corpusUpdateFailed'}),
    });
    await this.reload(false);
    const recovery = this.querySelector<DlFailedFileRecovery>('dl-failed-file-recovery');
    await recovery?.refresh(false);
  }

  /** The accepted Run's status is refused now; show the Corpus as it is. */
  async #mutationLost(error: unknown): Promise<void> {
    this.acceptedFiles = 0;
    this.#deleteRunId = null;
    requestToast(this, {
      message: error instanceof ApiError && error.errorType === 'auth'
        ? authRefusalMessage(error.status)
        : msg('Corpus update status is no longer available.', {
          id: 'inspectorFiles.corpusRunStatusUnavailable',
        }),
    });
    await this.reload(false);
    const recovery = this.querySelector<DlFailedFileRecovery>('dl-failed-file-recovery');
    await recovery?.refresh(false);
  }

  /** A mutation that followed no new Run hands polling back to the one this panel still follows. */
  #resumeFollowing(): void {
    if (!this.#session.mutating && this.active && this.isConnected) this.#tracker.wake();
  }

  /** Show news from elsewhere (a recovery Run settling) without cancelling
   *  the reader's own upload, deletion, or Workspace action: a reload takes the
   *  request slot and would abort it, so it waits until the mutation ends. */
  #reloadAfterMutations(): void {
    if (this.#session.mutating) {
      this.#reloadDeferred = true;
      return;
    }
    void this.reload(false);
  }

  #reloadIfDeferred(): void {
    if (!this.#reloadDeferred || this.#session.mutating) return;
    this.#reloadDeferred = false;
    if (this.active && this.isConnected) void this.reload(false);
  }

  async #resumeRepair(): Promise<void> {
    const outcome = await this.#tracker.resume();
    if (outcome === 'stale') return;
    const message = resumeRepairResult(outcome);
    if (outcome === 'failed') this.error = message;
    requestToast(this, {message});
  }

  #invalidateOlderFiles(): void {
    this.#olderFiles.cancel();
    this.#restoreOlderFocus = false;
  }

  #startRequest(): {controller: AbortController; generation: number} {
    this.#requestGeneration += 1;
    return {
      controller: this.#session.startRequest(),
      generation: this.#requestGeneration,
    };
  }

  #isCurrent(
    controller: AbortController,
    workspace: string,
    generation: number,
  ): boolean {
    return generation === this.#requestGeneration
      && this.#session.isCurrent(controller, workspace, this.handles.ingest.workspace);
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

  async #requestWorkspaceAction(
    kind: WorkspaceActionIntent['kind'],
    trigger: HTMLElement,
  ): Promise<void> {
    if (!this.active || this.loading || this.hasActiveMutation || this.#tracker.active) return;
    const workspace = this.handles.ingest.workspace;
    this.#actionReturnFocus = trigger;
    this.actionIntent = {kind, workspace};
    this.actionConfirmed = false;
    await this.updateComplete;
    const dialog = this.querySelector<HTMLDialogElement>('#workspace-action-dialog');
    if (!dialog || !this.active || !this.isConnected || workspace !== this.handles.ingest.workspace) return;
    const input = this.querySelector<HTMLInputElement>('#workspace-action-confirm-input');
    if (input) input.value = '';
    dialog.returnValue = '';
    showOwnedModal(this, dialog);
    window.requestAnimationFrame(() => {
      this.querySelector<HTMLInputElement>('#workspace-action-confirm-input')?.focus();
    });
  }

  #displayName(workspace: string): string {
    return this.handles.workspaces.records.find((record) => record.workspace === workspace)
      ?.displayName ?? workspace;
  }

  #workspaceActionDialog(): TemplateResult {
    const intent = this.actionIntent;
    const deleting = intent?.kind === 'delete';
    const displayName = this.#displayName(intent?.workspace ?? '');
    const title = deleting
      ? msg('Delete workspace', {id: 'inspectorFiles.deleteWorkspaceTitle'})
      : msg('Reset Corpus', {id: 'inspectorFiles.resetTitle'});
    return html`
      <dialog id="workspace-action-dialog" class="workspace-dialog"
              aria-labelledby="workspace-action-title" @cancel=${this.#actionCancelled}
              @close=${this.#actionClosed}>
        <form @submit=${this.#submitWorkspaceAction}>
          <h3 class="workspace-dialog-title" id="workspace-action-title">${title}</h3>
          ${deleting ? html`
            <p class="workspace-dialog-text">${msg('This will permanently delete the workspace and all of its Corpus data:', {id: 'inspectorFiles.deleteWorkspaceWarning'})} <strong>${displayName}</strong></p>
            <p class="workspace-dialog-text">${msg('Conversations and run history are kept.', {id: 'inspectorFiles.deleteWorkspaceKeeps'})}</p>
          ` : html`
            <p class="workspace-dialog-text">${msg('This will permanently remove all Corpus data while preserving workspace', {id: 'inspectorFiles.resetWarning'})} <strong>${displayName}</strong>.</p>
          `}
          <p class="workspace-dialog-text">${msg('Type the workspace name to confirm', {id: 'inspectorFiles.typeToConfirm'})}</p>
          <input type="text" id="workspace-action-confirm-input" class="dl-dialog-input"
                 autocomplete="off"
                 placeholder=${msg('Type workspace name...', {id: 'inspectorFiles.confirmPlaceholder'})}
                 aria-label=${msg(str`Type ${displayName} to confirm`, {id: 'inspectorFiles.typeNameToConfirmAria'})}
                 .readOnly=${this.actionPending}
                 @input=${this.#actionInput}>
          <div class="dl-dialog-actions">
            <button type="button" ?disabled=${this.actionPending}
                    @click=${() => this.querySelector<HTMLDialogElement>(
                      '#workspace-action-dialog',
                    )?.close()}>${msg('Cancel', {id: 'inspectorFiles.cancel'})}</button>
            <button type="submit" class="dl-dialog-danger"
                    ?disabled=${this.actionPending || !this.actionConfirmed}>
              ${deleting
                ? this.actionPending
                  ? msg('Accepting deletion…', {id: 'inspectorFiles.deletingWorkspace'})
                  : msg('Delete workspace', {id: 'inspectorFiles.deleteWorkspace'})
                : this.actionPending
                  ? msg('Accepting reset…', {id: 'inspectorFiles.resetting'})
                  : msg('Reset Corpus', {id: 'inspectorFiles.reset'})}
            </button>
          </div>
        </form>
      </dialog>
    `;
  }

  #actionInput = (event: Event): void => {
    const input = event.currentTarget as HTMLInputElement;
    const workspace = this.actionIntent?.workspace ?? '';
    const typed = input.value.trim();
    this.actionConfirmed = typed === this.#displayName(workspace) || typed === workspace;
  };

  #submitWorkspaceAction = async (event: SubmitEvent): Promise<void> => {
    event.preventDefault();
    const intent = this.actionIntent;
    if (!intent || this.actionPending || !this.actionConfirmed) return;
    const {kind, workspace} = intent;
    if (workspace !== this.handles.ingest.workspace || !this.active || this.hasActiveMutation
        || this.#tracker.active) return;
    this.#invalidateOlderFiles();
    const {controller, generation} = this.#startRequest();
    this.#beginMutation();
    this.actionPending = true;
    try {
      const receipt = kind === 'delete'
        ? await deleteWorkspaceRequest(workspace, controller.signal)
        : await resetWorkspaceRequest(workspace, controller.signal);
      if (!this.#isCurrent(controller, workspace, generation)) return;
      this.#deleteRunId = kind === 'delete' ? receipt.runId : null;
      this.#tracker.follow(receipt);
      this.querySelector<HTMLDialogElement>('#workspace-action-dialog')?.close();
      requestToast(this, {
        message: kind === 'delete'
          ? msg(str`Workspace deletion accepted for ${workspace}.`, {id: 'inspectorFiles.deleteWorkspaceAccepted'})
          : msg(str`Corpus reset accepted for ${workspace}.`, {id: 'inspectorFiles.resetAccepted'}),
      });
    } catch (error) {
      if (!isAbortError(error) && this.#isCurrent(controller, workspace, generation)) {
        requestToast(this, {
          message: apiErrorMessage(error, kind === 'delete'
            ? msg('Could not accept workspace deletion.', {id: 'inspectorFiles.deleteWorkspaceFailed'})
            : msg('Could not accept Corpus reset.', {id: 'inspectorFiles.resetFailed'})),
        });
      }
    } finally {
      this.#finishMutation();
      if (this.#session.finishRequest(controller)) {
        this.actionPending = false;
        await this.updateComplete;
        if (this.querySelector<HTMLDialogElement>('#workspace-action-dialog')?.open) {
          this.querySelector<HTMLInputElement>('#workspace-action-confirm-input')?.focus();
        }
      }
      this.#reloadIfDeferred();
    }
  };

  async #settleWorkspaceDelete(workspace: string, succeeded: boolean): Promise<void> {
    if (!succeeded) {
      requestToast(this, {
        message: msg('Workspace deletion did not finish.', {id: 'inspectorFiles.deleteWorkspaceDidNotFinish'}),
      });
      await this.reload(false);
      return;
    }
    const name = this.#displayName(workspace);
    // Retargeting Files reloads this panel for the Workspace that remains.
    this.handles.workspaces.remove(workspace);
    this.handles.ingest.resetToPrimary();
    requestToast(this, {
      message: msg(str`Workspace ${name} deleted.`, {id: 'inspectorFiles.workspaceDeleted'}),
    });
  }

  #actionCancelled = (event: Event): void => {
    if (this.actionPending) event.preventDefault();
  };

  #actionClosed = (): void => {
    publishModalState(this);
    this.actionIntent = null;
    this.actionPending = false;
    this.actionConfirmed = false;
    const returnFocus = this.#actionReturnFocus;
    this.#actionReturnFocus = null;
    const target = returnFocus?.isConnected && !returnFocus.inert
      && !returnFocus.closest('[hidden]')
      ? returnFocus
      : null;
    if (target?.isConnected && !target.inert) target.focus();
  };

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

  #progress(): TemplateResult | typeof nothing {
    const run = this.#tracker.run;
    if (run && this.#tracker.waitingForRepair) {
      return html`
        <div id="ingest-progress" role="status">
          <div class=${fileStyles['file-status']}>
            <span>${corpusRepairNotice(run)}</span>
            <button type="button" ?disabled=${this.hasActiveMutation}
                    aria-busy=${this.#tracker.resuming ? 'true' : 'false'}
                    @click=${() => { void this.#resumeRepair(); }}>
              ${resumeRepairLabel()}
            </button>
          </div>
        </div>
      `;
    }
    if (!this.#tracker.active) return nothing;
    return html`
      <div id="ingest-progress">
        <div class=${fileStyles['file-status']}>
          <div class=${fileStyles.spinner}></div>
          <span>${this.#deletingWorkspace
            ? msg('Deleting workspace…', {id: 'inspectorFiles.workspaceDeleting'})
            : msg('Corpus update in progress…', {id: 'inspectorFiles.corpusUpdateRunning'})}</span>
        </div>
      </div>
    `;
  }

  protected override render(): TemplateResult {
    const snapshot = this.snapshot;
    const files = snapshot?.files ?? [];
    const actionsBusy = this.loading || this.hasActiveMutation || this.#tracker.active;
    const workspace = this.handles.ingest.workspace;
    const changes = this.handles.workspaces.changes(workspace);
    const isDefault = workspace === this.handles.workspaces.deploymentDefault;
    const resettable = changes.includes('reset');
    const deletable = changes.includes('delete_workspace');
    return html`
      ${this.#progress()}
      ${this.error ? html`<div class="file-error" role="alert">${this.error}</div>` : nothing}
      ${changes.includes('ingest') ? html`
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
      ` : nothing}
      ${changes.includes('retry') ? html`
      <dl-failed-file-recovery
        .workspace=${workspace}
        .active=${this.active}
        @dl-failed-file-recovery-complete=${() => { this.#reloadAfterMutations(); }}
      ></dl-failed-file-recovery>
      ` : nothing}
      ${this.loading ? html`
        <div class=${fileStyles['file-status']}><div class=${fileStyles.spinner}></div><span>${msg('Loading files...', {id: 'inspectorFiles.loadingFiles'})}</span></div>
      ` : nothing}
      ${!this.loading ? html`
        <div id="file-list" role="list" aria-label=${msg('Processed files', {id: 'inspectorFiles.processedFilesAria'})} tabindex="-1">
          ${repeat(
            files,
            (file) => file.filePath,
            (file) => html`
              <div class=${fileStyles['file-item']} role="listitem" data-file-item>
                <span class=${fileStyles['file-name']} title=${file.filePath}>${file.fileName}</span>
                ${changes.includes('delete') ? html`
                <button class=${fileStyles['file-delete']} type="button" data-file-delete
                        aria-label=${msg(str`Delete ${file.fileName}`, {id: 'inspectorFiles.deleteFileAria'})}
                        @click=${(event: Event) => {
                          this.#deleteTrigger = event.currentTarget as HTMLElement;
                          void this.#deleteFile(file.filePath);
                        }}>
                  ${icon('close', {size: 'sm', className: fileStyles['file-delete-icon']})}
                </button>
                ` : nothing}
              </div>
            `,
          )}
        </div>
        ${loadOlderControl({
          list: 'files',
          pages: this.#olderFiles,
          label: msg('Load older files', {id: 'inspectorFiles.loadOlder'}),
          retryLabel: msg('Retry loading older files', {id: 'inspectorFiles.retryLoadOlder'}),
          loading: msg('Loading older files…', {id: 'inspectorFiles.loadingOlder'}),
          loaded: this.#appendedFiles === 1
            ? msg('Loaded 1 older file.', {id: 'inspectorFiles.loadedOneOlder'})
            : msg(str`Loaded ${this.#appendedFiles} older files.`, {id: 'inspectorFiles.loadedOlder'}),
          failed: msg('Older files could not be loaded.', {id: 'inspectorFiles.olderFilesFailed'}),
          onLoad: this.#loadOlderFiles,
          rowClass: fileStyles['file-page-control'],
        })}
      ` : nothing}
      ${!this.loading && !this.error && files.length === 0 && !this.#tracker.active ? html`
        <div class="empty-state">${msg(str`No files ingested in workspace “${this.#workspace}”.`, {id: 'inspectorFiles.emptyState'})}</div>
      ` : nothing}
      ${this.acceptedFiles > 0 && this.#tracker.active ? html`
        <div class=${`${fileStyles['ingest-queue-notice']} ${fileStyles['ingest-queue-notice--inline']}`}>
          ${msg(str`${this.acceptedFiles} new file(s) accepted for ingest`, {id: 'inspectorFiles.acceptedForIngest'})}
        </div>
      ` : nothing}
      ${resettable || deletable ? html`
      <details class="workspace-actions">
        <summary>${msg('Workspace actions', {id: 'inspectorFiles.workspaceActions'})}</summary>
        <div class="workspace-actions-body">
          ${resettable ? html`
          <button type="button" class="dl-btn dl-btn-danger-text" data-reset-workspace
                  ?disabled=${actionsBusy}
                  @click=${(event: Event) => { void this.#requestWorkspaceAction(
                    'reset', event.currentTarget as HTMLElement,
                  ); }}>${msg('Reset Corpus…', {id: 'inspectorFiles.resetAction'})}</button>
          ` : nothing}
          ${isDefault && resettable ? html`
            <p class="workspace-actions-note">${msg('The default workspace can be reset but not deleted.', {id: 'inspectorFiles.defaultWorkspaceKept'})}</p>
          ` : nothing}
          ${deletable ? html`
            <button type="button" class="dl-btn dl-btn-danger-text" data-delete-workspace
                    ?disabled=${actionsBusy}
                    @click=${(event: Event) => { void this.#requestWorkspaceAction(
                      'delete', event.currentTarget as HTMLElement,
                    ); }}>${msg('Delete workspace…', {id: 'inspectorFiles.deleteWorkspaceAction'})}</button>
          ` : nothing}
        </div>
      </details>
      ` : nothing}
      ${this.#workspaceActionDialog()}
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
