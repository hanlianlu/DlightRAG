// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, nothing, svg, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {WorkspaceApiError, deleteWorkspaceRequest} from '../api/workspaces.ts';
import type {WorkspaceRecord} from '../events/bus.ts';
import {LightElement, StoreController} from '../lib/lit_host.ts';
import {rovingArrowKeydown} from '../lib/listbox.ts';
import {createAutoDismiss} from '../lib/popover.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import workspaceStyles from '../styles/workspaces.module.css';
import {publishModalState, showOwnedModal} from './modal.ts';
import type {ToastRequestDetail} from './toast.ts';
import './workspace_create.ts';

const CARET = svg`<path d="M2.5 4 L5 6.5 L7.5 4"/>`;

/** Search-scope selection, workspace deletion, popover, and Dialog lifecycle. */
export class DlWorkspaceScope extends LightElement {
  static properties = {
    open: {state: true},
    deleteWorkspace: {state: true},
    deletePending: {state: true},
    deleteConfirmed: {state: true},
  };

  declare open: boolean;
  declare deleteWorkspace: string | null;
  declare deletePending: boolean;
  declare deleteConfirmed: boolean;

  #deleteOperation: AbortController | null = null;
  #deleteReturnFocus: HTMLElement | null = null;
  #restoreLoadMoreFocus = false;
  #settledFocusRestore = false;
  #loadMoreAnnouncement = '';
  #lastLoadMoreState: 'idle' | 'loading' | 'error' = 'idle';
  readonly #dismiss = createAutoDismiss({
    getAnchor: () => this,
    isOpen: () => this.open,
    onDismiss: (reason) => { this.#dismissPopover(reason === 'escape'); },
  });

  constructor() {
    super();
    this.open = false;
    this.deleteWorkspace = null;
    this.deletePending = false;
    this.deleteConfirmed = false;
    /** Store reads: records, active, primary. */
    new StoreController(this, workspaceStore);
  }

  override disconnectedCallback(): void {
    const dialog = this.querySelector<HTMLDialogElement>('#delete-workspace-dialog');
    if (dialog?.open) dialog.close();
    this.#deleteOperation?.abort();
    this.#deleteOperation = null;
    this.#deleteReturnFocus = null;
    this.open = false;
    this.deleteWorkspace = null;
    this.deletePending = false;
    this.deleteConfirmed = false;
    this.#dismiss.deactivate();
    publishModalState(this);
    super.disconnectedCallback();
  }

  close(): void {
    this.open = false;
  }

  protected override willUpdate(_changed: PropertyValues<this>): void {
    const state = workspaceStore.workspaceLoadMoreState;
    const previous = this.#lastLoadMoreState;
    if (state === previous) return;
    this.#lastLoadMoreState = state;
    if (state === 'loading') this.#loadMoreAnnouncement = 'Loading workspaces…';
    else if (state === 'error') this.#loadMoreAnnouncement = 'Workspaces could not be loaded.';
    else if (previous === 'loading') {
      this.#loadMoreAnnouncement = 'Loaded more workspaces.';
      this.#settledFocusRestore = true;
    }
  }

  protected override updated(): void {
    this.classList.toggle('open', this.open);
    if (this.open) this.#dismiss.activate();
    else this.#dismiss.deactivate();
    if (this.#settledFocusRestore) {
      this.#settledFocusRestore = false;
      if (this.#restoreLoadMoreFocus) {
        this.#restoreLoadMoreFocus = false;
        const control = this.querySelector<HTMLButtonElement>('[data-load-more-workspaces]');
        if (control) control.focus({preventScroll: true});
      }
    }
  }

  protected override render(): TemplateResult {
    const multi = workspaceStore.active.length > 1 || this.#allSelected;
    return html`
      <button class="workspace-selector-trigger" id="workspace-trigger" type="button"
              aria-label="Choose search workspaces" aria-haspopup="dialog"
              aria-expanded=${this.open ? 'true' : 'false'} aria-controls="workspace-popover"
              @click=${this.#togglePopover}>
        <span class="workspace-dot${multi ? ' multi' : ''}" id="workspace-dot"></span>
        <span class="workspace-label" id="workspace-label">${this.#label}</span>
        <svg class="workspace-caret" width="10" height="10" viewBox="0 0 10 10"
             fill="none" stroke="currentColor" stroke-width="1.5"
             stroke-linecap="round">${CARET}</svg>
      </button>
      ${this.#popover()}
      ${this.#deleteDialog()}
    `;
  }

  #trigger(): HTMLButtonElement | null {
    return this.querySelector<HTMLButtonElement>('#workspace-trigger');
  }

  #togglePopover = (): void => {
    if (this.open) {
      this.open = false;
      return;
    }
    this.open = true;
    void this.updateComplete.then(() => {
      const selected = this.querySelector<HTMLButtonElement>(
        '[data-workspace-choice][aria-pressed="true"]',
      );
      (selected ?? this.querySelector<HTMLButtonElement>('[data-workspace-choice]'))?.focus();
    });
  };

  #dismissPopover(restoreFocus: boolean): void {
    this.open = false;
    if (restoreFocus) {
      void this.updateComplete.then(() => { this.#trigger()?.focus(); });
    }
  }

  get #allSelected(): boolean {
    const known = workspaceStore.knownWorkspaces;
    const active = workspaceStore.active;
    return known.length > 0 && known.every((workspace) => active.includes(workspace));
  }

  get #label(): string {
    const total = workspaceStore.knownWorkspaces.length;
    const active = workspaceStore.active;
    if (active.length === 0 || this.#allSelected) {
      return total > 0 ? `All workspaces (${total})` : 'All workspaces';
    }
    const anchor = active.includes(workspaceStore.primary) ? workspaceStore.primary : active[0];
    const name = workspaceStore.records.find((record) => record.workspace === anchor)?.displayName
      ?? anchor;
    return active.length === 1 ? name : `${name} + ${active.length - 1}`;
  }

  #check(selected: boolean): TemplateResult {
    return html`<div class="${workspaceStyles.workspacePopoverCheck}${selected
      ? ` ${workspaceStyles.on}` : ''}"></div>`;
  }

  #popover(): TemplateResult {
    const sorted = [...workspaceStore.records]
      .sort((left, right) => left.displayName.localeCompare(right.displayName));
    return html`
      <div class="ui-popover ui-popover--workspace" id="workspace-popover"
           role="dialog" aria-label="Workspaces" ?hidden=${!this.open}
           @keydown=${(event: KeyboardEvent) => {
             rovingArrowKeydown(event, '[data-workspace-choice]');
           }}
           @dl-workspace-created=${this.#workspaceCreated}>
        ${this.#allOption()}
        ${repeat(sorted, (record) => record.workspace, (record) => this.#option(record))}
        ${this.#loadMoreControl()}
        <span class="sr-only" data-workspaces-status role="status" aria-live="polite">
          ${this.#loadMoreAnnouncement}
        </span>
        <dl-workspace-create></dl-workspace-create>
      </div>
    `;
  }

  #loadMoreControl(): TemplateResult | typeof nothing {
    if (!workspaceStore.hasMoreWorkspaces) return nothing;
    const state = workspaceStore.workspaceLoadMoreState;
    return html`
      <div class="workspace-load-more">
        <button type="button" data-load-more-workspaces class="ui-popover-item"
                aria-busy=${state === 'loading' ? 'true' : 'false'}
                ?disabled=${state === 'loading'} @click=${this.#loadMore}>
          ${state === 'error' ? 'Retry loading workspaces' : 'Load more workspaces'}
        </button>
      </div>
    `;
  }

  #loadMore = (event: Event): void => {
    const button = event.currentTarget as HTMLButtonElement;
    this.#restoreLoadMoreFocus = document.activeElement === button;
    void workspaceStore.loadMoreWorkspaces();
  };

  #allOption(): TemplateResult {
    const selected = this.#allSelected;
    const selectAll = (event: Event): void => {
      event.stopPropagation();
      workspaceStore.selectAll();
    };
    return html`
      <button class="ui-popover-item ${workspaceStyles.workspacePopoverAll}" type="button"
              data-workspace-choice data-workspace-all="true"
              aria-pressed=${selected ? 'true' : 'false'} @click=${selectAll}>
        ${this.#check(selected)}All workspaces
      </button>
    `;
  }

  #option(record: WorkspaceRecord): TemplateResult {
    const selected = workspaceStore.active.includes(record.workspace);
    const toggle = (event: Event): void => {
      event.stopPropagation();
      workspaceStore.toggle(record.workspace);
    };
    return html`
      <div class=${workspaceStyles.workspacePopoverItem}>
        <button class="ui-popover-item ${workspaceStyles.workspacePopoverOption}" type="button"
                data-workspace-choice aria-pressed=${selected ? 'true' : 'false'} @click=${toggle}>
          ${this.#check(selected)}
          <span class=${workspaceStyles.workspacePopoverName}>${record.displayName}</span>
        </button>
        <button type="button" class=${workspaceStyles.workspacePopoverDelete}
                title="Delete workspace" aria-label=${`Delete workspace ${record.displayName}`}
                @click=${(event: MouseEvent) => {
                  event.stopPropagation();
                  void this.#requestDelete(record.workspace, event.currentTarget as HTMLElement);
                }}>✕</button>
      </div>
    `;
  }

  #workspaceCreated = (): void => {
    const active = document.activeElement;
    const restoreFocus = active === document.body || this.contains(active);
    this.open = false;
    if (restoreFocus) {
      void this.updateComplete.then(() => { this.#trigger()?.focus(); });
    }
  };

  async #requestDelete(workspace: string, trigger: HTMLElement): Promise<void> {
    this.#deleteReturnFocus = trigger;
    this.open = false;
    this.deleteWorkspace = workspace;
    this.deleteConfirmed = false;
    await this.updateComplete;
    const dialog = this.querySelector<HTMLDialogElement>('#delete-workspace-dialog');
    if (!dialog) return;
    dialog.returnValue = '';
    showOwnedModal(this, dialog);
    window.requestAnimationFrame(() => {
      this.querySelector<HTMLInputElement>('#delete-workspace-confirm-input')?.focus();
    });
  }

  #deleteDialog(): TemplateResult {
    const workspace = this.deleteWorkspace ?? '';
    const displayName = workspaceStore.records.find((record) => record.workspace === workspace)
      ?.displayName ?? workspace;
    return html`
      <dialog id="delete-workspace-dialog" class="workspace-dialog"
              aria-labelledby="delete-workspace-title" @cancel=${this.#deleteCancelled}
              @close=${this.#deleteClosed}>
        <form @submit=${this.#submitDelete}>
          <h3 class="workspace-dialog-title" id="delete-workspace-title">Delete workspace</h3>
          <p class="workspace-dialog-text">This will permanently delete all data for</p>
          <p class="workspace-dialog-name">${displayName}</p>
          <p class="workspace-dialog-text">Type the workspace name to confirm</p>
          <input type="text" id="delete-workspace-confirm-input" class="workspace-dialog-input"
                 autocomplete="off" placeholder="Type workspace name..."
                 aria-label=${`Type ${displayName} to confirm`} .readOnly=${this.deletePending}
                 @input=${this.#deleteInput}>
          <div class="ui-dialog-actions">
            <button type="button" ?disabled=${this.deletePending}
                    @click=${() => this.querySelector<HTMLDialogElement>(
                      '#delete-workspace-dialog',
                    )?.close()}>Cancel</button>
            <button type="submit" class="ui-dialog-danger"
                    ?disabled=${this.deletePending || !this.deleteConfirmed}>
              ${this.deletePending ? 'Deleting…' : 'Delete'}
            </button>
          </div>
        </form>
      </dialog>
    `;
  }

  #deleteInput = (event: Event): void => {
    const input = event.currentTarget as HTMLInputElement;
    const workspace = this.deleteWorkspace ?? '';
    const displayName = workspaceStore.records.find((record) => record.workspace === workspace)
      ?.displayName ?? workspace;
    this.deleteConfirmed = input.value.trim() === displayName || input.value.trim() === workspace;
  };

  #submitDelete = async (event: SubmitEvent): Promise<void> => {
    event.preventDefault();
    const workspace = this.deleteWorkspace;
    if (!workspace || this.deletePending || !this.deleteConfirmed) return;
    const operation = new AbortController();
    this.#deleteOperation = operation;
    this.deletePending = true;
    try {
      const deleted = await deleteWorkspaceRequest(workspace, operation.signal);
      if (
        operation.signal.aborted || this.#deleteOperation !== operation
        || this.deleteWorkspace !== workspace
      ) return;
      workspaceStore.remove(deleted.workspace, deleted.next_workspace);
      this.querySelector<HTMLDialogElement>('#delete-workspace-dialog')?.close();
      this.#requestToast({message: `Workspace ${workspace} deleted.`});
    } catch (error) {
      if (!operation.signal.aborted && this.#deleteOperation === operation) {
        this.#requestToast({
          message: error instanceof WorkspaceApiError
            ? error.message
            : 'Could not delete workspace.',
          duration: 3000,
        });
      }
    } finally {
      if (this.#deleteOperation === operation) {
        this.#deleteOperation = null;
        this.deletePending = false;
        await this.updateComplete;
        if (this.querySelector<HTMLDialogElement>('#delete-workspace-dialog')?.open) {
          this.querySelector<HTMLInputElement>('#delete-workspace-confirm-input')?.focus();
        }
      }
    }
  };

  #deleteCancelled = (event: Event): void => {
    if (this.deletePending) event.preventDefault();
  };

  #deleteClosed = (): void => {
    publishModalState(this);
    this.#deleteOperation?.abort();
    this.#deleteOperation = null;
    this.deleteWorkspace = null;
    this.deletePending = false;
    this.deleteConfirmed = false;
    const returnFocus = this.#deleteReturnFocus;
    this.#deleteReturnFocus = null;
    const target = returnFocus?.isConnected && !returnFocus.inert
      && !returnFocus.closest('[hidden]')
      ? returnFocus
      : this.#trigger();
    if (target?.isConnected && !target.inert) target.focus();
  };

  #requestToast(detail: ToastRequestDetail): void {
    this.dispatchEvent(new CustomEvent<ToastRequestDetail>('dl-toast-request', {
      detail,
      bubbles: true,
      composed: true,
    }));
  }
}

customElements.define('dl-workspace-scope', DlWorkspaceScope);

declare global {
  interface HTMLElementTagNameMap {
    'dl-workspace-scope': DlWorkspaceScope;
  }
}
