// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges } from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {icon} from '../design-system/index.ts';
import {rovingArrowKeydown} from '../lib/listbox.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {createAutoDismiss} from '../lib/popover.ts';
import {type AppHandles, productionHandles } from '../stores/app-handles.ts';
import type {WorkspaceRecord} from '../stores/workspace-store.ts';
import workspaceStyles from '../styles/workspaces.module.css';
import './workspace-create.ts';

/** Search-scope selection and popover lifecycle. */
export class DlWorkspaceScope extends LightElement {
  static properties = {
    handles: {attribute: false},
    open: {state: true},
  };

  declare handles: AppHandles;
  declare open: boolean;

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
    updateWhenLocaleChanges(this);
    this.handles = productionHandles();
    this.open = false;
    /** Store reads: records, active, primary. */
    new StoreController(this, this.handles.workspaces);
  }

  override disconnectedCallback(): void {
    this.open = false;
    this.#dismiss.deactivate();
    super.disconnectedCallback();
  }

  close(): void {
    this.open = false;
  }

  protected override willUpdate(_changed: PropertyValues<this>): void {
    const state = this.handles.workspaces.workspaceLoadMoreState;
    const previous = this.#lastLoadMoreState;
    if (state === previous) return;
    this.#lastLoadMoreState = state;
    if (state === 'loading') {
      this.#loadMoreAnnouncement = msg('Loading workspaces…', {id: 'workspaceScope.loadingMore'});
    } else if (state === 'error') {
      this.#loadMoreAnnouncement = msg('Workspaces could not be loaded.', {id: 'workspaceScope.loadMoreFailed'});
    } else if (previous === 'loading') {
      this.#loadMoreAnnouncement = msg('Loaded more workspaces.', {id: 'workspaceScope.loadedMore'});
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
    const multi = this.handles.workspaces.active.length > 1 || this.#allSelected;
    return html`
      <button class="workspace-selector-trigger" id="workspace-trigger" type="button"
              aria-label=${msg('Choose search workspaces', {id: 'workspaceScope.chooseSearchWorkspaces'})}
              aria-haspopup="dialog"
              aria-expanded=${this.open ? 'true' : 'false'} aria-controls="workspace-popover"
              @click=${this.#togglePopover}>
        <span class="workspace-dot${multi ? ' multi' : ''}" id="workspace-dot">${multi
          ? html`<span class="workspace-pip"></span><span class="workspace-pip"></span
            ><span class="workspace-pip"></span>` : nothing}</span>
        <span class="workspace-label" id="workspace-label">${this.#label}</span>
        ${icon('chevron-down', {size: 'xs', className: 'workspace-caret'})}
      </button>
      ${this.#popover()}
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
    const known = this.handles.workspaces.knownWorkspaces;
    const active = this.handles.workspaces.active;
    return known.length > 0 && known.every((workspace) => active.includes(workspace));
  }

  get #label(): string {
    const total = this.handles.workspaces.knownWorkspaces.length;
    const active = this.handles.workspaces.active;
    if (active.length === 0 || this.#allSelected) {
      return total > 0
        ? msg(str`All workspaces (${total})`, {id: 'workspaceScope.allWithCount'})
        : msg('All workspaces', {id: 'workspaceScope.all'});
    }
    const anchor = active.includes(this.handles.workspaces.primary) ? this.handles.workspaces.primary : active[0];
    const name = this.handles.workspaces.records.find((record) => record.workspace === anchor)?.displayName
      ?? anchor;
    return active.length === 1 ? name : `${name} + ${active.length - 1}`;
  }

  #check(selected: boolean): TemplateResult {
    return html`<div class="${workspaceStyles.workspacePopoverCheck}${selected
      ? ` ${workspaceStyles.on}` : ''}"></div>`;
  }

  #popover(): TemplateResult {
    const sorted = [...this.handles.workspaces.records]
      .sort((left, right) => left.displayName.localeCompare(right.displayName));
    return html`
      <div class="dl-popover dl-popover--workspace" id="workspace-popover"
           role="dialog" aria-label=${msg('Workspaces', {id: 'workspaceScope.workspacesAria'})}
           ?hidden=${!this.open}
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
        <dl-workspace-create .handles=${this.handles}></dl-workspace-create>
      </div>
    `;
  }

  #loadMoreControl(): TemplateResult | typeof nothing {
    if (!this.handles.workspaces.hasMoreWorkspaces) return nothing;
    const state = this.handles.workspaces.workspaceLoadMoreState;
    return html`
      <div class="workspace-load-more">
        <button type="button" data-load-more-workspaces class="dl-popover-item"
                aria-busy=${state === 'loading' ? 'true' : 'false'}
                ?disabled=${state === 'loading'} @click=${this.#loadMore}>
          ${state === 'error'
            ? msg('Retry loading workspaces', {id: 'workspaceScope.retryLoadMore'})
            : msg('Load more workspaces', {id: 'workspaceScope.loadMore'})}
        </button>
      </div>
    `;
  }

  #loadMore = (event: Event): void => {
    const button = event.currentTarget as HTMLButtonElement;
    this.#restoreLoadMoreFocus = document.activeElement === button;
    void this.handles.workspaces.loadMoreWorkspaces();
  };

  #allOption(): TemplateResult {
    const selected = this.#allSelected;
    const selectAll = (event: Event): void => {
      event.stopPropagation();
      this.handles.workspaces.selectAll();
    };
    return html`
      <button class="dl-popover-item ${workspaceStyles.workspacePopoverAll}" type="button"
              data-workspace-choice data-workspace-all="true"
              aria-pressed=${selected ? 'true' : 'false'} @click=${selectAll}>
        ${this.#check(selected)}${msg('All workspaces', {id: 'workspaceScope.all'})}
      </button>
    `;
  }

  #option(record: WorkspaceRecord): TemplateResult {
    const selected = this.handles.workspaces.active.includes(record.workspace);
    const toggle = (event: Event): void => {
      event.stopPropagation();
      this.handles.workspaces.toggle(record.workspace);
    };
    return html`
      <div class=${workspaceStyles.workspacePopoverItem}>
        <button class="dl-popover-item ${workspaceStyles.workspacePopoverOption}" type="button"
                data-workspace-choice aria-pressed=${selected ? 'true' : 'false'} @click=${toggle}>
          ${this.#check(selected)}
          <span class=${workspaceStyles.workspacePopoverName}>${record.displayName}</span>
        </button>
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


}

customElements.define('dl-workspace-scope', DlWorkspaceScope);

declare global {
  interface HTMLElementTagNameMap {
    'dl-workspace-scope': DlWorkspaceScope;
  }
}
