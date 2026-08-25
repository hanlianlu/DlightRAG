// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import type {AnswerPresentation} from '../api/conversations.ts';
import {COMPACT_SHELL_MEDIA} from '../lib/breakpoints.ts';
import {wrapTabFocus} from '../lib/dom.ts';
import {LightElement} from '../lib/lit_host.ts';
import {ingestStore} from '../stores/ingestStore.ts';
import type {DlInspectorFiles} from './inspector_files.ts';
import './inspector_files.ts';
import type {
  DlInspectorSources,
  InspectorSourcesStateDetail,
} from './inspector_sources.ts';
import './inspector_sources.ts';
import './ingest_target.ts';

export type InspectorKind = 'files' | 'sources';

export interface InspectorStateDetail {
  open: boolean;
  kind: InspectorKind | null;
  compact: boolean;
}

export interface InspectorOpeningDetail {
  kind: InspectorKind;
}

/** Sources/Files pane state, accessibility, focus, and content composition. */
export class DlInspector extends LightElement {
  static properties = {
    kind: {state: true},
    presentation: {state: true},
    sourceHasItems: {state: true},
    sourcesExpanded: {state: true},
    shellInert: {state: true},
  };

  declare kind: InspectorKind | null;
  declare presentation: AnswerPresentation | null;
  declare sourceHasItems: boolean;
  declare sourcesExpanded: boolean;
  declare shellInert: boolean;

  #returnFocus: HTMLElement | null = null;
  #events: AbortController | null = null;
  #compactMedia: MediaQueryList | null = null;
  #stateSignature = '';
  #focusGeneration = 0;

  constructor() {
    super();
    this.kind = null;
    this.presentation = null;
    this.sourceHasItems = false;
    this.sourcesExpanded = false;
    this.shellInert = false;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    const events = new AbortController();
    this.#events = events;
    document.addEventListener('keydown', this.#documentKeydown, {signal: events.signal});
    this.#compactMedia = window.matchMedia(COMPACT_SHELL_MEDIA);
    this.#compactMedia.addEventListener('change', this.#compactChanged, {signal: events.signal});
    this.#syncHostState();
  }

  override disconnectedCallback(): void {
    this.#events?.abort();
    this.#events = null;
    this.#compactMedia = null;
    this.#focusGeneration += 1;
    this.#files()?.pause();
    this.#stateSignature = '';
    super.disconnectedCallback();
  }

  protected override updated(changed: PropertyValues<this>): void {
    this.#syncHostState();
    if (changed.has('kind') && this.kind === 'sources') {
      this.#sources()?.setSelection();
    }
    this.#publishState();
  }

  get open(): boolean {
    return this.kind !== null;
  }

  get hasActiveFileMutation(): boolean {
    return this.kind === 'files' && Boolean(this.#files()?.hasActiveMutation);
  }

  /** Open Sources at an optional cited source/chunk. */
  async openSources(
    presentation: AnswerPresentation,
    referenceId?: string,
    chunkId?: string,
    returnFocus?: HTMLElement | null,
  ): Promise<void> {
    if (!this.#beginOpen('sources', returnFocus)) return;
    this.presentation = presentation;
    this.sourceHasItems = presentation.sources.length > 0;
    this.sourcesExpanded = false;
    await this.updateComplete;
    this.#sources()?.setSelection(referenceId, chunkId);
    await this.#focusOnCompact();
  }

  /** Open workspace Files and refresh its content. */
  async openFiles(returnFocus?: HTMLElement | null): Promise<void> {
    if (!this.#beginOpen('files', returnFocus)) return;
    ingestStore.resetToPrimary();
    await this.updateComplete;
    await this.#focusOnCompact();
  }

  /** Open Files if needed, then upload through the Files content owner. */
  async uploadFiles(
    files: readonly File[],
    label?: string | null,
    returnFocus?: HTMLElement | null,
  ): Promise<void> {
    if (files.length === 0) return;
    if (this.kind !== 'files') await this.openFiles(returnFocus);
    if (this.kind !== 'files') return;
    const content = this.#files();
    if (!content) return;
    await content.updateComplete;
    await content.upload(files, label);
  }

  /** Close the Inspector and optionally restore the element that opened it. */
  close(restoreFocus = true): void {
    if (!this.open) return;
    const focusGeneration = ++this.#focusGeneration;
    this.#files()?.pause();
    this.kind = null;
    this.presentation = null;
    this.sourceHasItems = false;
    this.sourcesExpanded = false;
    this.#syncHostState();
    this.#publishState();
    const panel = this.querySelector<HTMLElement>('#panel');
    panel?.classList.remove('open');
    if (panel) {
      panel.inert = true;
      panel.setAttribute('aria-hidden', 'true');
      delete panel.dataset.panelKind;
    }
    const returnFocus = this.#returnFocus;
    this.#returnFocus = null;
    if (restoreFocus) {
      window.requestAnimationFrame(() => {
        if (focusGeneration !== this.#focusGeneration || this.open) return;
        if (returnFocus?.isConnected && !returnFocus.inert) returnFocus.focus();
      });
    }
  }

  /** Close conversation-scoped Sources while preserving workspace Files. */
  closeConversationContent(): void {
    if (this.kind !== 'sources') return;
    this.close(false);
    this.#returnFocus = null;
  }

  protected override render(): TemplateResult {
    const open = this.open;
    const compact = this.#isCompact();
    const files = this.kind === 'files';
    const sources = this.kind === 'sources';
    return html`
      <aside
        class="panel inspector-surface${open ? ' open' : ''}"
        id="panel"
        aria-label=${files ? 'Files' : 'Sources'}
        aria-hidden=${open ? nothing : 'true'}
        role=${open && compact ? 'dialog' : nothing}
        aria-modal=${open && compact ? 'true' : nothing}
        data-panel-kind=${this.kind ?? nothing}
        ?inert=${!open || this.shellInert}
        @keydown=${this.#panelKeydown}
      >
        <div class="panel-header">
          <span id="panel-title">${sources ? 'Sources' : ''}</span>
          <button
            class="source-toggle-all"
            id="source-toggle-all-btn"
            type="button"
            aria-pressed=${this.sourcesExpanded ? 'true' : 'false'}
            ?hidden=${!sources || !this.sourceHasItems}
            @click=${this.#toggleAllSources}
          >${this.sourcesExpanded ? 'Collapse all' : 'Show all'}</button>
          <dl-ingest-target
            class="ingest-target"
            id="ingest-target"
            .active=${files}
          ></dl-ingest-target>
          <button class="panel-close" id="panel-close-btn" type="button"
                  aria-label="Close panel" @click=${() => this.close()}>✕</button>
        </div>
        <div id="panel-content" class="panel-content">
          <dl-inspector-sources
            .sources=${this.presentation?.sources ?? []}
            ?hidden=${!sources}
            @dl-inspector-sources-state-change=${this.#sourcesStateChanged}
          ></dl-inspector-sources>
          <dl-inspector-files .active=${files} ?hidden=${!files}></dl-inspector-files>
        </div>
      </aside>
      <button
        class="inspector-backdrop"
        type="button"
        aria-label="Close panel"
        ?hidden=${!open || !compact}
        @click=${() => this.close()}
      ></button>
    `;
  }

  #beginOpen(kind: InspectorKind, returnFocus?: HTMLElement | null): boolean {
    const event = new CustomEvent<InspectorOpeningDetail>('dl-inspector-opening', {
      bubbles: true,
      composed: true,
      cancelable: true,
      detail: {kind},
    });
    if (!this.dispatchEvent(event)) return false;
    this.#focusGeneration += 1;
    if (returnFocus) this.#returnFocus = returnFocus;
    else if (!this.open && document.activeElement instanceof HTMLElement) {
      this.#returnFocus = document.activeElement;
    }
    this.kind = kind;
    this.#syncHostState();
    return true;
  }

  #sources(): DlInspectorSources | null {
    return this.querySelector<DlInspectorSources>('dl-inspector-sources');
  }

  #files(): DlInspectorFiles | null {
    return this.querySelector<DlInspectorFiles>('dl-inspector-files');
  }

  #toggleAllSources = (): void => {
    const sources = this.#sources();
    if (!sources) return;
    if (sources.fullyExpanded) sources.collapseAll();
    else sources.expandAll();
  };

  #sourcesStateChanged = (event: CustomEvent<InspectorSourcesStateDetail>): void => {
    this.sourceHasItems = event.detail.hasSources;
    this.sourcesExpanded = event.detail.fullyExpanded;
  };

  #syncHostState(): void {
    const open = this.open;
    this.classList.toggle('open', open);
    this.inert = !open || this.shellInert;
    if (open) this.removeAttribute('aria-hidden');
    else this.setAttribute('aria-hidden', 'true');
  }

  #publishState(): void {
    const detail: InspectorStateDetail = {
      open: this.open,
      kind: this.kind,
      compact: this.#isCompact(),
    };
    const signature = `${detail.open}:${detail.kind ?? ''}:${detail.compact}`;
    if (signature === this.#stateSignature) return;
    this.#stateSignature = signature;
    this.dispatchEvent(new CustomEvent<InspectorStateDetail>('dl-inspector-state-change', {
      bubbles: true,
      composed: true,
      detail,
    }));
  }

  async #focusOnCompact(): Promise<void> {
    if (!this.#isCompact()) return;
    await this.updateComplete;
    this.querySelector<HTMLButtonElement>('#panel-close-btn')?.focus();
  }

  #focusableElements(): HTMLElement[] {
    const panel = this.querySelector<HTMLElement>('#panel');
    if (!panel) return [];
    return Array.from(panel.querySelectorAll<HTMLElement>(
      'button:not([disabled]), a[href], input:not([disabled]), [tabindex]:not([tabindex="-1"])',
    )).filter((element) => !element.hidden && element.getClientRects().length > 0);
  }

  #isCompact(): boolean {
    return this.#compactMedia?.matches ?? window.matchMedia(COMPACT_SHELL_MEDIA).matches;
  }

  #compactChanged = (): void => {
    this.requestUpdate();
    this.#publishState();
    if (this.open && this.#isCompact() && !this.contains(document.activeElement)) {
      void this.#focusOnCompact();
    }
  };

  #documentKeydown = (event: KeyboardEvent): void => {
    if (!this.open || event.key !== 'Escape' || event.defaultPrevented) return;
    if (document.querySelector('dialog[open]')) return;
    event.preventDefault();
    this.close();
  };

  #panelKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'Tab' && this.#isCompact()) {
      wrapTabFocus(this.#focusableElements(), event);
    }
  };
}

customElements.define('dl-inspector', DlInspector);

declare global {
  interface HTMLElementTagNameMap {
    'dl-inspector': DlInspector;
  }

  interface HTMLElementEventMap {
    'dl-inspector-opening': CustomEvent<InspectorOpeningDetail>;
    'dl-inspector-state-change': CustomEvent<InspectorStateDetail>;
  }
}
