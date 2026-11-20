// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import type {AnswerPresentation} from '../api/conversations.ts';
import {isImeKey} from '../design-system/index.ts';
import {COMPACT_SHELL_MEDIA} from '../lib/breakpoints.ts';
import {raise, tabbables, wrapTabFocus} from '../lib/dom.ts';
import {FocusReturn} from '../lib/focus-return.ts';
import {LightElement, MediaController} from '../lib/lit-host.ts';
import {productionHandles, type AppHandles} from '../stores/app-handles.ts';
import type {AnswerRunEvent} from '../lib/run-controller.ts';
import type {TracesSource} from '../stores/traces-source.ts';
import type {DlInspectorTraces} from './inspector-traces.ts';
import './inspector-traces.ts';
import type {DlInspectorFiles} from './inspector-files.ts';
import './inspector-files.ts';
import './inspector-agentspace.ts';
import './inspector-live.ts';
import type {
  DlInspectorSources,
  InspectorSourcesStateDetail,
} from './inspector-sources.ts';
import './inspector-sources.ts';
import './ingest-target.ts';

export type InspectorKind = 'files' | 'sources' | 'traces' | 'live' | 'agentspace';

export interface InspectorStateDetail {
  open: boolean;
  kind: InspectorKind | null;
  compact: boolean;
}

interface InspectorOpeningDetail {
  kind: InspectorKind;
}

/** Sources/Files/Child agents pane state, accessibility, focus, and content composition. */
export class DlInspector extends LightElement {
  static properties = {
    handles: {attribute: false},
    kind: {state: true},
    presentation: {state: true},
    sourceHasItems: {state: true},
    sourcesExpanded: {state: true},
    tracesSource: {state: true},
    liveRunId: {state: true},
    shellInert: {state: true},
    answerFeedbackEnabled: {attribute: false},
  };

  declare handles: AppHandles;
  declare kind: InspectorKind | null;
  declare presentation: AnswerPresentation | null;
  declare sourceHasItems: boolean;
  declare sourcesExpanded: boolean;
  declare tracesSource: TracesSource | null;
  /** The Run whose browser the live dock shows; null while no live dock is open. */
  declare liveRunId: string | null;
  declare shellInert: boolean;
  /** Whether the reader is offered the thumbs for what a child answered. */
  declare answerFeedbackEnabled: boolean;

  readonly #focus = new FocusReturn();
  readonly #compact = new MediaController(this, COMPACT_SHELL_MEDIA, () => { this.#compactChanged(); });
  #stateSignature = '';

  constructor() {
    super();
    this.handles = productionHandles();
    this.kind = null;
    this.presentation = null;
    this.sourceHasItems = false;
    this.sourcesExpanded = false;
    this.tracesSource = null;
    this.liveRunId = null;
    this.shellInert = false;
    this.answerFeedbackEnabled = false;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    document.addEventListener('keydown', this.#documentKeydown, {signal: this.lifetime});
    this.#syncHostState();
  }

  override disconnectedCallback(): void {
    this.#focus.invalidate();
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
    this.handles.ingest.resetToPrimary();
    await this.updateComplete;
    await this.#focusOnCompact();
  }

  /** Open one Run's Child agents. Asked for the Run the dock already shows, it keeps its source, and
   * so the child the reader has open and what they have typed. */
  /** Open the conversation's Agentspace: its saved copies and the files each Run left. */
  async openAgentspace(returnFocus?: HTMLElement | null): Promise<void> {
    if (!this.#beginOpen('agentspace', returnFocus)) return;
    await this.updateComplete;
    await this.#focusOnCompact();
  }

  async openTraces(source: TracesSource, returnFocus?: HTMLElement | null): Promise<void> {
    const showing = this.kind === 'traces' && this.tracesSource?.runId === source.runId;
    if (!this.#beginOpen('traces', returnFocus)) return;
    if (!showing) this.tracesSource = source;
    await this.updateComplete;
    await this.#focusOnCompact();
  }

  /** Open the live browser preview of one Run, which subscribes to it until it closes. */
  async openLive(runId: string, returnFocus?: HTMLElement | null): Promise<void> {
    if (!this.#beginOpen('live', returnFocus)) return;
    this.liveRunId = runId;
    await this.updateComplete;
    await this.#focusOnCompact();
  }

  /** Hand an open Child agents dock what its Run just said, so what it draws stays live. */
  applyTracesEvents(runId: string, events: readonly AnswerRunEvent[]): void {
    this.#children()?.applyEvents(runId, events);
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
    this.#files()?.pause();
    this.kind = null;
    this.presentation = null;
    this.sourceHasItems = false;
    this.sourcesExpanded = false;
    this.tracesSource = null;
    this.liveRunId = null;
    this.#syncHostState();
    this.#publishState();
    const panel = this.querySelector<HTMLElement>('#panel');
    panel?.classList.remove('open');
    if (panel) {
      panel.inert = true;
      panel.setAttribute('aria-hidden', 'true');
      delete panel.dataset.panelKind;
    }
    if (restoreFocus) this.#focus.restore(() => this.open);
    else this.#focus.discard();
  }

  /** Close conversation-scoped Sources, Child agents and the live browser, preserving Files and
   *  the Agentspace, which follow the conversation the reader switches to. */
  closeConversationContent(): void {
    if (this.kind !== 'sources' && this.kind !== 'traces' && this.kind !== 'live') return;
    this.close(false);
  }

  protected override render(): TemplateResult {
    const open = this.open;
    const compact = this.#compact.matches;
    const files = this.kind === 'files';
    const sources = this.kind === 'sources';
    const children = this.kind === 'traces';
    const live = this.kind === 'live';
    const agentspace = this.kind === 'agentspace';
    return html`
      <aside
        class="panel inspector-surface${open ? ' open' : ''}"
        id="panel"
        aria-labelledby=${open ? 'panel-title' : nothing}
        aria-hidden=${open ? nothing : 'true'}
        role=${open && compact ? 'dialog' : nothing}
        aria-modal=${open && compact ? 'true' : nothing}
        data-panel-kind=${this.kind ?? nothing}
        ?inert=${!open || this.shellInert}
        @keydown=${this.#panelKeydown}
      >
        <div class="panel-header">
          <h2 id="panel-title">${this.#title()}</h2>
          <button
            class="source-toggle-all"
            id="source-toggle-all-btn"
            type="button"
            aria-pressed=${this.sourcesExpanded ? 'true' : 'false'}
            ?hidden=${!sources || !this.sourceHasItems}
            @click=${this.#toggleAllSources}
          >${this.sourcesExpanded ? msg('Collapse all', {id: 'inspector.collapseAll'}) : msg('Show all', {id: 'inspector.showAll'})}</button>
          <dl-ingest-target
            class="ingest-target"
            id="ingest-target"
            .handles=${this.handles}
            .active=${files}
          ></dl-ingest-target>
          <dl-icon-button class="panel-close" id="panel-close-btn" name="close" size="sm"
                  aria-label=${msg('Close panel', {id: 'inspector.closePanel'})} @click=${() => this.close()}></dl-icon-button>
        </div>
        <div id="panel-content" class="panel-content">
          <dl-inspector-sources
            .sources=${this.presentation?.sources ?? []}
            ?hidden=${!sources}
            @dl-inspector-sources-state-change=${this.#sourcesStateChanged}
          ></dl-inspector-sources>
          <dl-inspector-files .handles=${this.handles} .active=${files} ?hidden=${!files}></dl-inspector-files>
          <dl-inspector-traces .handles=${this.handles} .source=${this.tracesSource} .active=${children}
            .answerFeedbackEnabled=${this.answerFeedbackEnabled}
            ?hidden=${!children}></dl-inspector-traces>
          <dl-inspector-live .runId=${this.liveRunId} .active=${live}
            ?hidden=${!live}></dl-inspector-live>
          <dl-inspector-agentspace .handles=${this.handles} .active=${agentspace}
            ?hidden=${!agentspace}></dl-inspector-agentspace>
        </div>
      </aside>
      <div class="inspector-backdrop" aria-hidden="true"
           ?hidden=${!open || !compact}></div>
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
    this.#focus.invalidate();
    if (returnFocus || !this.open) this.#focus.remember(returnFocus);
    this.kind = kind;
    this.#syncHostState();
    return true;
  }

  #title(): string {
    switch (this.kind) {
      case 'files': return msg('Files', {id: 'inspector.files'});
      case 'sources': return msg('Sources', {id: 'inspector.sources'});
      case 'traces': return msg('Agent traces', {id: 'inspector.agentTraces'});
      case 'live': return msg('Live browser', {id: 'inspector.live'});
      case 'agentspace': return msg('Agentspace', {id: 'inspector.agentspace'});
      default: return '';
    }
  }

  #sources(): DlInspectorSources | null {
    return this.querySelector<DlInspectorSources>('dl-inspector-sources');
  }

  #files(): DlInspectorFiles | null {
    return this.querySelector<DlInspectorFiles>('dl-inspector-files');
  }

  #children(): DlInspectorTraces | null {
    return this.querySelector<DlInspectorTraces>('dl-inspector-traces');
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
      compact: this.#compact.matches,
    };
    const signature = `${detail.open}:${detail.kind ?? ''}:${detail.compact}`;
    if (signature === this.#stateSignature) return;
    this.#stateSignature = signature;
    raise(this, 'dl-inspector-state-change', detail);
  }

  async #focusOnCompact(): Promise<void> {
    if (!this.#compact.matches) return;
    await this.updateComplete;
    this.querySelector<HTMLElement>('#panel-close-btn')?.focus();
  }

  #focusableElements(): HTMLElement[] {
    const panel = this.querySelector<HTMLElement>('#panel');
    if (!panel) return [];
    return tabbables(panel);
  }

  #compactChanged(): void {
    this.#publishState();
    if (this.open && this.#compact.matches && !this.contains(document.activeElement)) {
      void this.#focusOnCompact();
    }
  }

  #documentKeydown = (event: KeyboardEvent): void => {
    // An Escape that cancels an IME composition belongs to the composition, not to the panel.
    if (!this.open || event.key !== 'Escape' || event.defaultPrevented || isImeKey(event)) return;
    if (document.querySelector('dialog[open]')) return;
    event.preventDefault();
    this.close();
  };

  #panelKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'Tab' && this.#compact.matches) {
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
