// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, type TemplateResult} from 'lit';
import {
  getWebBootstrap,
  type WebBootstrap,
} from '../api/bootstrap.ts';
import type {AnswerArtifact} from '../api/conversations.ts';
import {syncShellInert} from '../lib/dom.ts';
import {LightElement} from '../lib/lit_host.ts';
import type {AttachmentPolicy} from './attachment_policy.ts';
import type {DlArtifactCanvas} from './artifact_canvas.ts';
import './artifact_canvas.ts';
import type {
  ConversationSidebarStateDetail,
  DlConversationSidebar,
} from './conversation_sidebar.ts';
import './conversation_sidebar.ts';
import type {
  AnswerImageOpenDetail,
  AnswerPresentationElement,
  AnswerSourceOpenDetail,
} from './answer_presentation.ts';
import {hasActiveFileMutation} from './files-panel.ts';
import {openLightbox} from './images.ts';
import {closeConversationPanels, closePanel} from './panel.ts';
import {openAnswerSources} from './source-panel.ts';
import type {
  ChatContentChangeDetail,
  ChatRunActionDetail,
  ChatViewActionDetail,
  DlChatFeature,
} from './chat_feature.ts';
import './chat_feature.ts';
import {setupSettings} from './settings.ts';
import {syncPanelSplitState} from './split_panel.ts';
import {showToast} from './toast.ts';
import type {
  ContinuationResult,
  DlChildrenRoster,
  DlContinuationDialog,
} from './run_dialogs.ts';
import './run_dialogs.ts';

const EMPTY_BOOTSTRAP: WebBootstrap = {
  contract_version: 1,
  workspaces: [],
  primary_workspace: '',
  active_workspaces: [],
  answer_attachments: {
    count_limit: 0,
    image_max_bytes: 0,
    document_max_bytes: 0,
    extensions: [],
    image_capability: 'unknown',
    image_limit: 0,
    accept: '',
  },
  active_html_preview_enabled: true,
};

/** Vite-owned application document body and authenticated bootstrap lifecycle. */
export class DlApp extends LightElement {
  static properties = {
    bootState: {state: true},
  };

  declare bootState: 'loading' | 'ready' | 'error';
  #bootstrap: WebBootstrap = EMPTY_BOOTSTRAP;
  #controller: AbortController | null = null;
  #shellEvents: AbortController | null = null;
  #openSettings: (() => Promise<void>) | null = null;
  #pendingContinuation: {kind: 'follow-up' | 'fork'; runId: string} | null = null;
  readonly #ready: Promise<WebBootstrap>;
  #resolveReady!: (bootstrap: WebBootstrap) => void;
  #readyResolved = false;

  constructor() {
    super();
    this.bootState = 'loading';
    this.#ready = new Promise((resolve) => { this.#resolveReady = resolve; });
  }

  get ready(): Promise<WebBootstrap> {
    return this.#ready;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    if (!this.#shellEvents) {
      const events = new AbortController();
      this.#shellEvents = events;
      // Milestone 4 deletes this adapter with the legacy Inspector opening event.
      document.body.addEventListener('panelOpening', () => {
        void this.#conversationSidebar()?.close(false);
      }, {signal: events.signal});
    }
    if (!this.#controller && this.bootState !== 'ready') void this.#load();
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.#controller?.abort();
    this.#controller = null;
    this.#shellEvents?.abort();
    this.#shellEvents = null;
    document.body.classList.remove('conversation-sidebar-open', 'conversation-drawer-open');
  }

  /** Milestone 5 deletes this adapter with the imperative Settings setup. */
  setupSettingsAdapter(): void {
    this.#openSettings = setupSettings(() => this.#requestDeleteAllConversations());
  }

  async #requestDeleteAllConversations(): Promise<boolean> {
    return await this.#conversationSidebar()?.deleteAll() ?? false;
  }

  async #load(): Promise<void> {
    this.#controller?.abort();
    const controller = new AbortController();
    this.#controller = controller;
    this.bootState = 'loading';
    try {
      this.#bootstrap = await getWebBootstrap(controller.signal);
      if (this.#controller !== controller) return;
      this.bootState = 'ready';
      await this.updateComplete;
      if (!this.#readyResolved) {
        this.#readyResolved = true;
        this.#resolveReady(this.#bootstrap);
      }
    } catch (error) {
      if (controller.signal.aborted || this.#controller !== controller) return;
      this.bootState = 'error';
    } finally {
      if (this.#controller === controller) this.#controller = null;
    }
  }

  protected override render(): TemplateResult {
    const bootstrap = this.#bootstrap;
    const attachments = bootstrap.answer_attachments;
    const ready = this.bootState === 'ready';
    const chatFeature = this.querySelector<DlChatFeature>('dl-chat-feature');
    return html`
      <div
        class="app"
        id="app"
        @artifact-open=${this.#openArtifact}
        @answer-source-open=${this.#openAnswerSource}
        @answer-image-open=${this.#openAnswerImage}
        @dl-chat-view-action=${this.#chatViewAction}
        @dl-settings-request=${this.#settingsRequested}
        @dl-conversation-sidebar-opening=${this.#conversationSidebarOpening}
        @dl-conversation-sidebar-state-change=${this.#conversationSidebarStateChanged}
        @dl-conversation-route-change=${this.#conversationRouteChanged}
        aria-busy=${ready ? 'false' : 'true'}
        ?inert=${!ready}
      >
        <wa-split-panel class="panel-split" id="panel-split" primary="end"
                        position-in-pixels="0">
          <wa-split-panel class="panel-split" id="artifact-canvas-split" slot="start"
                          primary="end" position-in-pixels="0">
            <div class="primary-shell" slot="start">
              <div class="app-shell">
                <header class="topbar">
                  <dl-conversation-sidebar
                    .enabled=${ready}
                    .chatFeature=${chatFeature}
                  ></dl-conversation-sidebar>
                  <span class="topbar-scope-label">Search in:</span>
                  <workspace-scope
                    class="workspace-selector"
                    id="workspace-selector"
                    role="button"
                    tabindex="0"
                    aria-label="Choose search workspaces"
                    data-all=${JSON.stringify(bootstrap.workspaces)}
                    data-primary=${bootstrap.primary_workspace}
                    data-active=${JSON.stringify(bootstrap.active_workspaces)}
                  ></workspace-scope>
                  <div class="topbar-spacer"></div>
                  <button class="topbar-btn" id="files-btn" type="button">Files</button>
                  <div id="theme-control">
                    <button id="theme-trigger" type="button" aria-label="Appearance" title="Appearance"
                            aria-haspopup="menu" aria-controls="theme-menu" aria-expanded="false">
                      <svg class="theme-icon theme-icon-moon" width="17" height="17" viewBox="0 0 24 24"
                           fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"
                           stroke-linejoin="round" aria-hidden="true">
                        <path d="M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401"></path>
                      </svg>
                      <svg class="theme-icon theme-icon-sun" width="17" height="17" viewBox="0 0 24 24"
                           fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"
                           stroke-linejoin="round" aria-hidden="true">
                        <circle cx="12" cy="12" r="4"></circle>
                        <path d="M12 2v2"></path><path d="M12 20v2"></path>
                        <path d="m4.93 4.93 1.41 1.41"></path><path d="m17.66 17.66 1.41 1.41"></path>
                        <path d="M2 12h2"></path><path d="M20 12h2"></path>
                        <path d="m6.34 17.66-1.41 1.41"></path><path d="m19.07 4.93-1.41 1.41"></path>
                      </svg>
                    </button>
                    <div id="theme-menu" role="menu" aria-label="Appearance" hidden>
                      ${this.#themeOption('system', 'System', this.#systemIcon())}
                      ${this.#themeOption('light', 'Light', this.#sunIcon())}
                      ${this.#themeOption('dark', 'Dark', this.#moonIcon())}
                    </div>
                  </div>
                </header>

                <dl-chat-feature
                  .attachmentPolicy=${this.#attachmentPolicy()}
                  .attachmentAccept=${attachments.accept}
                  @dl-chat-content-change=${this.#chatContentChanged}
                  @dl-chat-run-action=${this.#chatRunAction}
                ></dl-chat-feature>

                <input class="hidden" type="file" id="folder-input" webkitdirectory directory multiple>
              </div>
            </div>
            <dl-artifact-canvas id="artifact-canvas" class="panel" slot="end"
              aria-label="Artifact Canvas" aria-hidden="true"
              .activePreviewEnabled=${bootstrap.active_html_preview_enabled}
            ></dl-artifact-canvas>
          </wa-split-panel>
          <aside class="panel" id="panel" slot="end">
            <div class="panel-header">
              <span id="panel-title"></span>
              <button class="source-toggle-all" id="source-toggle-all-btn" type="button"
                      aria-pressed="false" hidden>Show all</button>
              <ingest-target class="ingest-target" id="ingest-target"></ingest-target>
              <button class="panel-close" id="panel-close-btn" type="button" aria-label="Close panel">✕</button>
            </div>
            <div id="panel-content" class="panel-content"></div>
          </aside>
        </wa-split-panel>
        <div id="panel-backdrop" hidden></div>

        <div class="toast" id="toast" role="status" aria-live="polite" aria-atomic="true"></div>
        <div class="notify-offer" id="notify-offer" role="group"
             aria-label="Answer notifications" hidden>
          <span class="notify-offer-text">Notify you when an answer finishes?</span>
          <button class="ui-btn" id="notify-offer-accept" type="button">Enable</button>
          <button class="ui-btn" id="notify-offer-decline" type="button">Not now</button>
        </div>

        ${this.#dialogs()}
      </div>
      ${this.#bootstrapStatus()}
    `;
  }

  #conversationSidebar(): DlConversationSidebar | null {
    return this.querySelector<DlConversationSidebar>('dl-conversation-sidebar');
  }

  #chatViewAction(event: CustomEvent<ChatViewActionDetail>): void {
    this.#conversationSidebar()?.handleChatViewAction(event.detail.action);
  }

  #settingsRequested(): void {
    if (this.#openSettings) void this.#openSettings();
  }

  // Milestone 4 deletes these adapters when Inspector exposes typed commands.
  #conversationSidebarOpening(event: Event): void {
    const panel = this.querySelector<HTMLElement>('#panel');
    if (!panel?.classList.contains('open')) return;
    if (hasActiveFileMutation()) {
      event.preventDefault();
      showToast('Wait for the file change to finish before opening conversations.', 5000);
      return;
    }
    closePanel();
  }

  #conversationSidebarStateChanged(
    event: CustomEvent<ConversationSidebarStateDetail>,
  ): void {
    document.body.classList.toggle('conversation-sidebar-open', event.detail.expanded);
    document.body.classList.toggle(
      'conversation-drawer-open',
      event.detail.expanded && event.detail.compact,
    );
    syncPanelSplitState();
    syncShellInert();
  }

  #conversationRouteChanged(): void {
    closeConversationPanels();
  }

  #openArtifact(event: CustomEvent<{artifact: AnswerArtifact; returnFocus: HTMLElement}>): void {
    const canvas = this.querySelector<DlArtifactCanvas>('#artifact-canvas');
    if (!canvas) return;
    void canvas.open(event.detail.artifact, event.detail.returnFocus);
  }

  #openAnswerSource(event: CustomEvent<AnswerSourceOpenDetail>): void {
    const presentation = (event.target as AnswerPresentationElement | null)?.presentation;
    if (!presentation) return;
    const canvas = this.querySelector<DlArtifactCanvas>('#artifact-canvas');
    const sourceWasInCanvas = Boolean(canvas?.contains(event.detail.returnFocus));
    canvas?.prepareForInspector();
    const returnFocus = sourceWasInCanvas && !canvas?.classList.contains('open')
      ? document.activeElement instanceof HTMLElement ? document.activeElement : null
      : event.detail.returnFocus;
    openAnswerSources(
      presentation,
      event.detail.referenceId,
      event.detail.chunkId,
      returnFocus,
    );
  }

  #openAnswerImage(event: CustomEvent<AnswerImageOpenDetail>): void {
    event.detail.returnFocus.focus();
    openLightbox(event.detail.src);
  }

  #chatContentChanged(event: CustomEvent<ChatContentChangeDetail>): void {
    this.querySelector('.app')?.classList.toggle('has-messages', event.detail.hasMessages);
  }

  #attachmentPolicy(): AttachmentPolicy {
    const attachments = this.#bootstrap.answer_attachments;
    return {
      countLimit: attachments.count_limit,
      imageMaxBytes: attachments.image_max_bytes,
      documentMaxBytes: attachments.document_max_bytes,
      extensions: new Set(attachments.extensions),
      imageCapability: attachments.image_capability,
      imageLimit: attachments.image_limit,
    };
  }

  #chatRunAction(event: CustomEvent<ChatRunActionDetail>): void {
    const chat = this.querySelector<DlChatFeature>('dl-chat-feature');
    if (!chat) return;
    if (event.detail.action === 'children') {
      this.querySelector<DlChildrenRoster>('dl-children-roster')?.open(
        () => chat.loadRunChildren(event.detail.runId),
      );
      return;
    }
    this.#pendingContinuation = {
      kind: event.detail.action,
      runId: event.detail.runId,
    };
    this.querySelector<DlContinuationDialog>('dl-continuation-dialog')?.open(event.detail.action);
  }

  #continuationResult(event: CustomEvent<ContinuationResult>): void {
    const pending = this.#pendingContinuation;
    this.#pendingContinuation = null;
    if (!pending || !event.detail.query || event.detail.kind !== pending.kind) return;
    const chat = this.querySelector<DlChatFeature>('dl-chat-feature');
    if (chat) void chat.continueRun(pending.kind, pending.runId, event.detail.query);
  }

  #themeOption(value: string, label: string, icon: TemplateResult): TemplateResult {
    return html`
      <button type="button" role="menuitemradio" data-theme-value=${value}
              aria-checked=${value === 'system' ? 'true' : 'false'} tabindex="-1">
        <span class="theme-menu-icon" aria-hidden="true">${icon}</span>
        <span class="theme-menu-label">${label}</span>
        <span class="theme-menu-check" aria-hidden="true">✓</span>
      </button>
    `;
  }

  #systemIcon(): TemplateResult {
    return html`<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <rect width="20" height="14" x="2" y="3" rx="2"></rect>
      <line x1="8" x2="16" y1="21" y2="21"></line><line x1="12" x2="12" y1="17" y2="21"></line>
    </svg>`;
  }

  #sunIcon(): TemplateResult {
    return html`<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="4"></circle><path d="M12 2v2"></path><path d="M12 20v2"></path>
      <path d="m4.93 4.93 1.41 1.41"></path><path d="m17.66 17.66 1.41 1.41"></path>
      <path d="M2 12h2"></path><path d="M20 12h2"></path>
      <path d="m6.34 17.66-1.41 1.41"></path><path d="m19.07 4.93-1.41 1.41"></path>
    </svg>`;
  }

  #moonIcon(): TemplateResult {
    return html`<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
      <path d="M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401"></path>
    </svg>`;
  }

  #bootstrapStatus(): TemplateResult {
    if (this.bootState === 'ready') return html``;
    if (this.bootState === 'loading') {
      return html`<div class="bootstrap-status" role="status">Loading DlightRAG…</div>`;
    }
    return html`<div class="bootstrap-status bootstrap-status--error" role="alert">
      <span>DlightRAG could not load.</span>
      <button type="button" @click=${() => { void this.#load(); }}>Retry</button>
    </div>`;
  }

  #dialogs(): TemplateResult {
    return html`
      <dialog id="settings-dialog" class="settings-dialog" aria-labelledby="settings-title">
        <form id="settings-form" method="dialog">
          <div class="settings-drawer-body">
            <div class="settings-header">
              <h2 id="settings-title">Settings</h2>
              <button class="panel-close settings-close" type="submit" value="close-settings"
                      aria-label="Close settings">✕</button>
            </div>
            <section class="settings-section">
              <h3 id="settings-memory">Profile Memory</h3>
              <label class="ui-dialog-checkbox">
                <input type="checkbox" id="memory-enabled-toggle" />
                Activate profile memories
              </label>
              <p id="memory-active-count" class="settings-count" aria-live="polite" hidden></p>
              <div class="settings-actions">
                <button type="button" id="memory-clear-btn" class="ui-btn ui-btn-danger-text" hidden>Clear memory</button>
              </div>
            </section>
            <section class="settings-section">
              <h3>Active HTML Preview</h3>
              <p class="settings-note">
                ${this.#bootstrap.active_html_preview_enabled
                  ? 'Enabled by the operator. Interactive reports require an explicit open action.'
                  : 'Disabled by the operator. HTML Artifacts are shown with scripts disabled.'}
              </p>
            </section>
            <section class="settings-section">
              <h3 id="settings-data">Conversation Sessions</h3>
              <p class="settings-note">Conversations retain 365 days</p>
              <p id="conversation-count" class="settings-count" aria-live="polite"></p>
              <div class="settings-actions">
                <button type="button" id="delete-all-btn" class="ui-btn ui-btn-danger-text">Delete all conversations</button>
              </div>
            </section>
          </div>
        </form>
      </dialog>
      <dialog id="clear-memory-dialog" class="confirm-dialog" aria-labelledby="clear-memory-title">
        <form method="dialog">
          <h2 id="clear-memory-title">Clear Profile memory?</h2>
          <p>Remembered preferences and facts will be forgotten. Conversations are not affected.</p>
          <div class="ui-dialog-actions">
            <button type="submit" value="cancel">Cancel</button>
            <button type="submit" value="clear" class="ui-dialog-danger">Clear memory</button>
          </div>
        </form>
      </dialog>
      <dl-continuation-dialog
        @dl-continuation-result=${this.#continuationResult}
      ></dl-continuation-dialog>
      <dl-children-roster></dl-children-roster>
      <dialog id="delete-workspace-dialog" class="workspace-dialog">
        <form id="delete-workspace-form">
          <h3 class="workspace-dialog-title">Delete workspace</h3>
          <p class="workspace-dialog-text">This will permanently delete all data for</p>
          <p class="workspace-dialog-name" id="delete-workspace-name"></p>
          <p class="workspace-dialog-text">Type the workspace name to confirm</p>
          <input type="hidden" name="workspace_name" id="delete-workspace-id">
          <input type="text" name="confirm_name" id="delete-workspace-confirm-input"
                 class="workspace-dialog-input" autocomplete="off" placeholder="Type workspace name...">
          <div class="ui-dialog-actions">
            <button type="button" data-action="close-delete-workspace-dialog">Cancel</button>
            <button type="submit" id="delete-workspace-confirm-btn"
                    class="ui-dialog-danger" disabled>Delete</button>
          </div>
        </form>
      </dialog>
    `;
  }
}

customElements.define('dl-app', DlApp);

declare global {
  interface HTMLElementTagNameMap {
    'dl-app': DlApp;
  }
}
