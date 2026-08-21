// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, type TemplateResult} from 'lit';
import {
  getWebBootstrap,
  type WebBootstrap,
} from '../api/bootstrap.ts';
import {LightElement} from '../lib/lit_host.ts';

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
};

/** Vite-owned application document body and authenticated bootstrap lifecycle. */
export class DlApp extends LightElement {
  static properties = {
    bootState: {state: true},
  };

  declare bootState: 'loading' | 'ready' | 'error';
  #bootstrap: WebBootstrap = EMPTY_BOOTSTRAP;
  #controller: AbortController | null = null;
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
    if (!this.#controller && this.bootState !== 'ready') void this.#load();
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.#controller?.abort();
    this.#controller = null;
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
    return html`
      <div
        class="app"
        id="app"
        data-attachment-count-limit=${String(attachments.count_limit)}
        data-attachment-image-max-bytes=${String(attachments.image_max_bytes)}
        data-attachment-document-max-bytes=${String(attachments.document_max_bytes)}
        data-attachment-extensions=${JSON.stringify(attachments.extensions)}
        data-attachment-image-capability=${attachments.image_capability}
        data-attachment-image-limit=${String(attachments.image_limit)}
        aria-busy=${ready ? 'false' : 'true'}
        ?inert=${!ready}
      >
        <nav id="chat-sidebar" aria-label="Conversations">
          <div class="conversation-top-row">
            <button id="new-conversation-btn" type="button">
              <svg class="new-chat-icon" width="14" height="14" viewBox="0 0 24 24"
                   fill="none" stroke="currentColor" stroke-width="1.8"
                   stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"></path>
              </svg>
              New chat
            </button>
            <button id="conversation-sidebar-toggle" type="button"
                    aria-label="Collapse conversations" aria-controls="chat-sidebar">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="1.6" stroke-linecap="round"
                   stroke-linejoin="round" aria-hidden="true">
                <rect x="3" y="3" width="18" height="18" rx="2"></rect>
                <path d="M9 3v18"></path><path d="M14 10l-2 2 2 2"></path>
              </svg>
            </button>
          </div>
          <conversation-list id="conversation-list" role="list" aria-live="polite"></conversation-list>
          <button id="delete-all-conversations-btn" type="button"
                  aria-label="Delete all conversations">
            <span class="conversation-retention-note">Inactive conversations expire after 30 days.</span>
            <span class="conversation-delete-all-label" aria-hidden="true">Delete all conversations</span>
          </button>
        </nav>
        <div id="conversation-sidebar-backdrop" hidden></div>

        <wa-split-panel class="panel-split" id="panel-split" primary="end"
                        position-in-pixels="0">
          <wa-split-panel class="panel-split" id="report-panel-split" slot="start"
                          primary="end" position-in-pixels="0">
            <div class="primary-shell" slot="start">
              <div class="app-shell">
                <header class="topbar">
                  <button id="conversation-sidebar-open" type="button" aria-label="Open conversations"
                          aria-controls="chat-sidebar">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" stroke-width="1.6" stroke-linecap="round"
                         stroke-linejoin="round" aria-hidden="true">
                      <rect x="3" y="3" width="18" height="18" rx="2"></rect>
                      <path d="M9 3v18"></path><path d="M10 10l2 2-2 2"></path>
                    </svg>
                  </button>
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

                <main class="chat-area" id="chat-area">
                  <div id="chat-messages" class="chat-messages">
                    <div class="welcome" id="welcome">
                      <div class="welcome-brand">DlightRAG</div>
                      <div class="welcome-sub">Ask anything about your documents</div>
                    </div>
                  </div>
                </main>

                <div class="drop-overlay" id="drop-overlay">
                  <div class="drop-overlay-content">Drop files or folders here</div>
                </div>

                <div class="composer" id="composer">
                  <div class="composer-inner">
                    <div class="thumbnail-strip" id="thumbnail-strip"></div>
                    <form id="query-form" class="composer-form">
                      <button type="button" class="composer-plus" id="composer-plus" aria-label="Attach files">
                        <svg class="composer-plus-icon" width="24" height="24" viewBox="0 0 24 24"
                             fill="none" stroke="currentColor" stroke-linecap="round"
                             stroke-linejoin="round" aria-hidden="true" focusable="false">
                          <path d="M12 5v14"></path><path d="M5 12h14"></path>
                        </svg>
                      </button>
                      <textarea name="query" aria-label="Message" placeholder="Ask anything"
                                class="composer-input" rows="1" autocomplete="off"></textarea>
                      <div class="composer-mode">
                        <button type="button" class="composer-mode-trigger" id="composer-mode"
                                aria-haspopup="menu" aria-expanded="false" aria-label="Answer mode: Auto">Auto</button>
                        <div class="composer-mode-menu" id="composer-mode-menu" role="menu" hidden>
                          <button type="button" role="menuitemradio" data-mode="auto" aria-checked="true">Auto</button>
                          <button type="button" role="menuitemradio" data-mode="fast" aria-checked="false">Fast</button>
                          <button type="button" role="menuitemradio" data-mode="research" aria-checked="false">Research</button>
                        </div>
                      </div>
                      <button type="submit" class="composer-send" aria-label="Send">
                        <svg class="composer-send-icon composer-send-icon--send" width="18" height="18"
                             viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
                          <line x1="22" y1="2" x2="11" y2="13"></line>
                          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                        </svg>
                        <svg class="composer-send-icon composer-send-icon--stop" width="16" height="16"
                             viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">
                          <rect x="6" y="6" width="12" height="12" rx="2"></rect>
                        </svg>
                      </button>
                    </form>
                  </div>
                  <input class="hidden" type="file" id="attachment-input" accept=${attachments.accept} multiple>
                </div>

                <input class="hidden" type="file" id="folder-input" webkitdirectory directory multiple>
              </div>
            </div>
            <aside class="panel" id="report-panel" slot="end" aria-label="Report" aria-hidden="true">
              <div class="panel-header">
                <span id="report-panel-title">Report</span>
                <button class="panel-close" id="report-panel-close-btn" type="button"
                        aria-label="Close report">✕</button>
              </div>
              <div id="report-panel-content" class="panel-content"></div>
            </aside>
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
      <dialog id="delete-conversation-dialog" aria-labelledby="delete-conversation-title"
              aria-describedby="delete-conversation-message">
        <form method="dialog">
          <h2 id="delete-conversation-title">Delete conversation</h2>
          <p id="delete-conversation-message">This conversation and its history will be permanently deleted.</p>
          <p id="delete-conversation-draft-warning" hidden>Your unsent draft and attachments will also be discarded.</p>
          <div class="ui-dialog-actions">
            <button type="submit" value="cancel">Cancel</button>
            <button type="submit" value="delete" class="ui-dialog-danger">Delete</button>
          </div>
        </form>
      </dialog>
      <dialog id="delete-all-conversations-dialog" aria-labelledby="delete-all-conversations-title">
        <form method="dialog">
          <h2 id="delete-all-conversations-title">Delete all conversations?</h2>
          <p id="delete-all-conversations-draft-warning" hidden>Draft and attachments will also be deleted.</p>
          <div class="ui-dialog-actions">
            <button type="submit" value="cancel">Cancel</button>
            <button type="submit" value="delete-all" class="ui-dialog-danger">Delete all</button>
          </div>
        </form>
      </dialog>
      <dialog id="discard-draft-dialog" aria-labelledby="discard-draft-title">
        <form method="dialog">
          <h2 id="discard-draft-title">Discard draft?</h2>
          <p>Your unsent message and attachments will not move to another conversation.</p>
          <div class="ui-dialog-actions">
            <button type="submit" value="cancel">Keep editing</button>
            <button type="submit" value="discard">Discard and continue</button>
          </div>
        </form>
      </dialog>
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
