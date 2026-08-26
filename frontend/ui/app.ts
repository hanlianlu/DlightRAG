// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, type TemplateResult} from 'lit';
import {getWebBootstrap, type WebBootstrap} from '../api/bootstrap.ts';
import type {AnswerArtifact} from '../api/conversations.ts';
import {COMPACT_SHELL_MEDIA} from '../lib/breakpoints.ts';
import {LightElement} from '../lib/lit_host.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import type {AttachmentPolicy} from './attachment_policy.ts';
import type {
  ArtifactCanvasStateDetail,
  DlArtifactCanvas,
} from './artifact_canvas.ts';
import './artifact_canvas.ts';
import type {
  ConversationSidebarStateDetail,
  DlConversationSidebar,
} from './conversation_sidebar.ts';
import './conversation_sidebar.ts';
import type {AnswerSourceOpenDetail} from './answer_presentation.ts';
import type {ComposerWorkspaceDropDetail} from './chat_composer.ts';
import type {
  ChatContentChangeDetail,
  ChatMemoryOperationDetail,
  ChatRunActionDetail,
  ChatRunningChangeDetail,
  ChatViewActionDetail,
  DlChatFeature,
} from './chat_feature.ts';
import './chat_feature.ts';
import type {ImageOpenDetail} from './image_lightbox.ts';
import type {DlImageLightbox} from './image_lightbox.ts';
import './image_lightbox.ts';
import type {DlInspector, InspectorStateDetail} from './inspector.ts';
import './inspector.ts';
import type {ModalStateDetail} from './modal.ts';
import type {DlSettingsDialog} from './settings.ts';
import './settings.ts';
import {syncPanelSplitState} from './split_panel.ts';
import type {
  ContinuationResult,
  DlChildrenRoster,
  DlContinuationDialog,
} from './run_dialogs.ts';
import './run_dialogs.ts';
import './notifications.ts';
import './theme.ts';
import type {DlToastRegion, ToastRequestDetail} from './toast.ts';
import './toast.ts';
import './workspace_scope.ts';

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

/** Authenticated bootstrap, top-level capabilities, and Feature composition only. */
export class DlApp extends LightElement {
  static properties = {
    bootState: {state: true},
    hasMessages: {state: true},
    conversationExpanded: {state: true},
    conversationCompact: {state: true},
    inspectorOpen: {state: true},
    inspectorKind: {state: true},
    inspectorCompact: {state: true},
    canvasOpen: {state: true},
    canvasModal: {state: true},
    canvasOverlay: {state: true},
    chatRunning: {state: true},
    lightboxOpen: {state: true},
    nativeModalOpen: {state: true},
  };

  declare bootState: 'loading' | 'ready' | 'error';
  declare hasMessages: boolean;
  declare conversationExpanded: boolean;
  declare conversationCompact: boolean;
  declare inspectorOpen: boolean;
  declare inspectorKind: 'files' | 'sources' | null;
  declare inspectorCompact: boolean;
  declare canvasOpen: boolean;
  declare canvasModal: boolean;
  declare canvasOverlay: boolean;
  declare chatRunning: boolean;
  declare lightboxOpen: boolean;
  declare nativeModalOpen: boolean;

  #bootstrap: WebBootstrap = EMPTY_BOOTSTRAP;
  readonly #nativeModalOwners = new Set<HTMLElement>();
  #controller: AbortController | null = null;
  #pendingContinuation: {kind: 'follow-up' | 'fork'; runId: string} | null = null;
  readonly #ready: Promise<WebBootstrap>;
  #resolveReady!: (bootstrap: WebBootstrap) => void;
  #readyResolved = false;

  constructor() {
    super();
    this.bootState = 'loading';
    this.hasMessages = false;
    this.conversationExpanded = false;
    this.conversationCompact = false;
    this.inspectorOpen = false;
    this.inspectorKind = null;
    this.inspectorCompact = false;
    this.canvasOpen = false;
    this.canvasModal = false;
    this.canvasOverlay = false;
    this.chatRunning = false;
    this.lightboxOpen = false;
    this.nativeModalOpen = false;
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
    this.#nativeModalOwners.clear();
    this.nativeModalOpen = false;
    document.body.classList.remove(
      'conversation-sidebar-open',
      'conversation-drawer-open',
      'panel-open',
      'files-panel-open',
      'sources-panel-open',
      'panel-drawer-open',
      'artifact-canvas-open',
      'artifact-canvas-overlay',
      'artifact-canvas-modal',
      'settings-open',
    );
  }

  async #load(): Promise<void> {
    this.#controller?.abort();
    const controller = new AbortController();
    this.#controller = controller;
    this.bootState = 'loading';
    try {
      const bootstrap = await getWebBootstrap(controller.signal);
      if (this.#controller !== controller) return;
      this.#bootstrap = bootstrap;
      workspaceStore.init(
        bootstrap.workspaces.map((workspace) => ({
          workspace: workspace.workspace,
          displayName: workspace.display_name,
          embeddingModel: workspace.embedding_model,
        })),
        bootstrap.active_workspaces,
        bootstrap.primary_workspace,
      );
      this.bootState = 'ready';
      await this.updateComplete;
      if (!this.#readyResolved) {
        this.#readyResolved = true;
        this.#resolveReady(bootstrap);
      }
    } catch {
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
    const conversationModal = this.conversationExpanded && this.conversationCompact;
    const inspectorModal = this.inspectorOpen && this.inspectorCompact;
    const shellModal = conversationModal || inspectorModal || this.canvasModal
      || this.nativeModalOpen;
    return html`
      <div class="app${this.hasMessages ? ' has-messages' : ''}" id="app"
        @artifact-open=${this.#openArtifact}
        @answer-source-open=${this.#openAnswerSource}
        @dl-image-open=${this.#openImage}
        @dl-chat-view-action=${this.#chatViewAction}
        @dl-chat-memory-operation=${this.#memoryOperation}
        @dl-chat-running-change=${this.#runningChanged}
        @dl-settings-request=${this.#settingsRequested}
        @dl-toast-request=${this.#toastRequested}
        @dl-modal-state-change=${this.#modalStateChanged}
        @dl-conversation-sidebar-opening=${this.#conversationSidebarOpening}
        @dl-conversation-sidebar-state-change=${this.#conversationSidebarStateChanged}
        @dl-conversation-route-change=${this.#conversationRouteChanged}
        @dl-inspector-opening=${this.#inspectorOpening}
        @dl-inspector-state-change=${this.#inspectorStateChanged}
        @dl-artifact-canvas-state-change=${this.#canvasStateChanged}
        @dl-composer-workspace-drop=${this.#workspaceDrop}
        aria-busy=${ready ? 'false' : 'true'}
        ?inert=${!ready || this.lightboxOpen}
      >
        <wa-split-panel class="panel-split" id="panel-split" primary="end"
                        position-in-pixels="0">
          <wa-split-panel class="panel-split" id="artifact-canvas-split" slot="start"
                          primary="end" position-in-pixels="0">
            <div class="primary-shell" slot="start">
              <div class="app-shell">
                <header class="topbar" ?inert=${inspectorModal || this.canvasModal}>
                  <dl-conversation-sidebar .enabled=${ready} .chatFeature=${chatFeature}
                    .shellInert=${this.canvasModal}></dl-conversation-sidebar>
                  <span class="topbar-scope-label" ?inert=${shellModal}>Search in:</span>
                  <dl-workspace-scope class="workspace-selector" id="workspace-selector"
                    ?inert=${shellModal}></dl-workspace-scope>
                  <div class="topbar-spacer" ?inert=${shellModal}></div>
                  <button class="topbar-btn" id="files-btn" type="button"
                          ?inert=${shellModal} @click=${this.#openFiles}>Files</button>
                  <dl-theme-control id="theme-control" ?inert=${shellModal}></dl-theme-control>
                </header>

                <dl-chat-feature .attachmentPolicy=${this.#attachmentPolicy()}
                  .attachmentAccept=${attachments.accept} ?inert=${shellModal}
                  @dl-chat-content-change=${this.#chatContentChanged}
                  @dl-chat-background-click=${this.#chatBackgroundClick}
                  @dl-chat-run-action=${this.#chatRunAction}></dl-chat-feature>
              </div>
            </div>
            <dl-artifact-canvas id="artifact-canvas" class="panel" slot="end"
              aria-label="Artifact Canvas" aria-hidden="true"
              .activePreviewEnabled=${bootstrap.active_html_preview_enabled}
            ></dl-artifact-canvas>
          </wa-split-panel>
          <dl-inspector id="inspector" slot="end"
            .shellInert=${this.canvasModal}></dl-inspector>
        </wa-split-panel>

        <dl-toast-region class="toast" id="toast" role="status" aria-live="polite"
          aria-atomic="true"
          .shellInert=${shellModal || this.lightboxOpen}></dl-toast-region>
        <dl-notification-offer class="notify-offer" id="notify-offer" role="group"
          aria-label="Answer notifications" .running=${this.chatRunning}
          ?inert=${shellModal}></dl-notification-offer>
        <dl-settings-dialog
          .activeHtmlPreviewEnabled=${bootstrap.active_html_preview_enabled}
          .deleteAllConversations=${this.#requestDeleteAllConversations}
        ></dl-settings-dialog>
        ${this.#dialogs()}
      </div>
      <dl-image-lightbox id="image-lightbox"
        @dl-image-lightbox-state-change=${this.#lightboxStateChanged}></dl-image-lightbox>
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
    const settings = this.querySelector<DlSettingsDialog>('dl-settings-dialog');
    void settings?.open(document.activeElement instanceof HTMLElement ? document.activeElement : null);
  }

  #requestDeleteAllConversations = async (
    returnFocus?: HTMLElement | null,
  ): Promise<boolean> => {
    return await this.#conversationSidebar()?.deleteAll(returnFocus) ?? false;
  };

  #memoryOperation(event: CustomEvent<ChatMemoryOperationDetail>): void {
    this.querySelector<DlSettingsDialog>('dl-settings-dialog')
      ?.handleMemoryOperation(event.detail);
  }

  #toast(): DlToastRegion | null {
    return this.querySelector<DlToastRegion>('dl-toast-region');
  }

  #toastRequested(event: CustomEvent<ToastRequestDetail>): void {
    const toast = this.#toast();
    if (!toast) return;
    if (event.detail.action) toast.showAction(event.detail.message, event.detail.action);
    else toast.show(event.detail.message, event.detail.duration);
  }

  #runningChanged(event: CustomEvent<ChatRunningChangeDetail>): void {
    this.chatRunning = event.detail.active;
  }

  #modalStateChanged(event: CustomEvent<ModalStateDetail>): void {
    const owner = event.target;
    if (!(owner instanceof HTMLElement)) return;
    if (event.detail.open) this.#nativeModalOwners.add(owner);
    else this.#nativeModalOwners.delete(owner);
    this.nativeModalOpen = this.#nativeModalOwners.size > 0;
  }

  #inspector(): DlInspector | null {
    return this.querySelector<DlInspector>('dl-inspector');
  }

  #canvas(): DlArtifactCanvas | null {
    return this.querySelector<DlArtifactCanvas>('dl-artifact-canvas');
  }

  #conversationSidebarOpening(event: Event): void {
    const inspector = this.#inspector();
    if (!inspector?.open) return;
    if (inspector.hasActiveFileMutation) {
      event.preventDefault();
      this.#toast()?.show('Wait for the file change to finish before opening conversations.', 5000);
      return;
    }
    inspector.close();
    this.#canvas()?.close(false);
  }

  #inspectorOpening(): void {
    void this.#conversationSidebar()?.close(false);
  }

  #conversationSidebarStateChanged(
    event: CustomEvent<ConversationSidebarStateDetail>,
  ): void {
    this.conversationExpanded = event.detail.expanded;
    this.conversationCompact = event.detail.compact;
    this.#syncShellState();
  }

  #conversationRouteChanged(): void {
    this.#inspector()?.closeConversationContent();
    this.#canvas()?.close(false);
  }

  #inspectorStateChanged(event: CustomEvent<InspectorStateDetail>): void {
    this.inspectorOpen = event.detail.open;
    this.inspectorKind = event.detail.kind;
    this.inspectorCompact = event.detail.compact;
    this.#syncShellState();
  }

  #canvasStateChanged(event: CustomEvent<ArtifactCanvasStateDetail>): void {
    this.canvasOpen = event.detail.open;
    this.canvasModal = event.detail.modal;
    this.canvasOverlay = event.detail.overlay;
    this.#syncShellState();
  }

  #lightboxStateChanged(event: CustomEvent<{open: boolean}>): void {
    this.lightboxOpen = event.detail.open;
  }

  #syncShellState(): void {
    document.body.classList.toggle('conversation-sidebar-open', this.conversationExpanded);
    document.body.classList.toggle(
      'conversation-drawer-open',
      this.conversationExpanded && this.conversationCompact,
    );
    document.body.classList.toggle('panel-open', this.inspectorOpen || this.canvasOpen);
    document.body.classList.toggle('files-panel-open', this.inspectorKind === 'files');
    document.body.classList.toggle('sources-panel-open', this.inspectorKind === 'sources');
    document.body.classList.toggle(
      'panel-drawer-open',
      this.inspectorOpen && this.inspectorCompact,
    );
    document.body.classList.toggle('artifact-canvas-open', this.canvasOpen);
    document.body.classList.toggle('artifact-canvas-overlay', this.canvasOverlay);
    document.body.classList.toggle('artifact-canvas-modal', this.canvasModal);
    syncPanelSplitState();
  }

  #chatBackgroundClick = (): void => {
    if (window.matchMedia(COMPACT_SHELL_MEDIA).matches) return;
    if (document.body.hasAttribute('data-resizing')) return;
    if (!this.inspectorOpen && !this.canvasOpen) return;
    this.#inspector()?.close();
    this.#canvas()?.close(false);
  };

  #openFiles = (event: Event): void => {
    event.preventDefault();
    this.#canvas()?.close(false);
    const trigger = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    void this.#inspector()?.openFiles(trigger);
  };

  #workspaceDrop(event: CustomEvent<ComposerWorkspaceDropDetail>): void {
    const returnFocus = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    void this.#inspector()?.uploadFiles(event.detail.files, event.detail.folderName, returnFocus);
  }

  #openArtifact(event: CustomEvent<{artifact: AnswerArtifact; returnFocus: HTMLElement}>): void {
    const canvas = this.#canvas();
    if (canvas) void canvas.open(event.detail.artifact, event.detail.returnFocus);
  }

  #openAnswerSource(event: CustomEvent<AnswerSourceOpenDetail>): void {
    const canvas = this.#canvas();
    const sourceWasInCanvas = Boolean(canvas?.contains(event.detail.returnFocus));
    const canvasReturnFocus = canvas?.prepareForInspector() ?? null;
    void this.#inspector()?.openSources(
      event.detail.presentation,
      event.detail.referenceId,
      event.detail.chunkId,
      sourceWasInCanvas ? canvasReturnFocus : event.detail.returnFocus,
    );
  }

  #openImage(event: CustomEvent<ImageOpenDetail>): void {
    const lightbox = this.querySelector<DlImageLightbox>('dl-image-lightbox');
    void lightbox?.open(event.detail.src, event.detail.returnFocus, event.detail.gallery);
  }

  #chatContentChanged(event: CustomEvent<ChatContentChangeDetail>): void {
    this.hasMessages = event.detail.hasMessages;
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
    this.#pendingContinuation = {kind: event.detail.action, runId: event.detail.runId};
    this.querySelector<DlContinuationDialog>('dl-continuation-dialog')
      ?.open(event.detail.action);
  }

  #continuationResult(event: CustomEvent<ContinuationResult>): void {
    const pending = this.#pendingContinuation;
    this.#pendingContinuation = null;
    if (!pending || !event.detail.query || event.detail.kind !== pending.kind) return;
    const chat = this.querySelector<DlChatFeature>('dl-chat-feature');
    if (chat) void chat.continueRun(pending.kind, pending.runId, event.detail.query);
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
      <dl-continuation-dialog
        @dl-continuation-result=${this.#continuationResult}></dl-continuation-dialog>
      <dl-children-roster></dl-children-roster>
    `;
  }
}

customElements.define('dl-app', DlApp);

declare global {
  interface HTMLElementTagNameMap {
    'dl-app': DlApp;
  }
}
