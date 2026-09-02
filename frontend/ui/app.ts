// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import {getWebBootstrap, type WebBootstrap} from '../api/bootstrap.ts';
import {getWorkspacesPage} from '../api/workspaces.ts';
import {icon} from '../design-system/index.ts';
import type {AnswerArtifact} from '../api/conversations.ts';
import {LightElement} from '../lib/lit-host.ts';
import {workspaceStore} from '../stores/workspace-store.ts';
import type {AttachmentPolicy} from '../lib/attachment-policy.ts';
import type {
  ArtifactCanvasStateDetail,
  DlArtifactCanvas,
} from './artifact-canvas.ts';
import './artifact-canvas.ts';
import type {
  ConversationSidebarStateDetail,
  DlConversationSidebar,
} from './conversation-sidebar.ts';
import './conversation-sidebar.ts';
import type {AnswerSourceOpenDetail} from './answer-presentation.ts';
import type {ComposerWorkspaceDropDetail} from './chat-composer.ts';
import type {
  ChatContentChangeDetail,
  ChatMemoryOperationDetail,
  ChatRunActionDetail,
  ChatRunningChangeDetail,
  ChatViewActionDetail,
  DlChatFeature,
} from './chat-feature.ts';
import './chat-feature.ts';
import type {ImageOpenDetail} from './image-lightbox.ts';
import type {DlImageLightbox} from './image-lightbox.ts';
import './image-lightbox.ts';
import type {DlInspector, InspectorStateDetail} from './inspector.ts';
import './inspector.ts';
import type {ModalStateDetail} from './modal.ts';
import type {DlSettingsDialog} from './settings.ts';
import './settings.ts';
import {syncPanelSplitState} from './split-panel.ts';
import type {
  ContinuationResult,
  DlChildrenRoster,
  DlContinuationDialog,
} from './run-dialogs.ts';
import './run-dialogs.ts';
import './notifications.ts';
import './theme.ts';
import type {DlToastRegion, ToastRequestDetail} from './toast.ts';
import './toast.ts';
import './workspace-scope.ts';

const EMPTY_BOOTSTRAP: WebBootstrap = {
  contractVersion: 1,
  workspaces: [],
  primaryWorkspace: '',
  activeWorkspaces: [],
  workspacesNextCursor: null,
  knownWorkspaces: null,
  answerAttachments: {
    countLimit: 0,
    imageMaxBytes: 0,
    documentMaxBytes: 0,
    extensions: [],
    imageCapability: 'unknown',
    imageLimit: 0,
    accept: '',
  },
  activeHtmlPreviewEnabled: true,
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
    updateWhenLocaleChanges(this);
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
          displayName: workspace.displayName,
          embeddingModel: workspace.embeddingModel,
        })),
        bootstrap.activeWorkspaces,
        bootstrap.primaryWorkspace,
        (cursor, signal) => getWorkspacesPage(cursor, signal),
        bootstrap.workspacesNextCursor ?? null,
        bootstrap.knownWorkspaces ?? null,
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
    const attachments = bootstrap.answerAttachments;
    const ready = this.bootState === 'ready';
    const chatFeature = this.querySelector<DlChatFeature>('dl-chat-feature');
    const conversationModal = this.conversationExpanded && this.conversationCompact;
    const inspectorModal = this.inspectorOpen && this.inspectorCompact;
    const panelInteractionLocked = inspectorModal || this.canvasModal;
    const blockingShellModal = conversationModal || this.nativeModalOpen;
    const shellModal = blockingShellModal || panelInteractionLocked;
    return html`
      <div class="app${this.hasMessages ? ' has-messages' : ''}" id="app"
        @dl-artifact-open=${this.#openArtifact}
        @dl-answer-source-open=${this.#openAnswerSource}
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
        <dl-split-layout class="panel-split" id="panel-split" primary="end"
                         orientation="horizontal" size="0" min="320">
          <dl-split-layout class="panel-split" id="artifact-canvas-split" slot="start"
                           primary="end" orientation="horizontal" size="0" min="320">
            <div class="primary-shell" slot="start">
              <div class="app-shell">
                <header class="topbar" ?inert=${inspectorModal || this.canvasModal}>
                  <dl-conversation-sidebar .enabled=${ready} .chatFeature=${chatFeature}
                    .shellInert=${this.canvasModal}></dl-conversation-sidebar>
                  <span class="topbar-scope-label" ?inert=${shellModal}>${msg('Search in:', {id: 'app.searchIn'})}</span>
                  <dl-workspace-scope class="workspace-selector" id="workspace-selector"
                    ?inert=${shellModal}></dl-workspace-scope>
                  <div class="topbar-spacer" ?inert=${shellModal}></div>
                  <button class="topbar-btn" id="files-btn" type="button"
                          aria-label=${msg('Files', {id: 'app.files'})}
                          ?inert=${shellModal} @click=${this.#openFiles}>
                    ${icon('files', {size: 'sm', className: 'files-button-icon'})}
                    <span class="files-button-label">${msg('Files', {id: 'app.files'})}</span>
                  </button>
                  <dl-theme-control id="theme-control" ?inert=${shellModal}></dl-theme-control>
                </header>

                <dl-chat-feature .attachmentPolicy=${this.#attachmentPolicy()}
                  .attachmentAccept=${attachments.accept}
                  .interactionLocked=${panelInteractionLocked}
                  ?inert=${blockingShellModal}
                  @dl-chat-content-change=${this.#chatContentChanged}
                  @dl-chat-background-click=${this.#chatBackgroundClick}
                  @dl-chat-run-action=${this.#chatRunAction}></dl-chat-feature>
              </div>
            </div>
            <dl-artifact-canvas id="artifact-canvas" class="panel" slot="end"
              aria-label=${msg('Artifact canvas', {id: 'app.artifactCanvasLabel'})} aria-hidden="true"
              .activePreviewEnabled=${bootstrap.activeHtmlPreviewEnabled}
            ></dl-artifact-canvas>
          </dl-split-layout>
          <dl-inspector id="inspector" slot="end"
            .shellInert=${this.canvasModal}></dl-inspector>
        </dl-split-layout>

        <dl-toast-region class="toast" id="toast" role="status" aria-live="polite"
          aria-atomic="true"
          .shellInert=${shellModal || this.lightboxOpen}></dl-toast-region>
        <dl-notification-offer class="notify-offer" id="notify-offer" role="group"
          aria-label=${msg('Answer notifications', {id: 'app.answerNotificationsLabel'})} .running=${this.chatRunning}
          ?inert=${shellModal}></dl-notification-offer>
        <dl-settings-dialog
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
      this.#toast()?.show(
        msg('Wait for the file change to finish before opening conversations.', {
          id: 'app.waitForFileChange',
        }),
        3000,
      );
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
    const attachments = this.#bootstrap.answerAttachments;
    return {
      countLimit: attachments.countLimit,
      imageMaxBytes: attachments.imageMaxBytes,
      documentMaxBytes: attachments.documentMaxBytes,
      extensions: new Set(attachments.extensions),
      imageCapability: attachments.imageCapability,
      imageLimit: attachments.imageLimit,
    };
  }

  #chatRunAction(event: CustomEvent<ChatRunActionDetail>): void {
    const chat = this.querySelector<DlChatFeature>('dl-chat-feature');
    if (!chat) return;
    if (event.detail.action === 'children') {
      const runId = event.detail.runId;
      this.querySelector<DlChildrenRoster>('dl-children-roster')?.open(
        () => chat.loadRunChildren(runId),
        (cursor, signal) => chat.loadRunChildrenPage(runId, cursor, signal),
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
      return html`<div class="bootstrap-status" role="status">
        ${msg('Loading DlightRAG…', {id: 'bootstrap.loading'})}
      </div>`;
    }
    return html`<div class="bootstrap-status bootstrap-status--error" role="alert">
      <span>${msg('DlightRAG could not load.', {id: 'bootstrap.error'})}</span>
      <button type="button" @click=${() => { void this.#load(); }}>
        ${msg('Retry', {id: 'bootstrap.retry'})}
      </button>
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
