// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {getWebBootstrap, type WebBootstrap} from '../api/bootstrap.ts';
import type {AnswerArtifact} from '../api/conversations.ts';
import type {MemoryOperationEvent} from '../api/memory.ts';
import {onSignedOut} from '../api/wire.ts';
import {getWorkspacesPage} from '../api/workspaces.ts';
import {icon} from '../design-system/index.ts';
import {focusedElement} from '../lib/dom.ts';
import {authRefusal} from '../lib/errors.ts';
import {LatestRequest} from '../lib/latest-request.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {localizedText} from '../lib/localized-text.ts';
import '../styles/layout.css';
import type {AttachmentPolicy} from '../lib/attachment-policy.ts';
import {type AppHandles, productionHandles } from '../stores/app-handles.ts';
import type {RunActivityStore} from '../stores/run-activity-store.ts';
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
  ChatRunActivityDetail,
  ChatRunningChangeDetail,
  DlChatFeature,
} from './chat-feature.ts';
import './chat-feature.ts';
import type {ChatRunActionDetail, ChatViewActionDetail} from './chat-message-list.ts';
import {forgetStoredDraft} from './chat-composer.ts';
import {ConversationRouteController} from './conversation-route-controller.ts';
import type {DlImageLightbox, ImageLightboxStateDetail, ImageOpenDetail} from './image-lightbox.ts';
import './image-lightbox.ts';
import type {DlInspector, InspectorKind, InspectorStateDetail} from './inspector.ts';
import type {AgentspaceRunOpenDetail} from './inspector-agentspace.ts';
import './inspector.ts';
import type {ModalStateDetail} from './modal.ts';
import type {DlSettingsDialog} from './settings.ts';
import type {SettingsRequestDetail} from './settings-summary.ts';
import './settings.ts';
import type {
  ContinuationResult,
  DlContinuationDialog,
} from './run-dialogs.ts';
import {IDLE_SHELL_SPLIT_STATE, syncPanelSplitState} from './split-panel.ts';
import './run-dialogs.ts';
import './notifications.ts';
import './run-notifications.ts';
import './theme.ts';
import {type DlToastRegion, type ToastRequestDetail, type ToastTone, toastRegionOf} from './toast.ts';
import './toast.ts';
import './workspace-scope.ts';

const EMPTY_BOOTSTRAP: WebBootstrap = {
  contractVersion: 6,
  workspaces: [],
  primaryWorkspace: '',
  activeWorkspaces: [],
  workspacesNextCursor: null,
  knownWorkspaces: null,
  defaultWorkspace: '',
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
  agentEffort: {levels: [], default: null},
  answerFeedbackEnabled: false,
  observability: null,
};

const PRODUCT_NAME = 'DlightRAG';

/** The classes the Shell keeps on <body> for the stylesheets, each on while the panel it names is open.
 *  They are written together and cleared together, and every one is read by a stylesheet: the splits are
 *  told what they need directly (`syncPanelSplitState`), so nothing reads these back. */
const SHELL_BODY_CLASSES = [
  'conversation-sidebar-open',
  'conversation-drawer-open',
  'files-panel-open',
  'sources-panel-open',
  'traces-panel-open',
  'artifact-canvas-open',
] as const;

/** Authenticated bootstrap, top-level capabilities, and Feature composition only. */
export class DlApp extends LightElement {
  static properties = {
    handles: {attribute: false},
    bootState: {state: true},
    signedOut: {state: true},
    hasMessages: {state: true},
    conversationExpanded: {state: true},
    conversationCompact: {state: true},
    inspectorOpen: {state: true},
    inspectorKind: {state: true},
    inspectorCompact: {state: true},
    canvasOpen: {state: true},
    canvasModal: {state: true},
    canvasOverlay: {state: true},
    canvasWide: {state: true},
    chatRunning: {state: true},
    lightboxOpen: {state: true},
    nativeModalOpen: {state: true},
  };

  declare handles: AppHandles;
  declare bootState: 'loading' | 'ready' | 'error';
  /** The server no longer accepts this page's sign-in; nothing else renders. */
  declare signedOut: boolean;
  declare hasMessages: boolean;
  declare conversationExpanded: boolean;
  declare conversationCompact: boolean;
  declare inspectorOpen: boolean;
  declare inspectorKind: InspectorKind | null;
  declare inspectorCompact: boolean;
  declare canvasOpen: boolean;
  declare canvasModal: boolean;
  declare canvasOverlay: boolean;
  declare canvasWide: boolean;
  declare chatRunning: boolean;
  declare lightboxOpen: boolean;
  declare nativeModalOpen: boolean;

  #bootstrap: WebBootstrap = EMPTY_BOOTSTRAP;
  #policy: {from: WebBootstrap['answerAttachments']; policy: AttachmentPolicy} | null = null;
  readonly #nativeModalOwners = new Set<HTMLElement>();
  readonly #boot = new LatestRequest();
  #stopHearingSignOut: (() => void) | null = null;
  /** What the page was called before the Shell named it for the conversation it shows. */
  #titleBefore: string | null = null;
  /** Which conversation is open and what the Chat is told of it. The Shell supplies what the controller
   *  cannot reach itself: the Features it asks the reader through, and the panels that go with a conversation. */
  readonly #routes = new ConversationRouteController(this, {
    handles: () => this.handles,
    chat: () => this.querySelector<DlChatFeature>('dl-chat-feature'),
    ui: {
      confirmDiscardDraft: async (next) => await this.#conversationSidebar()?.confirmDiscardDraft(next) ?? false,
      waitForAcceptedQuestion: () => {
        this.#toast('error')?.show({
          message: msg('Wait for the current question to be accepted.', {
            id: 'conversationSidebar.waitForAcceptedQuestion',
          }),
        });
      },
      conversationLeft: () => {
        this.#inspector()?.closeConversationContent();
        this.#canvas()?.close(false);
      },
      closeDrawer: async () => { await this.#conversationSidebar()?.close(false); },
    },
  });
  #pendingFork: string | null = null;
  readonly #ready: Promise<WebBootstrap>;
  #resolveReady!: (bootstrap: WebBootstrap) => void;
  #readyResolved = false;
  /** The Run activity store this shell follows and has started, and the way to stop following it. */
  #activityStore: RunActivityStore | null = null;
  #stopFollowingRunActivity: (() => void) | null = null;
  /** The conversation whose settled runs were last marked seen; a change of it is the reader's look. */
  #seenConversationId: string | null | undefined = undefined;

  constructor() {
    super();
    this.handles = productionHandles();
    this.bootState = 'loading';
    this.signedOut = false;
    this.hasMessages = false;
    this.conversationExpanded = false;
    this.conversationCompact = false;
    this.inspectorOpen = false;
    this.inspectorKind = null;
    this.inspectorCompact = false;
    this.canvasOpen = false;
    this.canvasModal = false;
    this.canvasOverlay = false;
    this.canvasWide = false;
    this.chatRunning = false;
    this.lightboxOpen = false;
    this.nativeModalOpen = false;
    this.#ready = new Promise((resolve) => { this.#resolveReady = resolve; });
    // The open conversation's title and the Chat's view of it follow the conversation store.
    new StoreController(this, () => [this.handles.conversations]);
  }

  get ready(): Promise<WebBootstrap> {
    return this.#ready;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.#titleBefore = document.title;
    this.#stopHearingSignOut ??= onSignedOut(() => {
      // What this reader sent, rated, and read goes with the sign-in; the unsent message is the composer's.
      this.handles.reset();
      forgetStoredDraft();
      this.#routes.stop();
      this.signedOut = true;
    });
    this.#followRunActivity();
    if (!this.#boot.busy && this.bootState !== 'ready') void this.#load();
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.#stopHearingSignOut?.();
    this.#stopHearingSignOut = null;
    this.#stopFollowingRunActivity?.();
    this.#stopFollowingRunActivity = null;
    this.#activityStore?.stop();
    this.#activityStore = null;
    this.#boot.cancel();
    this.#nativeModalOwners.clear();
    this.nativeModalOpen = false;
    document.body.classList.remove(...SHELL_BODY_CLASSES);
    syncPanelSplitState(IDLE_SHELL_SPLIT_STATE);
    if (this.#titleBefore !== null) document.title = this.#titleBefore;
    this.#titleBefore = null;
  }

  protected override updated(): void {
    const live = this.bootState === 'ready' && !this.signedOut;
    if (live) {
      this.#followRunActivity();
      this.#startRunActivity();
      this.#routes.start();
    } else {
      this.#activityStore?.stop();
    }
    this.#syncSeenConversation();
    this.#syncTitle();
  }

  /** Follow the bag's Run activity: a Run that settles in the conversation the reader has open is
   *  read, so it gets no unread marker. A bag handed in after construction is followed instead. */
  #followRunActivity(): void {
    const store = this.handles.runActivity;
    if (store === this.#activityStore) return;
    this.#stopFollowingRunActivity?.();
    this.#activityStore?.stop();
    this.#activityStore = store;
    this.#stopFollowingRunActivity = store.subscribe(() => {
      const active = this.handles.conversations.activeConversationId;
      if (active !== null) store.markConversationSeen(active);
    });
  }

  /** Start the store once, and only for the bag currently followed. */
  #startRunActivity(): void {
    if (this.#activityStore === null) return;
    if (this.#activityStore !== this.handles.runActivity) return;
    this.#activityStore.start();
  }

  /** Opening a conversation is reading its settled answers: take their unread marker away. */
  #syncSeenConversation(): void {
    const active = this.signedOut ? null : this.handles.conversations.activeConversationId;
    if (active === this.#seenConversationId) return;
    this.#seenConversationId = active;
    if (active !== null) this.handles.runActivity.markConversationSeen(active);
  }

  async #load(): Promise<void> {
    const signal = this.#boot.start();
    this.bootState = 'loading';
    try {
      const bootstrap = await getWebBootstrap(signal);
      if (!this.#boot.isCurrent(signal)) return;
      this.#bootstrap = bootstrap;
      this.handles.workspaces.init(
        bootstrap.workspaces,
        bootstrap.activeWorkspaces,
        bootstrap.primaryWorkspace,
        (cursor, signal) => getWorkspacesPage(cursor, signal),
        bootstrap.workspacesNextCursor ?? null,
        bootstrap.knownWorkspaces ?? null,
        bootstrap.defaultWorkspace,
      );
      this.bootState = 'ready';
      await this.updateComplete;
      const callbackReturn = new URL(window.location.href);
      if (callbackReturn.searchParams.get('settings') === 'connections') {
        const settings = this.querySelector<DlSettingsDialog>('dl-settings-dialog');
        if (settings) await settings.open(null, 'connections');
        callbackReturn.searchParams.delete('settings');
        callbackReturn.searchParams.delete('authorization');
        window.history.replaceState(null, '', callbackReturn);
      }
      if (!this.#readyResolved) {
        this.#readyResolved = true;
        this.#resolveReady(bootstrap);
      }
    } catch {
      if (!this.#boot.isCurrent(signal)) return;
      this.bootState = 'error';
    } finally {
      this.#boot.finish(signal);
    }
  }

  /** The open conversation's title, if it has one: what the reader would call the page. */
  #conversationTitle(): string | null {
    const {conversations} = this.handles;
    const id = conversations.activeConversationId;
    if (id === null) return null;
    const summary = conversations.conversations.find((candidate) => candidate.conversationId === id);
    return summary?.title?.trim() || null;
  }

  /** Name the page for the conversation it shows, as the browser's tab and history should read it. */
  #syncTitle(): void {
    const title = this.signedOut ? null : this.#conversationTitle();
    document.title = title === null ? PRODUCT_NAME : `${title} · ${PRODUCT_NAME}`;
  }

  protected override render(): TemplateResult {
    if (this.signedOut) return this.#signedOutStatus();
    const bootstrap = this.#bootstrap;
    const attachments = bootstrap.answerAttachments;
    const ready = this.bootState === 'ready';
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
        @dl-profile-memory-change=${this.#profileMemoryChanged}
        @dl-chat-running-change=${this.#runningChanged}
        @dl-settings-request=${this.#settingsRequested}
        @dl-toast-request=${this.#toastRequested}
        @dl-answer-feedback-off=${this.#answerFeedbackOff}
        @dl-modal-state-change=${this.#modalStateChanged}
        @dl-conversation-sidebar-opening=${this.#conversationSidebarOpening}
        @dl-conversation-sidebar-state-change=${this.#conversationSidebarStateChanged}
        @dl-conversation-chosen=${this.#focusComposer}
        @dl-toast-focus-lost=${this.#focusComposer}
        @dl-inspector-opening=${this.#inspectorOpening}
        @dl-inspector-state-change=${this.#inspectorStateChanged}
        @dl-inspector-close-request=${this.#inspectorCloseRequested}
        @dl-artifact-canvas-state-change=${this.#canvasStateChanged}
        @dl-composer-workspace-drop=${this.#workspaceDrop}
        @dl-agentspace-open-run=${this.#openAgentspaceRun}
        aria-busy=${ready ? 'false' : 'true'}
        ?inert=${!ready || this.lightboxOpen}
      >
        <dl-split-layout class="panel-split" id="conversation-split" primary="start"
                         orientation="horizontal" size="0" min="240"
                         label=${msg('Resize conversations', {id: 'app.resizeConversations'})}>
          <dl-conversation-sidebar id="conversation-sidebar" slot="start"
            .handles=${this.handles} .enabled=${ready} .routes=${this.#routes}
            .shellInert=${this.canvasModal}></dl-conversation-sidebar>
          <dl-split-layout class="panel-split" id="panel-split" slot="end" primary="end"
                           orientation="horizontal" size="0" min="320"
                           label=${msg('Resize side panel', {id: 'app.resizeInspector'})}>
          <dl-split-layout class="panel-split" id="artifact-canvas-split" slot="start"
                           primary="end" orientation="horizontal" size="0" min="320"
                           label=${msg('Resize file preview', {id: 'app.resizeArtifactCanvas'})}>
            <div class="primary-shell" slot="start">
              <div class="app-shell">
                <h1 class="dl-sr-only">${this.#conversationTitle() ?? PRODUCT_NAME}</h1>
                <header class="topbar" ?inert=${inspectorModal || this.canvasModal}>
                  <button
                    id="conversation-sidebar-open"
                    type="button"
                    aria-label=${msg('Open conversations', {id: 'conversationSidebar.openConversations'})}
                    aria-controls="chat-sidebar"
                    aria-expanded=${this.conversationExpanded ? 'true' : 'false'}
                    aria-hidden=${conversationModal ? 'true' : nothing}
                    ?inert=${shellModal}
                    @click=${(event: MouseEvent) => {
                      void this.#conversationSidebar()?.open(event.currentTarget as HTMLElement);
                    }}
                  >
                    ${icon('panel-expand', {size: 'md'})}
                  </button>
                  <span class="topbar-scope-label" ?inert=${shellModal}>${msg('Search in:', {id: 'app.searchIn'})}</span>
                  <dl-workspace-scope class="workspace-selector" id="workspace-selector"
                    .handles=${this.handles} ?inert=${shellModal}></dl-workspace-scope>
                  <div class="topbar-spacer" ?inert=${shellModal}></div>
                  <button class="topbar-btn" id="agentspace-btn" type="button"
                          aria-label=${msg('Agentspace', {id: 'app.agentspace'})}
                          ?inert=${shellModal} @click=${this.#openAgentspace}>
                    ${icon('agentspace', {size: 'sm', className: 'files-button-icon'})}
                    <span class="files-button-label">${msg('Agentspace', {id: 'app.agentspace'})}</span>
                  </button>
                  <button class="topbar-btn" id="files-btn" type="button"
                          aria-label=${msg('Files', {id: 'app.files'})}
                          ?inert=${shellModal} @click=${this.#openFiles}>
                    ${icon('files', {size: 'sm', className: 'files-button-icon'})}
                    <span class="files-button-label">${msg('Files', {id: 'app.files'})}</span>
                  </button>
                  <dl-theme-control id="theme-control" ?inert=${shellModal}></dl-theme-control>
                </header>

                <dl-chat-feature .handles=${this.handles} .view=${this.#routes.view}
                  .attachmentPolicy=${this.#attachmentPolicy()}
                  .attachmentAccept=${attachments.accept}
                  .agentEffortOffer=${bootstrap.agentEffort}
                  .answerFeedbackEnabled=${bootstrap.answerFeedbackEnabled}
                  .interactionLocked=${panelInteractionLocked}
                  ?inert=${blockingShellModal || this.canvasWide}
                  @dl-chat-content-change=${this.#chatContentChanged}
                  @dl-chat-background-click=${this.#chatBackgroundClick}
                  @dl-chat-run-action=${this.#chatRunAction}
                  @dl-run-activity=${this.#runActivity}></dl-chat-feature>
              </div>
            </div>
            <dl-artifact-canvas id="artifact-canvas" class="panel" slot="end"
              aria-label=${msg('File preview', {id: 'app.artifactCanvasLabel'})} aria-hidden="true"
              .activePreviewEnabled=${bootstrap.activeHtmlPreviewEnabled}
            ></dl-artifact-canvas>
          </dl-split-layout>
          <dl-inspector id="inspector" slot="end"
            .handles=${this.handles} .shellInert=${this.canvasModal}
            .answerFeedbackEnabled=${bootstrap.answerFeedbackEnabled}></dl-inspector>
          </dl-split-layout>
        </dl-split-layout>

        <dl-toast-region class="toast"
          .shellInert=${shellModal || this.lightboxOpen}></dl-toast-region>
        <dl-toast-region class="toast" tone="error"
          .shellInert=${shellModal || this.lightboxOpen}></dl-toast-region>
        <dl-notification-offer class="notify-offer" id="notify-offer" role="group"
          aria-label=${msg('Answer notifications', {id: 'app.answerNotificationsLabel'})} .running=${this.chatRunning}
          ?inert=${shellModal}></dl-notification-offer>
        <dl-run-notifications .handles=${this.handles} ?enabled=${ready}></dl-run-notifications>
        <dl-settings-dialog
          .handles=${this.handles}
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
    this.#routes.chatViewAction(event.detail.action);
  }

  /** The reader chose where to be and the drawer is put away, or dismissed a notice that held focus and has
   *  nowhere to give it back: the Chat's box is theirs to type in. */
  #focusComposer(): void {
    this.querySelector<DlChatFeature>('dl-chat-feature')?.focusComposer();
  }

  #settingsRequested(event: CustomEvent<SettingsRequestDetail | undefined>): void {
    const settings = this.querySelector<DlSettingsDialog>('dl-settings-dialog');
    void settings?.open(focusedElement(), event.detail?.page);
  }

  #requestDeleteAllConversations = async (
    returnFocus?: HTMLElement | null,
  ): Promise<boolean> => {
    return await this.#conversationSidebar()?.deleteAll(returnFocus) ?? false;
  };

  /** The owner changed their Profile Memory in Settings: the notices under their answers read it again. */
  #profileMemoryChanged(): void {
    this.handles.memoryNotices.refresh();
  }

  #memoryOperation(event: CustomEvent<MemoryOperationEvent>): void {
    this.querySelector<DlSettingsDialog>('dl-settings-dialog')
      ?.handleMemoryOperation(event.detail);
  }

  /** The region that shows a notice of this tone. While Settings is open it holds the top layer and the
   *  shell's regions lie under its scrim, so a notice from anywhere shows in the dialog's own. */
  #toast(tone: ToastTone = 'receipt'): DlToastRegion | undefined {
    const settings = this.querySelector<DlSettingsDialog>('dl-settings-dialog');
    return toastRegionOf(settings?.openDialog ?? this, tone);
  }

  #toastRequested(event: CustomEvent<ToastRequestDetail>): void {
    this.#toast(event.detail.tone)?.show(event.detail);
  }

  /** A rating was refused because the operator has turned feedback off since this page loaded: the thumbs go
   *  everywhere, for as long as the page lives. */
  #answerFeedbackOff(): void {
    if (!this.#bootstrap.answerFeedbackEnabled) return;
    this.#bootstrap = {...this.#bootstrap, answerFeedbackEnabled: false};
    this.requestUpdate();
  }

  #runningChanged(event: CustomEvent<ChatRunningChangeDetail>): void {
    this.chatRunning = event.detail.active;
    // A submission started running: the activity store may be asleep, and this is its attention
    // signal. It reads once, and watches again only if it finds the Run under way.
    if (event.detail.active) this.handles.runActivity.wake();
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
      this.#toast('error')?.show({
        message: msg('Wait for the file change to finish before opening conversations.', {
          id: 'app.waitForFileChange',
        }),
      });
      return;
    }
    inspector.close();
    this.#canvas()?.close(false);
  }

  #inspectorOpening(): void {
    void this.#conversationSidebar()?.close(false);
  }

  /** A trace asked the Inspector to give the page back: below the wide shell the answer it points
   *  at is behind the dialog, and the reader asked for the answer. */
  #inspectorCloseRequested(): void {
    this.#inspector()?.close(false);
  }

  #conversationSidebarStateChanged(
    event: CustomEvent<ConversationSidebarStateDetail>,
  ): void {
    this.conversationExpanded = event.detail.expanded;
    this.conversationCompact = event.detail.compact;
    this.#syncShellState();
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
    this.canvasWide = event.detail.wide;
    this.#syncShellState();
  }

  #lightboxStateChanged(event: CustomEvent<ImageLightboxStateDetail>): void {
    this.lightboxOpen = event.detail.open;
  }

  #syncShellState(): void {
    const on: Record<(typeof SHELL_BODY_CLASSES)[number], boolean> = {
      'conversation-sidebar-open': this.conversationExpanded,
      'conversation-drawer-open': this.conversationExpanded && this.conversationCompact,
      'files-panel-open': this.inspectorKind === 'files',
      'sources-panel-open': this.inspectorKind === 'sources',
      'traces-panel-open': this.inspectorKind === 'traces',
      'artifact-canvas-open': this.canvasOpen,
    };
    for (const name of SHELL_BODY_CLASSES) document.body.classList.toggle(name, on[name]);
    syncPanelSplitState({
      conversationOpen: this.conversationExpanded,
      canvasOverlay: this.canvasOverlay,
      canvasWide: this.canvasWide,
      inspectorKind: this.inspectorKind,
    });
  }

  #chatBackgroundClick = (): void => {
    if (document.body.hasAttribute('data-resizing')) return;
    if (!this.inspectorOpen && !this.canvasOpen) return;
    // Child agents are watched beside the chat, so a click in the chat leaves them open. A compact shell
    // shows them as a modal drawer over the chat, where the click is on the scrim and closes them as it
    // closes Files and Sources.
    const monitoring = this.inspectorKind === 'traces' || this.inspectorKind === 'live';
    if (!monitoring || this.inspectorCompact) this.#inspector()?.close();
    this.#canvas()?.close(false);
  };

  #openAgentspace = (event: Event): void => {
    event.preventDefault();
    this.#canvas()?.close(false);
    const trigger = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    void this.#inspector()?.openAgentspace(trigger);
  };

  #openFiles = (event: Event): void => {
    event.preventDefault();
    this.#canvas()?.close(false);
    const trigger = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    void this.#inspector()?.openFiles(trigger);
  };

  #workspaceDrop(event: CustomEvent<ComposerWorkspaceDropDetail>): void {
    void this.#inspector()?.uploadFiles(event.detail.files, event.detail.folderName, focusedElement());
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

  /** The policy the bootstrap states, built once for it: the Shell draws again for every change of the
   *  conversation store, and a policy made anew each time would draw the Chat and its composer again too. */
  #attachmentPolicy(): AttachmentPolicy {
    const attachments = this.#bootstrap.answerAttachments;
    if (this.#policy?.from !== attachments) {
      this.#policy = {
        from: attachments,
        policy: {
          countLimit: attachments.countLimit,
          imageMaxBytes: attachments.imageMaxBytes,
          documentMaxBytes: attachments.documentMaxBytes,
          extensions: new Set(attachments.extensions),
          imageCapability: attachments.imageCapability,
          imageLimit: attachments.imageLimit,
        },
      };
    }
    return this.#policy.policy;
  }

  /** A saved copy names its Run: open that Run in the traces dock, where what it changed is read. */
  #openAgentspaceRun = (event: CustomEvent<AgentspaceRunOpenDetail>): void => {
    const chat = this.querySelector<DlChatFeature>('dl-chat-feature');
    if (!chat) return;
    const runId = event.detail.runId;
    const trigger = focusedElement();
    this.#canvas()?.close(false);
    void this.#inspector()?.openTraces(
      this.handles.traces.sourceFor(runId, () => chat.tracesSource(runId)),
      trigger,
    );
  };

  #chatRunAction(event: CustomEvent<ChatRunActionDetail>): void {
    const chat = this.querySelector<DlChatFeature>('dl-chat-feature');
    if (!chat) return;
    if (event.detail.action === 'live') {
      // Watching a Run's browser is a monitor, not a lookup: the chat stays where it is.
      const trigger = focusedElement();
      this.#canvas()?.close(false);
      void this.#inspector()?.openLive(event.detail.runId, trigger);
      return;
    }
    if (event.detail.action === 'traces') {
      const runId = event.detail.runId;
      const trigger = focusedElement();
      this.#canvas()?.close(false);
      void this.#inspector()?.openTraces(
        this.handles.traces.sourceFor(runId, () => chat.tracesSource(runId)),
        trigger,
      );
      return;
    }
    this.#pendingFork = event.detail.runId;
    this.querySelector<DlContinuationDialog>('dl-continuation-dialog')?.open();
  }

  #runActivity(event: CustomEvent<ChatRunActivityDetail>): void {
    this.#inspector()?.applyTracesEvents(event.detail.runId, event.detail.events);
  }

  #continuationResult(event: CustomEvent<ContinuationResult>): void {
    const runId = this.#pendingFork;
    this.#pendingFork = null;
    if (!runId || !event.detail.query) return;
    const chat = this.querySelector<DlChatFeature>('dl-chat-feature');
    if (chat) void chat.forkRun(runId, event.detail.query);
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

  /** The way back in once the server stops accepting this page's sign-in.

   * No Feature can do anything without it, so the shell stops rendering them:
   * a request that failed with the same 401 has nowhere left to report a
   * failure of its own. Signing in returns the reader to this page.
   */
  #signedOutStatus(): TemplateResult {
    const here = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    return html`<div class="bootstrap-status" role="alert">
      <span>${localizedText(authRefusal(401))}</span>
      <a class="dl-btn" href=${`/web/login?next=${encodeURIComponent(here)}`}>
        ${msg('Sign in', {id: 'app.signIn'})}
      </a>
    </div>`;
  }

  #dialogs(): TemplateResult {
    return html`
      <dl-continuation-dialog
        @dl-continuation-result=${this.#continuationResult}></dl-continuation-dialog>
    `;
  }
}

customElements.define('dl-app', DlApp);

declare global {
  interface HTMLElementTagNameMap {
    'dl-app': DlApp;
  }
}
