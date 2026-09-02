// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {html, type PropertyValues, type TemplateResult} from 'lit';
import {waitFor} from 'xstate';
import {BrowserAnswerSubmissionAdapter} from '../api/answerSubmission.ts';
import {
  continueAnswerRun,
  getAnswerRunChildren,
  getAnswerRunChildrenPage,
  steerAnswerRun,
  type ConversationAttachmentReference,
  type ConversationTurn,
} from '../api/conversations.ts';
import {localizedRunErrorPayload} from '../lib/run_errors.ts';
import {applyAnswerEvent} from '../lib/turn_projection.ts';
import {conversationRoute} from '../lib/router.ts';
import {
  RunController,
  type AnswerRunEvent,
  type FollowResult,
} from '../lib/run_controller.ts';
import {LightElement} from '../lib/lit_host.ts';
import {answerEventCursorStore} from '../stores/answerEventCursorStore.ts';
import {attachmentStore} from '../stores/attachmentStore.ts';
import {AnswerSubmissionController} from '../stores/answerSubmissionController.ts';
import {
  answerSubmissionSnapshot,
  type AnswerSubmissionActor,
} from '../stores/answerSubmissionMachine.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import type {AttachmentPolicy} from '../lib/attachment_policy.ts';
import type {
  AnswerMode,
  ComposerSteerDetail,
  ComposerSubmitDetail,
  DlChatComposer,
} from './chat_composer.ts';
import './chat_composer.ts';
import {
  ANSWER_RECONNECT_COPY,
  MAX_STEERING_MESSAGES,
  answerReconnectState,
  storedTurnView,
  type ChatReconnectDetail,
  type ChatToolTraceToggleDetail,
  type ChatView,
} from './chat_message_list.ts';
import './chat_message_list.ts';
import type {ChatTurnView} from '../lib/chat_views.ts';
import type {ToastRequestDetail} from './toast.ts';
import {webRouter} from './router.ts';

export type {ChatRunActionDetail, ChatView, ChatViewActionDetail} from './chat_message_list.ts';

function terminalTurn(turn: ChatTurnView): boolean {
  return turn.state === 'succeeded' || turn.state === 'failed' || turn.state === 'cancelled';
}

export interface ChatRunningChangeDetail {
  active: boolean;
}

export interface ChatContentChangeDetail {
  hasMessages: boolean;
}

export interface ChatMemoryOperationDetail {
  body?: string;
  change_id?: string | null;
  intent_id?: string;
  live?: boolean;
  operation?: 'remember' | 'forget' | 'undo';
  outcome?: 'changed' | 'unchanged' | 'rejected' | 'conflict';
}

function isMemoryOperation(value: unknown): value is ChatMemoryOperationDetail {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function loginHref(): string {
  const next = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  return `/web/login?next=${encodeURIComponent(next)}`;
}

function optimisticTurn(
  submissionId: string,
  query: string,
  attachments: readonly ConversationAttachmentReference[],
): ChatTurnView {
  return {
    id: `local-${submissionId}`,
    userText: query,
    userAttachments: attachments,
    runId: '',
    state: 'pending',
    streamText: '',
    presentation: null,
    usage: {},
    evidence: {},
    error: '',
    progress: '',
    liveStatus: '',
    sawChildren: false,
    cancelRequested: false,
    steeringMessages: [],
    toolRows: [],
    toolTotal: 0,
    toolExpanded: false,
  };
}

/** Chat composition root: owns submission, following, replay, and run intent. */
export class DlChatFeature extends LightElement {
  static properties = {
    view: {attribute: false},
    attachmentPolicy: {attribute: false},
    attachmentAccept: {type: String},
    interactionLocked: {attribute: false},
    turns: {state: true},
    runRevision: {state: true},
  };

  declare view: ChatView;
  declare attachmentPolicy: AttachmentPolicy | null;
  declare attachmentAccept: string;
  declare interactionLocked: boolean;
  declare turns: readonly ChatTurnView[];
  declare runRevision: number;

  readonly #runController: RunController;
  readonly #submissionController = new AnswerSubmissionController(this);
  readonly #submissionAdapter = new BrowserAnswerSubmissionAdapter();
  #submissionActor: AnswerSubmissionActor | null = null;
  #submissionTurnId: string | null = null;
  #continuationController: AbortController | null = null;
  #pendingResume: {conversationId: string; stored: ConversationTurn} | null = null;
  #scrollRequest = 0;
  #announcedActive = false;
  #announcedHasMessages: boolean | null = null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.view = {kind: 'new'};
    this.attachmentPolicy = null;
    this.attachmentAccept = '';
    this.interactionLocked = false;
    this.turns = [];
    this.runRevision = 0;
    this.#runController = new RunController({
      onStateChange: () => this.#runStateChanged(),
    });
  }

  get submissionPending(): boolean {
    return this.#submissionController.snapshots.some(
      ({status}) => status === 'submitting' || status === 'reconciling',
    );
  }

  get hasUnresolvedSubmission(): boolean {
    return this.#submissionController.snapshots.length > 0;
  }

  get hasDraft(): boolean {
    return this.#composer()?.hasDraft ?? attachmentStore.size > 0;
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this.#abortContinuation();
    this.#runController.disconnect();
  }

  detachRun(): void {
    this.#abortContinuation();
    this.#runController.detach();
  }

  clearDraft(): void {
    this.#composer()?.clearDraft();
    if (!this.#composer()) attachmentStore.clear();
  }

  focusComposer(): void {
    this.#composer()?.focusInput();
  }

  async loadRunChildren(runId: string) {
    try {
      return await getAnswerRunChildren(runId);
    } catch {
      return [];
    }
  }

  async loadRunChildrenPage(runId: string, cursor: string | null, signal?: AbortSignal) {
    return getAnswerRunChildrenPage(runId, cursor, signal);
  }

  async continueRun(
    kind: 'follow-up' | 'fork',
    runId: string,
    query: string,
  ): Promise<void> {
    this.#abortContinuation();
    const controller = new AbortController();
    this.#continuationController = controller;
    try {
      const descriptor = await continueAnswerRun(
        runId,
        kind,
        query,
        crypto.randomUUID(),
        controller.signal,
      );
      if (controller.signal.aborted || this.#continuationController !== controller) return;
      conversationStore.upsertSummary(descriptor.conversation);
      if (kind === 'fork') {
        await webRouter.navigate(conversationRoute(descriptor.conversation.conversation_id));
      } else {
        await conversationStore.open(descriptor.conversation.conversation_id, {
          showLoading: false,
          preserveOnError: true,
        });
      }
    } catch {
      if (!controller.signal.aborted && this.#continuationController === controller) {
        this.#requestToast({
          message: msg('The continuation could not be started.', {id: 'chatFeature.continuationFailed'}),
          duration: 3000,
        });
      }
    } finally {
      if (this.#continuationController === controller) this.#continuationController = null;
    }
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (!changed.has('view')) return;
    this.#pendingResume = null;
    const routeConversationId = this.view.kind === 'ready' ? this.view.conversationId : null;
    this.#submissionActor = this.#submissionController.actor(routeConversationId);
    this.#submissionTurnId = this.#submissionActor
      ? `local-${answerSubmissionSnapshot(this.#submissionActor).submissionId}`
      : null;
    if (this.view.kind === 'ready') {
      const stored = this.view.history.map(storedTurnView);
      const previousView = changed.get('view') as ChatView | undefined;
      if (
        previousView?.kind === 'ready'
        && previousView.conversationId === this.view.conversationId
      ) {
        const currentById = new Map(this.turns.map((turn) => [turn.id, turn]));
        const currentByRunId = new Map(
          this.turns.filter((turn) => turn.runId).map((turn) => [turn.runId, turn]),
        );
        const mergedStored = stored.map((turn) => {
          const current = currentById.get(turn.id) ?? currentByRunId.get(turn.runId);
          return current && !terminalTurn(current) && !terminalTurn(turn) ? current : turn;
        });
        const storedIds = new Set(stored.map((turn) => turn.id));
        const storedRunIds = new Set(stored.map((turn) => turn.runId).filter(Boolean));
        const live = this.turns.filter((turn) => (
          !terminalTurn(turn)
          && !storedIds.has(turn.id)
          && (!turn.runId || !storedRunIds.has(turn.runId))
        ));
        this.turns = [...mergedStored, ...live];
      } else {
        this.turns = stored;
      }
      const pending = [...this.view.history].reverse().find(
        (turn) => turn.status === 'queued' || turn.status === 'running',
      );
      if (pending) {
        this.#pendingResume = {
          conversationId: this.view.conversationId,
          stored: pending,
        };
      }
    } else {
      this.turns = [];
    }
    const submissionTurn = this.#submissionTurnForRoute();
    if (submissionTurn && !this.turns.some((turn) => turn.id === submissionTurn.id)) {
      this.turns = [...this.turns, submissionTurn];
    }
  }

  protected override updated(changed: PropertyValues<this>): void {
    const hasMessages = this.turns.length > 0;
    if (hasMessages !== this.#announcedHasMessages) {
      this.#announcedHasMessages = hasMessages;
      this.dispatchEvent(new CustomEvent<ChatContentChangeDetail>('dl-chat-content-change', {
        bubbles: true,
        composed: true,
        detail: {hasMessages},
      }));
    }
    if (!changed.has('view') || !this.#pendingResume) return;
    const pending = this.#pendingResume;
    this.#pendingResume = null;
    queueMicrotask(() => {
      if (!this.isConnected || this.view.kind !== 'ready'
          || this.view.conversationId !== pending.conversationId) return;
      const conversationId = conversationStore.activeConversationId || pending.conversationId;
      if (conversationId) void this.#resumeStoredTurn(conversationId, pending.stored);
    });
  }

  protected override render(): TemplateResult {
    void this.runRevision;
    return html`
      <dl-chat-message-list .view=${this.view} .turns=${this.turns}
        .scrollRequest=${this.#scrollRequest}
        .interactionLocked=${this.interactionLocked}
        @dl-chat-reconnect=${this.#reconnect}
        @dl-chat-tool-trace-toggle=${this.#toggleToolTrace}
        @dl-chat-load-older=${this.#loadOlderMessages}></dl-chat-message-list>
      ${this.#submissionFailureControls()}
      <dl-chat-composer
        ?inert=${this.interactionLocked}
        .running=${this.#runController.active}
        .submissionPending=${this.submissionPending}
        .stopping=${this.#runController.stopping}
        .attachmentPolicy=${this.attachmentPolicy}
        .attachmentAccept=${this.attachmentAccept}
        @dl-composer-submit=${this.#submit}
        @dl-composer-steer=${this.#steer}
        @dl-composer-cancel=${this.#cancel}
      ></dl-chat-composer>
    `;
  }

  #submissionFailureControls(): TemplateResult {
    if (!this.#submissionActor) return html``;
    const snapshot = answerSubmissionSnapshot(this.#submissionActor);
    if (!['editable', 'retryable', 'conflict', 'login'].includes(snapshot.status)) return html``;
    return html`
      <div role="alert" class="submission-failure">
        <span>${snapshot.error?.message || msg('The answer could not be submitted.', {id: 'chatFeature.submissionFailed'})}</span>
        ${snapshot.status === 'login' ? html`
          <a class="dl-btn" href=${loginHref()} @click=${this.#loginSubmission}>
            ${msg('Sign in', {id: 'chatFeature.submissionSignIn'})}
          </a>
        ` : html`
          ${snapshot.status === 'retryable' ? html`
            <button class="dl-btn" type="button" @click=${this.#retrySubmission}>
              ${msg('Retry', {id: 'chatFeature.submissionRetry'})}
            </button>
          ` : null}
          <button class="dl-btn" type="button" @click=${this.#editSubmission}>
            ${msg('Edit', {id: 'chatFeature.submissionEdit'})}
          </button>
        `}
        <button class="dl-btn" type="button" @click=${this.#discardSubmission}>
          ${msg('Discard', {id: 'chatFeature.submissionDiscard'})}
        </button>
      </div>
    `;
  }

  #loadOlderMessages = (event: Event): void => {
    event.stopPropagation();
    void conversationStore.loadOlderMessages();
  };

  #abortContinuation(): void {
    this.#continuationController?.abort();
    this.#continuationController = null;
  }

  #submissionTurnForRoute(): ChatTurnView | null {
    const actor = this.#submissionActor;
    if (!actor) return null;
    const snapshot = answerSubmissionSnapshot(actor);
    if (['accepted', 'handedOff', 'edited', 'discarded'].includes(snapshot.status)) return null;
    const {intent, lease} = actor.getSnapshot().context;
    const attachments: ConversationAttachmentReference[] = lease.items.map((item, index) => ({
      attachment_id: item.id,
      ordinal: index + 1,
      kind: item.kind,
      filename: item.file.name,
      mime_type: item.file.type,
      byte_size: item.file.size,
      url: item.objectUrl,
      thumbnail_url: item.objectUrl,
      label: item.file.name,
    }));
    const turn = optimisticTurn(intent.submissionId, intent.query, attachments);
    if (snapshot.status === 'submitting' || snapshot.status === 'reconciling') return turn;
    return {
      ...turn,
      state: 'failed',
      error: snapshot.error?.message || msg('The answer could not be submitted.', {
        id: 'chatFeature.submissionFailed',
      }),
    };
  }

  #runStateChanged(): void {
    this.runRevision += 1;
    const active = this.#runController.active || this.submissionPending;
    if (active === this.#announcedActive) return;
    this.#announcedActive = active;
    this.dispatchEvent(new CustomEvent<ChatRunningChangeDetail>('dl-chat-running-change', {
      bubbles: true,
      composed: true,
      detail: {active},
    }));
  }

  #composer(): DlChatComposer | null {
    return this.querySelector<DlChatComposer>('dl-chat-composer');
  }

  #requestToast(detail: ToastRequestDetail): void {
    this.dispatchEvent(new CustomEvent<ToastRequestDetail>('dl-toast-request', {
      detail,
      bubbles: true,
      composed: true,
    }));
  }

  #submit = (event: CustomEvent<ComposerSubmitDetail>): void => {
    event.stopPropagation();
    void this.#submitQuery(event.detail.query, event.detail.mode, event.detail.requestedSkill);
  };

  #steer = (event: CustomEvent<ComposerSteerDetail>): void => {
    event.stopPropagation();
    void this.#steerRun(event.detail.query);
  };

  #cancel = (event: Event): void => {
    event.stopPropagation();
    void this.#runController.cancel();
  };

  #reconnect = (event: CustomEvent<ChatReconnectDetail>): void => {
    event.stopPropagation();
    const turn = this.turns.find((candidate) => candidate.runId === event.detail.runId);
    const conversationId = conversationStore.activeConversationId;
    if (!turn || !conversationId || this.#runController.active) return;
    this.#setTurn(turn.id, {
      state: 'pending',
      error: '',
      progress: turn.cancelRequested
        ? msg('Stopping...', {id: 'chatFeature.stopping'})
        : msg('Answer in progress...', {id: 'chatFeature.answerInProgress'}),
    });
    answerEventCursorStore.trackRun(conversationId, turn.runId);
    if (!this.#runController.beginFollow(turn.runId, turn.cancelRequested)) return;
    void this.#followTurn(turn.id, conversationId, turn.runId);
  };

  async #submitQuery(
    query: string,
    mode: AnswerMode | null,
    requestedSkill: string | null = null,
  ): Promise<void> {
    if (this.#runController.active || this.#submissionActor) return;
    if (!conversationStore.canAnswer) {
      this.#requestToast({
        message: msg('Conversation service is unavailable. Please retry loading the conversation.', {
          id: 'chatFeature.conversationUnavailable',
        }),
        duration: 3000,
      });
      return;
    }
    const conversationId = conversationStore.answerConversationId;
    const lease = attachmentStore.leaseAll();
    const liveAttachmentRefs: ConversationAttachmentReference[] = lease.items.map((item, index) => ({
      attachment_id: item.id,
      ordinal: index + 1,
      kind: item.kind,
      filename: item.file.name,
      mime_type: item.file.type,
      byte_size: item.file.size,
      url: item.objectUrl,
      thumbnail_url: item.objectUrl,
      label: item.file.name,
    }));
    const submissionId = crypto.randomUUID();
    const turn = optimisticTurn(submissionId, query, liveAttachmentRefs);
    const actor = this.#submissionController.start({
      query,
      mode,
      conversationId,
      submissionId,
      workspaces: [...workspaceStore.active],
      ...(requestedSkill ? {requestedSkill} : {}),
    }, lease, this.#submissionAdapter);
    if (!actor) {
      lease.restore();
      return;
    }
    this.#submissionActor = actor;
    this.#submissionTurnId = turn.id;
    actor.subscribe(() => {
      this.requestUpdate();
      if (answerSubmissionSnapshot(actor).status !== 'accepted') this.#runStateChanged();
    });
    this.#scrollRequest += 1;
    this.turns = [...this.turns, turn];
    this.#runStateChanged();
    await this.#observeSubmission(actor, turn.id, conversationId);
  }

  async #observeSubmission(
    actor: AnswerSubmissionActor,
    turnId: string,
    expectedConversationId: string | null,
  ): Promise<void> {
    try {
      await waitFor(actor, (snapshot) => (
        !snapshot.matches('submitting') && !snapshot.matches('reconciling')
      ));
    } catch {
      return;
    }
    if (this.#submissionActor !== actor) return;
    const snapshot = answerSubmissionSnapshot(actor);
    if (snapshot.status !== 'accepted' || !snapshot.accepted) {
      this.#setTurnError(
        turnId,
        snapshot.error?.message || msg('The answer could not be submitted.', {
          id: 'chatFeature.submissionFailed',
        }),
      );
      this.#runStateChanged();
      return;
    }
    const accepted = snapshot.accepted;
    const acceptedConversationId = accepted.conversation.conversation_id;
    if (expectedConversationId && acceptedConversationId !== expectedConversationId) {
      this.#setTurnError(turnId, msg('The answer was accepted for an unexpected conversation.', {
        id: 'chatFeature.unexpectedConversation',
      }));
      actor.send({type: 'HANDOFF'});
      this.#submissionActor = null;
      this.#runStateChanged();
      return;
    }
    this.#submissionActor = null;
    this.#submissionTurnId = null;
    if (!expectedConversationId) {
      conversationStore.adoptCreatedConversation(accepted.conversation);
      await webRouter.navigate(conversationRoute(acceptedConversationId), {
        replace: true,
        notify: false,
        bypassGuard: true,
      });
    } else {
      conversationStore.upsertSummary(accepted.conversation);
    }
    const stored = accepted.turn;
    answerEventCursorStore.trackRun(acceptedConversationId, stored.answer_run_id);
    this.#replaceStoredTurn(turnId, stored);
    const following = this.#runController.beginFollow(
      stored.answer_run_id,
      stored.cancel_requested,
    );
    actor.send({type: 'HANDOFF'});
    if (!following) {
      this.#runStateChanged();
      return;
    }
    this.#runStateChanged();
    await this.#followTurn(turnId, acceptedConversationId, stored.answer_run_id);
  }

  #retrySubmission = (): void => {
    const actor = this.#submissionActor;
    const turnId = this.#submissionTurnId;
    if (!actor || !turnId) return;
    actor.send({type: 'RETRY'});
    this.#setTurn(turnId, {state: 'pending', error: ''});
    void this.#observeSubmission(
      actor,
      turnId,
      answerSubmissionSnapshot(actor).conversationId,
    );
  };

  #editSubmission = (): void => this.#finishFailedSubmission('EDIT');
  #discardSubmission = (): void => this.#finishFailedSubmission('DISCARD');
  #loginSubmission = (): void => this.#finishFailedSubmission('DISCARD');

  #finishFailedSubmission(type: 'EDIT' | 'DISCARD'): void {
    const actor = this.#submissionActor;
    if (!actor) return;
    const intent = actor.getSnapshot().context.intent;
    actor.send({type});
    if (type === 'EDIT') {
      workspaceStore.restoreActive(intent.workspaces);
      this.querySelector<DlChatComposer>('dl-chat-composer')
        ?.restoreSubmission(intent.query, intent.mode, intent.requestedSkill ?? null);
    }
    if (this.#submissionTurnId) {
      this.turns = this.turns.filter((turn) => turn.id !== this.#submissionTurnId);
    }
    this.#submissionActor = null;
    this.#submissionTurnId = null;
    this.#runStateChanged();
    if (type === 'EDIT') this.focusComposer();
  }

  async #resumeStoredTurn(conversationId: string, stored: ConversationTurn): Promise<void> {
    if (this.#runController.active) return;
    answerEventCursorStore.trackRun(conversationId, stored.answer_run_id);
    if (!this.#runController.beginFollow(stored.answer_run_id, stored.cancel_requested)) return;
    await this.#followTurn(
      stored.turn_id || stored.answer_run_id,
      conversationId,
      stored.answer_run_id,
    );
  }

  async #followTurn(turnId: string, conversationId: string, runId: string): Promise<void> {
    let result: FollowResult;
    try {
      result = await this.#runController.follow(
        conversationId,
        runId,
        (events) => this.#handleRunBatch(turnId, events),
      );
    } catch {
      result = {
        kind: 'error',
        message: msg('Connection error. Please try again.', {id: 'chatFeature.connectionError'}),
      };
    }
    if (this.#runController.runId !== runId) return;

    let finished = false;
    if (result.kind === 'terminal') {
      if (result.stored) this.#replaceStoredTurn(turnId, result.stored);
      finished = true;
    } else if (result.kind === 'retryable') {
      const cancelRequested = result.stored.cancel_requested;
      const reconnectState = answerReconnectState(cancelRequested);
      this.#setTurn(turnId, {
        state: 'retryable',
        error: msg(ANSWER_RECONNECT_COPY[reconnectState].status, {
          id: `chatMessageList.reconnect.${reconnectState}.status`,
        }),
        progress: '',
        liveStatus: '',
        cancelRequested,
      });
    } else if (result.kind === 'error') {
      this.#setTurnError(turnId, result.message);
      finished = true;
    }
    this.#runController.finish(runId);
    if (!finished) return;
    answerEventCursorStore.clear(conversationId);
    void conversationStore.refreshActive();
  }

  async #steerRun(query: string): Promise<void> {
    const runId = this.#runController.runId;
    if (!runId) return;
    const signal = this.#runController.signalFor(runId);
    if (!signal) return;
    try {
      await steerAnswerRun(runId, query, signal);
    } catch {
      if (!signal.aborted && this.#runController.runId === runId) {
        this.#requestToast({
          message: msg('This run can no longer be steered.', {id: 'chatFeature.steerUnavailable'}),
          duration: 3000,
        });
      }
      return;
    }
    if (signal.aborted || this.#runController.runId !== runId) return;
    const turn = [...this.turns].reverse().find((candidate) => candidate.runId === runId);
    if (turn) {
      this.#setTurn(turn.id, {
        steeringMessages: [...turn.steeringMessages, query].slice(-MAX_STEERING_MESSAGES),
      });
    }
    const composer = this.#composer();
    if (composer?.clearSubmittedText(query)) composer.focusInput();
  }

  #handleRunBatch(turnId: string, events: readonly AnswerRunEvent[]): void {
    const turnIndex = this.turns.findIndex((candidate) => candidate.id === turnId);
    const turn = this.turns[turnIndex];
    for (const event of events) {
      if (event.kind === 'memory' && isMemoryOperation(event.payload)) {
        this.dispatchEvent(new CustomEvent<ChatMemoryOperationDetail>(
          'dl-chat-memory-operation',
          {bubbles: true, composed: true, detail: event.payload},
        ));
      }
    }
    if (!turn) return;
    const projected = events.reduce(applyAnswerEvent, turn);
    if (projected === turn) return;
    const nextTurns = [...this.turns];
    nextTurns[turnIndex] = projected;
    this.turns = nextTurns;
  }

  #replaceStoredTurn(turnId: string, stored: ConversationTurn): void {
    const replacement = storedTurnView(stored);
    this.turns = this.turns.map((turn) => turn.id === turnId
      ? {...replacement, id: turnId, steeringMessages: turn.steeringMessages}
      : turn);
  }

  #setTurnError(turnId: string, message: string): void {
    this.#setTurn(turnId, {state: 'failed', error: message, progress: '', liveStatus: message});
  }

  #setTurn(turnId: string, patch: Partial<ChatTurnView>): void {
    this.turns = this.turns.map((turn) => turn.id === turnId ? {...turn, ...patch} : turn);
  }

  #toggleToolTrace = (event: CustomEvent<ChatToolTraceToggleDetail>): void => {
    event.stopPropagation();
    const turn = this.turns.find((candidate) => candidate.runId === event.detail.runId);
    if (!turn) return;
    this.#setTurn(turn.id, {toolExpanded: !turn.toolExpanded});
  };
}

customElements.define('dl-chat-feature', DlChatFeature);

declare global {
  interface HTMLElementTagNameMap {
    'dl-chat-feature': DlChatFeature;
  }
}
