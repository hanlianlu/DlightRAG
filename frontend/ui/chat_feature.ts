// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, type PropertyValues, type TemplateResult} from 'lit';
import {csrfHeaders} from '../api/csrf.ts';
import {
  continueAnswerRun,
  getAnswerRunChildren,
  steerAnswerRun,
  type AnswerRunDescriptor,
  type ConversationAttachmentReference,
  type ConversationTurn,
} from '../api/conversations.ts';
import {buildAnswerRequest} from '../lib/answer_request.ts';
import {answerErrorMessage} from '../lib/errors.ts';
import {conversationRoute} from '../lib/router.ts';
import {RunController, type FollowResult} from '../lib/run_controller.ts';
import {parseData} from '../lib/sse.ts';
import {LightElement} from '../lib/lit_host.ts';
import {answerRunStore, payloadFingerprint} from '../stores/answerRunStore.ts';
import {attachmentStore} from '../stores/attachmentStore.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import type {AttachmentPolicy} from './attachment_policy.ts';
import type {
  AnswerMode,
  ComposerSteerDetail,
  ComposerSubmitDetail,
  DlChatComposer,
} from './chat_composer.ts';
import './chat_composer.ts';
import {
  ANSWER_RECONNECT_COPY,
  MAX_CHAT_TURNS,
  MAX_STEERING_MESSAGES,
  answerReconnectState,
  storedTurnView,
  type ChatReconnectDetail,
  type ChatTurnView,
  type ChatView,
} from './chat_message_list.ts';
import './chat_message_list.ts';
import {webRouter} from './router.ts';

export type {ChatRunActionDetail, ChatView, ChatViewActionDetail} from './chat_message_list.ts';

const NEW_CHAT_RUN_KEY = '__new_chat__';
type AnswerPhase = 'routing' | 'planning' | 'searching' | 'researching' | 'generating';
export const ANSWER_PHASE_LABELS = {
  routing: 'Routing answer...',
  planning: 'Planning answer...',
  searching: 'Searching knowledge base...',
  researching: 'Researching sources...',
  generating: 'Generating answer...',
} as const satisfies Record<AnswerPhase, string>;

export type ToolEventType = 'tool_start' | 'tool_progress' | 'tool_end';
export const ANSWER_TOOL_EVENT_LABELS = {
  tool_start: 'Tool started...',
  tool_progress: 'Tool working...',
  tool_end: 'Tool finished...',
} as const satisfies Record<ToolEventType, string>;

export function answerPhaseLabel(phase: string): string | null {
  if (!Object.hasOwn(ANSWER_PHASE_LABELS, phase)) return null;
  return ANSWER_PHASE_LABELS[phase as AnswerPhase];
}

export function answerToolEventLabel(eventType: string): string | null {
  if (!Object.hasOwn(ANSWER_TOOL_EVENT_LABELS, eventType)) return null;
  return ANSWER_TOOL_EVENT_LABELS[eventType as ToolEventType];
}

interface DonePayload {
  status: 'succeeded' | 'cancelled';
  presentation: ConversationTurn['presentation'];
  usage?: Record<string, unknown>;
  evidence?: Record<string, number>;
}

interface ToolProgressPayload {
  tool_name?: string;
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

function isDonePayload(value: unknown): value is DonePayload {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const payload = value as Record<string, unknown>;
  return payload.status === 'succeeded' || payload.status === 'cancelled';
}

function isMemoryOperation(value: unknown): value is ChatMemoryOperationDetail {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function optimisticTurn(
  query: string,
  attachments: readonly ConversationAttachmentReference[],
): ChatTurnView {
  return {
    id: `local-${crypto.randomUUID()}`,
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
  #continuationController: AbortController | null = null;
  #pendingResume: {conversationId: string; stored: ConversationTurn} | null = null;
  #scrollRequest = 0;
  #announcedActive = false;
  #announcedHasMessages: boolean | null = null;

  constructor() {
    super();
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
    return this.#runController.submissionPending;
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
        window.alert('The continuation could not be started.');
      }
    } finally {
      if (this.#continuationController === controller) this.#continuationController = null;
    }
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (!changed.has('view')) return;
    this.#pendingResume = null;
    if (this.view.kind === 'ready') {
      const visibleHistory = this.view.history.slice(-MAX_CHAT_TURNS);
      this.turns = visibleHistory.map(storedTurnView);
      const pending = [...visibleHistory].reverse().find(
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
        @dl-chat-reconnect=${this.#reconnect}></dl-chat-message-list>
      <dl-chat-composer
        ?inert=${this.interactionLocked}
        .running=${this.#runController.active}
        .stopping=${this.#runController.stopping}
        .attachmentPolicy=${this.attachmentPolicy}
        .attachmentAccept=${this.attachmentAccept}
        @dl-composer-submit=${this.#submit}
        @dl-composer-steer=${this.#steer}
        @dl-composer-cancel=${this.#cancel}
      ></dl-chat-composer>
    `;
  }

  #abortContinuation(): void {
    this.#continuationController?.abort();
    this.#continuationController = null;
  }

  #runStateChanged(): void {
    this.runRevision += 1;
    const active = this.#runController.active;
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

  #submit = (event: CustomEvent<ComposerSubmitDetail>): void => {
    event.stopPropagation();
    void this.#submitQuery(event.detail.query, event.detail.mode);
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
      progress: turn.cancelRequested ? 'Stopping...' : 'Answer in progress...',
    });
    answerRunStore.trackRun(conversationId, turn.runId);
    if (!this.#runController.beginFollow(turn.runId, turn.cancelRequested)) return;
    void this.#followTurn(turn.id, conversationId, turn.runId);
  };

  async #submitQuery(query: string, mode: AnswerMode | null): Promise<void> {
    const signal = this.#runController.beginSubmission();
    if (!signal) return;
    let conversationId = conversationStore.answerConversationId;
    const pendingAttachments = [...attachmentStore.list()];
    const liveAttachmentRefs: ConversationAttachmentReference[] = pendingAttachments.map(
      (item, index) => {
        const previewUrl = URL.createObjectURL(item.file);
        return {
          attachment_id: item.id,
          ordinal: index + 1,
          kind: item.kind,
          filename: item.file.name,
          mime_type: item.file.type,
          byte_size: item.file.size,
          url: previewUrl,
          thumbnail_url: previewUrl,
          label: item.file.name,
        };
      },
    );
    const turn = optimisticTurn(query, liveAttachmentRefs);
    this.#scrollRequest += 1;
    this.turns = [...this.turns, turn].slice(-MAX_CHAT_TURNS);

    try {
      if (!conversationStore.canAnswer) {
        this.#setTurnError(
          turn.id,
          'Conversation service is unavailable. Please retry loading the conversation.',
        );
        return;
      }
      const activeWorkspaces = [...workspaceStore.active];
      const fingerprint = await payloadFingerprint({
        query,
        attachments: pendingAttachments.map((item) => ({
          name: item.file.name,
          size: item.file.size,
          type: item.file.type,
        })),
        workspaces: activeWorkspaces,
      });
      if (conversationStore.answerConversationId !== conversationId) {
        this.#setTurnError(turn.id, 'The active conversation changed before this answer started.');
        return;
      }
      const runKey = conversationId ?? NEW_CHAT_RUN_KEY;
      const submissionId = answerRunStore.getOrCreateSubmissionId(runKey, fingerprint);
      const {body, headers} = buildAnswerRequest(
        {
          query,
          workspaces: activeWorkspaces,
          conversationId,
          submissionId,
          ...(mode ? {mode} : {}),
        },
        pendingAttachments.map((item) => item.file),
      );
      attachmentStore.clear();
      const response = await fetch('/web/api/answer', {
        method: 'POST',
        headers: {...csrfHeaders(), ...(headers ?? {})},
        body,
        signal,
      });
      if (!response.ok) {
        if (response.status < 500) answerRunStore.clear(runKey);
        this.#setTurnError(turn.id, 'Service error. Please try again.');
        return;
      }
      const descriptor = await response.json() as AnswerRunDescriptor;
      const acceptedConversationId = descriptor.conversation.conversation_id;
      if (conversationId && acceptedConversationId !== conversationId) {
        answerRunStore.clear(runKey);
        this.#setTurnError(turn.id, 'The answer was accepted for an unexpected conversation.');
        return;
      }
      answerRunStore.attachRun(runKey, descriptor.run_id);
      if (!conversationId) {
        answerRunStore.transfer(runKey, acceptedConversationId);
        conversationStore.adoptCreatedConversation(descriptor.conversation);
        conversationId = acceptedConversationId;
        await webRouter.navigate(conversationRoute(conversationId), {
          replace: true,
          notify: false,
          bypassGuard: true,
        });
      } else {
        conversationStore.upsertSummary(descriptor.conversation);
      }
      this.#setTurn(turn.id, {
        runId: descriptor.run_id,
        state: 'pending',
        progress: descriptor.cancel_requested ? 'Stopping...' : '',
        cancelRequested: descriptor.cancel_requested,
      });
      this.#runController.acceptSubmission(descriptor.run_id, descriptor.cancel_requested);
      await this.#followTurn(turn.id, conversationId, descriptor.run_id);
    } catch {
      if (!signal.aborted) this.#setTurnError(turn.id, 'Connection error. Please try again.');
    } finally {
      if (this.#runController.submissionPending) this.#runController.finish();
    }
  }

  async #resumeStoredTurn(conversationId: string, stored: ConversationTurn): Promise<void> {
    if (this.#runController.active) return;
    answerRunStore.trackRun(conversationId, stored.answer_run_id);
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
        (eventType, data) => this.#handleRunEvent(turnId, eventType, data),
      );
    } catch {
      result = {kind: 'error', message: 'Connection error. Please try again.'};
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
        error: ANSWER_RECONNECT_COPY[reconnectState].status,
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
    answerRunStore.clear(conversationId);
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
        window.alert('This run can no longer be steered.');
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

  #handleRunEvent(turnId: string, eventType: string, data: string): void {
    if (eventType === 'memory_operation_settled') {
      const operation = parseData(data);
      if (isMemoryOperation(operation)) {
        this.dispatchEvent(new CustomEvent<ChatMemoryOperationDetail>(
          'dl-chat-memory-operation',
          {bubbles: true, composed: true, detail: operation},
        ));
      }
      return;
    }
    const turn = this.turns.find((candidate) => candidate.id === turnId);
    if (!turn) return;
    if (eventType === 'token') {
      const parsed = parseData(data);
      const token = typeof parsed === 'string' ? parsed : String(parsed);
      this.#setTurn(turnId, {
        state: 'streaming',
        streamText: turn.streamText + token,
        progress: turn.streamText ? turn.progress : '',
        error: '',
      });
      return;
    }
    if (eventType === 'reset') {
      this.#setTurn(turnId, {state: 'pending', streamText: '', progress: '', error: ''});
      return;
    }
    if (eventType === 'progress') {
      const payload = parseData(data) as {phase?: string};
      const phase = String(payload?.phase || '');
      const label = answerPhaseLabel(phase);
      if (label === null) return;
      this.#setTurn(turnId, {progress: label, liveStatus: label, error: ''});
      return;
    }
    if (eventType === 'tool_start' || eventType === 'tool_progress' || eventType === 'tool_end') {
      this.#handleToolProgress(turnId, eventType, parseData(data) as ToolProgressPayload);
      return;
    }
    if (eventType === 'error') {
      this.#setTurnError(turnId, answerErrorMessage(parseData(data)));
      return;
    }
    if (eventType !== 'done') return;
    const payload = parseData(data);
    if (!isDonePayload(payload)) {
      this.#setTurnError(turnId, 'Service error. Please try again.');
      return;
    }
    if (payload.status === 'cancelled') {
      this.#setTurn(turnId, {state: 'cancelled', progress: '', liveStatus: 'Answer stopped'});
      return;
    }
    if (!payload.presentation) {
      this.#setTurnError(turnId, 'Service error. Please try again.');
      return;
    }
    this.#setTurn(turnId, {
      state: 'succeeded',
      presentation: payload.presentation,
      streamText: payload.presentation.answer_text,
      usage: payload.usage ?? {},
      evidence: payload.evidence ?? {},
      progress: '',
      liveStatus: 'Answer ready',
    });
  }

  #handleToolProgress(turnId: string, eventType: ToolEventType, info: ToolProgressPayload): void {
    if (!info || typeof info.tool_name !== 'string') return;
    const turn = this.turns.find((candidate) => candidate.id === turnId);
    if (!turn) return;
    const label = answerToolEventLabel(eventType);
    if (label === null) return;
    this.#setTurn(turnId, {
      progress: label,
      liveStatus: label,
      sawChildren: turn.sawChildren || info.tool_name === 'spawn_agent',
      error: '',
    });
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
}

customElements.define('dl-chat-feature', DlChatFeature);

declare global {
  interface HTMLElementTagNameMap {
    'dl-chat-feature': DlChatFeature;
  }
}
