// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {guard} from 'lit/directives/guard.js';
import {repeat} from 'lit/directives/repeat.js';
import type {
  ConversationAttachmentReference,
  ConversationTurn,
} from '../api/conversations.ts';
import type {ChatTurnView} from '../lib/chat-views.ts';
import {icon} from '../design-system/index.ts';
import {localizedStoredRunError} from '../lib/run-errors.ts';
import {formatFileSize} from '../lib/file-size.ts';
import {toolRowText} from '../lib/tool-events.ts';
import {
  TURN_PLACEHOLDER_MIN_PX,
  turnIsLive,
  visibleTurnWindow,
} from '../lib/turn-window.ts';
import {LightElement} from '../lib/lit-host.ts';
import {safeImageSrc, safeSameOriginHref} from '../lib/urls.ts';
import chatStyles from '../styles/chat.module.css';
import './answer-presentation.ts';
import type {ImageOpenDetail} from './image-lightbox.ts';

export type ChatView =
  | {kind: 'new'}
  | {kind: 'loading'}
  | {
      kind: 'ready';
      conversationId: string;
      history: readonly ConversationTurn[];
      lineage: string | null;
      hasOlderMessages?: boolean;
      olderMessagesState?: 'idle' | 'loading' | 'error';
    }
  | {kind: 'unavailable'; hasRecent: boolean}
  | {kind: 'error'};


export interface ChatRunActionDetail {
  action: 'children' | 'follow-up' | 'fork';
  runId: string;
}

export interface ChatReconnectDetail {
  runId: string;
}

export interface ChatToolTraceToggleDetail {
  runId: string;
}

function formatToolDuration(durationMs: number | null): string {
  if (durationMs === null) return '';
  if (durationMs < 1000) return `${Math.round(durationMs)}ms`;
  return `${(durationMs / 1000).toFixed(1)}s`;
}

export interface ChatViewActionDetail {
  action: 'retry' | 'new' | 'recent';
}

export const MAX_STEERING_MESSAGES = 50;
const STICK_TO_BOTTOM_PX = 160;

export type AnswerReconnectState = 'running' | 'stopping';

export const ANSWER_RECONNECT_COPY = {
  running: {
    status: 'Connection lost while this answer is running.',
    action: 'Reconnect',
  },
  stopping: {
    status: 'Connection lost while this answer is stopping.',
    action: 'Reconnect',
  },
} as const satisfies Record<AnswerReconnectState, {status: string; action: string}>;

export function answerReconnectState(cancelRequested: boolean): AnswerReconnectState {
  return cancelRequested ? 'stopping' : 'running';
}

function terminal(state: ChatTurnView['state']): boolean {
  return state === 'succeeded' || state === 'failed' || state === 'cancelled';
}

function newlyCompletedTurn(
  previous: readonly ChatTurnView[],
  current: readonly ChatTurnView[],
): string | null {
  const previousStates = new Map(previous.map((turn) => [turn.id, turn.state]));
  return [...current].reverse().find((turn) => {
    const previousState = previousStates.get(turn.id);
    return previousState !== undefined && !terminal(previousState) && terminal(turn.state);
  })?.id ?? null;
}

function liveObjectUrls(turns: readonly ChatTurnView[]): Set<string> {
  const urls = new Set<string>();
  for (const turn of turns) {
    for (const attachment of turn.userAttachments) {
      if (attachment.url.startsWith('blob:')) urls.add(attachment.url);
    }
  }
  return urls;
}

function liveImageIds(turns: readonly ChatTurnView[]): Set<string> {
  const ids = new Set<string>();
  for (const turn of turns) {
    for (const attachment of turn.userAttachments) {
      if (attachment.kind === 'image') ids.add(attachment.attachmentId);
    }
  }
  return ids;
}

/** Lit-owned conversation messages, streaming state, and message intent. */
export class DlChatMessageList extends LightElement {
  static properties = {
    view: {attribute: false},
    turns: {attribute: false},
    scrollRequest: {attribute: false},
    interactionLocked: {attribute: false},
  };

  declare view: ChatView;
  declare turns: readonly ChatTurnView[];
  declare scrollRequest: number;
  declare interactionLocked: boolean;
  #imageErrors = new Set<string>();
  #imageLoaded = new Set<string>();
  #imageRevision = 0;
  #stickAfterUpdate = true;
  #scrollFrame = 0;
  #pendingTurnAnchor: string | null = null;
  #pendingPrependAnchor: {turnId: string; offset: number} | null = null;
  #restoreOlderFocus = false;
  #olderAnnouncement = '';
  #range = {start: 0, end: -1};
  readonly #heights = new Map<string, number>();
  #scrollArea: HTMLElement | null = null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.view = {kind: 'new'};
    this.turns = [];
    this.scrollRequest = 0;
    this.interactionLocked = false;
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    if (this.#scrollFrame) cancelAnimationFrame(this.#scrollFrame);
    this.#unbindScroll();
    for (const url of liveObjectUrls(this.turns)) URL.revokeObjectURL(url);
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    const previousView = changed.get('view') as ChatView | undefined;
    if (
      previousView?.kind === 'ready'
      && (this.view.kind !== 'ready'
        || previousView.conversationId !== this.view.conversationId)
    ) {
      this.#pendingPrependAnchor = null;
      this.#restoreOlderFocus = false;
      this.#olderAnnouncement = '';
      this.#heights.clear();
      this.#range = {start: 0, end: Number.MAX_SAFE_INTEGER};
    }
    if (
      previousView?.kind === 'ready'
      && this.view.kind === 'ready'
      && previousView.conversationId === this.view.conversationId
    ) {
      const previousState = previousView.olderMessagesState ?? 'idle';
      const currentState = this.view.olderMessagesState ?? 'idle';
      if (currentState === 'loading') {
        this.#olderAnnouncement = msg('Loading older messages…', {id: 'chatMessageList.olderLoading'});
      } else if (currentState === 'error') {
        this.#olderAnnouncement = msg('Older messages could not be loaded.', {
          id: 'chatMessageList.olderError',
        });
      } else if (previousState === 'loading') {
        this.#olderAnnouncement = msg('Loaded older messages.', {id: 'chatMessageList.olderLoaded'});
      }
    }
    const area = this.querySelector<HTMLElement>('#chat-area');
    this.#stickAfterUpdate = changed.has('scrollRequest') || !area
      || area.scrollHeight - area.scrollTop - area.clientHeight <= STICK_TO_BOTTOM_PX;
    if (changed.has('turns')) {
      const previous = (changed.get('turns') as readonly ChatTurnView[] | undefined) ?? [];
      this.#pendingTurnAnchor = newlyCompletedTurn(previous, this.turns)
        ?? this.#pendingTurnAnchor;
    }
    this.#recomputeWindow();
  }

  #bindScroll(): void {
    const area = this.querySelector<HTMLElement>('#chat-area');
    if (area === this.#scrollArea) return;
    this.#unbindScroll();
    this.#scrollArea = area;
    area?.addEventListener('scroll', this.#onScroll, {passive: true});
  }

  #unbindScroll(): void {
    this.#scrollArea?.removeEventListener('scroll', this.#onScroll);
    this.#scrollArea = null;
  }

  #onScroll = (): void => {
    if (this.#recomputeWindow()) this.requestUpdate();
  };

  #liveIndices(): Set<number> {
    const indices = new Set<number>();
    this.turns.forEach((turn, index) => {
      if (turnIsLive(turn.state)) indices.add(index);
    });
    return indices;
  }

  #recomputeWindow(): boolean {
    const area = this.querySelector<HTMLElement>('#chat-area');
    const next = visibleTurnWindow({
      count: this.turns.length,
      heights: this.turns.map((turn) => this.#heights.get(turn.id) ?? 0),
      scrollTop: area?.scrollTop ?? 0,
      viewportHeight: area?.clientHeight ?? 800,
      alwaysOn: this.#liveIndices(),
    });
    if (next.start === this.#range.start && next.end === this.#range.end) return false;
    this.#range = next;
    return true;
  }

  #measureSlots(): void {
    for (const slot of this.querySelectorAll<HTMLElement>('[data-turn-slot]')) {
      const id = slot.dataset.turnId;
      if (!id || slot.hasAttribute('data-placeholder')) continue;
      const height = slot.offsetHeight;
      if (height > 0) this.#heights.set(id, height);
    }
  }

  #mounted(index: number): boolean {
    return index >= this.#range.start && index <= this.#range.end;
  }

  protected override updated(changed: PropertyValues<this>): void {
    this.#bindScroll();
    if (changed.has('turns')) {
      const previous = (changed.get('turns') as readonly ChatTurnView[] | undefined) ?? [];
      const current = liveObjectUrls(this.turns);
      for (const url of liveObjectUrls(previous)) {
        if (!current.has(url)) URL.revokeObjectURL(url);
      }
      const currentImageIds = liveImageIds(this.turns);
      for (const id of this.#imageErrors) {
        if (!currentImageIds.has(id)) this.#imageErrors.delete(id);
      }
      for (const id of this.#imageLoaded) {
        if (!currentImageIds.has(id)) this.#imageLoaded.delete(id);
      }
    }
    this.#measureSlots();
    const windowChanged = this.#recomputeWindow();
    if (this.#showWelcome()) {
      if (windowChanged) this.requestUpdate();
      return;
    }
    const previousView = changed.get('view') as ChatView | undefined;
    const olderFlightSettled = previousView?.kind === 'ready'
      && this.view.kind === 'ready'
      && (previousView.olderMessagesState ?? 'idle') === 'loading'
      && (this.view.olderMessagesState ?? 'idle') !== 'loading';
    if (this.#pendingPrependAnchor && olderFlightSettled) {
      if (this.#scrollFrame) cancelAnimationFrame(this.#scrollFrame);
      const pending = this.#pendingPrependAnchor;
      this.#pendingPrependAnchor = null;
      this.#scrollFrame = requestAnimationFrame(() => {
        this.#scrollFrame = 0;
        const area = this.querySelector<HTMLElement>('#chat-area');
        const anchor = Array.from(
          this.querySelectorAll<HTMLElement>('[data-turn-id]'),
        ).find((element) => element.dataset.turnId === pending.turnId);
        if (area && anchor) {
          const nextOffset = anchor.getBoundingClientRect().top
            - area.getBoundingClientRect().top;
          area.scrollTop += nextOffset - pending.offset;
        }
        if (this.#restoreOlderFocus) {
          this.#restoreOlderFocus = false;
          const button = this.querySelector<HTMLButtonElement>('[data-load-older]');
          if (button) button.focus({preventScroll: true});
          else this.querySelector<HTMLElement>('#chat-messages')?.focus({preventScroll: true});
        }
      });
      if (windowChanged) this.requestUpdate();
      return;
    }
    if (this.#pendingTurnAnchor && this.#scrollFrame) {
      cancelAnimationFrame(this.#scrollFrame);
      this.#scrollFrame = 0;
    }
    if ((!this.#stickAfterUpdate && !this.#pendingTurnAnchor) || this.#scrollFrame) {
      if (windowChanged) this.requestUpdate();
      return;
    }
    const stickToBottom = this.#stickAfterUpdate;
    this.#scrollFrame = requestAnimationFrame(() => {
      this.#scrollFrame = 0;
      const area = this.querySelector<HTMLElement>('#chat-area');
      if (!area) return;
      const turnId = this.#pendingTurnAnchor;
      this.#pendingTurnAnchor = null;
      const anchor = turnId
        ? Array.from(this.querySelectorAll<HTMLElement>('[data-turn-id]')).find(
            (element) => element.dataset.turnId === turnId && !element.hasAttribute('data-turn-slot'),
          ) ?? Array.from(this.querySelectorAll<HTMLElement>('[data-turn-id]')).find(
            (element) => element.dataset.turnId === turnId,
          )
        : null;
      if (anchor) {
        area.scrollTop += anchor.getBoundingClientRect().top - area.getBoundingClientRect().top;
      } else if (stickToBottom) {
        area.scrollTop = area.scrollHeight;
      }
    });
    if (windowChanged) this.requestUpdate();
  }

  protected override render(): TemplateResult {
    const turns = this.turns;
    return html`
      <main class="chat-area" id="chat-area" aria-label=${msg('Chat', {id: 'chatMessageList.chatAria'})} @click=${this.#backgroundClick}>
        ${this.#olderMessagesControl()}
        <span class="sr-only" data-older-status role="status" aria-live="polite">
          ${this.#olderAnnouncement}
        </span>
        <div class="chat-messages" id="chat-messages" role="log" tabindex="-1"
             aria-label=${msg('Conversation messages', {id: 'chatMessageList.messagesAria'})}
             ?inert=${this.interactionLocked}>
          ${this.#lineage()}
          ${this.#viewState()}
          ${repeat(
            turns,
            (turn) => turn.id,
            (turn, index) => this.#turnSlot(turn, index),
          )}
          ${this.#showWelcome() ? html`
            <div class="welcome" id="welcome">
              <div class="welcome-brand">DlightRAG</div>
              <div class="welcome-sub">${msg('Ask anything about your documents', {id: 'chatMessageList.welcomeSub'})}</div>
            </div>
          ` : nothing}
        </div>
      </main>
    `;
  }

  #olderMessagesControl(): TemplateResult | typeof nothing {
    if (this.view.kind !== 'ready' || !this.view.hasOlderMessages) return nothing;
    const state = this.view.olderMessagesState ?? 'idle';
    return html`
      <div data-older-messages>
        <button type="button" data-load-older aria-busy=${state === 'loading' ? 'true' : 'false'}
                ?disabled=${state === 'loading'} @click=${this.#loadOlderMessages}>
          ${state === 'error'
            ? msg('Retry loading older messages', {id: 'chatMessageList.retryLoadOlder'})
            : msg('Load older messages', {id: 'chatMessageList.loadOlder'})}
        </button>
      </div>
    `;
  }

  #loadOlderMessages = (event: Event): void => {
    const button = event.currentTarget as HTMLButtonElement;
    const area = this.querySelector<HTMLElement>('#chat-area');
    if (area) {
      const areaTop = area.getBoundingClientRect().top;
      const anchor = Array.from(
        this.querySelectorAll<HTMLElement>('[data-turn-id]'),
      ).find((element) => element.getBoundingClientRect().bottom > areaTop);
      if (anchor?.dataset.turnId) {
        this.#pendingPrependAnchor = {
          turnId: anchor.dataset.turnId,
          offset: anchor.getBoundingClientRect().top - areaTop,
        };
      }
    }
    this.#restoreOlderFocus = document.activeElement === button;
    this.dispatchEvent(new CustomEvent<void>('dl-chat-load-older', {
      bubbles: true,
      composed: true,
    }));
  };

  #showWelcome(): boolean {
    return this.turns.length === 0 && (this.view.kind === 'new' || this.view.kind === 'ready');
  }

  #lineage(): TemplateResult | typeof nothing {
    if (this.view.kind !== 'ready' || !this.view.lineage) return nothing;
    return html`<div class=${chatStyles.conversationLineage}>${msg(
      str`Forked from ${this.view.lineage}`,
      {id: 'chatMessageList.forkedFrom'},
    )}</div>`;
  }

  #viewState(): TemplateResult | typeof nothing {
    if (this.view.kind === 'new' || this.view.kind === 'ready') return nothing;
    if (this.view.kind === 'loading') {
      return html`<div role="status" aria-live="polite">${msg('Loading conversation history…', {id: 'chatMessageList.loadingHistory'})}</div>`;
    }
    if (this.view.kind === 'error') {
      return html`
        <div class=${chatStyles.textError} role="alert" aria-live="assertive">
          ${msg('Conversation history is unavailable.', {id: 'chatMessageList.historyUnavailable'})}
          <button type="button" aria-label=${msg('Retry loading conversation history', {id: 'chatMessageList.retryHistoryAria'})}
                  @click=${() => this.#viewAction('retry')}>${msg('Retry conversation history', {id: 'chatMessageList.retryHistory'})}</button>
        </div>
      `;
    }
    return html`
      <div class=${chatStyles.textError} role="alert" aria-live="assertive">
        ${msg('Conversation unavailable.', {id: 'chatMessageList.conversationUnavailable'})}
        <button type="button" @click=${() => this.#viewAction('new')}>${msg('Start a new chat', {id: 'chatMessageList.startNewChat'})}</button>
        ${this.view.hasRecent ? html`
          <button type="button" @click=${() => this.#viewAction('recent')}>
            ${msg('Open recent conversation', {id: 'chatMessageList.openRecent'})}
          </button>
        ` : nothing}
      </div>
    `;
  }

  #turnSlot(turn: ChatTurnView, index: number): TemplateResult {
    if (!this.#mounted(index)) {
      const height = this.#heights.get(turn.id) ?? TURN_PLACEHOLDER_MIN_PX;
      return html`
        <div data-turn-id=${turn.id} data-turn-slot data-placeholder
             style=${`min-height:${height}px`}></div>
      `;
    }
    return html`
      <div data-turn-id=${turn.id} data-turn-slot>
        ${guard([turn, this.#imageRevision], () => this.#turn(turn))}
      </div>
    `;
  }

  #turn(turn: ChatTurnView): TemplateResult {
    return html`
      <div class=${chatStyles.userMessageWrapper} data-turn-id=${turn.id}>
        ${this.#attachments(turn.userAttachments)}
        <div class=${chatStyles.userMessage}>${turn.userText}</div>
      </div>
      <article class=${chatStyles.aiMessage} data-run-id=${turn.runId || nothing}>
        <header class=${chatStyles.aiMessageHeader}>
          <span class=${chatStyles.dot} aria-hidden="true">${icon('status-dot', {size: 'xs'})}</span> DlightRAG
        </header>
        <div class="${chatStyles.aiMessageContent} ${turn.state === 'failed' ? chatStyles.textError : ''}">
          ${this.#answer(turn)}
        </div>
        ${this.#runActions(turn)}
        ${turn.liveStatus ? html`
          <span class="sr-only" role="status" aria-live="polite">${turn.liveStatus}</span>
        ` : nothing}
      </article>
      ${turn.steeringMessages.slice(-MAX_STEERING_MESSAGES).map((message) => html`
        <div class=${chatStyles.userMessageWrapper} data-steer="true">
          <div class=${chatStyles.userMessage}>${message}</div>
        </div>
      `)}
    `;
  }

  #answer(turn: ChatTurnView): TemplateResult | typeof nothing {
    if (turn.sawChildren && turn.state !== 'succeeded' && turn.state !== 'failed'
        && turn.state !== 'cancelled') {
      return html`
        <button type="button" class=${chatStyles.childAgentChip}
                @click=${() => this.#runAction('children', turn.runId)}>
          ${msg('View child agents', {id: 'chatMessageList.viewChildAgents'})}
        </button>
        ${this.#answerBody(turn)}
      `;
    }
    return this.#answerBody(turn);
  }

  #answerBody(turn: ChatTurnView): TemplateResult | typeof nothing {
    if (turn.state === 'succeeded' && turn.presentation) {
      return html`<dl-answer-presentation .presentation=${turn.presentation}></dl-answer-presentation>${this.#toolSummary(turn)}`;
    }
    if (turn.state === 'failed') {
      return html`${turn.error || msg('Service error. Please try again.', {id: 'chatFeature.serviceError'})}${this.#toolSummary(turn)}`;
    }
    return html`
      ${turn.streamText ? html`<span class="stream-tail">${turn.streamText}</span>` : nothing}
      ${turn.progress ? html`
        <span class="${chatStyles.streamingDot} ${chatStyles.progressPhase}"
              data-phase=${turn.progress}></span>
      ` : nothing}
      ${turn.state === 'pending' && !turn.progress ? html`
        <span class="${chatStyles.streamingDot} ${chatStyles.progressPhase}"
              data-phase=${msg('Answer in progress...', {id: 'chatFeature.answerInProgress'})}></span>
      ` : nothing}
      ${turn.toolRows.length > 0 ? this.#toolTrace(turn) : nothing}
      ${turn.state === 'retryable' ? this.#reconnectNotice(turn) : nothing}
      ${turn.state === 'cancelled' ? html`
        <div class=${chatStyles.stoppedNote}>${msg('Stopped', {id: 'chatMessageList.stopped'})}</div>
      ` : nothing}
      ${turn.state === 'cancelled' ? this.#toolSummary(turn) : nothing}
    `;
  }

  #reconnectNotice(turn: ChatTurnView): TemplateResult {
    const state = answerReconnectState(turn.cancelRequested);
    const copy = ANSWER_RECONNECT_COPY[state];
    return html`
      <div class=${chatStyles.answerReconnect} data-reconnect-state=${state}>
        <span class=${chatStyles.answerReconnectStatus} role="status">
          ${turn.error || msg(copy.status, {id: `chatMessageList.reconnect.${state}.status`})}
        </span>
        <button class=${chatStyles.answerReconnectAction} type="button"
                aria-label=${msg('Reconnect to this answer', {id: 'chatMessageList.reconnectAria'})}
                @click=${() => this.#reconnect(turn.runId)}>${msg(copy.action, {id: `chatMessageList.reconnect.${state}.action`})}</button>
      </div>
    `;
  }

  #runActions(turn: ChatTurnView): TemplateResult | typeof nothing {
    if (!turn.runId) return nothing;
    const terminal = turn.state === 'succeeded' || turn.state === 'failed'
      || turn.state === 'cancelled';
    if (!terminal && !turn.sawChildren) return nothing;
    if (!terminal) {
      return html`
        <div class=${chatStyles.runActions}>
          <button type="button" @click=${() => this.#runAction('children', turn.runId)}>
            ${msg('Child agents', {id: 'chatMessageList.childAgents'})}
          </button>
        </div>
      `;
    }
    const evidenceCount = Number(turn.evidence.chunks || 0);
    const usageDetails = turn.usage.usage_details as Record<string, unknown> | undefined;
    const tokenCount = Number(usageDetails?.total_tokens || 0);
    return html`
      <div class=${chatStyles.runActions}>
        <button type="button" @click=${() => this.#runAction('follow-up', turn.runId)}>${msg('Follow up', {id: 'chatMessageList.followUp'})}</button>
        <button type="button" @click=${() => this.#runAction('fork', turn.runId)}>${msg('Fork', {id: 'chatMessageList.fork'})}</button>
        ${evidenceCount || tokenCount ? html`
          <span class=${chatStyles.runSummary}
                title=${[
                  evidenceCount ? msg(str`${evidenceCount} evidence chunks`, {id: 'chatMessageList.evidenceChunksTitle'}) : '',
                  tokenCount ? msg(str`${tokenCount} tokens`, {id: 'chatMessageList.tokensTitle'}) : '',
                ].filter(Boolean).join(' · ')}>
            ${evidenceCount ? msg(str`${evidenceCount} sources`, {id: 'chatMessageList.sources'}) : ''}
          </span>
        ` : nothing}
      </div>
    `;
  }

  #attachments(references: readonly ConversationAttachmentReference[]): TemplateResult | typeof nothing {
    if (references.length === 0) return nothing;
    const images = references.filter((reference) => reference.kind === 'image');
    const documents = references.filter((reference) => reference.kind === 'document');
    return html`
      ${images.length > 0 ? html`
        <div class=${chatStyles.messageImages}>
          ${repeat(images, (reference) => reference.attachmentId, (reference) => this.#image(reference))}
        </div>
      ` : nothing}
      ${documents.length > 0 ? html`
        <div class=${chatStyles.messageDocuments}>
          ${repeat(documents, (reference) => reference.attachmentId, (reference) => {
            const href = safeSameOriginHref(reference.url);
            const content = html`
              <span class=${chatStyles.documentChipInfo}>
                <span class=${chatStyles.documentChipName}>${reference.filename}</span>
                <span class=${chatStyles.documentChipMeta}>${formatFileSize(reference.byteSize)}</span>
              </span>
            `;
            return href
              ? html`<a class=${chatStyles.documentChip} data-document-attachment="true"
                         href=${href} download>${content}</a>`
              : html`<span class=${chatStyles.documentChip} data-document-attachment="true">
                  ${content}
                </span>`;
          })}
        </div>
      ` : nothing}
    `;
  }

  #image(reference: ConversationAttachmentReference): TemplateResult {
    const thumbnail = safeImageSrc(reference.thumbnailUrl || reference.url);
    const source = safeImageSrc(reference.url);
    const failed = !thumbnail || !source || this.#imageErrors.has(reference.attachmentId);
    const loaded = !failed && this.#imageLoaded.has(reference.attachmentId);
    return html`
      <div class=${chatStyles.historyImageCard}>
        <button type="button" class=${chatStyles.historyImageButton}
                aria-label=${msg(str`Open ${reference.label}`, {id: 'chatMessageList.openImageAria'})} ?disabled=${failed}
                @click=${(event: Event) => this.#openImage(
                  reference,
                  event.currentTarget as HTMLElement,
                )}>
          <img class=${chatStyles.messageImg} src=${failed ? nothing : thumbnail}
               alt=${reference.label} loading="lazy" decoding="async"
               ?hidden=${failed}
               @load=${() => this.#finishImage(reference.attachmentId)}
               @error=${() => this.#failImage(reference.attachmentId)}>
        </button>
        <span class=${chatStyles.historyImageStatus}
              role=${failed ? 'alert' : 'status'} ?hidden=${loaded}>
          ${failed
            ? msg(str`History image failed to load: ${reference.label}`, {id: 'chatMessageList.imageFailed'})
            : msg(str`Loading ${reference.label}`, {id: 'chatMessageList.imageLoading'})}
        </span>
        <button type="button" class=${chatStyles.historyImageRetry}
                aria-label=${msg(str`Retry image: ${reference.label}`, {id: 'chatMessageList.retryImageAria'})} ?hidden=${!failed}
                @click=${(event: Event) => this.#retryImage(reference, event.currentTarget as HTMLElement)}>
          ${msg('Retry image', {id: 'chatMessageList.retryImage'})}
        </button>
      </div>
    `;
  }

  #openImage(reference: ConversationAttachmentReference, returnFocus: HTMLElement): void {
    const source = safeImageSrc(reference.url);
    if (!source || this.#imageErrors.has(reference.attachmentId)) return;
    const gallery = this.turns.flatMap((turn) => turn.userAttachments)
      .filter((attachment) => attachment.kind === 'image')
      .map((attachment) => safeImageSrc(attachment.url))
      .filter(Boolean);
    this.dispatchEvent(new CustomEvent<ImageOpenDetail>('dl-image-open', {
      bubbles: true,
      composed: true,
      detail: {src: source, gallery: [...new Set(gallery)], returnFocus},
    }));
  }

  #finishImage(id: string): void {
    this.#imageLoaded.add(id);
    this.#imageRevision += 1;
    this.requestUpdate();
  }

  #failImage(id: string): void {
    this.#imageLoaded.delete(id);
    this.#imageErrors.add(id);
    this.#imageRevision += 1;
    this.requestUpdate();
  }

  async #retryImage(reference: ConversationAttachmentReference, target: HTMLElement): Promise<void> {
    const thumbnail = safeImageSrc(reference.thumbnailUrl || reference.url);
    const source = safeImageSrc(reference.url);
    if (!thumbnail || !source) return;
    this.#imageErrors.delete(reference.attachmentId);
    this.#imageLoaded.delete(reference.attachmentId);
    this.#imageRevision += 1;
    this.requestUpdate();
    await this.updateComplete;
    const card = target.closest(`.${chatStyles.historyImageCard}`);
    const image = card?.querySelector<HTMLImageElement>('img');
    if (!image) return;
    image.removeAttribute('src');
    requestAnimationFrame(() => { image.src = thumbnail; });
  }

  #backgroundClick = (event: MouseEvent): void => {
    if (event.defaultPrevented) return;
    const target = event.target instanceof Element ? event.target : null;
    if (target?.closest(
      'button, a[href], input, textarea, select, summary, [contenteditable="true"], '
      + '[role="button"], [role="link"], [role="menuitem"], [role="option"]',
    )) return;
    this.dispatchEvent(new CustomEvent<void>('dl-chat-background-click', {
      bubbles: true,
      composed: true,
    }));
  };

  #runAction(action: ChatRunActionDetail['action'], runId: string): void {
    this.dispatchEvent(new CustomEvent<ChatRunActionDetail>('dl-chat-run-action', {
      bubbles: true,
      composed: true,
      detail: {action, runId},
    }));
  }

  #reconnect(runId: string): void {
    this.dispatchEvent(new CustomEvent<ChatReconnectDetail>('dl-chat-reconnect', {
      bubbles: true,
      composed: true,
      detail: {runId},
    }));
  }

  #viewAction(action: ChatViewActionDetail['action']): void {
    this.dispatchEvent(new CustomEvent<ChatViewActionDetail>('dl-chat-view-action', {
      bubbles: true,
      composed: true,
      detail: {action},
    }));
  }

  #toggleToolTrace(runId: string): void {
    this.dispatchEvent(new CustomEvent<ChatToolTraceToggleDetail>('dl-chat-tool-trace-toggle', {
      bubbles: true,
      composed: true,
      detail: {runId},
    }));
  }

  #toolTrace(turn: ChatTurnView): TemplateResult {
    return html`
      <div class=${chatStyles.toolTrace} role="status"
           aria-label=${msg('Tool activity', {id: 'chatMessageList.toolActivity'})}>
        ${turn.toolRows.map((row) => html`
          <div class="${chatStyles.toolRow} ${row.state === 'failed' ? chatStyles.toolRowFailed : ''}">
            <span class=${chatStyles.toolState}>
              ${row.state === 'running' ? html`<span class=${chatStyles.toolSpinner}></span>`
                : row.state === 'failed' ? icon('close', {size: 'xs', className: 'tool-state-icon tool-state-icon--failed'})
                  : icon('check', {size: 'xs', className: 'tool-state-icon tool-state-icon--done'})}
            </span>
            <span class=${chatStyles.toolLabel}>${toolRowText(row)}</span>
            ${row.durationMs !== null
              ? html`<span class=${chatStyles.toolDuration}>${formatToolDuration(row.durationMs)}</span>`
              : nothing}
          </div>
        `)}
      </div>
    `;
  }

  #toolSummary(turn: ChatTurnView): TemplateResult | typeof nothing {
    if (turn.toolTotal === 0) return nothing;
    return html`
      <button type="button" class=${chatStyles.toolSummary}
              aria-expanded=${String(turn.toolExpanded)}
              aria-label=${msg('Tool activity', {id: 'chatMessageList.toolActivity'})}
              @click=${() => this.#toggleToolTrace(turn.runId)}>
        <span>${msg(str`${turn.toolTotal} tool call(s)`, {id: 'chatMessageList.toolSummaryCount'})}</span>
        <span class=${chatStyles.toolSummaryToggle}>${turn.toolExpanded
          ? icon('chevron-down', {size: 'xs'})
          : icon('disclosure', {size: 'xs'})}</span>
      </button>
      ${turn.toolExpanded ? this.#toolTrace(turn) : nothing}
    `;
  }
}

customElements.define('dl-chat-message-list', DlChatMessageList);

declare global {
  interface HTMLElementTagNameMap {
    'dl-chat-message-list': DlChatMessageList;
  }

  interface HTMLElementEventMap {
    'dl-chat-background-click': CustomEvent<void>;
    'dl-chat-tool-trace-toggle': CustomEvent<ChatToolTraceToggleDetail>;
  }
}

export function storedTurnView(stored: ConversationTurn): ChatTurnView {
  let state: ChatTurnView['state'] = 'pending';
  let error = '';
  if (stored.status === 'succeeded') {
    if (stored.presentation) state = 'succeeded';
    else {
      state = 'failed';
      error = msg('Stored answer presentation is unavailable.', {
        id: 'chatMessageList.storedPresentationUnavailable',
      });
    }
  } else if (stored.status === 'failed') {
    state = 'failed';
    error = localizedStoredRunError(stored.errorKind, stored.errorMessage);
  } else if (stored.status === 'cancelled') {
    state = 'cancelled';
  }
  return {
    id: stored.turnId || stored.answerRunId,
    userText: stored.userText,
    userAttachments: stored.userAttachments,
    runId: stored.answerRunId,
    state,
    streamText: '',
    presentation: stored.presentation,
    usage: stored.usage ?? {},
    evidence: stored.evidence ?? {},
    error,
    progress: (stored.status === 'queued' || stored.status === 'running')
      && stored.cancelRequested ? msg('Stopping...', {id: 'chatFeature.stopping'}) : '',
    liveStatus: '',
    sawChildren: false,
    cancelRequested: stored.cancelRequested,
    steeringMessages: [],
    toolRows: [],
    toolTotal: 0,
    toolExpanded: false,
  };
}
