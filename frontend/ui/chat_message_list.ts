// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {guard} from 'lit/directives/guard.js';
import {repeat} from 'lit/directives/repeat.js';
import type {
  AnswerPresentation,
  ConversationAttachmentReference,
  ConversationTurn,
} from '../api/conversations.ts';
import {answerErrorMessage} from '../lib/errors.ts';
import {formatFileSize} from '../lib/file_size.ts';
import {LightElement} from '../lib/lit_host.ts';
import {safeImageSrc, safeSameOriginHref} from '../lib/urls.ts';
import chatStyles from '../styles/chat.module.css';
import './answer_presentation.ts';

export type ChatView =
  | {kind: 'new'}
  | {kind: 'loading'}
  | {
      kind: 'ready';
      conversationId: string;
      history: readonly ConversationTurn[];
      lineage: string | null;
    }
  | {kind: 'unavailable'; hasRecent: boolean}
  | {kind: 'error'};

export interface ChatTurnView {
  id: string;
  userText: string;
  userAttachments: readonly ConversationAttachmentReference[];
  runId: string;
  state: 'pending' | 'streaming' | 'succeeded' | 'failed' | 'cancelled' | 'retryable';
  streamText: string;
  presentation: AnswerPresentation | null;
  usage: Record<string, unknown>;
  evidence: Record<string, number>;
  error: string;
  progress: string;
  liveStatus: string;
  sawChildren: boolean;
  cancelRequested: boolean;
  steeringMessages: readonly string[];
}

export interface ChatRunActionDetail {
  action: 'children' | 'follow-up' | 'fork';
  runId: string;
}

export interface ChatReconnectDetail {
  runId: string;
}

export interface ChatViewActionDetail {
  action: 'retry' | 'new' | 'recent';
}

export const MAX_CHAT_TURNS = 100;
export const MAX_STEERING_MESSAGES = 50;
const STICK_TO_BOTTOM_PX = 160;

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
      if (attachment.kind === 'image') ids.add(attachment.attachment_id);
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
  };

  declare view: ChatView;
  declare turns: readonly ChatTurnView[];
  declare scrollRequest: number;
  #imageErrors = new Set<string>();
  #imageLoaded = new Set<string>();
  #imageRevision = 0;
  #stickAfterUpdate = true;
  #scrollFrame = 0;

  constructor() {
    super();
    this.view = {kind: 'new'};
    this.turns = [];
    this.scrollRequest = 0;
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    if (this.#scrollFrame) cancelAnimationFrame(this.#scrollFrame);
    for (const url of liveObjectUrls(this.turns)) URL.revokeObjectURL(url);
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    const area = this.querySelector<HTMLElement>('#chat-area');
    this.#stickAfterUpdate = changed.has('scrollRequest') || !area
      || area.scrollHeight - area.scrollTop - area.clientHeight <= STICK_TO_BOTTOM_PX;
  }

  protected override updated(changed: PropertyValues<this>): void {
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
    if (this.#showWelcome()) return;
    if (!this.#stickAfterUpdate || this.#scrollFrame) return;
    this.#scrollFrame = requestAnimationFrame(() => {
      this.#scrollFrame = 0;
      const area = this.querySelector<HTMLElement>('#chat-area');
      if (area) area.scrollTop = area.scrollHeight;
    });
  }

  protected override render(): TemplateResult {
    const turns = this.turns;
    return html`
      <main class="chat-area" id="chat-area" aria-label="Chat">
        <div class="chat-messages" id="chat-messages" role="log" aria-label="Conversation messages">
          ${this.#lineage()}
          ${this.#viewState()}
          ${repeat(
            turns,
            (turn) => turn.id,
            (turn) => guard([turn, this.#imageRevision], () => this.#turn(turn)),
          )}
          ${this.#showWelcome() ? html`
            <div class="welcome" id="welcome">
              <div class="welcome-brand">DlightRAG</div>
              <div class="welcome-sub">Ask anything about your documents</div>
            </div>
          ` : nothing}
        </div>
      </main>
    `;
  }

  #showWelcome(): boolean {
    return this.turns.length === 0 && (this.view.kind === 'new' || this.view.kind === 'ready');
  }

  #lineage(): TemplateResult | typeof nothing {
    if (this.view.kind !== 'ready' || !this.view.lineage) return nothing;
    return html`<div class=${chatStyles.conversationLineage}>Forked from ${this.view.lineage}</div>`;
  }

  #viewState(): TemplateResult | typeof nothing {
    if (this.view.kind === 'new' || this.view.kind === 'ready') return nothing;
    if (this.view.kind === 'loading') {
      return html`<div role="status" aria-live="polite">Loading conversation history…</div>`;
    }
    if (this.view.kind === 'error') {
      return html`
        <div class=${chatStyles.textError} role="alert" aria-live="assertive">
          Conversation history is unavailable.
          <button type="button" aria-label="Retry loading conversation history"
                  @click=${() => this.#viewAction('retry')}>Retry conversation history</button>
        </div>
      `;
    }
    return html`
      <div class=${chatStyles.textError} role="alert" aria-live="assertive">
        Conversation unavailable.
        <button type="button" @click=${() => this.#viewAction('new')}>Start a new chat</button>
        ${this.view.hasRecent ? html`
          <button type="button" @click=${() => this.#viewAction('recent')}>
            Open recent conversation
          </button>
        ` : nothing}
      </div>
    `;
  }

  #turn(turn: ChatTurnView): TemplateResult {
    return html`
      <div class=${chatStyles.userMessageWrapper}>
        ${this.#attachments(turn.userAttachments)}
        <div class=${chatStyles.userMessage}>${turn.userText}</div>
      </div>
      <article class=${chatStyles.aiMessage} data-run-id=${turn.runId || nothing}>
        <header class=${chatStyles.aiMessageHeader}>
          <span class=${chatStyles.dot} aria-hidden="true">●</span> DlightRAG
        </header>
        <div class="${chatStyles.aiMessageContent} ${turn.state === 'failed' ? chatStyles.textError : ''}">
          ${this.#answer(turn)}
        </div>
        ${this.#runActions(turn)}
        <span class="sr-only" role="status" aria-live="polite">${turn.liveStatus}</span>
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
          Child agents working…
        </button>
        ${this.#answerBody(turn)}
      `;
    }
    return this.#answerBody(turn);
  }

  #answerBody(turn: ChatTurnView): TemplateResult | typeof nothing {
    if (turn.state === 'succeeded' && turn.presentation) {
      return html`<dl-answer-presentation .presentation=${turn.presentation}></dl-answer-presentation>`;
    }
    if (turn.state === 'failed') return html`${turn.error || 'Service error. Please try again.'}`;
    return html`
      ${turn.streamText ? html`<span class="stream-tail">${turn.streamText}</span>` : nothing}
      ${turn.progress ? html`
        <span class="${chatStyles.streamingDot} ${chatStyles.progressPhase}"
              data-phase=${turn.progress}></span>
      ` : nothing}
      ${turn.state === 'pending' && !turn.progress ? html`
        <span class="${chatStyles.streamingDot} ${chatStyles.progressPhase}"
              data-phase="Generating answer..."></span>
      ` : nothing}
      ${turn.state === 'retryable' ? html`
        <span class="answer-reconnect">
          <span role="status">${turn.error}</span>
          <button type="button" aria-label="Reconnect to this answer"
                  @click=${() => this.#reconnect(turn.runId)}>Reconnect</button>
        </span>
      ` : nothing}
      ${turn.state === 'cancelled' ? html`
        <div class=${chatStyles.stoppedNote}>Stopped</div>
      ` : nothing}
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
            Child agents
          </button>
        </div>
      `;
    }
    const evidenceCount = Number(turn.evidence.chunks || 0);
    const usageDetails = turn.usage.usage_details as Record<string, unknown> | undefined;
    const tokenCount = Number(usageDetails?.total_tokens || 0);
    return html`
      <div class=${chatStyles.runActions}>
        <button type="button" @click=${() => this.#runAction('follow-up', turn.runId)}>Follow up</button>
        <button type="button" @click=${() => this.#runAction('fork', turn.runId)}>Fork</button>
        ${evidenceCount || tokenCount ? html`
          <span class=${chatStyles.runSummary}
                title=${[
                  evidenceCount ? `${evidenceCount} evidence chunks` : '',
                  tokenCount ? `${tokenCount} tokens` : '',
                ].filter(Boolean).join(' · ')}>
            ${evidenceCount ? `${evidenceCount} sources` : ''}
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
          ${repeat(images, (reference) => reference.attachment_id, (reference) => this.#image(reference))}
        </div>
      ` : nothing}
      ${documents.length > 0 ? html`
        <div class=${chatStyles.messageDocuments}>
          ${repeat(documents, (reference) => reference.attachment_id, (reference) => {
            const href = safeSameOriginHref(reference.url);
            const content = html`
              <span class=${chatStyles.documentChipInfo}>
                <span class=${chatStyles.documentChipName}>${reference.filename}</span>
                <span class=${chatStyles.documentChipMeta}>${formatFileSize(reference.byte_size)}</span>
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
    const thumbnail = safeImageSrc(reference.thumbnail_url || reference.url);
    const source = safeImageSrc(reference.url);
    const failed = !thumbnail || !source || this.#imageErrors.has(reference.attachment_id);
    const loaded = !failed && this.#imageLoaded.has(reference.attachment_id);
    return html`
      <div class=${chatStyles.historyImageCard}>
        <button type="button" class=${chatStyles.historyImageButton}
                aria-label=${`Open ${reference.label}`} ?disabled=${failed}
                data-action=${failed ? nothing : 'open-lightbox'}
                data-full-src=${failed ? nothing : source}>
          <img class=${chatStyles.messageImg} src=${failed ? nothing : thumbnail}
               alt=${reference.label} loading="lazy" decoding="async"
               ?hidden=${failed}
               @load=${() => this.#finishImage(reference.attachment_id)}
               @error=${() => this.#failImage(reference.attachment_id)}>
        </button>
        <span class=${chatStyles.historyImageStatus}
              role=${failed ? 'alert' : 'status'} ?hidden=${loaded}>
          ${failed ? `History image failed to load: ${reference.label}` : `Loading ${reference.label}`}
        </span>
        <button type="button" class=${chatStyles.historyImageRetry}
                aria-label=${`Retry image: ${reference.label}`} ?hidden=${!failed}
                @click=${(event: Event) => this.#retryImage(reference, event.currentTarget as HTMLElement)}>
          Retry image
        </button>
      </div>
    `;
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
    const thumbnail = safeImageSrc(reference.thumbnail_url || reference.url);
    const source = safeImageSrc(reference.url);
    if (!thumbnail || !source) return;
    this.#imageErrors.delete(reference.attachment_id);
    this.#imageLoaded.delete(reference.attachment_id);
    this.#imageRevision += 1;
    this.requestUpdate();
    await this.updateComplete;
    const card = target.closest(`.${chatStyles.historyImageCard}`);
    const image = card?.querySelector<HTMLImageElement>('img');
    if (!image) return;
    image.removeAttribute('src');
    requestAnimationFrame(() => { image.src = thumbnail; });
  }

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
}

customElements.define('dl-chat-message-list', DlChatMessageList);

declare global {
  interface HTMLElementTagNameMap {
    'dl-chat-message-list': DlChatMessageList;
  }
}

export function storedTurnView(stored: ConversationTurn): ChatTurnView {
  let state: ChatTurnView['state'] = 'pending';
  let error = '';
  if (stored.status === 'succeeded') {
    if (stored.presentation) state = 'succeeded';
    else {
      state = 'failed';
      error = 'Stored answer presentation is unavailable.';
    }
  } else if (stored.status === 'failed') {
    state = 'failed';
    error = answerErrorMessage({message: stored.error_message});
  } else if (stored.status === 'cancelled') {
    state = 'cancelled';
  }
  return {
    id: stored.turn_id || stored.answer_run_id,
    userText: stored.user_text,
    userAttachments: stored.user_attachments,
    runId: stored.answer_run_id,
    state,
    streamText: '',
    presentation: stored.presentation,
    usage: stored.usage ?? {},
    evidence: stored.evidence ?? {},
    error,
    progress: (stored.status === 'queued' || stored.status === 'running')
      && stored.cancel_requested ? 'Stopping...' : '',
    liveStatus: '',
    sawChildren: false,
    cancelRequested: stored.cancel_requested,
    steeringMessages: [],
  };
}
