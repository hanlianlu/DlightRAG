// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {guard} from 'lit/directives/guard.js';
import {repeat} from 'lit/directives/repeat.js';
import type {
  ConversationAttachmentReference,
  ConversationTurn,
} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {getLocale} from '../i18n/locale.ts';
import {
  isTerminalTurnState,
  type ChatCommandView,
  type ChatTurnView,
  type ProcessEntry,
} from '../lib/chat-views.ts';
import {toolTime} from '../lib/date-format.ts';
import {raise} from '../lib/dom.ts';
import {formatFileSize} from '../lib/file-size.ts';
import {LightElement} from '../lib/lit-host.ts';
import type {KeysetPagerStatus} from '../lib/paged.ts';
import {catalogText, localizedText, type LocalizedText} from '../lib/localized-text.ts';
import {MAIN_AGENT} from '../lib/agent-id.ts';
import {SERVICE_ERROR_TEXT, storedRunErrorText} from '../lib/run-errors.ts';
import {
  processGroupSummary,
  rowDurationMs,
  toolRowText,
  type ProcessGroupSummary,
  type ToolRow,
} from '../lib/tool-events.ts';
import {toolVerbText} from '../lib/tool-display.ts';
import {answerPhaseLabel} from '../lib/turn-projection.ts';
import {TURN_PLACEHOLDER_MIN_PX, visibleTurnWindow} from '../lib/turn-window.ts';
import {safeImageSrc, safeSameOriginHref} from '../lib/urls.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import chatStyles from '../styles/chat.module.css';
import './answer-feedback.ts';
import './answer-presentation.ts';
import './memory-notice.ts';
import {loadOlderControl} from './load-older.ts';
import {tokenCountLabel} from './token-count.ts';

const NO_OLDER_PAGES: KeysetPagerStatus = {state: 'idle', starting: false, hasOlder: false, outcome: null};

export type ChatView =
  | {kind: 'new'}
  | {kind: 'loading'}
  | {
      kind: 'ready';
      conversationId: string;
      history: readonly ConversationTurn[];
      lineage: string | null;
      /** A snapshot of the history's next-page state, taken with this view. */
      olderMessages?: KeysetPagerStatus;
    }
  | {kind: 'unavailable'; hasRecent: boolean}
  | {kind: 'error'};


export interface ChatRunActionDetail {
  action: 'traces' | 'fork' | 'live';
  runId: string;
}

export interface ChatReconnectDetail {
  runId: string;
}

/** How often a live turn's process-group clock redraws; the same interval the time is shown at. */
const ELAPSED_TICK_MS = 1000;

function shownToolDuration(row: ToolRow, now: number): string {
  return toolTime(rowDurationMs(row, now), row.state === 'running');
}

export interface ChatViewActionDetail {
  action: 'retry' | 'new' | 'recent';
}

export const MAX_STEERING_MESSAGES = 50;
const STICK_TO_BOTTOM_PX = 160;

type AnswerReconnectState = 'running' | 'stopping';

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

function newlyCompletedTurn(
  previous: readonly ChatTurnView[],
  current: readonly ChatTurnView[],
): string | null {
  const previousStates = new Map(previous.map((turn) => [turn.id, turn.state]));
  return [...current].reverse().find((turn) => {
    const previousState = previousStates.get(turn.id);
    return previousState !== undefined && !isTerminalTurnState(previousState) && isTerminalTurnState(turn.state);
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

/** One line of the conversation the list draws: a turn, or a command the server kept. */
type ConversationEntry =
  | {readonly kind: 'turn'; readonly key: string; readonly sequence: number; readonly turn: ChatTurnView; readonly index: number}
  | {readonly kind: 'command'; readonly key: string; readonly sequence: number; readonly command: ChatCommandView};

/** The page's turns and commands in the order they happened. They share one counter — a turn's
 *  `sequence` is its `turn_number` — so a command falls between the turns around it. A turn the
 *  server has not numbered yet (one the reader has just sent) sorts last. Commands are small and
 *  bounded, so every command on the page is drawn while the turns keep their window. */
function conversationEntries(
  turns: readonly ChatTurnView[],
  commands: readonly ChatCommandView[],
): ConversationEntry[] {
  const ordered = [...commands].sort((left, right) => left.sequence - right.sequence);
  const entries: ConversationEntry[] = [];
  let next = 0;
  for (const [index, turn] of turns.entries()) {
    const sequence = turn.sequence ?? Number.POSITIVE_INFINITY;
    while (next < ordered.length && ordered[next].sequence < sequence) {
      const command = ordered[next];
      entries.push({kind: 'command', key: `command-${command.id}`, sequence: command.sequence, command});
      next += 1;
    }
    entries.push({kind: 'turn', key: turn.id, sequence, turn, index});
  }
  for (; next < ordered.length; next += 1) {
    const command = ordered[next];
    entries.push({kind: 'command', key: `command-${command.id}`, sequence: command.sequence, command});
  }
  return entries;
}

/** Lit-owned conversation messages, streaming state, and message intent. */
export class DlChatMessageList extends LightElement {
  static properties = {
    handles: {attribute: false},
    view: {attribute: false},
    turns: {attribute: false},
    commands: {attribute: false},
    scrollRequest: {attribute: false},
    interactionLocked: {attribute: false},
    answerFeedbackEnabled: {attribute: false},
  };

  declare handles: AppHandles;
  declare view: ChatView;
  declare turns: readonly ChatTurnView[];
  /** The server-executed commands on this page, drawn between the turns by sequence. */
  declare commands: readonly ChatCommandView[];
  declare scrollRequest: number;
  declare interactionLocked: boolean;
  /** Whether the reader is offered the thumbs under a settled answer. */
  declare answerFeedbackEnabled: boolean;
  #imageErrors = new Set<string>();
  #imageLoaded = new Set<string>();
  #imageRevision = 0;
  /** The turns whose process group the reader has opened or shut: their choice, not the render's.
   *  A turn the reader has not chosen for opens on its trace while it runs, so the latest steps are
   *  seen without a click; a settled turn keeps the line shut, because its trace is gone. */
  readonly #processOpen = new Map<string, boolean>();
  /** The live traces the reader is following: absent or true means the newest row stays in view,
   *  false means they scrolled up into the history and a redraw must not pull them back down. */
  readonly #processFollow = new Map<string, boolean>();
  #stickAfterUpdate = true;
  #scrollFrame = 0;
  #elapsedTimer: ReturnType<typeof setInterval> | null = null;
  #elapsedTick = 0;
  #now = performance.now();
  #pendingTurnAnchor: string | null = null;
  #pendingPrependAnchor: {turnId: string; offset: number} | null = null;
  #restoreOlderFocus = false;
  #range = {start: 0, end: -1};
  readonly #heights = new Map<string, number>();
  #scrollArea: HTMLElement | null = null;

  constructor() {
    super();
    this.handles = productionHandles();
    this.view = {kind: 'new'};
    this.turns = [];
    this.commands = [];
    this.scrollRequest = 0;
    this.interactionLocked = false;
    this.answerFeedbackEnabled = false;
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    if (this.#scrollFrame) cancelAnimationFrame(this.#scrollFrame);
    this.#stopElapsedTimer();
    this.#unbindScroll();
    for (const url of liveObjectUrls(this.turns)) URL.revokeObjectURL(url);
  }

  /** One view clock for every live process group: it exists only while a mounted turn
   *  still works, so an idle transcript schedules nothing. */
  #syncElapsedTimer(): void {
    const running = this.turns.some((turn) => this.#counting(turn));
    if (running && this.#elapsedTimer === null) {
      this.#elapsedTimer = setInterval(() => {
        this.#now = performance.now();
        this.#elapsedTick += 1;
        this.requestUpdate();
      }, ELAPSED_TICK_MS);
    } else if (!running) {
      this.#stopElapsedTimer();
    }
  }

  /** Whether this turn draws a clock that has to redraw on its own: a live turn's process group
   *  times the whole group from the first call, whether or not one is still in flight. */
  #counting(turn: ChatTurnView): boolean {
    return turn.toolRows.length > 0 && !isTerminalTurnState(turn.state);
  }

  #stopElapsedTimer(): void {
    if (this.#elapsedTimer === null) return;
    clearInterval(this.#elapsedTimer);
    this.#elapsedTimer = null;
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
      this.#heights.clear();
      this.#range = {start: 0, end: Number.MAX_SAFE_INTEGER};
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

  /** Turns whose Run has not settled stay mounted however far they scroll away. */
  #liveIndices(): Set<number> {
    const indices = new Set<number>();
    this.turns.forEach((turn, index) => {
      if (!isTerminalTurnState(turn.state)) indices.add(index);
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
    this.#syncElapsedTimer();
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
      // A process group the reader opened or shut belongs to its turn: another turn's transcript
      // must not inherit it. The follow state goes with the turn for the same reason.
      const currentTurnIds = new Set(this.turns.map((turn) => turn.id));
      for (const id of this.#processOpen.keys()) {
        if (!currentTurnIds.has(id)) this.#processOpen.delete(id);
      }
      for (const id of this.#processFollow.keys()) {
        if (!currentTurnIds.has(id)) this.#processFollow.delete(id);
      }
    }
    this.#measureSlots();
    this.#followProcessTraces();
    if (this.#showWelcome()) {
      return;
    }
    const previousView = changed.get('view') as ChatView | undefined;
    const olderFlightSettled = previousView?.kind === 'ready'
      && this.view.kind === 'ready'
      && previousView.olderMessages?.state === 'loading'
      && this.view.olderMessages?.state !== 'loading';
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
          const button = this.querySelector<HTMLButtonElement>('[data-load-older="messages"]');
          if (button) button.focus({preventScroll: true});
          else this.querySelector<HTMLElement>('#chat-messages')?.focus({preventScroll: true});
        }
      });
      return;
    }
    if (this.#pendingTurnAnchor && this.#scrollFrame) {
      cancelAnimationFrame(this.#scrollFrame);
      this.#scrollFrame = 0;
    }
    if ((!this.#stickAfterUpdate && !this.#pendingTurnAnchor) || this.#scrollFrame) {
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
  }

  protected override render(): TemplateResult {
    const entries = conversationEntries(this.turns, this.commands);
    return html`
      <main class="chat-area" id="chat-area" aria-label=${msg('Chat', {id: 'chatMessageList.chatAria'})} @click=${this.#backgroundClick}>
        ${loadOlderControl({
          list: 'messages',
          pages: this.view.kind === 'ready' ? this.view.olderMessages ?? NO_OLDER_PAGES : NO_OLDER_PAGES,
          label: msg('Load older messages', {id: 'chatMessageList.loadOlder'}),
          retryLabel: msg('Retry loading older messages', {id: 'chatMessageList.retryLoadOlder'}),
          loading: msg('Loading older messages…', {id: 'chatMessageList.olderLoading'}),
          loaded: msg('Loaded older messages.', {id: 'chatMessageList.olderLoaded'}),
          failed: msg('Older messages could not be loaded.', {id: 'chatMessageList.olderError'}),
          onLoad: this.#loadOlderMessages,
        })}
        <div class="chat-messages" id="chat-messages" role="log" tabindex="-1"
             aria-label=${msg('Conversation messages', {id: 'chatMessageList.messagesAria'})}
             ?inert=${this.interactionLocked}>
          ${this.#lineage()}
          ${this.#viewState()}
          ${repeat(
            entries,
            (entry) => entry.key,
            (entry) => entry.kind === 'command'
              ? entry.command.tone === 'error'
                ? this.#commandNode(entry.command)
                : nothing
              : this.#turnSlot(entry.turn, entry.index),
          )}
          ${this.#showWelcome() ? html`
            <div class="welcome" id="welcome">
              <div class="welcome-brand">DlightRAG</div>
              <div class="welcome-sub">${msg('Deep work, delegated.', {id: 'chatMessageList.welcomeSub'})}</div>
            </div>
          ` : nothing}
        </div>
      </main>
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
    raise(this, 'dl-chat-load-older');
  };

  #showWelcome(): boolean {
    return this.turns.length === 0 && this.commands.length === 0
      && (this.view.kind === 'new' || this.view.kind === 'ready');
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

  /** One mounted turn, drawn from the cache key its own content depends on.
   *
   * The locale is part of that key: a settled turn draws words — its Run actions,
   * its tool rows, its usage — and nothing else in the key reports a language
   * change, so without it a redraw would hand back the old language. So is whether
   * the thumbs are offered, which the page can learn is off after it drew the turn,
   * and whether the reader has the turn's process group opened. */
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
        ${guard(
          [
            turn,
            this.#imageRevision,
            this.#elapsedDependency(turn),
            this.#processOpen.get(turn.id),
            getLocale(),
            this.answerFeedbackEnabled,
          ],
          () => this.#turn(turn),
        )}
      </div>
    `;
  }

  /** The cache key that lets one running tool row redraw its counter.
   *
   * The turn template is cached by identity, so a ticking view clock alone
   * changes nothing. A turn that shows no counter keeps the stable key and stays
   * cached; only the turn that is actually counting joins the clock. */
  #elapsedDependency(turn: ChatTurnView): number {
    return this.#counting(turn) ? this.#elapsedTick : 0;
  }

  /** The phase the server named for this turn, if it named one: the process group falls back on
   *  it until the first call lands. */
  #phase(turn: ChatTurnView): string | null {
    const activity = turn.activity;
    return activity?.kind === 'phase' ? activity.phase : null;
  }

  /** What this turn's one-line process group says, or null when it has no work to show yet. */
  #processSummary(turn: ChatTurnView): ProcessGroupSummary | null {
    return processGroupSummary(turn.toolRows, this.#phase(turn), this.#now);
  }

  /** The status line a live turn draws, in the language showing now: the phase the server
   *  named, or the copy the client owns. A turn with tool calls draws its process group
   *  instead, so this line never says the same thing twice. */
  #activity(turn: ChatTurnView): string {
    const activity = turn.activity;
    if (activity === null) return '';
    if (activity.kind === 'tool') return '';
    if (activity.kind === 'copy') return localizedText(activity);
    const label = answerPhaseLabel(activity.phase);
    return label === null ? '' : msg(label, {id: `chatFeature.phase.${activity.phase}`});
  }

  /** The words a process group leads with: the newest call's verb, in the language showing now. */
  #processState(summary: ProcessGroupSummary): string {
    return toolVerbText(summary.verb, summary.verbId);
  }

  /** How many calls the group has counted, in the language showing now. */
  #callsText(count: number): string {
    return count === 1
      ? msg('1 tool call', {id: 'chatMessageList.processOneCall'})
      : msg(str`${count} tool calls`, {id: 'chatMessageList.processCalls'});
  }

  /** What a screen reader hears for this turn: what it is doing, or how it ended.
   *  Drawn from the turn's machine state, so it follows a language change too. The
   *  clock is left out: it redraws every second and would be read out every second. */
  #announcement(turn: ChatTurnView): string {
    if (turn.state === 'succeeded') return msg('Answer ready', {id: 'chatFeature.answerReady'});
    if (turn.state === 'cancelled') return msg('Answer stopped', {id: 'chatFeature.answerStopped'});
    if (turn.state === 'failed') {
      return turn.error ? localizedText(turn.error) : localizedText(SERVICE_ERROR_TEXT);
    }
    // The reconnect notice carries a status role of its own, so this one stays quiet.
    if (turn.state === 'retryable') return '';
    const summary = this.#processSummary(turn);
    if (summary !== null && summary.count > 0) {
      return `${this.#processState(summary)} · ${this.#callsText(summary.count)}`;
    }
    const activity = this.#activity(turn);
    // A turn waiting on its first word draws only the dots, which are silent, so the status says it is
    // under way. The words do not change from one frame to the next, so they are read out once.
    return activity === '' && turn.state === 'pending' ? this.#inProgress() : activity;
  }

  /** What a turn says while it waits and has nothing more particular to say. */
  #inProgress(): string {
    return msg('Answer in progress...', {id: 'chatFeature.answerInProgress'});
  }

  #turn(turn: ChatTurnView): TemplateResult {
    const announcement = this.#announcement(turn);
    return html`
      <div class=${chatStyles.userMessageWrapper} data-turn-id=${turn.id}>
        ${this.#attachments(turn.userAttachments)}
        <div class=${chatStyles.userMessage}>${turn.userText}</div>
      </div>
      ${turn.steeringMessages.slice(-MAX_STEERING_MESSAGES).map((message) => html`
        <div class=${chatStyles.userMessageWrapper} data-steer="true">
          <div class=${chatStyles.userMessage}>${message}</div>
        </div>
      `)}
      <article class=${chatStyles.aiMessage} data-run-id=${turn.runId || nothing}>
        <header class=${chatStyles.aiMessageHeader}>
          <span class=${chatStyles.dot} aria-hidden="true">${icon('status-dot', {size: 'xs'})}</span> DlightRAG
        </header>
        <div class="${chatStyles.aiMessageContent} ${turn.state === 'failed' ? chatStyles.textError : ''}">
          ${this.#answer(turn)}
        </div>
        ${this.#runActions(turn)}
        ${announcement ? html`
          <span class="dl-sr-only" role="status" aria-live="polite">${announcement}</span>
        ` : nothing}
      </article>
    `;
  }

  /** One command the server executed and kept, drawn as a one-line system notice: the slash line the
   *  reader sent in monospace, then its result on the same muted line. The goal bar is where the
   *  goal's state lives, so this line is a record of the handshake and never a card repeating that
   *  state. It is not a model answer — no author header, Run actions, traces, fork, or judgement. */
  #commandNode(command: ChatCommandView): TemplateResult {
    const line = command.input === ''
      ? `/${command.name}`
      : `/${command.name} ${command.input}`;
    return html`
      <div class="command-node" data-command-id=${command.id} data-tone=${command.tone}>
        <span class="command-node-line">${line}</span>
        <span class="command-node-text">${command.result()}</span>
      </div>
    `;
  }

  #answer(turn: ChatTurnView): TemplateResult | typeof nothing {
    if (turn.state === 'succeeded' && turn.presentation) {
      return html`<dl-answer-presentation .presentation=${turn.presentation}
                                      .runId=${turn.runId ?? null}></dl-answer-presentation>`;
    }
    if (turn.state === 'failed') {
      return html`${turn.error ? localizedText(turn.error) : localizedText(SERVICE_ERROR_TEXT)}`;
    }
    // A turn with process to show — reasoning folds and Tool rows in one rolling trace — draws it
    // as one process group; before that, the phase the server named (or the dots) stands in its
    // place. A failed turn keeps its words only, and a succeeded one keeps its answer only.
    const summary = this.#processSummary(turn);
    const showsProcess = turn.processEntries.length > 0 || (summary !== null && summary.count > 0);
    const activity = showsProcess ? '' : this.#activity(turn);
    // The log around the turns is polite, and text that grows every frame must not be read out as it
    // grows: the tail and the dots opt out, and the status under the answer says how the turn ended.
    // The Thinking folds opt out too, for the same reason.
    return html`
      ${turn.streamText ? html`<span class="stream-tail" aria-live="off">${turn.streamText}</span>` : nothing}
      ${activity ? html`
        <span class="${chatStyles.streamingDot} ${chatStyles.progressPhase}" aria-live="off"
              data-phase=${activity}></span>
      ` : nothing}
      ${turn.state === 'pending' && !activity && !showsProcess ? html`
        <span class="${chatStyles.streamingDot} ${chatStyles.progressPhase}" aria-live="off"
              data-phase=${this.#inProgress()}></span>
      ` : nothing}
      ${showsProcess ? this.#processGroup(turn, summary) : nothing}
      ${turn.state === 'retryable' ? this.#reconnectNotice(turn) : nothing}
      ${turn.state === 'cancelled' ? html`
        <div class=${chatStyles.stoppedNote}>${msg('Stopped', {id: 'chatMessageList.stopped'})}</div>
      ` : nothing}
    `;
  }

  /** What the process group says while the model is thinking and has not called a Tool yet:
   *  the phase the server named when it named one, and Thinking otherwise. */
  #thinkingState(summary: ProcessGroupSummary | null): string {
    const phase = summary?.phase ?? null;
    const label = phase === null ? null : answerPhaseLabel(phase);
    return label ?? msg('Thinking', {id: 'chatMessageList.thinking'});
  }

  /** One entry of a running turn's merged process trace, drawn where it happened: a reasoning fold
   *  (open while it is still being written) or a Tool row.
   *
   *  The reasoning text grows every frame, so the fold sits outside the polite log's announcements,
   *  exactly like the answer tail; the Tool rows are statements and the trace around them stays a
   *  live region. */
  #processEntry(entry: ProcessEntry): TemplateResult {
    if (entry.kind === 'tool') return this.#toolRow(entry.row);
    return html`
      <details class=${chatStyles.thinkingBlock} ?open=${entry.open} data-thinking aria-live="off">
        <summary class=${chatStyles.thinkingSummary}>
          <span class=${chatStyles.thinkingLabel}>${msg('Thinking', {id: 'chatMessageList.thinking'})}</span>
          ${icon('disclosure', {size: 'xs', className: chatStyles.thinkingChevron})}
        </summary>
        <div class=${chatStyles.thinkingText}>${entry.text}</div>
      </details>
    `;
  }

  #toolRow(row: ToolRow): TemplateResult {
    const rowDuration = shownToolDuration(row, this.#now);
    return html`
      <div class="${chatStyles.toolRow} ${row.state === 'failed' ? chatStyles.toolRowFailed : ''}">
        <span class=${chatStyles.toolState}>
          ${row.state === 'running' ? html`<span class=${chatStyles.toolSpinner}></span>`
            : row.state === 'failed' ? icon('close', {size: 'xs', className: 'tool-state-icon tool-state-icon--failed'})
              : icon('check', {size: 'xs', className: 'tool-state-icon tool-state-icon--done'})}
        </span>
        <span class=${chatStyles.toolLabel}>${toolRowText(row)}</span>
        ${rowDuration
          ? html`<span class=${chatStyles.toolDuration}
                  aria-hidden=${String(row.state === 'running')}>${rowDuration}</span>`
          : nothing}
      </div>
    `;
  }

  #reconnectNotice(turn: ChatTurnView): TemplateResult {
    const state = answerReconnectState(turn.cancelRequested);
    const copy = ANSWER_RECONNECT_COPY[state];
    return html`
      <div class=${chatStyles.answerReconnect} data-reconnect-state=${state}>
        <span class=${chatStyles.answerReconnectStatus} role="status">
          ${turn.error ? localizedText(turn.error) : msg(copy.status, {id: `chatMessageList.reconnect.${state}.status`})}
        </span>
        <button class=${chatStyles.answerReconnectAction} type="button"
                aria-label=${msg('Reconnect to this answer', {id: 'chatMessageList.reconnectAria'})}
                @click=${() => this.#reconnect(turn.runId)}>${msg(copy.action, {id: `chatMessageList.reconnect.${state}.action`})}</button>
      </div>
    `;
  }

  #runActions(turn: ChatTurnView): TemplateResult | typeof nothing {
    if (!turn.runId) return nothing;
    const terminal = isTerminalTurnState(turn.state);
    const traces = html`
      <button type="button" class=${chatStyles.tracesButton}
              @click=${() => this.#runAction('traces', turn.runId)}>
        ${terminal ? nothing : html`<span class=${chatStyles.tracesLive} aria-hidden="true">${icon('status-dot', {size: 'xs'})}</span>`}
        ${msg('Agent traces', {id: 'chatMessageList.agentTraces'})}
      </button>
    `;
    if (!terminal) {
      return html`<div class="dl-action-row ${chatStyles.runActions}">${traces}${this.#browsing(turn)}</div>`;
    }
    const usageDetails = turn.usage.usage_details as Record<string, unknown> | undefined;
    const tokenCount = Number(usageDetails?.total_tokens || 0);
    // What the Run spent, said plainly: the References section already carries
    // how many sources the answer cites, so this row does not repeat it.
    return html`
      <div class="dl-action-row ${chatStyles.runActions}">
        ${turn.state === 'failed' ? html`
          <button type="button"
                  aria-label=${msg('Retry this message', {id: 'chatMessageList.retryFailedTurnAria'})}
                  @click=${() => this.#retryFailedTurn(turn)}>
            ${msg('Retry', {id: 'chatMessageList.retryFailedTurn'})}
          </button>
        ` : nothing}
        ${traces}
        <button type="button" @click=${() => this.#runAction('fork', turn.runId)}>${msg('Fork', {id: 'chatMessageList.fork'})}</button>
        ${this.#feedback(turn)}
        ${tokenCount > 0 ? html`
          <span class=${chatStyles.runUsage}>${tokenCountLabel(tokenCount)}</span>
        ` : nothing}
        ${this.#memoryNotice(turn)}
      </div>
    `;
  }

  /** What the Run left in Profile Memory, once it has succeeded: the last part of the row, drawn only when the
   *  Run left something. The element reads it itself, from the page's store. */
  #memoryNotice(turn: ChatTurnView): TemplateResult | typeof nothing {
    if (turn.state !== 'succeeded') return nothing;
    return html`<dl-memory-notice .handles=${this.handles} .runId=${turn.runId}></dl-memory-notice>`;
  }

  /** The way into a Run's live browser: a small, persistent indicator while it browses.
   *
   * It is shown only for a Run that is using a browser, and what it names is what the Run's
   * own tool rows already carry — the newest `browser` call's object, which the server
   * labels with the page's host and path. Nothing opens the dock by itself: a Run may open
   * and close a browser a dozen times, most of the time nobody needs to watch, and on a
   * narrow screen an auto-opened panel would cover the conversation. Auto-opening it when
   * the agent explicitly asks for a person's help is deliberately out of scope: that needs
   * a signal the agent can raise, and this indicator is not one.
   *
   * The chat sees a browser only through the Run's tool events, so the indicator's life is
   * the live turn's: a settled Run has released its lease, and the row is gone.
   */
  #browsing(turn: ChatTurnView): TemplateResult | typeof nothing {
    if (!turn.runId) return nothing;
    const rows = turn.toolRows.filter((row) => row.name === 'browser');
    const row = rows[rows.length - 1];
    if (row === undefined) return nothing;
    const host = row.object.split('/')[0];
    const label = host === ''
      ? msg('Live browser', {id: 'chatMessageList.liveBrowser'})
      : host;
    return html`
      <button type="button" class=${chatStyles.liveButton}
              aria-label=${host === ''
                ? msg('Open the live browser preview', {id: 'chatMessageList.liveBrowserOpenAria'})
                : msg(str`Live browser on ${host} — open the preview`, {id: 'chatMessageList.liveBrowserHostAria'})}
              @click=${() => this.#runAction('live', turn.runId)}>
        <span class=${chatStyles.liveGlyph} aria-hidden="true">${icon('system', {size: 'xs'})}</span>
        <span class=${chatStyles.liveHost}>${label}</span>
      </button>
    `;
  }

  /** The thumbs for the Run's final answer, once the Run has succeeded and the server has said which message
   *  that is: a turn that has just finished waits for the history that states it. */
  #feedback(turn: ChatTurnView): TemplateResult | typeof nothing {
    if (turn.state !== 'succeeded' || turn.answerSequence === null) return nothing;
    return html`
      <dl-answer-feedback layout="row" .handles=${this.handles} .runId=${turn.runId} .agent=${MAIN_AGENT}
        .sequence=${turn.answerSequence} .judgement=${turn.feedback}
        .enabled=${this.answerFeedbackEnabled}></dl-answer-feedback>
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
    raise(this, 'dl-image-open', {src: source, gallery: [...new Set(gallery)], returnFocus});
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
    raise(this, 'dl-chat-background-click');
  };

  #runAction(action: ChatRunActionDetail['action'], runId: string): void {
    raise(this, 'dl-chat-run-action', {action, runId});
  }

  #reconnect(runId: string): void {
    raise(this, 'dl-chat-reconnect', {runId});
  }

  /** A failed Run kept the reader's words but not their work: offer them again as a new send. */
  #retryFailedTurn(turn: ChatTurnView): void {
    raise(this, 'dl-chat-retry-turn', {query: turn.userText});
  }

  #viewAction(action: ChatViewActionDetail['action']): void {
    raise(this, 'dl-chat-view-action', {action});
  }

  /** One live turn's process that opens on one line: what the turn is doing, how many calls it has
   *  taken, and how long the group has run. Its body is the merged trace — the reasoning folds and
   *  the Tool rows interleaved in the order they happened — in one bounded, internally scrolling
   *  viewport that holds the newest steps in view. The block is open while the turn runs unless the
   *  reader shut it, so a growing thought is read inside the block and never runs down the page. */
  #processGroup(turn: ChatTurnView, summary: ProcessGroupSummary | null): TemplateResult {
    const live = !isTerminalTurnState(turn.state);
    const calls = summary?.count ?? 0;
    const duration = calls > 0 && summary !== null ? toolTime(summary.durationMs, live) : '';
    const open = this.#processOpen.get(turn.id) ?? live;
    return html`
      <details class=${chatStyles.processGroup} data-process-group
               data-process-turn=${turn.id} ?open=${open} @toggle=${this.#toggledProcess}>
        <summary class=${chatStyles.processSummary} aria-expanded=${String(open)}>
          ${live ? html`<span class=${chatStyles.streamingDot} aria-hidden="true"></span>` : nothing}
          ${calls > 0 && summary !== null
            ? html`
              <span class=${chatStyles.processState}>${this.#processState(summary)}</span>
              <span class=${chatStyles.processSeparator} aria-hidden="true">·</span>
              <span class=${chatStyles.processCalls}>${this.#callsText(calls)}</span>
              ${duration
                ? html`<span class=${chatStyles.processSeparator} aria-hidden="true">·</span>
                  <span class=${chatStyles.processDuration}
                        aria-hidden=${live ? 'true' : nothing}>${duration}</span>`
                : nothing}`
            : html`<span class=${chatStyles.processState}>${this.#thinkingState(summary)}</span>`}
          ${icon('disclosure', {size: 'xs', className: chatStyles.processChevron})}
        </summary>
        ${turn.processEntries.length === 0 ? nothing : html`
          <div class=${chatStyles.toolTrace} role="status"
               data-process-trace data-process-turn=${turn.id} data-live=${live ? 'true' : nothing}
               aria-label=${msg('Tool activity', {id: 'chatMessageList.toolActivity'})}
               @scroll=${this.#processScrolled}>
            ${turn.processEntries.map((entry) => this.#processEntry(entry))}
          </div>`}
      </details>
    `;
  }

  /** Whether a trace is scrolled to its newest row, allowing for sub-pixel rounding. */
  #atTraceEnd(trace: HTMLElement): boolean {
    return trace.scrollHeight - trace.scrollTop - trace.clientHeight <= 8;
  }

  /** The reader scrolled a live trace: follow the newest row only while they are at its end. */
  #processScrolled = (event: Event): void => {
    const trace = event.currentTarget as HTMLElement;
    const turnId = trace.dataset.processTurn;
    if (turnId === undefined) return;
    this.#processFollow.set(turnId, this.#atTraceEnd(trace));
  };

  /** Hold every followed live trace at its newest row. A reader who scrolled up into the history
   *  keeps their place across redraws, so the trace is never yanked back while it is being read. */
  #followProcessTraces(): void {
    for (const trace of this.querySelectorAll<HTMLElement>('[data-process-trace][data-live]')) {
      const turnId = trace.dataset.processTurn;
      if (turnId === undefined || this.#processFollow.get(turnId) === false) continue;
      trace.scrollTop = trace.scrollHeight;
    }
  }

  /** The reader opened or shut one process group: their choice, not the render's, so it outlives
   *  the clock that redraws the turn every second. */
  #toggledProcess = (event: Event): void => {
    const group = event.currentTarget as HTMLDetailsElement;
    const turnId = group.dataset.processTurn;
    if (turnId === undefined) return;
    this.#processOpen.set(turnId, group.open);
    this.requestUpdate();
  };

}

customElements.define('dl-chat-message-list', DlChatMessageList);

export interface ChatRetryTurnDetail {
  query: string;
}

declare global {
  interface HTMLElementTagNameMap {
    'dl-chat-message-list': DlChatMessageList;
  }

  interface HTMLElementEventMap {
    'dl-chat-background-click': CustomEvent<void>;
    'dl-chat-load-older': CustomEvent<void>;
    'dl-chat-run-action': CustomEvent<ChatRunActionDetail>;
    'dl-chat-reconnect': CustomEvent<ChatReconnectDetail>;
    'dl-chat-retry-turn': CustomEvent<ChatRetryTurnDetail>;
    'dl-chat-view-action': CustomEvent<ChatViewActionDetail>;
  }
}

export function storedTurnView(stored: ConversationTurn): ChatTurnView {
  let state: ChatTurnView['state'] = 'pending';
  let error: LocalizedText | null = null;
  if (stored.status === 'succeeded') {
    if (stored.presentation) state = 'succeeded';
    else {
      state = 'failed';
      error = catalogText(
        'Stored answer presentation is unavailable.',
        'chatMessageList.storedPresentationUnavailable',
      );
    }
  } else if (stored.status === 'failed') {
    state = 'failed';
    error = storedRunErrorText(stored.errorKind, stored.errorMessage);
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
    processEntries: [],
    presentation: stored.presentation,
    usage: stored.usage ?? {},
    error,
    activity: (stored.status === 'queued' || stored.status === 'running')
      && stored.cancelRequested
      ? {kind: 'copy', source: 'Stopping...', id: 'chatFeature.stopping'}
      : null,
    cancelRequested: stored.cancelRequested,
    steeringMessages: stored.steeringMessages,
    toolRows: [],
    /** The turn's place among the command nodes and the turns around it. */
    sequence: stored.turnNumber,
    answerSequence: stored.answerSequence,
    feedback: stored.feedback,
  };
}
