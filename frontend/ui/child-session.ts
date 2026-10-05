// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** One child agent in the Child agents dock: what it is doing, what it concluded, and the one box
 * that steers or continues it.
 *
 * The element shows one child at a time and is handed the next through `childSessionId`, so what the
 * reader typed for a child waits while they look at another. It observes the child and sends the
 * commands; once a command has settled it tells the roster, which refreshes the child's row.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {keyed} from 'lit/directives/keyed.js';
import {live} from 'lit/directives/live.js';
import {repeat} from 'lit/directives/repeat.js';
import {
  CHILD_TRANSCRIPT_LIMIT,
  ChildControlRejectedError,
  type AgentChildStatus,
  type ChildControlReceipt,
  type ChildObservation,
  type ChildQuestion,
} from '../api/conversations.ts';
import {ApiError} from '../api/wire.ts';
import {icon} from '../design-system/index.ts';
import {getLocale} from '../i18n/locale.ts';
import {projectActivity, type ActivityStep} from '../lib/child-activity.ts';
import {raise} from '../lib/dom.ts';
import {isAbortError} from '../lib/errors.ts';
import {LightElement} from '../lib/lit-host.ts';
import styles from '../styles/child-session.module.css';
import {childElapsed, childGlyph, childStateText} from './child-status.ts';
import type {ChildrenSource} from './inspector-children.ts';

const MINUTE_MILLISECONDS = 60_000;

type ChildAction = 'steer' | 'continue' | 'cancel' | 'reply';

/** What the reader has typed for one box, kept until its command is accepted. */
interface Draft {
  text: string;
  reauthorize: boolean;
}

const NO_DRAFT: Draft = {text: '', reauthorize: false};

/** What a command answered, and the child and Operation it answered for. */
interface Outcome {
  code: string;
  childSessionId: string;
  operationId: string | null;
}

function modelText(role: string | undefined): string {
  switch (role) {
    case undefined:
    case '': return '';
    case 'query': return msg('Query model', {id: 'childSession.model.query'});
    case 'extract': return msg('Extract model', {id: 'childSession.model.extract'});
    case 'keyword': return msg('Keyword model', {id: 'childSession.model.keyword'});
    case 'vlm': return msg('Vision model', {id: 'childSession.model.vlm'});
    case 'default': return msg('Default model', {id: 'childSession.model.default'});
    default: return role;
  }
}

function tokensText(child: AgentChildStatus): string {
  const total = child.usage?.total_tokens;
  if (typeof total !== 'number' || !Number.isFinite(total)) return '';
  const count = new Intl.NumberFormat(getLocale(), {notation: 'compact', maximumFractionDigits: 1}).format(total);
  return msg(str`${count} tokens`, {id: 'childSession.tokens'});
}

function outcomeText(code: string): string {
  switch (code) {
    case 'queued': return msg('Queued. The child has not necessarily followed it yet.', {id: 'childSession.outcome.queued'});
    case 'consumed': return msg('Consumed at a safe checkpoint. This does not prove the model complied.', {id: 'childSession.outcome.consumed'});
    case 'accepted': return msg('Continuation accepted as a new operation.', {id: 'childSession.outcome.accepted'});
    case 'cancellation_requested': return msg('Cancellation requested.', {id: 'childSession.outcome.cancellationRequested'});
    case 'replied': return msg('Reply sent.', {id: 'childSession.outcome.replied'});
    case 'terminal_child': return msg('This child is already terminal and was not revived.', {id: 'childSession.outcome.terminalChild'});
    case 'run_terminal': return msg('The parent run is terminal, so this child cannot continue.', {id: 'childSession.outcome.runTerminal'});
    case 'child_running': return msg('This child is still running.', {id: 'childSession.outcome.childRunning'});
    case 'reauthorization_required': return msg('User-cancelled work needs explicit reauthorization.', {id: 'childSession.outcome.reauthorizationRequired'});
    case 'queue_full': return msg('The pending control queue is full.', {id: 'childSession.outcome.queueFull'});
    case 'idempotency_conflict': return msg('This submission id was already used for a different request.', {id: 'childSession.outcome.idempotencyConflict'});
    case 'already_replied': return msg('This question was already answered.', {id: 'childSession.outcome.alreadyReplied'});
    case 'expired': return msg('This question expired before the reply arrived.', {id: 'childSession.outcome.expired'});
    case 'cancelled': return msg('This question was cancelled.', {id: 'childSession.outcome.cancelled'});
    case 'unknown_outcome': return msg('The child outcome is unknown.', {id: 'childSession.outcome.unknownOutcome'});
    case 'failed': return msg('The child intervention could not be sent.', {id: 'childSession.interventionFailed'});
    default: return code;
  }
}

/** Whether the reader has scrolled a page down to within one line of its bottom. A page that has not been
 * scrolled down is not at a bottom the reader chose, however short it is. */
function scrolledToBottom(page: HTMLElement): boolean {
  const line = Number.parseFloat(getComputedStyle(page).lineHeight);
  return page.scrollTop > 0
    && page.scrollHeight - page.scrollTop - page.clientHeight <= (Number.isFinite(line) ? line : 1);
}

/** What a Run that is over says in the place of the box that would steer or continue its children. */
function finishedText(): string {
  return msg('This answer has finished, so its child agents can no longer be steered or continued.', {id: 'childSession.finished'});
}

function senderText(origin: string | null): string {
  switch (origin) {
    case 'user': return msg('You', {id: 'childSession.you'});
    case 'parent': return msg('Parent', {id: 'childSession.parent'});
    default: return '';
  }
}

/** Whether a row of this child differs from the last one in what an observation shows. */
function rowChanged(before: AgentChildStatus | null | undefined, after: AgentChildStatus): boolean {
  return !before
    || before.status !== after.status
    || before.operationId !== after.operationId
    || before.summary !== after.summary
    || before.pendingQuestions !== after.pendingQuestions;
}

export class DlChildSession extends LightElement {
  static properties = {
    source: {attribute: false},
    childSessionId: {attribute: false},
    entry: {attribute: false},
    now: {attribute: false},
    commandable: {attribute: false},
    observation: {state: true},
    failed: {state: true},
    missing: {state: true},
    outcome: {state: true},
    objectiveOpen: {state: true},
    confirming: {state: true},
    answering: {state: true},
  };

  declare source: ChildrenSource | null;
  /** The child to show; empty while the roster shows its list instead. */
  declare childSessionId: string;
  /** The roster's row for this child, which heads the page before its observation arrives. A child the
   * roster no longer lists is null. */
  declare entry: AgentChildStatus | null;
  /** The roster's clock, in epoch milliseconds. */
  declare now: number;
  /** Whether the Run can still be steered. Once it is over the page has no box and no Cancel: the server
   * would refuse what they send. */
  declare commandable: boolean;
  declare observation: ChildObservation | null;
  /** The first observation of this child failed, so there is nothing to show yet. */
  declare failed: boolean;
  /** The server no longer knows this child. */
  declare missing: boolean;
  declare outcome: Outcome | null;
  declare objectiveOpen: boolean;
  declare confirming: boolean;
  /** The question whose reply box is open. */
  declare answering: string | null;

  #observing: {controller: AbortController; again: boolean} | null = null;
  readonly #drafts = new Map<string, Draft>();
  readonly #sending = new Set<string>();
  #allowNextLineBreak = false;
  /** The Run ended while this page was open to its children, which the page says out loud. */
  #runEnded = false;
  /** The reader has scrolled a running child's page to its bottom and is still there, so the page keeps
   * its bottom in view as steps arrive. Only the reader's own scrolling sets it: opening a child never jumps. */
  #following = false;
  /** Measures the title's clamp again whenever the title is resized: a settled child has no clock to redraw
   * the page, and the dock can be narrowed at any time. */
  readonly #titleSize = new ResizeObserver(() => { this.#measureTitle(); });
  #watched: HTMLElement | null = null;

  constructor() {
    super();
    this.source = null;
    this.childSessionId = '';
    this.entry = null;
    this.now = Date.now();
    this.commandable = true;
    this.observation = null;
    this.failed = false;
    this.missing = false;
    this.outcome = null;
    this.objectiveOpen = false;
    this.confirming = false;
    this.answering = null;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.lifetime.addEventListener('abort', () => {
      this.#titleSize.disconnect();
      this.#watched = null;
    }, {once: true});
  }

  /** Give the title focus: where a reader who has opened this child belongs. */
  focusTitle(): void {
    this.querySelector<HTMLElement>('[data-title]')?.focus();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    // A Run's children share nothing with another's: what was typed for one is gone with it.
    if (changed.has('source')) this.#drafts.clear();
    if (changed.has('source') || changed.has('childSessionId')) {
      this.#observing?.controller.abort();
      this.#observing = null;
      this.observation = null;
      this.failed = false;
      this.missing = false;
      this.outcome = null;
      this.objectiveOpen = false;
      this.confirming = false;
      this.answering = null;
      if (this.source && this.childSessionId) void this.#observe();
    } else if (changed.has('entry') && this.entry && this.entry.childSessionId === this.childSessionId) {
      // A fresh row: a running child's transcript grows without its row changing.
      const before = changed.get('entry') as AgentChildStatus | null | undefined;
      if (this.entry.status === 'running' || rowChanged(before, this.entry)) this.#reobserve();
    }
    // The Run ending under a reader who has its children open is said once; opening a child of a Run
    // that is already over says nothing, since the note on the page is there to read.
    if (changed.has('commandable')) this.#runEnded = !this.commandable && changed.get('commandable') === true;
    // Only a running child of a Run that is still going can be cancelled, so a question about it lapses
    // when either ends.
    if (this.confirming && (!this.commandable || this.#child()?.status !== 'running')) this.confirming = false;
    // Measured before the update adds to the page: a reader who has left the bottom, or a child that has
    // settled, is no longer followed.
    const page = this.#page();
    if (this.#following && !(page && this.#child()?.status === 'running' && scrolledToBottom(page))) {
      this.#following = false;
    }
  }

  protected override updated(changed: PropertyValues<this>): void {
    this.#measureTitle();
    const title = this.querySelector<HTMLElement>('[data-title]');
    if (title !== this.#watched) {
      this.#titleSize.disconnect();
      this.#watched = title;
      if (title) this.#titleSize.observe(title);
    }
    for (const field of this.querySelectorAll<HTMLTextAreaElement>('textarea')) this.#fit(field);
    const page = this.#page();
    if (!page) return;
    // A child opens on its title, and a page that is followed keeps its bottom in view.
    if (changed.has('source') || changed.has('childSessionId')) page.scrollTop = 0;
    else if (this.#following) page.scrollTop = page.scrollHeight;
  }

  /** The toggle shows only where the title is cut off, or has been opened. */
  #measureTitle(): void {
    const title = this.querySelector<HTMLElement>('[data-title]');
    const more = this.querySelector<HTMLElement>('[data-more]');
    if (title && more) more.hidden = !this.objectiveOpen && title.scrollHeight <= title.clientHeight + 1;
  }

  #page(): HTMLElement | null {
    return this.querySelector<HTMLElement>('[data-page]');
  }

  #scrolled = (event: Event): void => {
    this.#following = this.#child()?.status === 'running' && scrolledToBottom(event.currentTarget as HTMLElement);
  };

  /** Read the child where it stands; a read already in flight is not interrupted, one more follows it. */
  #reobserve(): void {
    if (this.#observing) this.#observing.again = true;
    else void this.#observe();
  }

  async #observe(): Promise<void> {
    const source = this.source;
    const childSessionId = this.childSessionId;
    if (!source || !childSessionId) return;
    const flight = {controller: new AbortController(), again: false};
    this.#observing = flight;
    try {
      const observation = await source.observe(
        childSessionId,
        AbortSignal.any([this.lifetime, flight.controller.signal]),
      );
      if (this.#observing !== flight) return;
      this.observation = observation;
      this.failed = false;
      this.missing = false;
    } catch (error) {
      if (this.#observing !== flight || isAbortError(error)) return;
      if (error instanceof ApiError && error.status === 404) this.missing = true;
      // A read that fails while the child is on screen leaves what is shown until the next one.
      else if (this.observation === null) this.failed = true;
    } finally {
      if (this.#observing === flight) {
        this.#observing = null;
        if (flight.again && !this.lifetime.aborted) void this.#observe();
      }
    }
  }

  #retry = (): void => {
    this.failed = false;
    void this.#observe();
  };

  /** The row, or the observation when it is newer: whichever tells the child's state now. */
  #child(): AgentChildStatus | null {
    return this.observation?.child ?? this.entry;
  }

  /** No child to show: the roster's latest first page no longer lists it (`entry` is null; the server may
   * still know it, but a Run has a handful of children, so a listed child does not drop off a page), or
   * the server answered 404 (`missing`). */
  #gone(): boolean {
    return this.missing || this.entry === null;
  }

  /** The box of a child's Operation. It steers the Operation while the child runs and continues it once the
   * child settles, so what was typed stays with the Operation when the child settles under the reader. */
  #composerKey(operationId: string | null): string {
    return `${this.childSessionId}:${operationId ?? ''}`;
  }

  #replyKey(requestId: string): string {
    return `${this.childSessionId}:reply:${requestId}`;
  }

  #draft(key: string): Draft {
    return this.#drafts.get(key) ?? NO_DRAFT;
  }

  // The page holds what the reader typed and binds it with `live()`, which writes a box only when the
  // box differs from the draft: writing back text the box already holds, such as an IME's composition,
  // would interrupt the composition.
  #typed = (event: Event): void => {
    const field = event.currentTarget as HTMLTextAreaElement;
    const key = field.dataset.draft!;
    this.#drafts.set(key, {...this.#draft(key), text: field.value});
    this.requestUpdate();
  };

  #reauthorized = (event: Event): void => {
    const box = event.currentTarget as HTMLInputElement;
    const key = box.dataset.draft!;
    this.#drafts.set(key, {...this.#draft(key), reauthorize: box.checked});
    this.requestUpdate();
  };

  // Enter sends and Shift+Enter breaks the line, as in the chat composer; Enter that commits an
  // IME composition is the composition's, never a send.
  #keydown = (event: KeyboardEvent): void => {
    if (event.key === 'Enter') this.#allowNextLineBreak = event.shiftKey;
  };

  #beforeInput = (event: InputEvent): void => {
    if (event.inputType !== 'insertLineBreak') return;
    if (event.isComposing || this.#allowNextLineBreak) {
      this.#allowNextLineBreak = false;
      return;
    }
    event.preventDefault();
    this.#allowNextLineBreak = false;
    (event.currentTarget as HTMLTextAreaElement).form?.requestSubmit();
  };

  #keyup = (event: KeyboardEvent): void => {
    if (event.key === 'Enter') this.#allowNextLineBreak = false;
  };

  /** Grow a box with what it holds, up to the height the stylesheet allows, then scroll it. */
  #fit(field: HTMLTextAreaElement): void {
    const limit = Number.parseFloat(getComputedStyle(field).maxHeight);
    field.style.height = 'auto';
    field.style.height = `${Number.isFinite(limit) ? Math.min(field.scrollHeight, limit) : field.scrollHeight}px`;
    field.style.overflowY = Number.isFinite(limit) && field.scrollHeight > limit ? 'auto' : 'hidden';
  }

  #submitComposer = (event: Event): void => {
    event.preventDefault();
    const child = this.#child();
    if (!child) return;
    const action = child.status === 'running' ? 'steer' : 'continue';
    const key = this.#composerKey(child.operationId);
    const {text, reauthorize} = this.#draft(key);
    const content = text.trim();
    if (!content) return;
    void this.#execute(action, null, key, (source) => source.control(
      this.childSessionId, action, content, action === 'continue' && reauthorize, child.operationId,
    ));
  };

  #cancelChild = (): void => {
    const child = this.#child();
    if (!child) return;
    void this.#execute('cancel', null, null, (source) => source.control(
      this.childSessionId, 'cancel', '', false, child.operationId,
    ));
  };

  #submitReply = (event: Event): void => {
    event.preventDefault();
    const requestId = (event.currentTarget as HTMLFormElement).dataset.request!;
    const key = this.#replyKey(requestId);
    const content = this.#draft(key).text.trim();
    if (!content) return;
    void this.#execute('reply', requestId, key, (source) => source.reply(requestId, content));
  };

  /** Send one command and show what came of it, unless the reader has moved on to another child or
   * Operation by then; one command at a time goes out for each child, Operation, and question. */
  async #execute(
    action: ChildAction,
    requestId: string | null,
    draftKey: string | null,
    send: (source: ChildrenSource) => Promise<ChildControlReceipt>,
  ): Promise<void> {
    const source = this.source;
    const childSessionId = this.childSessionId;
    const operationId = this.#child()?.operationId ?? null;
    const key = this.#commandKey(action, requestId);
    if (!source || this.#sending.has(key)) return;
    const shown = (): boolean => this.source === source
      && this.childSessionId === childSessionId
      && (this.#child()?.operationId ?? null) === operationId;
    this.#sending.add(key);
    this.outcome = null;
    this.requestUpdate();
    try {
      const receipt = await send(source);
      // What was typed for an accepted command is spent, wherever the reader is by now.
      if (draftKey) this.#drafts.delete(draftKey);
      if (this.source !== source) return;
      raise(this, 'dl-child-command-settled');
      if (!shown()) return;
      this.outcome = {
        code: receipt.outcome,
        childSessionId,
        // A continuation that was accepted started the Operation the child shows next.
        operationId: action === 'continue' && receipt.outcome === 'accepted' && receipt.operationId
          ? receipt.operationId
          : operationId,
      };
      if (action === 'reply') this.answering = null;
      if (action === 'cancel') {
        this.confirming = false;
        void this.updateComplete.then(() => { this.focusTitle(); });
      }
      this.#reobserve();
    } catch (error) {
      if (this.source !== source || isAbortError(error)) return;
      // A refusal means the child or the question is not as the page shows it, so the roster is told and
      // the page reads the child again.
      const refused = error instanceof ChildControlRejectedError;
      if (refused) raise(this, 'dl-child-command-settled');
      if (!shown()) return;
      if (error instanceof ApiError && error.status === 404) {
        this.missing = true;
      } else {
        this.outcome = {code: refused ? error.outcome : 'failed', childSessionId, operationId};
        if (refused) this.#reobserve();
      }
    } finally {
      this.#sending.delete(key);
      this.requestUpdate();
    }
  }

  /** What one command is called while it is in flight: its child, Operation, and question. */
  #commandKey(action: ChildAction, requestId: string | null): string {
    return `${this.childSessionId}:${action}:${this.#child()?.operationId ?? ''}:${requestId ?? ''}`;
  }

  #sendingNow(action: ChildAction, requestId: string | null = null): boolean {
    return this.#sending.has(this.#commandKey(action, requestId));
  }

  /** What the page says out loud: that the child is gone, else the latest answer to a command. */
  #announcement(): string {
    if (this.#gone()) return msg('That child is no longer available.', {id: 'childSession.gone'});
    if (this.#runEnded) return finishedText();
    const outcome = this.outcome;
    const shown = outcome
      && outcome.childSessionId === this.childSessionId
      && outcome.operationId === (this.#child()?.operationId ?? null);
    return shown ? outcomeText(outcome.code) : '';
  }

  #toggleObjective = (): void => {
    this.objectiveOpen = !this.objectiveOpen;
  };

  #askConfirmation = (): void => {
    this.confirming = true;
    void this.updateComplete.then(() => { this.querySelector<HTMLElement>('[data-keep]')?.focus(); });
  };

  #keepRunning = (): void => {
    this.confirming = false;
    void this.updateComplete.then(() => { this.querySelector<HTMLElement>('[data-cancel]')?.focus(); });
  };

  #openAnswer = (event: Event): void => {
    this.answering = (event.currentTarget as HTMLElement).dataset.request!;
    void this.updateComplete.then(() => { this.querySelector<HTMLTextAreaElement>('[data-reply]')?.focus(); });
  };

  #closeAnswer = (): void => {
    const requestId = this.answering;
    this.answering = null;
    void this.updateComplete.then(() => {
      this.querySelector<HTMLElement>(`[data-request="${CSS.escape(requestId ?? '')}"]`)?.focus();
    });
  };

  /** A question the parent is still waiting to answer. */
  #live(question: ChildQuestion): boolean {
    return question.status === 'pending'
      && (question.expiresAt === null || Date.parse(question.expiresAt) > this.now);
  }

  protected override render(): TemplateResult | typeof nothing {
    if (!this.childSessionId) return nothing;
    // The status region outlives every view, so a change in it is announced.
    return html`${this.#view()}<span class="dl-sr-only" role="status">${this.#announcement()}</span>`;
  }

  #view(): TemplateResult {
    const child = this.#child();
    if (this.#gone() || !child) {
      return html`
        <div class=${styles.gone}>
          <p class=${styles.goneTitle}>${msg('That child is no longer available.', {id: 'childSession.gone'})}</p>
          <p class=${styles.goneHint}>${msg('Pick another from the list.', {id: 'childSession.goneHint'})}</p>
        </div>
      `;
    }
    const running = child.status === 'running';
    return html`
      <section class=${styles.session} aria-labelledby="child-session-title">
        <div class=${styles.scroll} data-page @scroll=${this.#scrolled}>
          ${keyed(this.childSessionId, html`
            ${this.#heading(child)}
            ${this.#statusLine(child, running)}
            ${this.confirming ? this.#confirmation() : nothing}
            ${this.observation ? this.#questions(this.observation) : nothing}
            ${running ? nothing : this.#result(child)}
            ${this.observation ? this.#activity(this.observation, child, running) : this.#pending()}
            ${this.observation ? this.#history(this.observation) : nothing}
          `)}
        </div>
        ${this.commandable ? this.#composer(child, running) : this.#finished()}
      </section>
    `;
  }

  #heading(child: AgentChildStatus): TemplateResult {
    return html`
      <h3 id="child-session-title" class="${styles.title} ${this.objectiveOpen ? '' : styles.clamped}"
          data-title tabindex="-1">${child.objective || this.childSessionId}</h3>
      <button type="button" class=${styles.more} data-more hidden
              aria-expanded=${this.objectiveOpen ? 'true' : 'false'}
              aria-controls="child-session-title" @click=${this.#toggleObjective}>
        ${this.objectiveOpen
          ? msg('Show less', {id: 'childSession.showLess'})
          : msg('Show full objective', {id: 'childSession.showFull'})}
      </button>
    `;
  }

  #statusLine(child: AgentChildStatus, running: boolean): TemplateResult {
    const meta = [childElapsed(child, this.now), modelText(child.modelRole), tokensText(child)]
      .filter(Boolean).join(' · ');
    return html`
      <div class=${styles.status}>
        ${childGlyph(child)}
        <b class=${styles.state}>${childStateText(child)}</b>
        <span>${meta}</span>
        ${running && this.commandable && !this.confirming ? html`
          <button type="button" class="dl-btn dl-btn-danger-text ${styles.cancel}" data-cancel
                  @click=${this.#askConfirmation}>${msg('Cancel child', {id: 'childSession.cancel'})}</button>
        ` : nothing}
      </div>
    `;
  }

  #confirmation(): TemplateResult {
    const busy = this.#sendingNow('cancel');
    return html`
      <div class=${styles.confirm} role="group" aria-labelledby="child-session-confirm">
        <p id="child-session-confirm">${msg('Cancel this child? Its work so far is kept.', {id: 'childSession.cancelConfirm'})}</p>
        <button type="button" class="dl-btn dl-btn-danger-text" aria-disabled=${busy ? 'true' : nothing}
                @click=${() => { if (!busy) this.#cancelChild(); }}>${msg('Cancel child', {id: 'childSession.cancel'})}</button>
        <button type="button" class="dl-btn" data-keep @click=${this.#keepRunning}>${msg('Keep running', {id: 'childSession.keepRunning'})}</button>
      </div>
    `;
  }

  #questions(observation: ChildObservation): TemplateResult | typeof nothing {
    const waiting = observation.questions.filter((question) => this.#live(question));
    if (waiting.length === 0) return nothing;
    return html`${repeat(waiting, (question) => question.requestId, (question) => this.#card(question))}`;
  }

  #card(question: ChildQuestion): TemplateResult {
    const minutes = question.expiresAt === null
      ? 0
      : Math.max(1, Math.ceil((Date.parse(question.expiresAt) - this.now) / MINUTE_MILLISECONDS));
    const label = question.expiresAt === null
      ? msg('Asking the parent', {id: 'childSession.asking'})
      : msg(str`Asking the parent · expires in ${minutes} min`, {id: 'childSession.askingExpires'});
    const key = this.#replyKey(question.requestId);
    const draft = this.#draft(key);
    const busy = this.#sendingNow('reply', question.requestId);
    return html`
      <div class=${styles.card} role="group" aria-labelledby=${`child-question-${question.requestId}`}>
        <p class=${styles.cap} id=${`child-question-${question.requestId}`}>${label}</p>
        <p class=${styles.asked}>${question.question}</p>
        ${this.answering === question.requestId ? html`
          <form class=${styles.reply} data-request=${question.requestId} @submit=${this.#submitReply}>
            <div class=${styles.field}>
              <textarea rows="1" class=${styles.input} data-reply data-draft=${key}
                        aria-label=${msg('Your answer', {id: 'childSession.answerLabel'})}
                        .value=${live(draft.text)} ?readonly=${busy}
                        @input=${this.#typed} @keydown=${this.#keydown}
                        @beforeinput=${this.#beforeInput} @keyup=${this.#keyup}></textarea>
              ${this.#sendButton(draft.text, busy)}
            </div>
            <button type="button" class="dl-btn" @click=${this.#closeAnswer}>${msg('Cancel answer', {id: 'childSession.cancelAnswer'})}</button>
          </form>
        ` : html`
          <button type="button" class="dl-btn" data-request=${question.requestId}
                  @click=${this.#openAnswer}>${msg('Answer instead', {id: 'childSession.answerInstead'})}</button>
        `}
      </div>
    `;
  }

  #sendButton(text: string, busy: boolean): TemplateResult {
    return html`
      <button type="submit" class=${styles.send} aria-label=${msg('Send', {id: 'childSession.send'})}
              aria-disabled=${busy || !text.trim() ? 'true' : nothing}>${icon('send', {size: 'sm'})}</button>
    `;
  }

  #result(child: AgentChildStatus): TemplateResult | typeof nothing {
    const summary = child.summary || this.observation?.result?.summary || '';
    const handles = this.observation?.result?.handles ?? child.resultHandles;
    if (!summary && handles.length === 0) return nothing;
    return html`
      <section class=${styles.section}>
        <h4 class=${styles.label}>${msg('Result', {id: 'childSession.result'})}</h4>
        ${summary ? html`<p class=${styles.resultText}>${summary}</p>` : nothing}
        ${handles.length > 0 ? html`
          <details class=${styles.fold}>
            <summary class=${styles.summary}>${icon('disclosure', {size: 'xs', className: styles.chevron})}
              ${msg(str`Evidence · ${handles.length}`, {id: 'childSession.evidence'})}</summary>
            <ul class=${styles.mono}>${handles.map((handle) => html`<li>${handle}</li>`)}</ul>
          </details>
        ` : nothing}
      </section>
    `;
  }

  /** Until the observation arrives the page has only the roster's row to show. */
  #pending(): TemplateResult {
    return this.failed
      ? html`
        <div class=${styles.failed}>
          <p role="alert">${msg('Child details could not be loaded.', {id: 'childSession.loadFailed'})}</p>
          <button type="button" class="dl-btn" @click=${this.#retry}>${msg('Retry', {id: 'childSession.retry'})}</button>
        </div>`
      : html`<p class=${styles.quiet}>${msg('Loading child details…', {id: 'childSession.loading'})}</p>`;
  }

  #activity(observation: ChildObservation, child: AgentChildStatus, running: boolean): TemplateResult {
    const steps = projectActivity(observation.transcript, {
      objective: child.objective ?? '',
      childRunning: running,
    });
    const title = running
      ? msg('Activity', {id: 'childSession.activity'})
      : steps.length === 1
        ? msg('Activity · 1 step', {id: 'childSession.activityOneStep'})
        : msg(str`Activity · ${steps.length} steps`, {id: 'childSession.activitySteps'});
    return html`
      <details class=${styles.fold} ?open=${running}>
        <summary class=${styles.summary}>${icon('disclosure', {size: 'xs', className: styles.chevron})} ${title}</summary>
        ${observation.transcript.length >= CHILD_TRANSCRIPT_LIMIT
          ? html`<p class=${styles.caption}>${msg(str`Latest ${steps.length} steps`, {id: 'childSession.latestSteps'})}</p>`
          : nothing}
        ${steps.length === 0
          ? html`<p class=${styles.quiet}>${msg('No activity yet.', {id: 'childSession.noActivity'})}</p>`
          : html`<ol class=${styles.steps}>${repeat(steps, (step) => step.key, (step) => this.#step(step))}</ol>`}
      </details>
    `;
  }

  #step(step: ActivityStep): TemplateResult {
    if (step.kind === 'say') {
      return html`<li class=${styles.step}><p class=${styles.say}>${step.text}</p></li>`;
    }
    if (step.kind === 'instruction') {
      const sender = this.#sender(step.text);
      return html`
        <li class=${styles.step}>
          ${sender.label ? html`<p class=${styles.told}>${sender.label}</p>` : nothing}
          <p class=${styles.toldText}>${sender.text}</p>
        </li>
      `;
    }
    const state = step.state === 'running'
      ? msg('Running', {id: 'childSession.state.running'})
      : step.state === 'failed' ? msg('Failed', {id: 'childSession.state.failed'}) : '';
    const line = html`
      ${state ? html`<span class="dl-sr-only">${state}: </span>` : nothing}
      <span class=${styles.verb}>${step.verb}</span>
      ${step.excerpt ? html`<span class=${styles.excerpt}>${step.excerpt}</span>` : nothing}
    `;
    const tone = step.state === 'running' ? styles.stepRunning : step.state === 'failed' ? styles.stepFailed : '';
    // A result that is no more than its excerpt has nothing to open.
    return step.full.trim() === step.excerpt
      ? html`<li class="${styles.step} ${tone}">${line}</li>`
      : html`
        <li class="${styles.step} ${tone}">
          <details>
            <summary class=${styles.stepSummary}>
              <span class=${styles.stepLine}>${line}</span>
              ${icon('disclosure', {size: 'xs', className: styles.chevron})}
            </summary>
            <pre class=${styles.full} tabindex="0">${step.full}</pre>
          </details>
        </li>
      `;
  }

  /** A steer the control records name shows as its sender wrote it, under who sent it. The runtime writes a
   * steer into the transcript as "<Origin> steer: <content>", so a record is matched by exactly that. */
  #sender(text: string): {label: string; text: string} {
    const control = this.observation?.controls.find((record) => {
      const origin = record.origin || 'unknown';
      return text === `${origin.charAt(0).toUpperCase()}${origin.slice(1).toLowerCase()} steer: ${record.content}`;
    });
    return control ? {label: senderText(control.origin), text: control.content} : {label: '', text};
  }

  #history(observation: ChildObservation): TemplateResult {
    const asked = observation.questions.filter((question) => !this.#live(question));
    const controls = observation.controls;
    return html`
      ${asked.length > 0 ? html`
        <section class=${styles.section}>
          <h4 class=${styles.label}>${msg('Questions', {id: 'childSession.questions'})}</h4>
          <ul class=${styles.history}>
            ${asked.map((question) => this.#asked(question))}
          </ul>
        </section>
      ` : nothing}
      ${controls.length > 0 ? html`
        <details class=${styles.fold}>
          <summary class=${styles.summary}>${icon('disclosure', {size: 'xs', className: styles.chevron})}
            ${msg(str`Control history · ${controls.length}`, {id: 'childSession.controlHistory'})}</summary>
          <ul class=${styles.history}>
            ${controls.map((record) => html`
              <li>
                <p class=${styles.meta}>${[
                  record.consumed
                    ? msg('Consumed', {id: 'childSession.control.consumed'})
                    : msg('Queued', {id: 'childSession.control.queued'}),
                  senderText(record.origin),
                ].filter(Boolean).join(' · ')}</p>
                <p class=${styles.toldText}>${record.content}</p>
              </li>
            `)}
          </ul>
        </details>
      ` : nothing}
    `;
  }

  #asked(question: ChildQuestion): TemplateResult {
    // A question still marked pending past its expiry is shown as the expired one it is.
    const status = question.status === 'pending' ? 'expired' : question.status;
    const word = status === 'replied'
      ? msg('Answered', {id: 'childSession.question.replied'})
      : status === 'cancelled'
        ? msg('Cancelled', {id: 'childSession.question.cancelled'})
        : msg('Expired', {id: 'childSession.question.expired'});
    return html`
      <li>
        <p class=${styles.toldText}>${question.question}</p>
        <p class=${styles.meta}>${[word, senderText(question.replyOrigin)].filter(Boolean).join(' · ')}</p>
        ${question.reply ? html`<p class=${styles.toldText}>${question.reply}</p>` : nothing}
      </li>
    `;
  }

  #finished(): TemplateResult {
    return html`<p class=${styles.finished}>${finishedText()}</p>`;
  }

  #composer(child: AgentChildStatus, running: boolean): TemplateResult {
    const mode = running ? 'steer' : 'continue';
    const key = this.#composerKey(child.operationId);
    const draft = this.#draft(key);
    const busy = this.#sendingNow(mode);
    const userCancelled = !running && child.status === 'cancelled' && child.cancellationOrigin === 'user';
    const placeholder = running
      ? msg('Steer this child…', {id: 'childSession.steerPlaceholder'})
      : msg('Continue this child…', {id: 'childSession.continuePlaceholder'});
    const said = this.#announcement();
    return html`
      <form class=${styles.composer} @submit=${this.#submitComposer}>
        ${said ? html`<p class=${styles.outcome}>${said}</p>` : nothing}
        ${userCancelled ? html`
          <label class="dl-dialog-checkbox ${styles.reauthorize}">
            <input type="checkbox" data-draft=${key} .checked=${live(draft.reauthorize)} @change=${this.#reauthorized}>
            ${msg('Reauthorize this user-cancelled work', {id: 'childSession.reauthorize'})}
          </label>
        ` : nothing}
        <div class=${styles.field}>
          <textarea rows="1" class=${styles.input} data-draft=${key} aria-label=${placeholder}
                    aria-describedby="child-session-hint" placeholder=${placeholder}
                    .value=${live(draft.text)} ?readonly=${busy}
                    @input=${this.#typed} @keydown=${this.#keydown}
                    @beforeinput=${this.#beforeInput} @keyup=${this.#keyup}></textarea>
          ${this.#sendButton(draft.text, busy)}
        </div>
        <p class=${styles.hint} id="child-session-hint">${running
          ? msg('Delivered at the child\'s next safe checkpoint.', {id: 'childSession.steerHint'})
          : msg('Starts a new operation on this child.', {id: 'childSession.continueHint'})}</p>
      </form>
    `;
  }
}

customElements.define('dl-child-session', DlChildSession);

declare global {
  interface HTMLElementTagNameMap {
    'dl-child-session': DlChildSession;
  }

  interface HTMLElementEventMap {
    'dl-child-command-settled': CustomEvent<void>;
  }
}
