// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** One agent of a Run, as the Inspector reads it: what it is doing, what it concluded, what it was
 *  asked, and the activity timeline it worked through.
 *
 *  The page is read-only. Steering, continuing, cancelling, and answering a child all belong to the
 *  conversation's composer — the surface a reader acts on — so this page states, and never commands;
 *  a question to the parent reads as who it asks and when it lapses, with no box to answer it here.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {keyed} from 'lit/directives/keyed.js';
import {repeat} from 'lit/directives/repeat.js';
import type {
  ActivityMessage,
  AgentChildStatus,
  ChildControlRecord,
  ChildObservation,
  ChildQuestion,
} from '../api/conversations.ts';
import {ApiError} from '../api/wire.ts';
import {icon} from '../design-system/index.ts';
import {ClockController} from '../lib/clock.ts';
import {isAbortError} from '../lib/errors.ts';
import {LightElement} from '../lib/lit-host.ts';
import {agentIdOf, MAIN_AGENT, RUN_AGENT, sameAgent} from '../lib/agent-id.ts';
import {agentName} from '../lib/agent-name.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import type {TracesSource} from '../stores/traces-source.ts';
import styles from '../styles/agent-session.module.css';
import {type DlActivity, senderText } from './activity.ts';
import {agentElapsed, agentGlyph, agentIsCounting, agentStateText} from './agent-status.ts';
import './answer-feedback.ts';
import {evidenceFold} from './evidence-fold.ts';
import {tokenCountLabel} from './token-count.ts';

const MINUTE_MILLISECONDS = 60_000;

const NO_CONTROLS: readonly ChildControlRecord[] = [];

function tokensText(child: AgentChildStatus): string {
  const total = child.usage?.total_tokens;
  if (typeof total !== 'number' || !Number.isFinite(total)) return '';
  return tokenCountLabel(total);
}

/** Whether a row of this child differs from the last one in what an observation shows. */
function rowChanged(before: AgentChildStatus | null | undefined, after: AgentChildStatus): boolean {
  return !before
    || before.status !== after.status
    || before.operationId !== after.operationId
    || before.summary !== after.summary
    || before.pendingQuestions !== after.pendingQuestions;
}

export class DlAgentSession extends LightElement {
  static properties = {
    handles: {attribute: false},
    source: {attribute: false},
    childSessionId: {attribute: false},
    entry: {attribute: false},
    answerFeedbackEnabled: {attribute: false},
    observation: {state: true},
    failed: {state: true},
    missing: {state: true},
    objectiveOpen: {state: true},
    // What a search hands this page; the transcript view needs them to redraw at all.
    terms: {attribute: false},
    found: {attribute: false},
    // The step a finding asked this page to show, when it belongs to this agent. (`focus` is the
    // DOM's own method, so the step a finding names is `stepFocus`.)
    stepFocus: {attribute: false},
  };

  declare handles: AppHandles;
  declare source: TracesSource | null;
  /** The agent to show; the Run's own agent is spelled `main`. */
  declare childSessionId: string;
  /** The roster's row for this agent, which heads the page before its observation arrives. A child
   *  the roster no longer lists is null. */
  declare entry: AgentChildStatus | null;

  /** The messages a search found in this agent, which are the pane while it is on. */
  declare found: readonly ActivityMessage[] | null;
  /** The words being searched for, marked in what is shown. */
  declare terms: readonly string[];

  /** The step a finding sent the reader to, when it belongs to this agent: its agent's wire key and
   *  its sequence in that agent's own Session. The activity pane scrolls to it and rings it. Named
   *  `stepFocus` because `focus` is the DOM's own method on an element. */
  declare stepFocus: {agent: string; sequence: number} | null;
  /** Whether the reader is offered the thumbs for what a child answered. */
  declare answerFeedbackEnabled: boolean;
  declare observation: ChildObservation | null;
  /** The first observation of this child failed, so there is nothing to show yet. */
  declare failed: boolean;
  /** The server no longer knows this child. */
  declare missing: boolean;
  declare objectiveOpen: boolean;

  #observing: {controller: AbortController; again: boolean} | null = null;
  /** Measures the title's clamp again whenever the title is resized: a settled child has no clock to redraw
   * the page, and the dock can be narrowed at any time. */
  readonly #titleSize = new ResizeObserver(() => { this.#measureTitle(); });
  #watched: HTMLElement | null = null;
  /** The clock every time this page draws is drawn against: the page owns it, so a dock that no longer
   *  re-renders every second still shows a running time move. */
  #now = Date.now();
  /** Whether the roster has listed this agent: a null `entry` then means it dropped off, while one
   *  before the roster's first page is the page still waiting for it. */
  #listed = false;

  constructor() {
    super();
    // The host holds the controller for as long as the element lives, so it needs no field here.
    new ClockController(this, () => this.#moving());
    this.handles = productionHandles();
    this.source = null;
    this.childSessionId = '';
    this.entry = null;
    this.answerFeedbackEnabled = false;
    this.observation = null;
    this.failed = false;
    this.missing = false;
    this.objectiveOpen = false;
    // See DlActivity: a class field would shadow Lit's accessor and updates would stop firing.
    this.terms = [];
    this.found = null;
    this.stepFocus = null;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.lifetime.addEventListener('abort', () => {
      this.#titleSize.disconnect();
      this.#watched = null;
    }, {once: true});
  }

  /** Give the title focus: where a reader who has opened this agent belongs. */
  focusTitle(): void {
    this.querySelector<HTMLElement>('[data-title]')?.focus();
  }

  /** Read again what this page shows, for a Run that the dock follows: the child's own row and details,
   *  and the activity pane either way. The dock's events say the Run moved, not which agent did, so the
   *  page reads when the dock asks rather than on every render. */
  refresh(): void {
    if (!this.source || !this.childSessionId) return;
    this.#reobserve();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    // One answer per frame for every time this render draws, the tick's and the dock's redraws alike.
    this.#now = Date.now();
    if (changed.has('source') || changed.has('childSessionId')) {
      this.#observing?.controller.abort();
      this.#observing = null;
      this.observation = null;
      this.failed = false;
      this.missing = false;
      this.objectiveOpen = false;
      this.#listed = this.entry !== null && this.entry.childSessionId === this.childSessionId;
      if (this.source && this.childSessionId) void this.#observe();
    } else if (changed.has('entry') && this.entry && this.entry.childSessionId === this.childSessionId) {
      this.#listed = true;
      // Only a row that states something new is worth a read. A dock that re-renders hands a new row
      // object every time, and reading on identity alone was one transcript read per render; a running
      // child's transcript is refreshed when the dock says the Run moved, through refresh().
      const before = changed.get('entry') as AgentChildStatus | null | undefined;
      if (rowChanged(before, this.entry)) this.#reobserve();
    }
  }

  /** Whether this page draws a time that still moves, which is the only reason it needs a clock. */
  #moving(): boolean {
    const child = this.#child();
    return child !== null && agentIsCounting(child);
  }

  protected override updated(changed: PropertyValues<this>): void {
    this.#measureTitle();
    const title = this.querySelector<HTMLElement>('[data-title]');
    if (title !== this.#watched) {
      this.#titleSize.disconnect();
      this.#watched = title;
      if (title) this.#titleSize.observe(title);
    }
    // A child opens on its title.
    if (changed.has('source') || changed.has('childSessionId')) {
      this.querySelector<HTMLElement>('[data-scroller]')?.scrollTo({top: 0});
    }
  }

  /** The toggle shows only where the title is cut off, or has been opened. */
  #measureTitle(): void {
    const title = this.querySelector<HTMLElement>('[data-title]');
    const more = this.querySelector<HTMLElement>('[data-more]');
    if (title && more) more.hidden = !this.objectiveOpen && title.scrollHeight <= title.clientHeight + 1;
  }

  /** Read the child where it stands, and its activity with it; a read already in flight is not interrupted,
   * one more follows it. */
  #reobserve(): void {
    if (this.#observing) this.#observing.again = true;
    else void this.#observe();
    this.querySelector<DlActivity>('dl-activity')?.refresh();
  }

  async #observe(): Promise<void> {
    const source = this.source;
    const childSessionId = this.childSessionId;
    if (!source || !childSessionId || childSessionId === MAIN_AGENT) return;
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

  /** No child to show: the roster that had listed it no longer does (`entry` is null; the server may
   *  still know it, but a Run has a handful of children, so a listed child does not drop off a page),
   *  or the server answered 404 (`missing`). A null `entry` before the roster arrives is a page that
   *  is still loading, not a missing child: the dock mounts the page and the roster together. */
  #gone(): boolean {
    return this.missing || (this.#listed && this.entry === null);
  }

  /** What the page says out loud: the one state a reader who was reading it should hear. */
  #announcement(): string {
    return this.#gone() ? msg('That child is no longer available.', {id: 'agentSession.gone'}) : '';
  }

  #toggleObjective = (): void => {
    this.objectiveOpen = !this.objectiveOpen;
  };

  /** A question the parent is still waiting to answer. */
  #live(question: ChildQuestion): boolean {
    return question.status === 'pending'
      && (question.expiresAt === null || Date.parse(question.expiresAt) > this.#now);
  }

  protected override render(): TemplateResult | typeof nothing {
    if (!this.childSessionId) return nothing;
    // The status region outlives every view, so a change in it is announced.
    return html`${this.#view()}<span class="dl-sr-only" role="status">${this.#announcement()}</span>`;
  }

  #view(): TemplateResult {
    if (this.#gone()) {
      return html`
        <div class=${styles.gone}>
          <p class=${styles.goneTitle}>${msg('That child is no longer available.', {id: 'agentSession.gone'})}</p>
          <p class=${styles.goneHint}>${msg('Pick another from the list.', {id: 'agentSession.goneHint'})}</p>
        </div>
      `;
    }
    const child = this.#child();
    // The dock mounts this page as soon as the Run opens, with the roster read beside it, so a page
    // whose roster row has not arrived is still a page: the quiet line stands where the row will, and
    // the activity below reads its transcript now rather than after the roster.
    return html`
      <section class=${styles.session} aria-labelledby="agent-session-title">
        <div class=${styles.scroll} data-scroller>
          ${keyed(this.childSessionId, html`
            ${child === null ? this.#waiting() : html`
              ${this.#heading(child)}
              ${this.#statusLine(child)}
              ${this.observation ? this.#questions(this.observation) : nothing}
              ${child.status === 'running' ? nothing : this.#result(child)}
              ${this.observation || this.#main() ? nothing : this.#pending()}
              ${this.observation ? this.#history(this.observation) : nothing}
            `}
            ${this.#activity(child)}
          `)}
        </div>
      </section>
    `;
  }

  /** The page before the roster's row or the child's own observation has arrived: a heading that names
   *  the agent the dock asked for, and the quiet line that says its details are being read. */
  #waiting(): TemplateResult {
    return html`
      <h3 id="agent-session-title" class=${styles.title} data-title tabindex="-1">${agentName(agentIdOf(this.childSessionId))}</h3>
      <p class="dl-note ${styles.quiet}" role="status">${msg('Loading child details…', {id: 'agentSession.loading'})}</p>
    `;
  }

  /** The activity pane, which reads its own transcript: it mounts with the page so the transcript and
   *  the roster load in parallel, and it is the same pane whether the rest of the page is waiting or not. */
  #activity(child: AgentChildStatus | null): TemplateResult {
    return html`
      <dl-activity .source=${this.source} .terms=${this.terms} .found=${this.found}
        .brief=${child?.summary ?? ''} .now=${this.#now} .focusSequence=${this.#focusSequence()}
        .agent=${this.#main() ? null : this.childSessionId}
        .objective=${this.#main() ? '' : child?.objective ?? ''} .controls=${this.observation?.controls ?? NO_CONTROLS}></dl-activity>
    `;
  }

  /** The Run's main agent has this page too, with nothing to steer: its Run is steered from the chat. */
  #main(): boolean {
    return this.childSessionId === MAIN_AGENT;
  }

  /** The step this page is asked to ring, when the focus names the agent this page shows. */
  #focusSequence(): number | null {
    const focus = this.stepFocus;
    if (focus === null) return null;
    return sameAgent(agentIdOf(focus.agent), agentIdOf(this.childSessionId)) ? focus.sequence : null;
  }

  #heading(child: AgentChildStatus): TemplateResult {
    return html`
      ${this.#main() ? html`<p class=${styles.label}>${agentName(RUN_AGENT)}</p>` : nothing}
      <h3 id="agent-session-title" class="${styles.title} ${this.objectiveOpen ? '' : styles.clamped}"
          data-title tabindex="-1">${child.objective || agentName(agentIdOf(this.childSessionId))}</h3>
      <button type="button" class="dl-text-action dl-text-action--disclosure ${styles.more}" data-more hidden
              aria-expanded=${this.objectiveOpen ? 'true' : 'false'}
              aria-controls="agent-session-title" @click=${this.#toggleObjective}>
        ${this.objectiveOpen
          ? msg('Show less', {id: 'agentSession.showLess'})
          : msg('Show full objective', {id: 'agentSession.showFull'})}
        ${icon('chevron-down', {size: 'xs'})}
      </button>
    `;
  }

  #statusLine(child: AgentChildStatus): TemplateResult {
    const meta = [agentElapsed(child, this.#now), tokensText(child)]
      .filter(Boolean).join(' · ');
    return html`
      <div class=${styles.status}>
        ${agentGlyph(child)}
        <b class=${styles.state}>${agentStateText(child)}</b>
        <span>${meta}</span>
        ${this.#feedback(child)}
      </div>
    `;
  }

  /** The thumbs for what a child answered, once it has settled and the server has said which message that
   *  is. The Run's own agent is judged under the chat's turn, not here. */
  #feedback(child: AgentChildStatus): TemplateResult | typeof nothing {
    const runId = this.source?.runId;
    if (!runId || this.#main() || child.status !== 'succeeded' || child.resultSequence === null) return nothing;
    return html`
      <dl-answer-feedback layout="block" .handles=${this.handles} .runId=${runId} .agent=${this.childSessionId}
        .sequence=${child.resultSequence} .judgement=${child.feedback}
        .enabled=${this.answerFeedbackEnabled}></dl-answer-feedback>
    `;
  }

  #questions(observation: ChildObservation): TemplateResult | typeof nothing {
    const waiting = observation.questions.filter((question) => this.#live(question));
    if (waiting.length === 0) return nothing;
    return html`${repeat(waiting, (question) => question.requestId, (question) => this.#card(question))}`;
  }

  /** A question a child asked its parent: a state a reader can see, answered from the conversation's own
   *  composer, never from this review surface. */
  #card(question: ChildQuestion): TemplateResult {
    const minutes = question.expiresAt === null
      ? 0
      : Math.max(1, Math.ceil((Date.parse(question.expiresAt) - this.#now) / MINUTE_MILLISECONDS));
    const label = question.expiresAt === null
      ? msg('Asking the parent', {id: 'agentSession.asking'})
      : msg(str`Asking the parent · expires in ${minutes} min`, {id: 'agentSession.askingExpires'});
    return html`
      <div class=${styles.card} role="group" aria-labelledby=${`child-question-${question.requestId}`}>
        <p class=${styles.cap} id=${`child-question-${question.requestId}`}>${label}</p>
        <p class=${styles.asked}>${question.question}</p>
      </div>
    `;
  }

  /** What a Run failed at, and what a child produced.
   *
   *  The Run's own agent says nothing here about its answer: the chat shows that answer and the
   *  sources it cites, and this pane stands beside the chat. What a reader of traces is asking
   *  instead is why a Run stopped, so a failure is said here too. A child's evidence is said
   *  nowhere else — the references beside the answer are the Run's, not a child's — so a child
   *  keeps its own, and says nothing else.
   */
  #result(child: AgentChildStatus): TemplateResult | typeof nothing {
    const main = this.#main();
    const failed = main && child.status === 'failed';
    const words = failed ? child.summary || '' : '';
    const handles = this.observation?.result?.handles ?? child.resultHandles;
    const evidence = main ? [] : handles;
    if (!words && evidence.length === 0) return nothing;
    return html`
      <section class=${styles.section}>
        ${words
          ? html`<h4 class=${styles.label}>${msg('Result', {id: 'agentSession.result'})}</h4>
              <p class=${styles.resultText} tabindex="0">${words}</p>`
          : nothing}
        ${evidence.length > 0 ? evidenceFold(this, evidence, this.source?.presentation() ?? null) : nothing}
      </section>
    `;
  }

  /** Until the observation arrives the page has only the roster's row to show. */
  #pending(): TemplateResult {
    return this.failed
      ? html`
        <div class=${styles.failed}>
          <p role="alert">${msg('Child details could not be loaded.', {id: 'agentSession.loadFailed'})}</p>
          <button type="button" class="dl-btn" @click=${this.#retry}>${msg('Retry', {id: 'agentSession.retry'})}</button>
        </div>`
      : html`<p class="dl-note ${styles.quiet}">${msg('Loading child details…', {id: 'agentSession.loading'})}</p>`;
  }

  #history(observation: ChildObservation): TemplateResult {
    const asked = observation.questions.filter((question) => !this.#live(question));
    const controls = observation.controls;
    return html`
      ${asked.length > 0 ? html`
        <section class=${styles.section}>
          <h4 class=${styles.label}>${msg('Questions', {id: 'agentSession.questions'})}</h4>
          <ul class=${styles.history}>
            ${asked.map((question) => this.#asked(question))}
          </ul>
        </section>
      ` : nothing}
      ${controls.length > 0 ? html`
        <details class=${styles.fold}>
          <summary class=${styles.summary}>${icon('disclosure', {size: 'xs', className: styles.chevron})}
            ${msg(str`Instructions received · ${controls.length}`, {id: 'agentSession.instructionsReceived'})}</summary>
          <ul class=${styles.history}>
            ${controls.map((record) => html`
              <li>
                <p class=${styles.meta}>${[
                  senderText(record.origin),
                  record.consumed
                    ? msg('Handled', {id: 'agentSession.control.handled'})
                    : msg('Pending', {id: 'agentSession.control.pending'}),
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
      ? msg('Answered', {id: 'agentSession.question.replied'})
      : status === 'cancelled'
        ? msg('Cancelled', {id: 'agentSession.question.cancelled'})
        : msg('Expired', {id: 'agentSession.question.expired'});
    return html`
      <li>
        <p class=${styles.toldText}>${question.question}</p>
        <p class=${styles.meta}>${[word, senderText(question.replyOrigin)].filter(Boolean).join(' · ')}</p>
        ${question.reply ? html`<p class=${styles.toldText}>${question.reply}</p>` : nothing}
      </li>
    `;
  }
}

customElements.define('dl-agent-session', DlAgentSession);

declare global {
  interface HTMLElementTagNameMap {
    'dl-agent-session': DlAgentSession;
  }
}
