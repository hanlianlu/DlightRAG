// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** One agent's activity as a fold in its page: what it said, each tool it called beside what came back, and
 * what it was told.
 *
 * It reads the agent's transcript a page at a time, newest page first, and shows older pages on request, so
 * a reader can go back through the whole of it. The fold is open while the agent works, and its steps scroll
 * in a pane of their own that opens on the newest step. A reader at the bottom of a running agent is kept
 * there as steps arrive; one who has scrolled up is left where they are.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {ActivityMessage, ActivityPage} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {type ActivityControl, type ActivityStep, joinActivity, projectActivity} from '../lib/activity.ts';
import {toolTime} from '../lib/date-format.ts';
import {LightElement} from '../lib/lit-host.ts';
import {KeysetPager} from '../lib/paged.ts';
import {roomBelow} from '../lib/window-fit.ts';
import styles from '../styles/activity.module.css';
import sessionStyles from '../styles/agent-session.module.css';
import {foldView} from './fold.ts';
import {loadOlderControl} from './load-older.ts';
import {markedText} from './marked-text.ts';

/** Where a timeline reads from: one page of an agent's transcript, the Run's main agent when `agent` is null. */
interface ActivityReader {
  activity(agent: string | null, cursor: string | null, signal: AbortSignal): Promise<ActivityPage>;
  /** The messages written after a sequence, when the reader can be handed them that way. */
  since?(agent: string | null, sequence: number, signal: AbortSignal): Promise<ActivityPage>;
}

/** Who wrote an instruction, as the timeline labels it. */
export function senderText(origin: string | null): string {
  switch (origin) {
    case 'user': return msg('You', {id: 'agentSession.you'});
    case 'parent': return msg('Parent', {id: 'agentSession.parent'});
    default: return '';
  }
}

/** Whether a pane is scrolled to within one line of its bottom, as a pane that fits its content is. */
function scrolledToBottom(pane: HTMLElement): boolean {
  const line = Number.parseFloat(getComputedStyle(pane).lineHeight);
  return pane.scrollHeight - pane.scrollTop - pane.clientHeight <= (Number.isFinite(line) ? line : 1);
}

/** What a window can hold without becoming a scroll of its own: about eight lines of words. */
const WINDOW_WORDS = 600;

/** The least a window may shrink to, when the page it sits in has almost no room left. */
const MIN_WINDOW = 160;

/** How often the pane reads itself forward while its agent is working.
 *
 * The dock asks its pages to refresh on Run events, but that leaves the pane dependent on a reader
 * above it: a continuation Run whose roster/agent state makes the dock think the page is settled, or
 * a follow path that never reaches this pane, leaves a live transcript frozen on its first (empty)
 * page. The pane already knows whether the agent is working from the page it read, so it keeps
 * itself current and treats the dock's refresh as an accelerator rather than the only pulse.
 */
const FOLLOW_MILLISECONDS = 1500;

/** Whether these words are laid out or opened in a window. A search is about finding words, so
 *  nothing that holds them is folded away while one is on — which is how a match appeared in an
 *  agent's row and then could not be seen in its page. */
function folded(text: string, searching: boolean): boolean {
  return !searching && (text.length > WINDOW_WORDS || text.split('\n').length > 8);
}

/** The first line of a long message, so the fold says what it holds. */
function opening(text: string): string {
  const line = text.split('\n').find((one) => one.trim() !== '')?.trim() ?? '';
  return line.length > 110 ? `${line.slice(0, 109)}…` : line;
}

export class DlActivity extends LightElement {
  static properties = {
    source: {attribute: false},
    agent: {attribute: false},
    controls: {attribute: false},
    objective: {attribute: false},
    messages: {state: true},
    running: {state: true},
    loaded: {state: true},
    failed: {state: true},
    // What a search hands this pane: without these the pane keeps drawing the page it read, and a
    // search looks like it did nothing at all.
    terms: {attribute: false},
    brief: {attribute: false},
    found: {attribute: false},
    now: {attribute: false},
    // The step a finding asked this pane to show: its sequence in this agent's own Session.
    focusSequence: {attribute: false},
  };

  declare source: ActivityReader | null;
  /** The child to show; null for the Run's main agent. */
  declare agent: string | null;
  /** The agent's control records, which name who sent each steer. */
  declare controls: readonly ActivityControl[];
  /** The agent's task, which its first user message records and the timeline leaves out. */
  declare objective: string;
  /** The transcript as read so far, oldest first. */
  declare messages: readonly ActivityMessage[];
  /** Whether the agent is working, as the latest newest page said. */
  declare running: boolean;

  /** The messages a search found in this agent. While they are here, they are the pane: a search
   *  scopes what a reader is already reading, rather than opening something beside it. */
  declare found: readonly ActivityMessage[] | null;

  /** What this agent ended with — a Run's answer, or the brief a child handed back. Folded like a
   *  tool call, because it is the end of the work and not the work: a reader opens it when they
   *  want the conclusion, and reads the steps while it stays shut. */
  declare brief: string;

  /** The words a reader searched for: marked in the steps, so a narrowed pane says why. */
  declare terms: readonly string[];

  /** The clock this render holds, handed down by the page that owns it: a Tool call still in flight
   *  is timed against it, so a reader watching one sees it climb. */
  declare now: number;

  /** The step a finding sent the reader to, by its sequence in this agent's Session; null when no
   *  step is asked for. The pane opens to it and rings it, and a page that does not hold it yet
   *  rings it when the step arrives. */
  declare focusSequence: number | null;

  /** Whether the reader has this pane's work open. It starts from what the agent is doing, and
   *  belongs to the reader from their first press on: a re-render — and a running agent redraws
   *  every second — must never shut a fold the reader opened. */
  #open: boolean | null = null;

  declare loaded: boolean;
  /** The first page could not be read, so there is nothing to show yet. */
  declare failed: boolean;

  /** The page of words this pane has already gone to, so it goes there once. */
  #revealed: string | null = null;

  /** Whether a newly asked step still has to be scrolled to: it may not be loaded yet. */
  #focusPending = false;

  readonly #pager = new KeysetPager<ActivityPage>(
    // A pane starts a read only while it has a source; one whose source was taken away mid-read ends as a failed load.
    (cursor, signal) => this.source === null
      ? Promise.reject(new Error('The activity pane has no source to read from'))
      : this.source.activity(this.agent, cursor, signal),
    () => { this.requestUpdate(); },
  );
  #head: AbortController | null = null;
  #scroller: HTMLElement | null = null;
  /** The pane has been put on the newest step since it was last folded away. */
  #placed = false;
  /** The self-follow timer: a live pane reads itself when the dock above it has gone quiet. */
  #follow: ReturnType<typeof setTimeout> | null = null;
  /** When this pane last read (or was asked to read), for the self-follow's quiet check. */
  #lastPoke = 0;
  /** The reader is at the bottom of an agent that is working, so the bottom stays in view as steps arrive. */
  #following = false;
  /** Where the pane stood before older steps came in above what the reader is looking at. */
  #anchor: {height: number; top: number} | null = null;

  /** The scroller this window is fitted to, and what watches it for changes. */
  #room: HTMLElement | null = null;
  #roomObserver: ResizeObserver | null = null;

  constructor() {
    super();
    this.source = null;
    this.agent = null;
    this.controls = [];
    this.objective = '';
    this.messages = [];
    this.running = false;
    this.loaded = false;
    this.failed = false;
    // Assigned here, not as class field initialisers: under `[[Define]]` semantics a field would
    // shadow the accessor Lit installs, and the pane would never redraw when a search hands it
    // the messages it found.
    this.terms = [];
    this.brief = '';
    this.found = null;
    // A page that owns no clock of its own still has one: the host redraws it with every tick.
    this.now = Date.now();
    this.focusSequence = null;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.classList.add(styles.root);
  }

  override disconnectedCallback(): void {
    this.#stopFollow();
    this.#head?.abort();
    this.#head = null;
    this.#scroller?.removeEventListener('scroll', this.#scrolled);
    this.#scroller = null;
    super.disconnectedCallback();
  }

  /** Read the newest page again and join it to what is shown, which stays until it lands. A read already in
   * flight is replaced. The host decides how often. */
  refresh(): void {
    if (!this.source) return;
    this.#lastPoke = performance.now();
    // A pane holding what a search found is not waiting for a page: those messages are it.
    if (!this.loaded && this.found === null) {
      if (this.failed) void this.#start();
      return;
    }
    void this.#join(this.source);
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    // A search replaces what this pane holds and takes it back when it is over: from the messages
    // it found, and back to the Session's own pages, which are read again.
    if (changed.has('source') || changed.has('agent') || changed.has('found')) {
      this.#stopFollow();
      this.#lastPoke = 0;
      this.#pager.reset(null);
      this.#head?.abort();
      this.#head = null;
      this.messages = [];
      this.running = false;
      this.loaded = false;
      this.failed = false;
      this.#placed = false;
      this.#following = false;
      this.#anchor = null;
      if (this.source && this.found === null) void this.#start();
    }
    // A finding opens the page on its step: the fold opens so the ring can be seen, and the scroll
    // waits until the step is in the page (a later page load retries it).
    if (changed.has('focusSequence')) {
      if (this.focusSequence !== null) this.#open = true;
      this.#focusPending = true;
    }
    // Measured before the update adds to the pane: a reader who has left the bottom, or an agent that has
    // settled, is no longer followed.
    const pane = this.#scroller;
    if (this.#following && !(this.running && pane && scrolledToBottom(pane))) this.#following = false;
  }

  protected override updated(): void {
    this.#revealFirstMark();
    const pane = this.#bindScroller();
    if (!pane) {
      this.#anchor = null;
      return;
    }
    this.#fitWindow(pane);
    // A folded pane has no place to keep, so it opens on the newest step again.
    if (pane.clientHeight === 0) this.#placed = false;
    if (this.#anchor) {
      pane.scrollTop = this.#anchor.top + (pane.scrollHeight - this.#anchor.height);
      this.#anchor = null;
    } else if (!this.#placed && this.loaded && pane.clientHeight > 0) {
      this.#placed = true;
      this.#following = this.running;
      pane.scrollTop = pane.scrollHeight;
    } else if (this.#following && !(pane?.matches(':hover') ?? false)) {
      // A pane that moves under the pointer takes the line a reader was pressing with it: while the
      // pointer is over the pane, the follow waits, and it picks up again when the pointer leaves.

      pane.scrollTop = pane.scrollHeight;
    }
    this.#revealFocus();
  }

  /** Ring the step a finding asked for, and put it in view, once it is in the page. A step that is
   *  not loaded yet is left to the render that will hold it; clearing the focus takes the ring away. */
  #revealFocus(): void {
    const wanted = this.focusSequence;
    if (wanted === null) {
      for (const step of this.querySelectorAll<HTMLElement>('[data-focus]')) {
        step.removeAttribute('data-focus');
      }
      this.#focusPending = false;
      return;
    }
    const step = [...this.querySelectorAll<HTMLElement>('[data-sequence]')]
      .find((candidate) => Number(candidate.dataset.sequence) === wanted) ?? null;
    if (step === null) return;
    for (const other of this.querySelectorAll<HTMLElement>('[data-focus]')) {
      if (other !== step) other.removeAttribute('data-focus');
    }
    step.setAttribute('data-focus', 'true');
    if (this.#focusPending) {
      step.scrollIntoView({block: 'center'});
      this.#focusPending = false;
    }
  }

  /**
   * The window is as tall as the room its page has left, not as tall as the viewport. A fold opened
   * near the bottom of a page that is capped by `88vh` hangs past the end of it, and the page — not
   * the window — becomes the thing that scrolls.
   *
   * The room cannot come from CSS: a `details` lays its content out in a slot, so no flex chain
   * reaches the window. It is measured instead, from the fold's top to the bottom of whatever
   * scrolls this page, in the scroller's own coordinates so that its scroll position cannot make the
   * answer — and with it the window's height — move.
   */
  #fitWindow(pane: HTMLElement): void {
    const scroller = this.#scrollerAbove();
    if (scroller === null) {
      pane.style.removeProperty('max-height');
      return;
    }
    const room = roomBelow({
      height: scroller.clientHeight,
      top: pane.getBoundingClientRect().top,
      scrollerTop: scroller.getBoundingClientRect().top,
      paddingBottom: Number.parseFloat(getComputedStyle(scroller).paddingBottom) || 0,
    });
    pane.style.maxHeight = `${Math.max(MIN_WINDOW, Math.round(room))}px`;
    if (this.#room !== scroller) {
      this.#roomObserver?.disconnect();
      this.#room = scroller;
      this.#roomObserver = new ResizeObserver(() => { this.#fitWindow(pane); });
      this.#roomObserver.observe(scroller);
    }
  }

  /** Whatever scrolls this page: the pane's own window is inside it, so look above this element. */
  #scrollerAbove(): HTMLElement | null {
    for (let node = this.parentElement; node !== null; node = node.parentElement) {
      const overflow = getComputedStyle(node).overflowY;
      if (overflow === 'auto' || overflow === 'scroll') return node;
    }
    return null;
  }

  #bindScroller(): HTMLElement | null {
    const pane = this.querySelector<HTMLElement>('[data-activity-pane]');
    if (pane !== this.#scroller) {
      this.#scroller?.removeEventListener('scroll', this.#scrolled);
      this.#scroller = pane;
      pane?.addEventListener('scroll', this.#scrolled, {passive: true});
    }
    return pane;
  }

  #scrolled = (): void => {
    const pane = this.#scroller;
    this.#following = this.running && pane !== null && scrolledToBottom(pane);
  };

  /** The fold opened or closed, which only an update can see in the pane's size. */
  #toggled = (event: Event): void => {
    // Remember what the reader chose: from here the fold opens and shuts because they said so,
    // not because the agent it belongs to drew another second.
    this.#open = (event.currentTarget as HTMLDetailsElement).open;
    this.requestUpdate();
  };

  /** Keep a live pane current without waiting to be poked: one quiet read per interval. */
  #scheduleFollow(): void {
    this.#stopFollow();
    if (!this.running || !this.isConnected) return;
    this.#follow = setTimeout(() => {
      this.#follow = null;
      if (!this.isConnected || !this.running || this.source === null) return;
      // The dock asked for a read inside this interval: it is the pulse, and a second one here
      // would be the duplicate the dock's own throttle exists to avoid.
      if (performance.now() - this.#lastPoke >= FOLLOW_MILLISECONDS / 2) {
        this.#lastPoke = performance.now();
        void this.#join(this.source);
      }
      this.#scheduleFollow();
    }, FOLLOW_MILLISECONDS);
  }

  #stopFollow(): void {
    if (this.#follow !== null) clearTimeout(this.#follow);
    this.#follow = null;
  }

  async #start(): Promise<void> {
    // A search's messages are the whole pane: there is no page behind them to read.
    if (this.found !== null) return;
    this.failed = false;
    await this.#pager.start((page) => {
      this.messages = page.messages;
      this.running = page.running;
      this.loaded = true;
      this.#lastPoke = performance.now();
    }, () => {
      this.failed = true;
    });
    this.#scheduleFollow();
  }

  /** The newest sequence this pane holds, or 0 when it holds nothing. */
  #newestSequence(): number {
    return this.messages.reduce((newest, message) => Math.max(newest, message.sequence), 0);
  }

  async #join(source: ActivityReader): Promise<void> {
    this.#head?.abort();
    const read = new AbortController();
    this.#head = read;
    try {
      // The read aborts on disconnect through this controller; composing the element's lifetime
      // signal as well used to abort every forward read without a request when that signal was
      // spent, and a read that never left the page looked exactly like an empty transcript.
      const signal = read.signal;
      const held = this.#newestSequence();
      // A delta is what this pane does not hold yet, and it starts where the pane left off: the
      // join's reach-back rule would read that as a hole and throw the history away, so a delta is
      // appended, and only a page goes through the join.
      if (held > 0 && source.since) {
        const delta = await source.since(this.agent, held, signal);
        if (this.#head !== read) return;
        this.messages = [
          ...this.messages,
          ...delta.messages.filter((message) => message.sequence > held),
        ];
        this.running = delta.running;
        return;
      }
      const page = await source.activity(this.agent, null, signal);
      if (this.#head !== read) return;
      const joined = joinActivity(this.messages, page.messages);
      // A page that leaves a gap starts over, and the older pages shown give way to its own cursor.
      if (joined === null) this.#pager.reset(page.nextCursor);
      this.messages = joined ?? page.messages;
      this.running = page.running;
    } catch {
      // A read that fails leaves the steps on screen until the next one.
    } finally {
      if (this.#head === read) {
        this.#head = null;
        this.#lastPoke = performance.now();
        this.#scheduleFollow();
      }
    }
  }

  #revealFirstMark(): void {
    if (this.terms.length === 0) {
      this.#revealed = null;
      return;
    }
    const first = this.messages[0]?.sequence ?? 0;
    const last = this.messages[this.messages.length - 1]?.sequence ?? 0;
    const key = `${first}:${last}:${this.terms.join(',')}`;
    if (this.#revealed === key) return;
    const mark = this.querySelector('mark');
    if (mark === null) return;
    this.#revealed = key;
    mark.scrollIntoView({block: 'center'});
  }

  #loadOlder = (): void => {
    void this.#loadOlderPage();
  };

  #loadOlderPage = (): Promise<void> => this.#pager.loadNext((page) => {
    const pane = this.#scroller;
    if (pane) this.#anchor = {height: pane.scrollHeight, top: pane.scrollTop};
    this.messages = [...page.messages, ...this.messages];
  });

  #retry = (): void => {
    void this.#start();
  };

  protected override render(): TemplateResult | typeof nothing {
    if (!this.source) return nothing;
    const messages = this.found ?? this.messages;
    const steps = projectActivity(messages, {
      objective: this.objective,
      running: this.running,
      controls: this.controls,
      terms: this.terms,
      brief: this.brief,
      now: this.now,
    });

    return html`
      <details class=${sessionStyles.fold} ?open=${this.#open ?? (this.running || this.terms.length > 0)}
               @toggle=${this.#toggled}>
        <summary class=${sessionStyles.summary}>${icon('disclosure', {size: 'xs', className: sessionStyles.chevron})}
          ${this.#title(steps.filter((step) => step.kind !== 'brief').length)}</summary>
        <div class=${styles.pane} data-activity-pane tabindex="0" role="region"
             aria-label=${msg('Agent timeline', {id: 'activity.title'})}>${this.#timeline(steps)}</div>
      </details>
    `;
  }

  #title(count: number): string {
    if (this.running || !this.loaded) return msg('Agent timeline', {id: 'activity.title'});
    if (this.#pager.hasOlder) return msg(str`Agent timeline · ${count}+ steps`, {id: 'activity.stepsMore'});
    return count === 1
      ? msg('Agent timeline · 1 step', {id: 'activity.oneStep'})
      : msg(str`Agent timeline · ${count} steps`, {id: 'activity.steps'});
  }

  #timeline(steps: readonly ActivityStep[]): TemplateResult {
    // A pane holding what a search found is not waiting for a page: those messages are it.
    if (!this.loaded && this.found === null) {
      return this.failed
        ? html`
          <div class=${sessionStyles.failed}>
            <p role="alert">${msg('Activity could not be loaded.', {id: 'activity.loadFailed'})}</p>
            <button type="button" class="dl-btn" @click=${this.#retry}>${msg('Retry', {id: 'activity.retry'})}</button>
          </div>`
        : html`<p class="dl-note ${sessionStyles.quiet}" role="status">${msg('Loading activity…', {id: 'activity.loading'})}</p>`;
    }
    const toolSequences = this.#toolSequences();
    return html`
      ${this.found !== null ? nothing : loadOlderControl({
        list: 'activity',
        pages: this.#pager,
        label: msg('Show earlier steps', {id: 'activity.loadOlder'}),
        retryLabel: msg('Retry showing earlier steps', {id: 'activity.retryLoadOlder'}),
        loading: msg('Loading earlier steps…', {id: 'activity.loadingOlder'}),
        loaded: msg('Earlier steps loaded.', {id: 'activity.loadedOlder'}),
        failed: msg('Earlier steps could not be loaded.', {id: 'activity.olderFailed'}),
        onLoad: this.#loadOlder,
        rowClass: styles.older,
        buttonClass: 'dl-btn',
      })}
      ${steps.length === 0 && this.brief.trim() === ''
        ? html`<p class="dl-note ${sessionStyles.quiet}">${msg('No activity yet.', {id: 'activity.none'})}</p>`
        : html`<ol class=${styles.steps}>
            ${repeat(steps, (step) => step.key, (step) => this.#step(step, toolSequences))}
          </ol>`}
    `;
  }

  /** Where the messages behind the Tool steps are, so a Tool row can carry the sequence a finding
   *  names: a Tool step is keyed by its call id, which says nothing about where it sits. */
  #toolSequences(): ReadonlyMap<string, number> {
    const found = new Map<string, number>();
    for (const message of this.messages) {
      if (message.role !== 'assistant') continue;
      message.toolCalls.forEach((call, order) => {
        found.set(call.id ? `tool:${call.id}` : `tool:${message.sequence}.${order}`, message.sequence);
      });
    }
    return found;
  }

  /** Where a step sits in its Session, when it is one message's own words: what opens a hit. A
   *  thinking step names a message too, so a searching row can read it where it happened even when
   *  the message said nothing but chose Tools. */
  #sequenceOf(step: ActivityStep, toolSequences: ReadonlyMap<string, number>): number | typeof nothing {
    const match = /^(?:say|thinking|told|delivery):(\d+)$/.exec(step.key);
    if (match !== null) return Number(match[1]);
    return toolSequences.get(step.key) ?? nothing;
  }

  #step(step: ActivityStep, toolSequences: ReadonlyMap<string, number>): TemplateResult {
    const sequence = this.#sequenceOf(step, toolSequences);
    if (step.kind === 'brief') {
      // The same fold a tool call opens, pinned to the top of the pane: a list can be a hundred
      // steps long, and the words a reader came for are the ones it ends on. The reasoning that
      // produced those words folds above them, inside the same window.
      // A window holding a fold cannot be a `pre`: the reasoning is a block, so the words and the
      // fold share the same rich body the Tool windows use, and the sticky `li.brief` stays.
      return foldView({
        className: styles.brief,
        label: html`<span class=${styles.verb}>${this.agent === null
          ? msg('Final answer', {id: 'activity.finalAnswer'})
          : msg('Brief', {id: 'activity.brief'})}</span>`,
        body: step.reasoning === undefined
          ? step.text
          : html`${this.#thinkingFold(step.reasoning)}${step.text}`,
        wordless: step.reasoning !== undefined,
        stepKind: 'brief',
      });
    }
    if (step.kind === 'thinking') {
      // Reasoning with no words of its own to hang on: a step before the calls it explains, in the
      // same fold a say step's reasoning is. It names the message it came from, like the words do.
      return html`<li class=${styles.step} data-sequence=${sequence} data-step-kind="thinking">${this.#thinkingFold(step.text)}</li>`;
    }
    if (step.kind === 'say') {
      // The reasoning this message showed before it spoke is how the step was reached, so it is a
      // fold above the words, shut until a reader wants it. The templates below stay on one line
      // inside the `li`: literal layout whitespace would become text between adjacent steps.
      const thinking = step.reasoning === undefined ? nothing : this.#thinkingFold(step.reasoning);
      // Words a window cannot hold are not laid out: a page is as long as its steps, and a reader
      // who wants the whole of one opens it, the way they open a tool result.
      return folded(step.text, this.terms.length > 0)
        ? html`<li class=${styles.step} data-sequence=${sequence} data-step-kind="say">${thinking}${foldView({
          as: 'div',
          label: markedText(opening(step.text), this.terms),
          body: markedText(step.text, this.terms),
          wordless: true, prose: true,
        })}</li>`
        : html`<li class=${styles.step} data-sequence=${sequence} data-step-kind="say">${thinking}<p class=${styles.say}>${markedText(step.text, this.terms)}</p></li>`;
    }
    if (step.kind === 'delivery') {
      return html`
        <li class=${styles.step} data-sequence=${sequence} data-step-kind="delivery">
          <p class=${styles.deliveryCheck}>${msg('Delivery note', {id: 'activity.deliveryNote'})}</p>
          <p class=${styles.deliveryText}>${markedText(step.text, this.terms)}</p>
        </li>
      `;
    }
    if (step.kind === 'instruction') {
      const sender = senderText(step.sender);
      if (folded(step.text, this.terms.length > 0)) {
        return foldView({
          label: html`${sender ? html`<span class=${styles.told}>${sender}</span> ` : nothing}${markedText(opening(step.text), this.terms)}`,
          body: markedText(step.text, this.terms),
          wordless: true, prose: true, sequence, stepKind: 'instruction',
        });
      }
      return html`
        <li class=${styles.step} data-sequence=${sequence} data-step-kind="instruction">
          ${sender ? html`<p class=${styles.told}>${sender}</p>` : nothing}
          <p class=${styles.toldText}>${markedText(step.text, this.terms)}</p>
        </li>
      `;
    }
    const state = step.state === 'running'
      ? msg('Running', {id: 'agentSession.state.running'})
      : step.state === 'failed' ? msg('Failed', {id: 'agentSession.state.failed'}) : '';
    // How long this step took, by the one rule every surface that times a Tool call uses.
    const duration = toolTime(step.durationMs, step.state === 'running');
    const line = html`
      ${state ? html`<span class="dl-sr-only">${state}: </span>` : nothing}
      <span class=${styles.verb}>${step.verb} ${duration === ''
        ? nothing
        : html`<span class=${styles.duration}>${duration}</span>`}</span>
      ${step.subject ? html`<span class=${styles.subject}>${step.subject}</span>` : nothing}
      ${step.excerpt ? html`<span class=${styles.excerpt}>${step.excerpt}</span>` : nothing}
    `;
    const tone = step.state === 'running' ? styles.stepRunning : step.state === 'failed' ? styles.stepFailed : '';
    // A call id names the Tool call; the key of a call whose id was lost carries its order instead.
    const callId = step.key.startsWith('tool:') && !/^\d+\.\d+$/.test(step.key.slice(5))
      ? step.key.slice(5)
      : undefined;
    // A result that is no more than its excerpt has nothing to open.
    return step.full.trim() === step.excerpt
      ? html`<li class="${styles.step} ${tone}" data-sequence=${sequence} data-call-id=${callId ?? nothing} data-step-kind="tool">${line}</li>`
      : foldView({label: line, body: step.full, tone, callId, sequence, stepKind: 'tool'});
  }

  /** The reasoning behind one step, folded shut: how the agent got to what it did, not the thing it
   *  did. One helper, so a say step's reasoning, a standalone thinking step, and a conclusion's
   *  reasoning look and behave the same. */
  #thinkingFold(text: string): TemplateResult {
    return html`<div class=${styles.thinking}>${foldView({
      as: 'div',
      label: html`<span class=${styles.verb}>${msg('Thinking', {id: 'activity.thinking'})}</span>`,
      body: text,
      wordless: true,
      prose: true,
    })}</div>`;
  }
}

customElements.define('dl-activity', DlActivity);

declare global {
  interface HTMLElementTagNameMap {
    'dl-activity': DlActivity;
  }
}
