// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** One agent's activity as a fold in its page: what it said, each tool it called beside what came back, and
 * what it was told.
 *
 * It reads the agent's transcript a page at a time, newest page first, and shows older pages on request, so
 * a reader can go back through the whole of it. The fold is open while the agent works. The page around it
 * owns the scroll (its nearest `[data-scroller]` ancestor): the reader's place is left alone until they
 * scroll to the bottom of a running agent, which keeps the bottom in view as steps arrive.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {ActivityMessage, ActivityPage} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {joinActivity, projectActivity, type ActivityControl, type ActivityStep} from '../lib/activity.ts';
import {LightElement} from '../lib/lit-host.ts';
import {KeysetPager} from '../lib/paged.ts';
import styles from '../styles/activity.module.css';
import sessionStyles from '../styles/agent-session.module.css';
import {loadOlderControl} from './load-older.ts';

/** Where a timeline reads from: one page of an agent's transcript, the Run's main agent when `agent` is null. */
export interface ActivityReader {
  activity(agent: string | null, cursor: string | null, signal: AbortSignal): Promise<ActivityPage>;
}

/** Who wrote an instruction, as the timeline labels it. */
export function senderText(origin: string | null): string {
  switch (origin) {
    case 'user': return msg('You', {id: 'agentSession.you'});
    case 'parent': return msg('Parent', {id: 'agentSession.parent'});
    default: return '';
  }
}

/** Whether the reader has scrolled a pane down to within one line of its bottom. A pane that has not been
 * scrolled down is not at a bottom the reader chose, however short it is. */
function scrolledToBottom(pane: HTMLElement): boolean {
  const line = Number.parseFloat(getComputedStyle(pane).lineHeight);
  return pane.scrollTop > 0
    && pane.scrollHeight - pane.scrollTop - pane.clientHeight <= (Number.isFinite(line) ? line : 1);
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
  declare loaded: boolean;
  /** The first page could not be read, so there is nothing to show yet. */
  declare failed: boolean;

  readonly #pager = new KeysetPager<ActivityPage>(
    (cursor, signal) => this.source!.activity(this.agent, cursor, signal),
    () => { this.requestUpdate(); },
  );
  #head: AbortController | null = null;
  #scroller: HTMLElement | null = null;
  /** The reader is at the bottom of an agent that is working, so the bottom stays in view as steps arrive. */
  #following = false;
  /** Where the pane stood before older steps came in above what the reader is looking at. */
  #anchor: {height: number; top: number} | null = null;

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
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.classList.add(styles.root!);
  }

  override disconnectedCallback(): void {
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
    if (!this.loaded) {
      if (this.failed) void this.#start();
      return;
    }
    void this.#join(this.source);
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (changed.has('source') || changed.has('agent')) {
      this.#pager.reset(null);
      this.#head?.abort();
      this.#head = null;
      this.messages = [];
      this.running = false;
      this.loaded = false;
      this.failed = false;
      this.#following = false;
      this.#anchor = null;
      if (this.source) void this.#start();
    }
    // Measured before the update adds to the pane: a reader who has left the bottom, or an agent that has
    // settled, is no longer followed.
    const pane = this.#scroller;
    if (this.#following && !(this.running && pane && scrolledToBottom(pane))) this.#following = false;
  }

  protected override updated(): void {
    const pane = this.#bindScroller();
    if (!pane) {
      this.#anchor = null;
      return;
    }
    if (this.#anchor) {
      pane.scrollTop = this.#anchor.top + (pane.scrollHeight - this.#anchor.height);
      this.#anchor = null;
    } else if (this.#following) {
      pane.scrollTop = pane.scrollHeight;
    }
  }

  #bindScroller(): HTMLElement | null {
    const pane = this.closest<HTMLElement>('[data-scroller]');
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

  async #start(): Promise<void> {
    this.failed = false;
    await this.#pager.start((page) => {
      this.messages = page.messages;
      this.running = page.running;
      this.loaded = true;
    }, () => {
      this.failed = true;
    });
  }

  async #join(source: ActivityReader): Promise<void> {
    this.#head?.abort();
    const read = new AbortController();
    this.#head = read;
    try {
      const page = await source.activity(this.agent, null, AbortSignal.any([this.lifetime, read.signal]));
      if (this.#head !== read) return;
      const joined = joinActivity(this.messages, page.messages);
      // A page that leaves a gap starts over, and the older pages shown give way to its own cursor.
      if (joined === null) this.#pager.reset(page.nextCursor);
      this.messages = joined ?? page.messages;
      this.running = page.running;
    } catch {
      // A read that fails leaves the steps on screen until the next one.
    } finally {
      if (this.#head === read) this.#head = null;
    }
  }

  #loadOlder = (): void => {
    void this.#pager.loadNext((page) => {
      const pane = this.#scroller;
      if (pane) this.#anchor = {height: pane.scrollHeight, top: pane.scrollTop};
      this.messages = [...page.messages, ...this.messages];
    });
  };

  #retry = (): void => {
    void this.#start();
  };

  protected override render(): TemplateResult | typeof nothing {
    if (!this.source) return nothing;
    const steps = projectActivity(this.messages, {
      objective: this.objective,
      running: this.running,
      controls: this.controls,
    });
    return html`
      <details class=${sessionStyles.fold} ?open=${this.running}>
        <summary class=${sessionStyles.summary}>${icon('disclosure', {size: 'xs', className: sessionStyles.chevron})}
          ${this.#title(steps.length)}</summary>
        ${this.#timeline(steps)}
      </details>
    `;
  }

  #title(count: number): string {
    if (this.running || !this.loaded) return msg('Activity', {id: 'activity.title'});
    if (this.#pager.hasOlder) return msg(str`Activity · ${count}+ steps`, {id: 'activity.stepsMore'});
    return count === 1
      ? msg('Activity · 1 step', {id: 'activity.oneStep'})
      : msg(str`Activity · ${count} steps`, {id: 'activity.steps'});
  }

  #timeline(steps: readonly ActivityStep[]): TemplateResult {
    if (!this.loaded) {
      return this.failed
        ? html`
          <div class=${sessionStyles.failed}>
            <p role="alert">${msg('Activity could not be loaded.', {id: 'activity.loadFailed'})}</p>
            <button type="button" class="dl-btn" @click=${this.#retry}>${msg('Retry', {id: 'activity.retry'})}</button>
          </div>`
        : html`<p class=${sessionStyles.quiet} role="status">${msg('Loading activity…', {id: 'activity.loading'})}</p>`;
    }
    return html`
      ${loadOlderControl({
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
      ${steps.length === 0
        ? html`<p class=${sessionStyles.quiet}>${msg('No activity yet.', {id: 'activity.none'})}</p>`
        : html`<ol class=${styles.steps}>${repeat(steps, (step) => step.key, (step) => this.#step(step))}</ol>`}
    `;
  }

  #step(step: ActivityStep): TemplateResult {
    if (step.kind === 'say') {
      return html`<li class=${styles.step}><p class=${styles.say}>${step.text}</p></li>`;
    }
    if (step.kind === 'instruction') {
      const sender = senderText(step.sender);
      return html`
        <li class=${styles.step}>
          ${sender ? html`<p class=${styles.told}>${sender}</p>` : nothing}
          <p class=${styles.toldText}>${step.text}</p>
        </li>
      `;
    }
    const state = step.state === 'running'
      ? msg('Running', {id: 'agentSession.state.running'})
      : step.state === 'failed' ? msg('Failed', {id: 'agentSession.state.failed'}) : '';
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
}

customElements.define('dl-activity', DlActivity);

declare global {
  interface HTMLElementTagNameMap {
    'dl-activity': DlActivity;
  }
}
