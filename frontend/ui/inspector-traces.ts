// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Inspector's Run traces: one Run as a dossier and its agents as pages.
 *
 *  Wide, the dock is two columns. The left column is the Run overview: a 44px band stating the Run's
 *  shape, duration and spend, then the dossier — the composition bar, what the Run delivered, its
 *  agents as a ledger whose rows carry each agent's span over the Run's clock, and what needs
 *  attention. The right column is a one-line agent switcher over the selected agent's read-only
 *  page, whose Agent timeline is the only place a Run renders its steps. A reader's search for words
 *  is the one step list the Run panel carries: typing switches the left column to the Run's matching
 *  steps, and clearing it restores the dossier it holds in memory.
 *
 *  Narrow, the dossier folds to one summary line above the same switcher and page, so the page keeps
 *  the height it needs; a search shows its results above them instead.
 *
 *  The dock reads: it never steers, continues, cancels, or answers a child — those acts belong to
 *  the conversation's composer. A followed Run keeps its pages current: its own events state facts
 *  the pages hold, and the left column re-reads the Run's overview (which revalidates by version) on
 *  the cadence the roster refreshes at. A finding a reader picks opens its step in its agent's page:
 *  the dock selects that agent and hands the page the step to ring.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import type {
  ActivityMessage,
  AgentChildRosterPage,
  AgentChildStatus,
  AgentRunStatus,
  AnswerPresentation,
  RunOverview,
  RunStep,
} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {type AgentId, agentIdOf, agentKey, agentOfRow, MAIN_AGENT, RUN_AGENT, sameAgent} from '../lib/agent-id.ts';
import {agentName} from '../lib/agent-name.ts';
import {elapsed, timeOfDay, toolTime} from '../lib/date-format.ts';
import {LightElement, NarrowController} from '../lib/lit-host.ts';
import {KeysetPager} from '../lib/paged.ts';
import {type AnswerRunEvent, statesSomething} from '../lib/run-controller.ts';
import {runDurationMs, runTokens} from '../lib/run-overview.ts';
import {readStored, writeStored} from '../lib/storage.ts';
import {toolDisplay, toolVerbText} from '../lib/tool-display.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import type {TracesSource} from '../stores/traces-source.ts';
import styles from '../styles/inspector-traces.module.css';
import {getLocale} from '../i18n/locale.ts';
import {tokenCountLabel} from './token-count.ts';
import './agent-session.ts';
import type {DlAgentSession} from './agent-session.ts';
import './agent-switcher.ts';
import type {AgentSwitchDetail, AgentSwitcherPaging} from './agent-switcher.ts';
import type {TraceQueryDetail} from './trace-search.ts';
import './trace-search.ts';
import './run-overview.ts';
import type {ListedChild} from './run-overview.ts';

/** A followed roster refetches at most this often while its run streams child activity. */
const FOLLOW_REFRESH_INTERVAL_MS = 1000;
/** From this width, in rem (736px), the dock shows the Run panel beside the agent page. */
const WIDE_REM = 46;
/** How often a followed Run's pages are read again with nothing to say they moved: the events are
 *  the cadence, and this only covers what a stream cannot see (a child finishing while the Run's
 *  own agent is quiet, a dropped frame, a reconnect). */
const RECONCILE_MILLISECONDS = 15_000;
/** Where a Run remembers the agent its reader last looked at. */
const SELECTED_AGENT_PREFIX = 'dlightrag-inspector-agent:';

/** What a scope of copy the client owns is called in each language. */
interface Copy {
  readonly source: string;
  readonly id: string;
}

const OVERVIEW: Copy = {source: 'Run overview', id: 'tracesPanel.overview'};
const OVERVIEW_LOADING: Copy = {source: 'Loading the Run overview…', id: 'tracesPanel.overviewLoading'};
const OVERVIEW_FAILED: Copy = {source: 'The Run overview could not be loaded.', id: 'tracesPanel.overviewFailed'};
const STEP_ONE: Copy = {source: '1 step', id: 'tracesPanel.overviewOneStep'};
const AGENT_ONE: Copy = {source: '1 agent', id: 'tracesPanel.overviewOneAgent'};
const SEARCH_RESULTS: Copy = {source: 'Search results', id: 'tracesPanel.searchResults'};
const SEARCH_LOADING: Copy = {source: 'Searching the Run’s steps…', id: 'tracesPanel.searchLoading'};
const SEARCH_FAILED: Copy = {source: 'The step search could not be loaded.', id: 'tracesPanel.searchFailed'};
const SEARCH_NO_MATCH: Copy = {source: 'No step in this Run holds those words.', id: 'tracesPanel.searchNoMatch'};
const SEARCH_ONE_MATCH: Copy = {source: '1 matching step', id: 'tracesPanel.searchOneMatch'};
const SEARCH_LOAD_OLDER: Copy = {source: 'Load earlier steps', id: 'tracesPanel.searchLoadOlder'};
const SEARCH_SAID: Copy = {source: 'Said', id: 'tracesPanel.searchSaid'};
const SEARCH_ANSWERED: Copy = {source: 'Answered', id: 'tracesPanel.searchAnswered'};
const SEARCH_INSTRUCTION: Copy = {source: 'Instruction', id: 'tracesPanel.searchInstruction'};

/** Why a read of the Run's pages happens: a newly shown Run's first read, a followed Run's tick, or
 *  the periodic full read that covers what a stream of events cannot see. */
type ReloadMode = 'open' | 'follow' | 'reconcile';

/** The step a finding sent the reader to: its agent, and where the step sits in that agent's Session.
 *  The dock holds it until the reader picks another agent or another finding. */
interface RunFocus {
  readonly agent: AgentId;
  readonly sequence: number;
}

function listed(children: readonly AgentChildStatus[]): ListedChild[] {
  return children.filter((child): child is ListedChild => Boolean(child.childSessionId));
}

/** An agent whose words hold none of the searched words shows nothing. */
const EMPTY_HITS: readonly ActivityMessage[] = [];

/** What the Run's transcript search found, grouped by the agent that said it: one read, every count. */
function hitsByAgent(
  messages: readonly ActivityMessage[],
): Map<string, readonly ActivityMessage[]> {
  const byAgent = new Map<string, ActivityMessage[]>();
  for (const message of messages) {
    const key = agentKey(message.agent);
    const held = byAgent.get(key);
    if (held === undefined) byAgent.set(key, [message]);
    else held.push(message);
  }
  return byAgent;
}

export class DlInspectorTraces extends LightElement {
  static properties = {
    handles: {attribute: false},
    source: {attribute: false},
    active: {attribute: false},
    entries: {state: true},
    failed: {state: true},
    picked: {state: true},
    run: {state: true},
    answerFeedbackEnabled: {attribute: false},
  };

  declare handles: AppHandles;
  declare source: TracesSource | null;
  /** Whether the Inspector is showing this content: a dock that is not does no background work. */
  declare active: boolean;
  /** The roster as loaded, newest first; null until its first page arrives. */
  declare entries: readonly ListedChild[] | null;
  declare failed: boolean;
  /** The agent the reader chose, or null while the Run's own agent is shown. */
  declare picked: AgentId | null;
  /** Where the Run stood at the latest refresh; null before the first, or from a server that does not say. */
  declare run: AgentRunStatus | null;
  /** Whether the reader is offered the thumbs for what a child answered. */
  declare answerFeedbackEnabled: boolean;

  /** The Run's whole-Run overview: the dossier the left column draws. Null before the first read. */
  #overview: RunOverview | null = null;
  #overviewFailed = false;
  #overviewLoading = false;
  #overviewQueued = false;
  #overviewAbort: AbortController | null = null;
  /** Whether the reader unfolded the narrow dossier; a refresh leaves their choice where it was. */
  #overviewOpen = false;
  #searchSteps: readonly RunStep[] = [];
  #searchNextBefore: string | null = null;
  #searchMatched = 0;
  #searchLoading = false;
  #searchFailed = false;
  #searchAbort: AbortController | null = null;

  /** The step a finding asked the agent page to ring; the page clears it when the reader moves on. */
  #focus: RunFocus | null = null;
  /** Whether the agent on show was restored from storage and still has to be checked against the roster. */
  #restoredPick = false;

  readonly #narrow = new NarrowController(this, WIDE_REM);
  readonly #pager = new KeysetPager<AgentChildRosterPage>(
    async (cursor, signal) => {
      const source = this.source;
      if (!source) throw new Error('no Run is open');
      return await source.page(cursor, signal);
    },
    () => { this.requestUpdate(); },
  );
  #refreshing = false;
  #refreshQueued = false;
  #lastRefresh = Number.NEGATIVE_INFINITY;
  #followTimer: ReturnType<typeof setTimeout> | null = null;
  #reconcile: ReturnType<typeof setInterval> | null = null;
  /** The words the transcript search settled on: what the switcher and pages are narrowed to. */
  #query = '';
  /** The matches of those words, grouped by the agent whose words they are; null when not searching. */
  #hits: readonly ActivityMessage[] | null = null;
  /** The same matches, once per agent: a narrowed pane is handed an array that does not change on
   *  every redraw, or the pane would take each new array for new words and read its pages again. */
  #byAgent = new Map<string, readonly ActivityMessage[]>();
  /** The words the server read, which is what a match was made of and so what is marked. */
  #terms: readonly string[] = [];

  constructor() {
    super();
    this.handles = productionHandles();
    this.source = null;
    this.active = false;
    this.entries = null;
    this.failed = false;
    this.picked = null;
    this.run = null;
    this.answerFeedbackEnabled = false;
  }

  override disconnectedCallback(): void {
    this.#cancelFollow();
    this.#stopReconcile();
    this.#overviewAbort?.abort();
    this.#overviewAbort = null;
    this.#searchAbort?.abort();
    this.#searchAbort = null;
    super.disconnectedCallback();
  }

  /** What the followed Run just said, as the reason to read its pages again.

   *  A token says the Run is still writing, and the chat is where those words are drawn: the row
   *  this dock draws already counts its own time while it runs, so a token needs no read at all.
   *  An event that states something — a tool step, a phase, the end — is a fact the pages hold, and
   *  this asks for them at the interval a followed Run's own activity passes through.
   */
  applyEvents(runId: string, events: readonly AnswerRunEvent[]): void {
    if (!this.#following(runId)) return;
    // A child on show is the one thing whose own work the Run's log cannot state: while it runs,
    // the pane it fills is read at the interval, whatever the batch was about.
    if (this.#showsRunningChild()) {
      this.#refreshIfFollowing(runId);
      return;
    }
    if (statesSomething(events)) this.#refreshIfFollowing(runId);
  }

  /** Whether the agent on show is a child of this Run that is still working. */
  #showsRunningChild(): boolean {
    const shown = this.picked;
    if (shown === null || shown.kind !== 'child') return false;
    return this.entries?.some((child) => (
      agentKey(agentOfRow(child)) === agentKey(shown) && child.status === 'running'
    )) ?? false;
  }

  /** Refresh for child activity of the followed run: at once, then at most once per interval. */
  #refreshIfFollowing(runId: string): void {
    if (!this.#following(runId) || this.#followTimer !== null) return;
    const wait = this.#lastRefresh + FOLLOW_REFRESH_INTERVAL_MS - performance.now();
    if (wait <= 0) {
      this.#followRefresh();
      return;
    }
    // One trailing refresh carries every activity inside the interval.
    this.#followTimer = setTimeout(() => {
      this.#followTimer = null;
      if (this.#following(runId)) this.#followRefresh();
    }, wait);
  }

  /** The tick a followed Run's activity is read on: the roster, the Run's overview, and the page on
   *  show when it can still grow. A settled page is read when its own row changes, not on another
   *  agent's activity — following would otherwise re-read it once per tick. */
  #followRefresh(): void {
    this.#refreshQueued = true;
    void this.#flushRefresh();
    if (this.#showsMovingPage()) this.#agentPage()?.refresh();
  }

  /** Whether the page on show can still grow: the Run's own page while the Run runs, or a child that
   *  is still working. */
  #showsMovingPage(): boolean {
    return this.picked === null ? this.run?.status === 'running' : this.#showsRunningChild();
  }

  /** The agent page on show, which follows the Run the dock follows. */
  #agentPage(): DlAgentSession | null {
    return this.querySelector<DlAgentSession>('dl-agent-session');
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (changed.has('source')) {
      this.#pager.reset(null);
      this.#cancelFollow();
      this.#refreshQueued = false;
      this.entries = null;
      this.failed = false;
      this.picked = null;
      this.run = null;
      this.#overviewAbort?.abort();
      this.#overviewAbort = null;
      this.#overview = null;
      this.#overviewFailed = false;
      this.#overviewLoading = false;
      this.#overviewQueued = false;
      this.#overviewOpen = false;
      this.#searchAbort?.abort();
      this.#searchAbort = null;
      this.#resetSearch();
      // Another Run is not narrowed by the words the reader searched the last one for: the box
      // clears itself with the source, and the panel goes back to the Run's own dossier.
      this.#query = '';
      this.#hits = null;
      this.#terms = [];
      this.#byAgent = new Map();
      this.#focus = null;
      if (this.source) {
        this.picked = this.#rememberedAgent();
        this.#restoredPick = this.picked !== null;
        void this.#refresh('open');
      }
    }
    if (!this.active) this.#cancelFollow();
  }

  protected override updated(): void {
    // The events are the cadence while a Run is on screen; this reads the pages again when they
    // have had nothing to say, which is what a child settling quietly or a dropped frame looks like.
    const following = this.active && this.source !== null && this.run?.status === 'running';
    if (following && this.#reconcile === null) {
      this.#reconcile = setInterval(() => { void this.#refresh('reconcile'); }, RECONCILE_MILLISECONDS);
    } else if (!following) {
      this.#stopReconcile();
    }
  }

  #stopReconcile(): void {
    if (this.#reconcile !== null) clearInterval(this.#reconcile);
    this.#reconcile = null;
  }

  #following(runId: string): boolean {
    return this.active && this.source?.runId === runId;
  }

  #cancelFollow(): void {
    if (this.#followTimer !== null) clearTimeout(this.#followTimer);
    this.#followTimer = null;
  }

  async #flushRefresh(): Promise<void> {
    if (this.#refreshing) return;
    this.#refreshing = true;
    try {
      while (this.#refreshQueued) {
        this.#refreshQueued = false;
        await this.#refresh();
      }
    } finally {
      this.#refreshing = false;
      if (this.#refreshQueued) void this.#flushRefresh();
    }
  }

  /** Read the roster and the Run's overview again. `open` is a newly shown Run's first read, and
   *  `reconcile` the periodic full read; both the roster and the overview are whole-Run reads. */
  async #refresh(mode: ReloadMode = 'follow'): Promise<void> {
    const source = this.source;
    if (!source) return;
    // Opening, a retry, and followed activity share one throttle window.
    this.#lastRefresh = performance.now();
    this.failed = false;
    this.#requestOverview(mode);
    await this.#pager.start((page) => {
      this.entries = listed(page.children);
      this.run = page.run;
      // A remembered agent the first page does not hold (and no older page could) is gone; the
      // next open falls back to the Run's own agent. A child the reader chose in this visit is
      // left as it is: the page says it is unavailable rather than moving them elsewhere.
      if (this.#restoredPick) {
        this.#restoredPick = false;
        const picked = this.picked;
        if (picked?.kind === 'child' && !this.#pager.hasOlder
          && !this.entries.some((child) => sameAgent(agentOfRow(child), picked))) {
          this.picked = null;
        }
      }
    }, () => {
      this.failed = true;
    });
  }

  /** Ask for the Run's overview. One read is in flight at a time; a read asked for meanwhile is
   *  answered after it rather than dropped, and a Run whose version has not moved revalidates
   *  without a body. */
  #requestOverview(mode: ReloadMode): void {
    if (this.#overviewLoading) {
      this.#overviewQueued = true;
      return;
    }
    void this.#readOverview(mode);
  }

  async #readOverview(mode: ReloadMode): Promise<void> {
    const source = this.source;
    if (!source) return;
    const request = new AbortController();
    this.#overviewAbort = request;
    // Only a first read says it is loading on screen; a follow read leaves the dossier standing
    // until the new one lands.
    this.#overviewLoading = this.#overview === null;
    if (mode === 'open') this.#overviewFailed = false;
    this.requestUpdate();
    try {
      const overview = await source.overview(request.signal);
      if (request.signal.aborted || this.source !== source || overview.runId !== source.runId) return;
      this.#overview = overview;
      this.#overviewFailed = false;
    } catch {
      if (request.signal.aborted) return;
      // A dock whose overview could not be read is not a failed dock: its agents are readable on
      // their own, and a held dossier stands until a later read replaces it.
      if (this.#overview === null) this.#overviewFailed = true;
    } finally {
      if (this.#overviewAbort === request) {
        this.#overviewAbort = null;
        this.#overviewLoading = false;
        if (this.#overviewQueued) {
          this.#overviewQueued = false;
          if (this.source === source) this.#requestOverview('follow');
        }
      }
      this.requestUpdate();
    }
  }

  /** Where the Run's own clock and spend put the header summary: state, duration, agents, steps and
   *  tokens, in the order a reader takes them in.
   *
   *  A running Run states no duration here: the value would freeze between reads, and the agent
   *  switcher beside the page already draws the Run's moving clock (it owns a ClockController).
   *  A settled Run's duration is a fixed fact, so the band states it once. */
  #overviewSummary(): string {
    const overview = this.#overview;
    if (overview === null) return '';
    const parts: string[] = [];
    const state = this.#runStateText();
    if (state !== '') parts.push(state);
    const duration = this.run !== null && this.run.status !== 'running'
      ? runDurationMs(overview, this.run, Date.now())
      : null;
    if (duration !== null) parts.push(elapsed(duration, getLocale()));
    const agents = overview.agents.length;
    parts.push(agents === 1
      ? msg(AGENT_ONE.source, {id: AGENT_ONE.id})
      : msg(str`${agents} agents`, {id: 'tracesPanel.overviewAgents'}));
    const steps = overview.totals.steps;
    parts.push(steps === 1
      ? msg(STEP_ONE.source, {id: STEP_ONE.id})
      : msg(str`${steps} steps`, {id: 'tracesPanel.overviewSteps'}));
    const tokens = runTokens(overview).total;
    if (tokens !== null) parts.push(tokenCountLabel(tokens));
    return parts.join(' · ');
  }

  /** The Run's state in the reader's words, or '' while the roster does not say. */
  #runStateText(): string {
    switch (this.run?.status) {
      case 'running': return msg('Running', {id: 'agentSession.state.running'});
      case 'succeeded': return msg('Done', {id: 'agentSession.state.done'});
      case 'failed': return msg('Failed', {id: 'agentSession.state.failed'});
      case 'cancelled': return msg('Cancelled', {id: 'agentSession.state.cancelled'});
      default: return '';
    }
  }

  /** Where a Run remembers the agent its reader last read; main when it remembers none. */
  #rememberedAgent(): AgentId | null {
    const runId = this.source?.runId;
    if (!runId) return null;
    const stored = readStored(`${SELECTED_AGENT_PREFIX}${runId}`);
    if (stored === null || stored === '') return null;
    const agent = agentIdOf(stored);
    return sameAgent(agent, RUN_AGENT) ? null : agent;
  }

  #rememberSelected(agent: AgentId): void {
    const runId = this.source?.runId;
    if (!runId) return;
    writeStored(
      `${SELECTED_AGENT_PREFIX}${runId}`,
      sameAgent(agent, RUN_AGENT) ? MAIN_AGENT : agentKey(agent),
    );
  }

  /** The agent the dock is showing: the reader's choice, or the Run's own agent. A choice the roster
   *  no longer holds stays on show as the unavailable agent it is (the page says so), rather than
   *  silently moving the reader somewhere else. */
  #shown(): AgentId {
    return this.picked ?? RUN_AGENT;
  }

  /** Read an agent's page ahead of the click, so the switch does not wait on the wire. */
  #warmAgent = (agent: AgentId): void => {
    if (agent.kind === 'child') this.source?.warm?.(agent.id);
  };

  /** The reader picked an agent: it is what the page shows, and what the Run remembers. A pick ends
   *  whatever step a finding had sent them to. */
  #setPicked(agent: AgentId): void {
    this.#focus = null;
    this.picked = sameAgent(agent, RUN_AGENT) ? null : agent;
    this.#restoredPick = false;
    this.#rememberSelected(agent);
    this.requestUpdate();
  }

  #switched = (event: CustomEvent<AgentSwitchDetail>): void => {
    this.#setPicked(event.detail.agent);
  };

  /** The Run panel selected an agent: open its page. */
  #agentSelected = (event: CustomEvent<{agent: string}>): void => {
    this.#setPicked(agentIdOf(event.detail.agent));
  };

  /** A finding was picked: open its agent's page and send the page the step to ring. */
  #findingSelected = (event: CustomEvent<{agent: string; sequence: number}>): void => {
    const agent = agentIdOf(event.detail.agent);
    this.picked = sameAgent(agent, RUN_AGENT) ? null : agent;
    this.#restoredPick = false;
    this.#rememberSelected(agent);
    this.#focus = {agent, sequence: event.detail.sequence};
    this.requestUpdate();
  };

  /** What the agent page is told to ring, in the wire's spelling; null when nothing is focused. */
  #passFocus(): {agent: string; sequence: number} | null {
    const focus = this.#focus;
    return focus === null ? null : {agent: agentKey(focus.agent), sequence: focus.sequence};
  }

  #loadOlderChildren = (): void => {
    if (this.#pager.state === 'loading') return;
    void this.#pager.loadNext((page) => {
      const known = new Set(this.entries?.map((child) => child.childSessionId));
      const older = listed(page.children).filter((child) => {
        if (known.has(child.childSessionId)) return false;
        known.add(child.childSessionId);
        return true;
      });
      this.entries = [...(this.entries ?? []), ...older];
    });
  };

  /** The switcher's roster state: another page of children, and how reading it went. */
  #switcherPaging(): AgentSwitcherPaging {
    return {
      hasOlder: this.#pager.hasOlder,
      loading: this.#pager.state === 'loading',
      failed: this.#pager.outcome === 'failed',
    };
  }

  /** The matches one agent's words hold, by agent key, while a search narrows the Run. */
  #hitCounts(): ReadonlyMap<string, number> | null {
    if (this.#hits === null) return null;
    const counts = new Map<string, number>();
    for (const [key, messages] of this.#byAgent) counts.set(key, messages.length);
    return counts;
  }

  /** The reader settled on words: the panes narrow to them, and the Run panel becomes the Run's
   *  matching steps. Clearing the words restores the dossier the dock holds, at once. */
  #asked = (event: CustomEvent<TraceQueryDetail>): void => {
    const terms = event.detail.terms;
    const changed = this.#query !== terms;
    this.#query = terms;
    if (changed) this.#restartSearch();
    if (terms === '') {
      this.#hits = null;
      this.#terms = [];
      this.#byAgent = new Map();
      this.requestUpdate();
      return;
    }
    const source = this.source;
    if (!source?.search) return;
    void source.search(terms).then((answer) => {
      // A later search owns the panel now; this answer is about words nobody is looking at.
      if (this.#query !== terms) return;
      this.#hits = answer.messages;
      this.#terms = answer.terms ?? [terms];
      // New words: whatever source the last search narrowed to is about those words, not these.
      this.#byAgent = hitsByAgent(this.#hits);
      const first = this.#firstAgentWithMatches();
      // Stay where the reader is while the words are there; otherwise go to the agent that has them.
      const here = this.#shown();
      if ((this.#byAgent.get(agentKey(here))?.length ?? 0) === 0 && first !== undefined) {
        this.#setPicked(first);
      }
      this.requestUpdate();
    });
  };

  /** The matches of one agent's words, or null when no search is narrowing the pane. */
  #hitsFor(agent: AgentId | null): readonly ActivityMessage[] | null {
    if (this.#hits === null) return null;
    return this.#byAgent.get(agentKey(agent ?? RUN_AGENT)) ?? EMPTY_HITS;
  }

  #firstAgentWithMatches(): AgentId | undefined {
    return (this.#hits ?? [])[0]?.agent;
  }

  /** Start the Run's step search again from its first page, or clear it when the words went away. */
  #restartSearch(): void {
    this.#searchAbort?.abort();
    this.#searchAbort = null;
    this.#resetSearch();
    if (this.#query.trim() !== '') void this.#readSearch(null, 'open');
    else this.requestUpdate();
  }

  /** Forget the held search results; the words, the abort, and the request belong to the caller. */
  #resetSearch(): void {
    this.#searchSteps = [];
    this.#searchNextBefore = null;
    this.#searchMatched = 0;
    this.#searchLoading = false;
    this.#searchFailed = false;
  }

  /** Read one page of the Run's matching steps; `before` reads the page before the held ones. */
  async #readSearch(before: string | null, mode: 'open' | 'older'): Promise<void> {
    const source = this.source;
    if (!source) return;
    const words = this.#query;
    const request = new AbortController();
    this.#searchAbort = request;
    this.#searchLoading = true;
    if (mode === 'open') this.#searchFailed = false;
    this.requestUpdate();
    try {
      const page = await source.searchSteps(words, before, request.signal);
      if (request.signal.aborted || this.source !== source || this.#query !== words) return;
      this.#searchSteps = mode === 'older' ? [...page.steps, ...this.#searchSteps] : page.steps;
      this.#searchNextBefore = page.nextBefore;
      this.#searchMatched = page.matched;
      this.#searchFailed = false;
    } catch {
      if (request.signal.aborted) return;
      // The words are still what the reader asked for: the panel says the read failed and offers it again.
      this.#searchFailed = true;
    } finally {
      if (this.#searchAbort === request) {
        this.#searchAbort = null;
        this.#searchLoading = false;
      }
      this.requestUpdate();
    }
  }

  #loadEarlierSearch = (): void => {
    if (this.#searchLoading || this.#searchNextBefore === null) return;
    void this.#readSearch(this.#searchNextBefore, 'older');
  };

  #retrySearch = (): void => {
    this.#restartSearch();
  };

  /** A step a search found: open the agent that stated it. */
  #searchStepSelected(row: RunStep): void {
    this.#setPicked(agentIdOf(row.agent));
  }

  #toggledOverview = (event: Event): void => {
    this.#overviewOpen = (event.currentTarget as HTMLDetailsElement).open;
  };

  /** Whether the left column has anything to draw: the dossier of a Run with steps, its loading or
   *  failed state, or the step search a reader asked for. A Run with no steps draws no panel. */
  #panelVisible(): boolean {
    if (this.#query.trim() !== '') return true;
    if (this.#overviewLoading || this.#overviewFailed) return true;
    return this.#overview !== null && this.#overview.totals.steps > 0;
  }

  protected override render(): TemplateResult | typeof nothing {
    const source = this.source;
    if (!source) return nothing;
    const all = this.entries ?? [];
    const narrow = this.#narrow.narrow;
    const shown = this.#shown();
    // The main agent's row is drawn from the turn as it is now, so its words — a
    // failure's reason among them — are in the language showing now too. Its own
    // row comes from the page that states every row, this one and each child's.
    const lead = source.mainAgent(this.run);
    const entry = sameAgent(shown, RUN_AGENT)
      ? lead
      : all.find((child) => sameAgent(agentOfRow(child), shown)) ?? null;
    const switcher = html`
      <dl-agent-switcher class=${styles.switcher} .active=${this.active}
        .main=${lead} .entries=${all} .selected=${shown}
        .hits=${this.#hitCounts()} .paging=${this.#switcherPaging()} .warm=${this.#warmAgent}
        @dl-agent-switch=${this.#switched}
        @dl-agent-load-older=${this.#loadOlderChildren}></dl-agent-switcher>
    `;
    // The roster states the switcher and the row the page heads itself with; the page is mounted
    // anyway, so its transcript reads in parallel with the roster rather than after it.
    const loading = this.entries === null && !this.failed
      ? html`<p class="dl-note ${styles.quiet}" role="status" data-roster-loading>${msg('Loading agents…', {id: 'tracesPanel.loading'})}</p>`
      : nothing;
    const page = html`
      <dl-agent-session class=${styles.detail}
        .handles=${this.handles} .source=${source} .childSessionId=${agentKey(shown)} .entry=${entry}
        .terms=${this.#terms} .found=${this.#hitsFor(shown)} .stepFocus=${this.#passFocus()}
        .answerFeedbackEnabled=${this.answerFeedbackEnabled}></dl-agent-session>
    `;
    const panel = this.#panelVisible() ? this.#panel(narrow) : null;
    return html`
      <div class=${styles.root}>
        ${this.failed ? this.#failure() : nothing}
        ${all.length > 0 ? this.#toolbar(all) : nothing}
        <dl-trace-search .source=${source.search ? source : null}
                         @dl-trace-query=${this.#asked}></dl-trace-search>
        ${narrow ? html`
          <div class=${styles.narrowPane}>
            ${panel === null ? nothing : html`
              <div class=${styles.narrowPanel}>${panel}</div>
            `}
            ${switcher}${loading}
            <div class=${styles.agentPane}>${page}</div>
          </div>
        ` : html`
          <div class=${styles.widePane}>
            ${panel === null ? nothing : html`
              <div class=${styles.overviewColumn}>${panel}</div>
            `}
            <div class=${styles.agentColumn}>${switcher}${loading}${page}</div>
          </div>
        `}
      </div>
    `;
  }

  /** The left column, in the mode the reader is in: the Run's dossier, or its matching steps. */
  #panel(narrow: boolean): TemplateResult {
    if (this.#query.trim() !== '') return this.#searchPanel();
    return narrow ? this.#narrowOverview() : this.#wideOverview();
  }

  /** The wide Run panel: a 44px band (title and the one summary line) over the dossier. */
  #wideOverview(): TemplateResult {
    return html`
      <section class=${styles.overview} data-overview aria-labelledby="run-overview-title">
        <div class=${styles.overviewHeader} data-overview-band>
          <span class=${styles.overviewTitle} data-overview-title id="run-overview-title">${msg(OVERVIEW.source, {id: OVERVIEW.id})}</span>
          <span class=${styles.overviewSummary} data-overview-summary>${this.#overviewSummary()}</span>
        </div>
        <div class=${styles.overviewBody} data-overview-body>${this.#overviewBody()}</div>
      </section>
    `;
  }

  /** The narrow Run panel: the same band as one summary line that opens the dossier above the page. */
  #narrowOverview(): TemplateResult {
    return html`
      <details class=${styles.overviewFold} data-overview
               ?open=${this.#overviewOpen || this.#overviewFailed} @toggle=${this.#toggledOverview}>
        <summary class=${styles.overviewSummaryLine}>
          <span class=${styles.overviewChevron}>${icon('disclosure', {size: 'xs'})}</span>
          <span class=${styles.overviewTitle} data-overview-title>${msg(OVERVIEW.source, {id: OVERVIEW.id})}</span>
          <span class=${styles.overviewSummary} data-overview-summary>${this.#overviewSummary()}</span>
        </summary>
        <div class=${styles.overviewFoldBody}>${this.#overviewBody()}</div>
      </details>
    `;
  }

  /** What the Run panel says under its band: the dossier, its loading line, or its failure. */
  #overviewBody(): TemplateResult | typeof nothing {
    if (this.#overviewFailed && this.#overview === null) {
      return html`
        <p class="dl-note ${styles.quiet}" role="status" data-overview-failed>
          ${msg(OVERVIEW_FAILED.source, {id: OVERVIEW_FAILED.id})}
        </p>
      `;
    }
    const overview = this.#overview;
    if (overview === null) {
      return this.#overviewLoading
        ? html`
          <p class="dl-note ${styles.quiet}" role="status" data-overview-loading>
            ${msg(OVERVIEW_LOADING.source, {id: OVERVIEW_LOADING.id})}
          </p>
        `
        : nothing;
    }
    return html`
      <dl-run-overview .overview=${overview} .loading=${this.#overviewLoading}
        .failed=${this.#overviewFailed} .entries=${this.entries ?? []}
        .lead=${this.source?.mainAgent(this.run) ?? null} .run=${this.run}
        .presentation=${this.#presentation()} .selected=${agentKey(this.#shown())}
        @dl-run-agent-select=${this.#agentSelected}
        @dl-run-finding-select=${this.#findingSelected}></dl-run-overview>
    `;
  }

  /** The Run's answer, which only the chat holds: what the delivery card draws. */
  #presentation(): AnswerPresentation | null {
    return this.source?.presentation() ?? null;
  }

  /** The left column while a reader searches: the Run's matching steps, a page at a time. */
  #searchPanel(): TemplateResult {
    return html`
      <section class=${styles.searchPanel} data-step-search
               aria-labelledby="run-search-title">
        <div class=${styles.overviewHeader}>
          <span class=${styles.overviewTitle} data-overview-title id="run-search-title">
            ${msg(SEARCH_RESULTS.source, {id: SEARCH_RESULTS.id})}
          </span>
          <span class=${styles.overviewSummary} data-search-count>${this.#searchCountText()}</span>
        </div>
        ${this.#searchBody()}
      </section>
    `;
  }

  /** How many of the Run's steps hold the words, or what the search is doing instead. */
  #searchCountText(): string {
    if (this.#searchLoading && this.#searchSteps.length === 0) {
      return msg(SEARCH_LOADING.source, {id: SEARCH_LOADING.id});
    }
    if (this.#searchFailed) return msg(SEARCH_FAILED.source, {id: SEARCH_FAILED.id});
    return this.#searchMatched === 1
      ? msg(SEARCH_ONE_MATCH.source, {id: SEARCH_ONE_MATCH.id})
      : msg(str`${this.#searchMatched} matching steps`, {id: 'tracesPanel.searchMatches'});
  }

  /** The search results: their rows, how to read older ones, and the states in between. */
  #searchBody(): TemplateResult | typeof nothing {
    if (this.#searchFailed && this.#searchSteps.length === 0) {
      return html`
        <p class="dl-note ${styles.quiet}" role="status" data-search-failed>
          ${msg(SEARCH_FAILED.source, {id: SEARCH_FAILED.id})}
          <button type="button" class="dl-btn" @click=${this.#retrySearch}>
            ${msg('Retry', {id: 'tracesPanel.retry'})}
          </button>
        </p>
      `;
    }
    if (this.#searchSteps.length === 0) {
      return this.#searchLoading
        ? html`
          <p class="dl-note ${styles.quiet}" role="status" data-search-loading>
            ${msg(SEARCH_LOADING.source, {id: SEARCH_LOADING.id})}
          </p>
        `
        : html`
          <p class="dl-note ${styles.quiet}" role="status" data-search-no-match>
            ${msg(SEARCH_NO_MATCH.source, {id: SEARCH_NO_MATCH.id})}
          </p>
        `;
    }
    return html`
      <ol class=${styles.searchRows} role="list" data-search-rows>
        ${this.#searchSteps.map((row) => this.#searchRow(row))}
      </ol>
      ${this.#searchNextBefore === null ? nothing : html`
        <button type="button" class=${styles.searchOlder} data-search-older
                aria-disabled=${String(this.#searchLoading)} @click=${this.#loadEarlierSearch}>
          ${this.#searchLoading
            ? msg('Loading earlier steps…', {id: 'activity.loadingOlder'})
            : msg(SEARCH_LOAD_OLDER.source, {id: SEARCH_LOAD_OLDER.id})}
        </button>
      `}
    `;
  }

  /** One step a search found: who stated it, what it did, the start of what came back, and when.
   *  The whole row opens the agent that stated it. */
  #searchRow(row: RunStep): TemplateResult {
    const duration = toolTime(row.durationMs, false);
    const time = timeOfDay(row.at, getLocale(), true);
    return html`
      <li class=${styles.searchStep}>
        <button type="button" class=${styles.searchRow} data-search-step
                data-agent=${row.agent} data-sequence=${row.sequence} data-kind=${row.kind}
                @click=${() => this.#searchStepSelected(row)}>
          <span class=${styles.searchHead}>
            <span class=${styles.searchTitle}>
              <span class=${styles.searchVerb}>${this.#searchVerb(row)}</span>
              ${row.subject === null || row.subject === ''
                ? nothing
                : html`<span class=${styles.searchSubject}>${row.subject}</span>`}
            </span>
            ${duration === '' ? nothing : html`<span class=${styles.searchDuration}>${duration}</span>`}
          </span>
          <span class=${styles.searchMeta}>
            <span class=${styles.searchAgent}>${this.#searchAgentName(row)}</span>
            ${time === '' ? nothing : html`<span class=${styles.searchTime}>${time}</span>`}
            ${row.excerpt === ''
              ? nothing
              : html`<span class=${styles.searchExcerpt} title=${row.excerpt}>${row.excerpt}</span>`}
          </span>
        </button>
      </li>
    `;
  }

  /** The verb a matching step leads with: the Tool's own, or the kind of step it is. */
  #searchVerb(row: RunStep): string {
    if (row.kind === 'tool' && row.name !== null) {
      const display = toolDisplay(row.name);
      return toolVerbText(display.verb, display.verbId);
    }
    switch (row.kind) {
      case 'say': return msg(SEARCH_SAID.source, {id: SEARCH_SAID.id});
      case 'answer': return msg(SEARCH_ANSWERED.source, {id: SEARCH_ANSWERED.id});
      case 'instruction': return msg(SEARCH_INSTRUCTION.source, {id: SEARCH_INSTRUCTION.id});
      case 'delivery': return msg('Delivery note', {id: 'activity.deliveryNote'});
      default: return row.kind;
    }
  }

  /** The agent a matching step names: the Run's own agent in the product's words, a child by the
   *  roster label the search carried. */
  #searchAgentName(row: RunStep): string {
    const agent = agentIdOf(row.agent);
    return agent.kind === 'run' ? agentName(agent) : row.agentLabel ?? agentName(agent);
  }

  #failure(): TemplateResult {
    return html`
      <div class=${styles.empty}>
        <p class=${styles.emptyTitle} role="alert">${msg('Agents could not be loaded.', {id: 'tracesPanel.loadFailed'})}</p>
        <button type="button" class="dl-btn" @click=${this.#retry}>${msg('Retry', {id: 'tracesPanel.retry'})}</button>
      </div>
    `;
  }

  #retry = (): void => {
    void this.#refresh('open');
  };

  #toolbar(entries: readonly ListedChild[]): TemplateResult {
    const total = `${entries.length}${this.#pager.hasOlder ? '+' : ''}`;
    const running = entries.filter((child) => child.status === 'running').length;
    return html`
      <div class=${styles.toolbar}>
        <span>${entries.length === 1 && !this.#pager.hasOlder
          ? msg('1 child', {id: 'tracesPanel.oneChild'})
          : msg(str`${total} children`, {id: 'tracesPanel.children'})}</span>
        ${running > 0 ? html`
          <span aria-hidden="true">·</span>
          <span class=${styles.live}>${msg(str`${running} running`, {id: 'tracesPanel.running'})}</span>
        ` : nothing}
      </div>
    `;
  }
}

customElements.define('dl-inspector-traces', DlInspectorTraces);

declare global {
  interface HTMLElementTagNameMap {
    'dl-inspector-traces': DlInspectorTraces;
  }
}
