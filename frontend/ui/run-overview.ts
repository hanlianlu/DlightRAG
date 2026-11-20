// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Run overview: one Run stated as a dossier rather than as its step list.
 *
 *  This is the panel beside an agent's page: the Run's shape as a composition bar, what it delivered,
 *  its agents as a ledger whose rows carry each agent's span over the Run's clock, and a calm Run
 *  quality line whose technical details — the marks a reader can open in an agent's page — fold
 *  behind one disclosure. The default surface is written for the reader of the answer, not for a
 *  developer triaging the Run. The per-agent page's timeline stays the only place a Run renders its
 *  steps; this surface renders aggregates, structure, outcome facts and links.
 *
 *  It owns its sections but not the panel around them: the Inspector draws the title band and the
 *  narrow fold, and hands this element the overview read, the roster page and the Run's answer. The
 *  element raises two intents upward — `dl-run-agent-select` when a reader picks an agent (its ledger
 *  row or its legend entry) and `dl-run-finding-select` when they pick a finding — and the Inspector
 *  decides what selecting one means.
 *
 *  A screen reader hears the same facts: the composition bar itself is decorative and hidden, and its
 *  legend names every segment with its count; every agent segment and finding is a button, so a
 *  colour is never the only thing a reader is given.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import type {
  AgentChildStatus,
  AgentRunStatus,
  AnswerPresentation,
  DecisionMark,
  RunFinding,
  RunOverview,
  TimelineKind,
} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {agentIdOf, MAIN_AGENT} from '../lib/agent-id.ts';
import {agentName} from '../lib/agent-name.ts';
import {elapsed, timeOfDay} from '../lib/date-format.ts';
import {
  attentionSummary,
  barSegments,
  deliveryStates,
  laneGeometry,
  laneRange,
  ledgerRows,
  runIsRunning,
  SEVERITY_ORDER,
  type AttentionSummary,
  type BarDimension,
  type BarSegment,
  type DeliverySummary,
  type RunLedgerRow,
  type TimeRange,
} from '../lib/run-overview.ts';
import {getLocale} from '../i18n/locale.ts';
import {ClockController} from '../lib/clock.ts';
import {LightElement} from '../lib/lit-host.ts';
import {toolDisplay, toolVerbText} from '../lib/tool-display.ts';
import styles from '../styles/run-overview.module.css';
import {agentGlyph, agentStateText} from './agent-status.ts';
import {artifactDeliveryRecord} from './artifact-attestation.ts';
import {markBadge, markLabel} from './run-mark.ts';
import {tokenCountLabel} from './token-count.ts';

/** A roster row a reader can open: the wire leaves the id optional, and a row without one has nothing
 *  to open. */
export type ListedChild = AgentChildStatus & {childSessionId: string};

/** What the delivery section says a Run did, from its state and its Answer. */
type DeliveryState = 'running' | 'stopped' | 'failed' | 'delivered' | 'empty' | 'unknown';

/** One word the client owns: its English source and the catalog id that translates it. */
interface Copy {
  readonly source: string;
  readonly id: string;
}

const SRC: Copy = {source: 'Sources', id: 'runOverview.sources'};
const ARTIFACTS: Copy = {source: 'Files', id: 'runOverview.artifacts'};
const ATTESTATION_NONE: Copy = {source: 'No record', id: 'runOverview.attestationNone'};
const UNAVAILABLE: Copy = {source: 'Unavailable', id: 'runOverview.unavailable'};
const ARTIFACT_ISSUES: Copy = {source: 'File issues', id: 'runOverview.artifactIssues'};
const DELIVERED: Copy = {source: 'Delivered', id: 'runOverview.delivered'};
const QUALITY: Copy = {source: 'Run quality', id: 'runOverview.quality'};
const QUALITY_CLEAN: Copy = {
  source: 'The Run completed with nothing to flag.',
  id: 'runOverview.qualityClean',
};

/** The six step kinds, named for the legend and for a finding that is not a Tool call. A count of one
 *  takes the singular words; the legend's number stands beside them. */
const KIND_TEXT: Record<TimelineKind, {readonly one: Copy; readonly many: Copy}> = {
  thinking: {
    one: {source: 'Thought', id: 'runOverview.kindThought'},
    many: {source: 'Thoughts', id: 'runOverview.kindThoughts'},
  },
  tool: {
    one: {source: 'Tool call', id: 'runOverview.kindToolCall'},
    many: {source: 'Tool calls', id: 'runOverview.kindToolCalls'},
  },
  say: {
    one: {source: 'Message', id: 'runOverview.kindMessage'},
    many: {source: 'Messages', id: 'runOverview.kindMessages'},
  },
  answer: {
    one: {source: 'Answer', id: 'runOverview.kindAnswer'},
    many: {source: 'Answers', id: 'runOverview.kindAnswers'},
  },
  instruction: {
    one: {source: 'Instruction', id: 'runOverview.kindInstruction'},
    many: {source: 'Instructions', id: 'runOverview.kindInstructions'},
  },
  delivery: {
    one: {source: 'Delivery note', id: 'runOverview.kindDeliveryNote'},
    many: {source: 'Delivery notes', id: 'runOverview.kindDeliveryNotes'},
  },
};

/** The honest one-line states a Run that has not delivered can be in. */
const DELIVERY_SENTENCES: Record<Exclude<DeliveryState, 'delivered'>, Copy> = {
  running: {source: 'The Run is still going, so nothing is delivered yet.', id: 'runOverview.deliveryRunning'},
  stopped: {source: 'The Run was stopped before it delivered.', id: 'runOverview.deliveryStopped'},
  failed: {source: 'The Run failed, so nothing was delivered.', id: 'runOverview.deliveryFailed'},
  empty: {source: 'The Run finished without delivering an answer.', id: 'runOverview.deliveryEmpty'},
  unknown: {source: 'No delivery record yet.', id: 'runOverview.deliveryUnknown'},
};

/** One metric of a ledger row, kept small enough that a row's numbers scan as a line. */
const METRICS_SEPARATOR = '·';

/** How many agent tints the stylesheet names: more agents than this repeat the palette. */
const AGENT_TINTS = 6;

/** Which of the bar's ends a tip sits at, so the first and last cannot clip at the bar's edges. */
function edgeOf(index: number, count: number): 'first' | 'last' | typeof nothing {
  if (index === 0) return 'first';
  if (index === count - 1) return 'last';
  return nothing;
}

/** The status row a decision word needs: the agent words are shared with the agent pages, and a
 *  ledger row carries the two fields they read. */
function statusRowOf(row: RunLedgerRow): AgentChildStatus {
  return {
    childSessionId: row.main ? MAIN_AGENT : row.agent,
    status: row.status ?? '',
    objective: row.label ?? undefined,
    modelRole: undefined,
    usage: null,
    operationId: null,
    operationSequence: null,
    operationStatus: null,
    cancellationOrigin: row.cancellationOrigin,
    summary: null,
    resultHandles: [],
    startedAt: null,
    finishedAt: null,
    pendingQuestions: 0,
    resultSequence: null,
    feedback: null,
  };
}

export class DlRunOverview extends LightElement {
  static override properties = {
    overview: {attribute: false},
    loading: {attribute: false},
    failed: {attribute: false},
    entries: {attribute: false},
    lead: {attribute: false},
    run: {attribute: false},
    presentation: {attribute: false},
    selected: {attribute: false},
  };

  /** The Run's whole-Run read; null before it arrives. */
  declare overview: RunOverview | null;
  /** Whether the first read is still in flight. */
  declare loading: boolean;
  /** Whether the last read was refused: the section says so instead of drawing an empty dossier. */
  declare failed: boolean;
  /** The roster's child rows, as the Inspector holds them. */
  declare entries: readonly ListedChild[];
  /** The Run's own agent as a status row, which only the roster read can state. */
  declare lead: ListedChild | null;
  /** Where the Run stands, for the delivery card and the clock; null when the server does not say. */
  declare run: AgentRunStatus | null;
  /** The Run's Answer, when it has one: what the delivery card draws. */
  declare presentation: AnswerPresentation | null;
  /** The agent whose page is on show, by its wire key (`main` or a child Session's id): the bar,
   *  the legend and the ledger mark it, because the reader is looking at it. */
  declare selected: string | null;

  /** Which breakdown the composition bar states. A reader's choice, held while they stay. */
  #dimension: BarDimension = 'kind';
  /** Whether the delivery record's evidence is showing. */
  #recordExpanded = false;
  /** Whether the reader opened the Run quality details; the calm line is the default. */
  #qualityOpen = false;
  /** The clock this render draws "so far" times against: a running Run's lane and duration move. */
  #now = Date.now();

  constructor() {
    super();
    // A Run still going has a lane and a duration that move: the page owns the clock it draws them
    // against, and the controller exists only while something on it is still running.
    new ClockController(this, () => this.#moving());
    this.overview = null;
    this.loading = false;
    this.failed = false;
    this.entries = [];
    this.lead = null;
    this.run = null;
    this.presentation = null;
    this.selected = null;
  }

  /** Whether the page draws a time that moves: the Run is still going, or one of its agents is. */
  #moving(): boolean {
    if (runIsRunning(this.run?.status ?? null)) return true;
    if (this.lead?.status === 'running') return true;
    return (this.entries ?? []).some((row) => row.status === 'running');
  }

  protected override willUpdate(): void {
    // One clock for the render: every "so far" time is drawn against this value, so a tick and a
    // parent's update meeting in one frame cannot disagree about what time it is.
    this.#now = Date.now();
  }

  protected override updated(changed: PropertyValues<this>): void {
    super.updated(changed);
    // A record folded under one Answer is not a claim the next Answer stated.
    if (changed.has('presentation')) this.#recordExpanded = false;
  }

  protected override render(): TemplateResult | typeof nothing {
    if (this.failed) {
      return html`
        <div class=${styles.overview} data-run-overview data-state="failed">
          <p class="dl-note ${styles.note}" role="status" data-overview-failed>
            ${msg('The Run overview could not be read.', {id: 'runOverview.failed'})}
          </p>
        </div>
      `;
    }
    const overview = this.overview;
    if (overview === null) {
      if (!this.loading) return nothing;
      return html`
        <div class=${styles.overview} data-run-overview data-state="loading">
          <p class="dl-note ${styles.note}" role="status" data-overview-loading>
            ${msg('Reading the Run overview…', {id: 'runOverview.loading'})}
          </p>
        </div>
      `;
    }
    const rows = ledgerRows(overview, this.entries ?? [], this.lead ?? null, this.run, this.#now);
    const range = laneRange(overview, this.run, this.#now);
    const delivery = deliveryStates(this.presentation);
    return html`
      <div class=${styles.overview} data-run-overview>
        ${this.#composition(overview, rows)}
        ${this.#delivery(delivery)}
        ${this.#ledger(rows, range)}
        ${this.#quality(attentionSummary(overview), rows)}
      </div>
    `;
  }

  /** The Run's shape: a bar and the legend that names every part of it, switchable between the step
   *  kinds that made the Run and the agents that did. */
  #composition(overview: RunOverview, rows: readonly RunLedgerRow[]): TemplateResult {
    const dimension = this.#dimension;
    const segments = barSegments(overview, dimension);
    const head = html`
      <div class=${styles.sectionHead}>
        <h3 class=${styles.sectionTitle} id="run-overview-composition">
          ${msg('Composition', {id: 'runOverview.composition'})}
        </h3>
        ${segments.length === 0 ? nothing : html`
          <div class=${styles.dimensions} role="group"
               aria-label=${msg('Group the steps by', {id: 'runOverview.groupBy'})}>
            <button type="button" class=${styles.dimensionButton} data-dimension="kind"
                    aria-pressed=${String(dimension === 'kind')} @click=${() => this.#setDimension('kind')}>
              ${msg('By kind', {id: 'runOverview.byKind'})}
            </button>
            <button type="button" class=${styles.dimensionButton} data-dimension="agent"
                    aria-pressed=${String(dimension === 'agent')} @click=${() => this.#setDimension('agent')}>
              ${msg('By agent', {id: 'runOverview.byAgent'})}
            </button>
          </div>
        `}
      </div>
    `;
    if (segments.length === 0) {
      return html`
        <section class=${styles.section} aria-labelledby="run-overview-composition">
          ${head}
          <p class="dl-note ${styles.note}" data-composition-empty>${msg('No steps yet.', {id: 'runOverview.noSteps'})}</p>
        </section>
      `;
    }
    return html`
      <section class=${styles.section} aria-labelledby="run-overview-composition" data-composition>
        ${head}
        ${dimension === 'agent'
          ? html`
            <div class=${styles.bar} role="group" aria-labelledby="run-overview-composition"
                 data-bar data-dimension="agent">
              ${segments.map((segment, index) => this.#agentBarPart(segment, index, segments.length, rows))}
            </div>`
          : html`
            <div class=${styles.bar} aria-hidden="true" data-bar data-dimension="kind">
              ${segments.map((segment, index) => this.#kindBarPart(segment, index, segments.length))}
            </div>`}
        ${dimension === 'agent'
          ? this.#agentLegend(segments, rows)
          : this.#kindLegend(segments)}
      </section>
    `;
  }

  /** One kind's part of the decorative bar: its width is its count, its tip names it, and there is
   *  nothing to select a kind of. */
  #kindBarPart(segment: BarSegment, index: number, count: number): TemplateResult {
    return html`
      <span class=${styles.barPart} data-bar-segment data-key=${segment.key}
            data-edge=${edgeOf(index, count)}
            style=${`flex-grow: ${segment.count}`}>
        <span class=${styles.barTip} aria-hidden="true">${this.#kindTip(segment)}</span>
      </span>
    `;
  }

  /** One agent's part of the bar: a button that selects that agent's page, with the same name the
   *  legend states. The visual bar is six pixels; the pointer target reaches above and below it. */
  #agentBarPart(segment: BarSegment, index: number, count: number, rows: readonly RunLedgerRow[]): TemplateResult {
    const name = this.#agentName(segment.key, rows);
    const current = this.selected !== null && this.selected === segment.key;
    return html`
      <button type="button" class=${styles.barPart} data-bar-segment data-key=${segment.key}
              data-agent-tint=${String(index % AGENT_TINTS)} data-edge=${edgeOf(index, count)}
              aria-label=${this.#agentSegmentAria(name, segment.count)}
              aria-current=${current ? 'true' : nothing}
              style=${`flex-grow: ${segment.count}`}
              @click=${() => this.#selectAgent(segment.key)}>
        <span class=${styles.barTip} aria-hidden="true">${name} · ${this.#stepCount(segment.count)}</span>
      </button>
    `;
  }

  /** What a kind's tip says: the kind's name and its exact count. */
  #kindTip(segment: BarSegment): string {
    return `${this.#kindNoun(segment.key as TimelineKind, segment.count)} · ${segment.count}`;
  }

  /** What an agent's segment and legend entry are called: its name and its step count. */
  #agentSegmentAria(name: string, count: number): string {
    return count === 1
      ? msg(str`${name}: 1 step`, {id: 'runOverview.agentSegmentOneAria'})
      : msg(str`${name}: ${count} steps`, {id: 'runOverview.agentSegmentManyAria'});
  }

  /** The kinds' legend: every kind the Run holds, named with its exact count. Text, not controls —
   *  there is nothing to select a kind of. */
  #kindLegend(segments: readonly BarSegment[]): TemplateResult {
    return html`
      <ul class=${styles.legend} role="list" data-legend="kind">
        ${segments.map((segment) => html`
          <li class=${styles.legendItem} data-legend-kind=${segment.key}>
            <span class=${styles.swatch} data-kind=${segment.key} aria-hidden="true"></span>
            <span class=${styles.legendName}>${this.#kindNoun(segment.key as TimelineKind, segment.count)}</span>
            <span class=${styles.legendCount}>${segment.count}</span>
          </li>
        `)}
      </ul>
    `;
  }

  /** The agents' legend: every agent with steps in the Run, each a button that selects its page. */
  #agentLegend(segments: readonly BarSegment[], rows: readonly RunLedgerRow[]): TemplateResult {
    return html`
      <ul class=${styles.legend} role="list" data-legend="agent">
        ${segments.map((segment, index) => {
          const name = this.#agentName(segment.key, rows);
          const current = this.selected !== null && this.selected === segment.key;
          return html`
            <li class=${styles.legendItem}>
              <button type="button" class=${styles.legendAgent} data-legend-agent=${segment.key}
                      aria-label=${this.#agentSegmentAria(name, segment.count)}
                      aria-current=${current ? 'true' : nothing}
                      @click=${() => this.#selectAgent(segment.key)}>
                <span class=${styles.swatch} data-agent-tint=${String(index % AGENT_TINTS)} aria-hidden="true"></span>
                <span class=${styles.legendName}>${name}</span>
                <span class=${styles.legendCount}>${segment.count}</span>
              </button>
            </li>
          `;
        })}
      </ul>
    `;
  }

  /** What the Run delivered, said honestly: a Run still going has nothing to show, and a Run that
   *  failed delivered nothing — neither is a green verdict. */
  #delivery(delivery: DeliverySummary | null): TemplateResult {
    const state = this.#deliveryState(delivery);
    const sentence = this.#deliverySentence(state);
    return html`
      <section class=${styles.section} aria-labelledby="run-overview-delivery" data-delivery data-delivery-state=${state}>
        <div class=${styles.sectionHead}>
          <h3 class=${styles.sectionTitle} id="run-overview-delivery">
            ${msg('Delivery', {id: 'runOverview.delivery'})}
          </h3>
          ${state === 'delivered'
            ? html`<span class=${styles.stateWord}>${msg(DELIVERED.source, {id: DELIVERED.id})}</span>`
            : nothing}
        </div>
        ${sentence === null
          ? (delivery === null ? nothing : this.#deliveryRecord(delivery))
          : html`<p class="dl-note ${styles.note}" data-delivery-sentence>${sentence}</p>`}
      </section>
    `;
  }

  /** The one honest line a Run that has not delivered states; null when it did deliver. */
  #deliverySentence(state: DeliveryState): string | null {
    if (state === 'delivered') return null;
    const copy = DELIVERY_SENTENCES[state];
    return msg(copy.source, {id: copy.id});
  }

  /** What a delivered Run states: the counts a reader can act on, its Artifacts with their
   *  records, and its issues. A number is only drawn when there is something to count; a text-only
   *  answer says that in one quiet line instead of two zeroes. */
  #deliveryRecord(delivery: DeliverySummary): TemplateResult {
    const counts = delivery.sources > 0 || delivery.artifacts.length > 0;
    return html`
      <div class=${styles.deliveryCard}>
        ${counts ? html`
          <dl class=${styles.deliveryCounts} data-delivery-counts>
            ${delivery.sources === 0
              ? nothing
              : this.#count(msg(SRC.source, {id: SRC.id}), delivery.sources, 'sources')}
            ${delivery.artifacts.length === 0
              ? nothing
              : this.#count(msg(ARTIFACTS.source, {id: ARTIFACTS.id}), delivery.artifacts.length, 'artifacts')}
          </dl>
        ` : html`
          <p class="dl-note ${styles.note}" data-delivery-text-only>
            ${msg('Text-only answer: no cited sources or published files.', {id: 'runOverview.deliveryTextOnly'})}
          </p>
        `}
        ${delivery.artifacts.length === 0
          ? nothing
          : html`
            <ul class=${styles.artifacts} role="list" data-artifacts>
              ${delivery.artifacts.map((state) => html`
                <li class=${styles.artifact} data-artifact data-status=${state.artifact.status}>
                  <span class=${styles.artifactText}>
                    <span class=${styles.artifactLabel} title=${state.artifact.label}>${state.artifact.label}</span>
                    <span class=${styles.artifactFile}>${state.artifact.filename}</span>
                  </span>
                  ${state.available
                    ? nothing
                    : html`<span class=${styles.artifactState}>${msg(UNAVAILABLE.source, {id: UNAVAILABLE.id})}</span>`}
                </li>
              `)}
            </ul>
          `}
        ${this.#attestationBuckets(delivery)}
        ${delivery.outcome === 'complete'
          ? nothing
          : html`
            <p class=${styles.warning} data-delivery-outcome>
              ${msg('Some requested files could not be published.', {id: 'answerPresentation.publicationWarning'})}
            </p>
          `}
        ${delivery.issues.length === 0 ? nothing : html`
          <div class=${styles.issues} data-artifact-issues>
            <p class=${styles.issueHeading}>${msg(ARTIFACT_ISSUES.source, {id: ARTIFACT_ISSUES.id})}</p>
            <ul class=${styles.issueList} role="list">
              ${delivery.issues.map((issue) => html`
                <li class=${styles.issue}>${issue.description === '' ? issue.kind : issue.description}</li>
              `)}
            </ul>
          </div>
        `}
        ${artifactDeliveryRecord(delivery.record, {
          expanded: this.#recordExpanded,
          evidenceId: 'run-overview-delivery-evidence',
          onToggle: this.#toggleRecord,
        })}
      </div>
    `;
  }

  /** How the Run's Artifacts stand when they do not all stand alike.
   *
   *  The server stamps one record on every Artifact of a Run, so a homogeneous Run states it once
   *  through :func:`artifactDeliveryRecord`; a bucket line here would repeat that record once per
   *  status. It is drawn only when the Artifacts disagree (a mixed or older payload), where the
   *  counts are the fact a single record cannot carry. */
  #attestationBuckets(delivery: DeliverySummary): TemplateResult | typeof nothing {
    const buckets: Array<{count: number; copy: Copy; key: string}> = [
      {count: delivery.attestation.attested, key: 'attested', copy: {source: 'Attested', id: 'artifactAttestation.attested'}},
      {count: delivery.attestation.unattested, key: 'unattested', copy: {source: 'Not attested', id: 'artifactAttestation.notAttested'}},
      {count: delivery.attestation.unknown, key: 'unknown', copy: {source: 'Attestation unknown', id: 'artifactAttestation.unknown'}},
      {count: delivery.attestation.none, key: 'none', copy: ATTESTATION_NONE},
    ].filter((bucket) => bucket.count > 0);
    if (buckets.length < 2) return nothing;
    return html`
      <p class=${styles.buckets} data-attestation-buckets>
        ${buckets.map((bucket) => html`
          <span class=${styles.bucket} data-bucket=${bucket.key}>
            ${msg(bucket.copy.source, {id: bucket.copy.id})} ${bucket.count}
          </span>
        `)}
      </p>
    `;
  }

  /** One number beside the word that says what it counts. */
  #count(label: string, count: number, name: string): TemplateResult {
    return html`
      <div class=${styles.deliveryCount} data-delivery-count=${name}>
        <dt>${label}</dt>
        <dd>${count}</dd>
      </div>
    `;
  }

  /** The agents' ledger: one row per agent, each fusing the whole-Run rollup with the roster's live
   *  status, clock and token count, and drawing the agent's span over the Run's clock. */
  #ledger(rows: readonly RunLedgerRow[], range: TimeRange | null): TemplateResult | typeof nothing {
    if (rows.length === 0) return nothing;
    return html`
      <section class=${styles.section} aria-labelledby="run-overview-agents" data-ledger>
        <div class=${styles.sectionHead}>
          <h3 class=${styles.sectionTitle} id="run-overview-agents">
            ${msg('Agents', {id: 'runOverview.agents'})}
          </h3>
          <span class=${styles.sectionCount} data-agent-count>${rows.length}</span>
        </div>
        <ul class=${styles.ledger} role="list">
          ${rows.map((row) => {
            const current = this.selected !== null && this.selected === row.agent;
            return html`
              <li class=${styles.ledgerItem}>
                <button type="button" class=${styles.ledgerRow} data-agent-row data-agent=${row.agent}
                        aria-current=${current ? 'true' : nothing}
                        @click=${() => this.#selectAgent(row.agent)}>
                  ${this.#ledgerHead(row, rows)}
                  ${this.#lane(row, range, rows)}
                  ${this.#metrics(row)}
                </button>
              </li>
            `;
          })}
        </ul>
      </section>
    `;
  }

  /** A ledger row's first line: the glyph and name, then where the agent stands. */
  #ledgerHead(row: RunLedgerRow, rows: readonly RunLedgerRow[]): TemplateResult {
    const name = this.#agentName(row.agent, rows);
    return html`
      <span class=${styles.rowHead}>
        ${row.status === null
          ? html`<span class=${styles.glyphBox} aria-hidden="true"></span>`
          : agentGlyph(statusRowOf(row))}
        <span class=${styles.rowName} title=${row.main ? name : row.label ?? name}>${name}</span>
        ${row.status === null
          ? nothing
          : html`<span class=${styles.rowState} data-state=${row.status}>${agentStateText(statusRowOf(row))}</span>`}
      </span>
    `;
  }

  /** The agent's span over the Run's clock, with a dot per decision it states. Decorative: the row's
   *  own words state its marks, and the lane only shows where the span sits in the Run. */
  #lane(row: RunLedgerRow, range: TimeRange | null, rows: readonly RunLedgerRow[]): TemplateResult {
    const geometry = laneGeometry(row, range);
    const agent = this.#agentName(row.agent, rows);
    const took = row.durationMs === null ? '' : elapsed(row.durationMs, getLocale());
    const spanTitle = took === '' ? agent : `${agent} · ${took}`;
    return html`
      <span class=${styles.laneRow} aria-hidden="true">
        <span class=${styles.lane} data-lane>
          <span class=${styles.laneTrack}></span>
          ${row.start === null || row.end === null || range === null
            ? nothing
            : html`
              <span class=${styles.laneSpan} data-lane-span data-running=${String(row.running)}
                    title=${spanTitle}
                    style=${`inset-inline-start: ${geometry.left}%; width: ${geometry.width}%`}></span>
              ${geometry.dots.map((dot) => html`
                <span class=${styles.laneDot} data-lane-dot data-mark=${dot.mark ?? 'delivery'}
                      title=${this.#dotTitle(dot.mark, dot.at)}
                      style=${`inset-inline-start: ${dot.left}%`}></span>
              `)}
            `}
        </span>
      </span>
    `;
  }

  /** What one decision dot is: its mark's word and when it happened. Supplement only — the row's
   *  words state the same decisions, and the lane is decoration for a screen reader. */
  #dotTitle(mark: DecisionMark | null, at: number | null): string {
    const word = mark === null
      ? this.#kindNoun('delivery', 1)
      : markLabel(mark);
    return at === null ? word : `${word} · ${timeOfDay(at, getLocale(), true)}`;
  }

  /** The row's readable numbers: how long, what it spent, and how much it did. The marks it states
   *  are the quality section's details, folded away, so the ledger's default line carries no
   *  developer vocabulary. */
  #metrics(row: RunLedgerRow): TemplateResult {
    const metrics: TemplateResult[] = [];
    if (row.durationMs !== null) {
      metrics.push(html`
        <span class=${styles.metric} data-metric="duration" title=${msg('Duration', {id: 'runOverview.duration'})}>
          ${elapsed(row.durationMs, getLocale())}
        </span>
      `);
    }
    if (row.tokens !== null) {
      metrics.push(html`<span class=${styles.metric} data-metric="tokens">${tokenCountLabel(row.tokens)}</span>`);
    }
    metrics.push(html`<span class=${styles.metric} data-metric="steps">${this.#stepCount(row.steps)}</span>`);
    metrics.push(html`
      <span class=${styles.metric} data-metric="tool-calls">${this.#toolCallCount(row.toolCalls)}</span>
    `);
    return html`
      <span class=${styles.metrics} data-metrics>
        ${metrics.map((metric, index) => (index === 0 ? metric : html`
          <span class=${styles.metricSeparator} aria-hidden="true">${METRICS_SEPARATOR}</span>${metric}
        `))}
      </span>
    `;
  }

  /** The Run's quality as one calm line: whether anything was worth flagging, said in the reader's
   *  words. What the marks and findings are — the developer-facing detail — folds behind one
   *  disclosure, so the default dossier stays quiet while nothing is removed. */
  #quality(attention: AttentionSummary, rows: readonly RunLedgerRow[]): TemplateResult {
    const total = attention.total;
    const clean = total === 0;
    const state = clean ? 'clean' : 'noted';
    return html`
      <section class=${styles.section} aria-labelledby="run-overview-quality"
               data-attention data-quality data-quality-state=${state}>
        <div class=${styles.sectionHead} data-quality-head>
          <h3 class=${styles.sectionTitle} id="run-overview-quality">
            ${msg(QUALITY.source, {id: QUALITY.id})}
          </h3>
          ${clean ? nothing : html`
            <button type="button" class="dl-text-action dl-text-action--disclosure"
                    data-quality-toggle aria-expanded=${String(this.#qualityOpen)}
                    aria-controls="run-overview-quality-details" @click=${this.#toggleQuality}>
              ${this.#qualityOpen
                ? msg('Hide details', {id: 'runOverview.qualityHideDetails'})
                : msg('Show details', {id: 'runOverview.qualityShowDetails'})}
              ${icon('chevron-down', {size: 'xs'})}
            </button>
          `}
        </div>
        <p class=${styles.qualityLine} data-quality-line>
          <span class=${styles.qualityGlyph} data-quality-glyph=${state} aria-hidden="true">
            ${icon(clean ? 'check' : 'info', {size: 'sm'})}
          </span>
          <span>${clean
            ? msg(QUALITY_CLEAN.source, {id: QUALITY_CLEAN.id})
            : msg(str`${total} details were recorded during the Run (retries, degraded searches, or delivery notes).`, {id: 'runOverview.qualityNoted'})}</span>
        </p>
        ${clean ? nothing : html`
          <div class=${styles.qualityDetails} id="run-overview-quality-details"
               role="region" tabindex="0"
               aria-label=${msg('Run details', {id: 'runOverview.qualityDetailsAria'})}
               ?hidden=${!this.#qualityOpen}>
            ${this.#qualityDetails(attention, rows)}
          </div>
        `}
      </section>
    `;
  }

  /** What sits behind the quality line: the exact counts and the findings a reader can open, in
   *  severity order. Shown only when the reader asks for it. */
  #qualityDetails(attention: AttentionSummary, rows: readonly RunLedgerRow[]): TemplateResult {
    const counts = attention.counts;
    const hasCounts = SEVERITY_ORDER.some((mark) => counts[mark] > 0) || counts.delivery > 0;
    return html`
      ${hasCounts ? html`
        <div class=${styles.counts} data-attention-counts>
          ${SEVERITY_ORDER.map((mark) => counts[mark] > 0 ? html`
            <span class=${styles.countChip} data-count-mark=${mark}>${markLabel(mark)} ${counts[mark]}</span>
          ` : nothing)}
          ${counts.delivery > 0 ? html`
            <span class=${styles.countChip} data-count-delivery>
              ${this.#kindNoun('delivery', counts.delivery)} ${counts.delivery}
            </span>
          ` : nothing}
        </div>
      ` : nothing}
      ${attention.findings.length === 0
        ? nothing
        : html`
          <ul class=${styles.findings} role="list">
            ${attention.findings.map((finding) => this.#finding(finding, rows))}
          </ul>
        `}
      ${attention.truncated > 0 ? html`
        <p class="dl-note ${styles.note}" data-attention-truncated>
          ${msg(str`Showing the first ${attention.findings.length} of ${attention.total} findings.`, {id: 'runOverview.findingsTruncated'})}
        </p>
      ` : nothing}
    `;
  }

  /** One finding: its marks, who stated it, what it did, the start of what came back, and when. The
   *  whole row is one button that asks the Inspector to open that step in its agent's page. */
  #finding(finding: RunFinding, rows: readonly RunLedgerRow[]): TemplateResult {
    const title = this.#findingTitle(finding);
    const time = timeOfDay(finding.at, getLocale(), true);
    return html`
      <li class=${styles.finding}>
        <button type="button" class=${styles.findingRow} data-finding
                data-agent=${finding.agent} data-sequence=${finding.sequence} data-kind=${finding.kind}
                @click=${() => this.#selectFinding(finding)}>
          <span class=${styles.findingHead}>
            <span class=${styles.findingTitle}>
              ${finding.marks.length === 0
                ? html`
                  <span class=${styles.findingKind} data-run-mark=${finding.kind}>
                    ${this.#kindNoun(finding.kind, 1)}
                  </span>
                `
                : finding.marks.map((mark) => markBadge(mark))}
              ${finding.kind === 'tool'
                ? html`<span class=${styles.findingVerb}>${title}</span>`
                : nothing}
              ${finding.subject === null || finding.subject === ''
                ? nothing
                : html`<span class=${styles.findingSubject}>${finding.subject}</span>`}
            </span>
            ${time === '' ? nothing : html`<span class=${styles.findingTime}>${time}</span>`}
          </span>
          <span class=${styles.findingMeta}>
            <span class=${styles.findingAgent}>${this.#agentName(finding.agent, rows)}</span>
            ${finding.excerpt === ''
              ? nothing
              : html`<span class=${styles.findingExcerpt} title=${finding.excerpt}>${finding.excerpt}</span>`}
          </span>
        </button>
      </li>
    `;
  }

  /** The verb a Tool finding leads with; a delivery finding's kind badge already says what it is. */
  #findingTitle(finding: RunFinding): string {
    if (finding.kind === 'tool' && finding.name !== null) {
      const display = toolDisplay(finding.name);
      return toolVerbText(display.verb, display.verbId);
    }
    return '';
  }

  /** An agent's name as the ledger and the finding rows say it: the Run's own agent in the product's
   *  words, a child by its objective, and a child the roster page does not name by its place. */
  #agentName(agent: string, rows: readonly RunLedgerRow[]): string {
    const id = agentIdOf(agent);
    if (id.kind === 'run') return agentName(id);
    const row = rows.find((candidate) => candidate.agent === agent) ?? null;
    const label = row?.label ?? null;
    if (label !== null && label !== '' && label !== MAIN_AGENT) return label;
    const index = rows.filter((candidate) => !candidate.main).findIndex((candidate) => candidate.agent === agent);
    return index >= 0
      ? msg(str`Child ${index + 1}`, {id: 'runOverview.childNumber'})
      : agentName(id);
  }

  /** A kind's words for a count: the singular when there is one of it. */
  #kindNoun(kind: TimelineKind, count: number): string {
    const words = KIND_TEXT[kind];
    return count === 1
      ? msg(words.one.source, {id: words.one.id})
      : msg(words.many.source, {id: words.many.id});
  }

  /** How many steps a row states, with its own word. */
  #stepCount(count: number): string {
    return count === 1
      ? msg('1 step', {id: 'runOverview.stepOne'})
      : msg(str`${count} steps`, {id: 'runOverview.steps'});
  }

  /** How many Tool calls a row states, with its own word. */
  #toolCallCount(count: number): string {
    return count === 1
      ? msg('1 tool call', {id: 'runOverview.toolCallOne'})
      : msg(str`${count} tool calls`, {id: 'runOverview.toolCalls'});
  }

  /** Where the Run stands for the delivery card's honest sentence. */
  #deliveryState(delivery: DeliverySummary | null): DeliveryState {
    const status = this.run?.status ?? null;
    if (status !== null && runIsRunning(status)) return 'running';
    if (status === 'failed') return 'failed';
    if (status === 'cancelled') return 'stopped';
    if (delivery !== null) return 'delivered';
    if (status === 'succeeded') return 'empty';
    return 'unknown';
  }

  #setDimension = (dimension: BarDimension): void => {
    if (this.#dimension === dimension) return;
    this.#dimension = dimension;
    this.requestUpdate();
  };

  #toggleRecord = (): void => {
    this.#recordExpanded = !this.#recordExpanded;
    this.requestUpdate();
  };

  #toggleQuality = (): void => {
    this.#qualityOpen = !this.#qualityOpen;
    this.requestUpdate();
  };

  #selectAgent = (agent: string): void => {
    this.dispatchEvent(new CustomEvent('dl-run-agent-select', {
      detail: {agent},
      bubbles: true,
      composed: true,
    }));
  };

  #selectFinding = (finding: RunFinding): void => {
    this.dispatchEvent(new CustomEvent('dl-run-finding-select', {
      detail: {agent: finding.agent, sequence: finding.sequence},
      bubbles: true,
      composed: true,
    }));
  };
}

customElements.define('dl-run-overview', DlRunOverview);

declare global {
  interface HTMLElementTagNameMap {
    'dl-run-overview': DlRunOverview;
  }
}
