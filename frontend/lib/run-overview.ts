// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Run panel's view models: the overview's aggregates turned into the facts its sections draw.
 *
 *  The panel states the whole Run, never one step of it: its bar is a breakdown, its ledger is one
 *  row per agent with that agent's span over the Run's clock, and its attention list is the capped
 *  findings the server states. Everything here is pure arithmetic over the wire — the same inputs
 *  always give the same percentages — so the geometry is tested without a page, and no word of
 *  language is resolved (the component applies the words).
 *
 *  A missing fact stays missing. An absent token count is `null`, not zero, and a Run whose clock is
 *  unknown draws no lane; a panel that guessed would read as a record of something the server never
 *  stated.
 */

import type {
  AgentChildStatus,
  AgentRunStatus,
  AnswerPresentation,
  ArtifactAttestation,
  DecisionMark,
  RunAgentRollup,
  RunFinding,
  RunMarkCounts,
  RunOverview,
  TimelineKind,
} from '../api/conversations.ts';
import {MAIN_AGENT} from './agent-id.ts';

/** The two breakdowns the composition bar offers: by step kind, or by agent. */
export type BarDimension = 'kind' | 'agent';

/** One segment of the composition bar: what it holds, and what share of the Run it is. */
export interface BarSegment {
  /** The kind token or the agent key ('main' or a child Session's id). */
  readonly key: string;
  readonly dimension: BarDimension;
  readonly count: number;
  /** The part of the whole Run this segment is, from 0 to 1. */
  readonly share: number;
}

/** The order the product lists the six step kinds in: the model's thinking leads, then its actions,
 *  then what it delivered. A kind the Run does not hold is left out of the bar. */
export const KIND_ORDER: readonly TimelineKind[] = [
  'thinking', 'tool', 'say', 'answer', 'instruction', 'delivery',
];

/** The marks in the order the attention list ranks them: what failed first, then what degraded, then
 *  what the Run had to correct, then the calls it repeated. Lower ranks sort first. */
export const SEVERITY_ORDER: readonly DecisionMark[] = ['failed', 'degraded', 'redirect', 'repeated'];

/** The rank a finding sorts by; a delivery step holds no mark and ranks after every marked step. */
export function severityRank(finding: Pick<RunFinding, 'marks'>): number {
  let rank = 0;
  for (const mark of SEVERITY_ORDER) {
    if (finding.marks.includes(mark)) return rank;
    rank += 1;
  }
  return rank;
}

/** The one mark that speaks for a finding: its most severe, or null for a delivery step. */
function findingMark(finding: Pick<RunFinding, 'marks'>): DecisionMark | null {
  for (const mark of SEVERITY_ORDER) {
    if (finding.marks.includes(mark)) return mark;
  }
  return null;
}

/** Every instant the Run states one way: ISO-8601 text from the wire as epoch milliseconds. */
function instantOf(text: string | null | undefined): number | null {
  if (text === null || text === undefined || text === '') return null;
  const instant = Date.parse(text);
  return Number.isNaN(instant) ? null : instant;
}

/** What the Run's steps break down to in one dimension, biggest fact first: a zero count is not a
 *  segment, and a Run with no steps has no bar. */
export function barSegments(
  overview: RunOverview | null,
  dimension: BarDimension,
): readonly BarSegment[] {
  if (overview === null) return [];
  const entries: {key: string; count: number}[] = dimension === 'kind'
    ? KIND_ORDER.map((kind) => ({key: kind, count: overview.totals.kinds[kind]}))
    : [...overview.agents]
      .sort((one, other) => (agentRank(one) - agentRank(other)) || (one.order - other.order))
      .map((agent) => ({key: agent.agent, count: agent.steps}));
  const whole = entries.reduce((total, entry) => total + Math.max(0, entry.count), 0);
  if (whole <= 0) return [];
  return entries
    .filter((entry) => entry.count > 0)
    .map((entry) => ({
      key: entry.key,
      dimension,
      count: entry.count,
      share: entry.count / whole,
    }));
}

/** The Run's own agent sorts before every child, whatever its stated order. */
function agentRank(agent: RunAgentRollup): number {
  return agent.agent === MAIN_AGENT ? 0 : 1;
}

/** The Run's spent tokens, split the way the Run record states them: its own agent, its children,
 *  and the two together. A Run whose record states nothing has three nulls — unknown, not free. */
export interface RunTokenTriple {
  readonly total: number | null;
  readonly main: number | null;
  readonly children: number | null;
}

/** The total-token counter one usage detail set states, or null when it states none. */
function tokenTotal(details: Readonly<Record<string, number>> | null | undefined): number | null {
  const total = details?.total_tokens;
  return typeof total === 'number' && Number.isFinite(total) ? total : null;
}

/** The Run's token total, its main agent's share, and its children's share. */
export function runTokens(overview: RunOverview | null): RunTokenTriple {
  const usage = overview?.usage ?? null;
  return {
    total: tokenTotal(usage?.inclusiveUsageDetails),
    main: tokenTotal(usage?.usageDetails),
    children: tokenTotal(usage?.childUsageDetails),
  };
}

/** A span of instants in epoch milliseconds; both ends are known. */
export interface TimeRange {
  readonly start: number;
  readonly end: number;
}

/** Whether a Run state still moves: anything but a settled state is running for the clock's sake. */
export function runIsRunning(status: string | null): boolean {
  return status !== null && status !== '' && status !== 'succeeded'
    && status !== 'failed' && status !== 'cancelled';
}

/** The Run's clock: the roster row's own start and finish when it states them, else the span every
 *  step it holds covers, ended at `now` while the Run is still going. Null when no end can be known —
 *  a panel with no clock draws no lane rather than one from zero. */
export function laneRange(
  overview: RunOverview | null,
  run: AgentRunStatus | null,
  now: number,
): TimeRange | null {
  const starts: number[] = [];
  const ends: number[] = [];
  const runStart = instantOf(run?.startedAt);
  if (runStart !== null) starts.push(runStart);
  for (const agent of overview?.agents ?? []) {
    const first = instantOf(agent.firstAt);
    const last = instantOf(agent.lastAt);
    if (first !== null) starts.push(first);
    if (last !== null) ends.push(last);
  }
  if (starts.length === 0) return null;
  const start = Math.min(...starts);
  let end = instantOf(run?.finishedAt);
  if (end === null && run !== null && runIsRunning(run.status)) end = now;
  if (end === null && ends.length > 0) end = Math.max(...ends);
  if (end === null) return null;
  return {start, end: Math.max(start, end)};
}

/** How long the Run has taken, or took: the width of its clock. Null when the clock is unknown. */
export function runDurationMs(
  overview: RunOverview | null,
  run: AgentRunStatus | null,
  now: number,
): number | null {
  const range = laneRange(overview, run, now);
  return range === null ? null : range.end - range.start;
}

/** One agent's row in the ledger: the whole-Run rollup merged with what the roster states about the
 *  same agent. The server's overview is the count; the roster is where its status, clock and token
 *  count still moving come from, so a row can be current between overview reads. */
export interface RunLedgerRow {
  readonly agent: string;
  readonly main: boolean;
  /** What the server calls it: the Run's own agent's label is 'main', a child's its objective. */
  readonly label: string | null;
  /** Where the roster says it stands; null for an agent the roster page has not named. */
  readonly status: string | null;
  /** Who stopped it, when a cancellation states one; null otherwise. */
  readonly cancellationOrigin: string | null;
  /** Whether its clock still moves, so the component can time a date that grows. */
  readonly running: boolean;
  /** The lane's left edge: its earliest step, or the roster's start when it has none yet. */
  readonly start: number | null;
  /** The lane's right edge: its latest step, or the render's clock while it runs. */
  readonly end: number | null;
  readonly durationMs: number | null;
  /** The tokens the roster states for it; null when it states none. */
  readonly tokens: number | null;
  readonly steps: number;
  readonly toolCalls: number;
  readonly marks: RunMarkCounts;
  /** Every mark the agent's steps state, which can exceed the steps that state one. */
  readonly markTotal: number;
  /** This agent's rows of the capped findings list, in merge order: the lane's decision dots. */
  readonly findings: readonly RunFinding[];
  readonly order: number;
}

const ZERO_MARKS: RunMarkCounts = {failed: 0, degraded: 0, repeated: 0, redirect: 0};

/** The roster row that stands for an agent, if the roster page holds it. */
function rosterRowFor(
  agent: string,
  rosterRows: readonly (AgentChildStatus & {childSessionId: string})[],
  lead: (AgentChildStatus & {childSessionId: string}) | null,
): (AgentChildStatus & {childSessionId: string}) | null {
  if (agent === MAIN_AGENT) return lead;
  return rosterRows.find((row) => row.childSessionId === agent) ?? null;
}

/** The tokens a roster row states for its agent: its provider's own total when it states one. */
function rosterTokens(row: AgentChildStatus | null): number | null {
  const total = row?.usage?.total_tokens;
  return typeof total === 'number' && Number.isFinite(total) ? total : null;
}

/** The lane's start and end for one agent, and whether either still moves. */
function laneSpan(
  agent: RunAgentRollup | null,
  roster: AgentChildStatus | null,
  running: boolean,
  now: number,
): {start: number | null; end: number | null} {
  const start = instantOf(agent?.firstAt) ?? instantOf(roster?.startedAt);
  let end = instantOf(agent?.lastAt) ?? instantOf(roster?.finishedAt);
  if (running) end = end === null ? now : Math.max(end, now);
  return {start, end};
}

/** The ledger rows: main first, then the roster's children, each fusing the overview's whole-Run
 *  rollup with the roster's live status and clock.
 *
 *  The union matters in both directions: an agent the roster names before it writes a step still gets
 *  a row (zero counts, its own status), and a roster row's clock, state and token count stay current
 *  while the overview read behind them settles. An agent that only the overview names — a Session
 *  past the roster page — is still a row of the Run's structure, drawn with what the aggregate says. */
export function ledgerRows(
  overview: RunOverview | null,
  rosterRows: readonly (AgentChildStatus & {childSessionId: string})[],
  lead: (AgentChildStatus & {childSessionId: string}) | null,
  run: AgentRunStatus | null,
  now: number,
): readonly RunLedgerRow[] {
  const rows: RunLedgerRow[] = [];
  const named = new Set<string>();
  const rollups = [...(overview?.agents ?? [])].sort(
    (one, other) => (agentRank(one) - agentRank(other)) || (one.order - other.order),
  );
  const findings = overview?.findings ?? [];
  for (const agent of rollups) {
    named.add(agent.agent);
    const roster = rosterRowFor(agent.agent, rosterRows, lead);
    const main = agent.agent === MAIN_AGENT;
    const status = roster?.status ?? (main ? run?.status ?? null : null);
    const running = runIsRunning(status);
    const span = laneSpan(agent, roster, running, now);
    rows.push({
      agent: agent.agent,
      main,
      label: agent.label,
      status,
      cancellationOrigin: roster?.cancellationOrigin ?? null,
      running,
      start: span.start,
      end: span.end,
      durationMs: span.start !== null && span.end !== null
        ? Math.max(0, span.end - span.start)
        : null,
      steps: agent.steps,
      toolCalls: agent.toolCalls,
      marks: agent.marks,
      markTotal: markTotal(agent.marks),
      tokens: rosterTokens(roster),
      findings: findings.filter((finding) => finding.agent === agent.agent),
      order: agent.order,
    });
  }
  // The Run's own agent always has a row: the roster states it even before it writes a step.
  if (!named.has(MAIN_AGENT) && lead !== null) {
    named.add(MAIN_AGENT);
    rows.unshift(rosterOnlyRow(lead, run, findings, now, 0));
  }
  for (const row of rosterRows) {
    if (named.has(row.childSessionId)) continue;
    named.add(row.childSessionId);
    rows.push(rosterOnlyRow(row, run, findings, now, rows.length));
  }
  return rows;
}

/** One row for a roster agent the overview holds no rollup for: its live row, and no counts. */
function rosterOnlyRow(
  row: AgentChildStatus & {childSessionId: string},
  run: AgentRunStatus | null,
  findings: readonly RunFinding[],
  now: number,
  order: number,
): RunLedgerRow {
  const main = row.childSessionId === MAIN_AGENT;
  const status = row.status !== '' ? row.status : (main ? run?.status ?? null : null);
  const running = runIsRunning(status);
  const span = laneSpan(null, row, running, now);
  return {
    agent: row.childSessionId,
    main,
    label: main ? MAIN_AGENT : row.objective ?? null,
    status,
    cancellationOrigin: row.cancellationOrigin,
    running,
    start: span.start,
    end: span.end,
    durationMs: span.start !== null && span.end !== null
      ? Math.max(0, span.end - span.start)
      : null,
    tokens: rosterTokens(row),
    steps: 0,
    toolCalls: 0,
    marks: ZERO_MARKS,
    markTotal: 0,
    findings: findings.filter((finding) => finding.agent === row.childSessionId),
    order,
  };
}

/** How many marks a mark-count set holds. */
function markTotal(marks: RunMarkCounts): number {
  return marks.failed + marks.degraded + marks.repeated + marks.redirect;
}

/** One decision dot of a lane: where on the lane, and which mark it states. */
interface LaneDot {
  /** Where the dot sits across the lane, from 0 to 100. */
  readonly left: number;
  /** The finding's most severe mark; null for a delivery step, which states no mark. */
  readonly mark: DecisionMark | null;
  /** When the finding happened, in epoch milliseconds; null when the wire states no instant. */
  readonly at: number | null;
}

/** Where one agent's span and decisions sit on the Run's lane, as percentages of it. */
export interface LaneGeometry {
  readonly left: number;
  readonly width: number;
  readonly dots: readonly LaneDot[];
}

const EMPTY_GEOMETRY: LaneGeometry = {left: 0, width: 0, dots: []};

/** Clamp a percentage into the lane, so a clock that runs past the Run's range cannot paint outside. */
function percentOf(instant: number, range: TimeRange): number {
  const span = range.end - range.start;
  if (span <= 0) return 0;
  return Math.min(100, Math.max(0, ((instant - range.start) / span) * 100));
}

/** Where a ledger row's span and decision dots sit on the Run's clock. A row with no span (an agent
 *  that has not acted yet) or a Run with no clock draws no bar and no dots. */
export function laneGeometry(row: RunLedgerRow, range: TimeRange | null): LaneGeometry {
  if (range === null || row.start === null || row.end === null) return EMPTY_GEOMETRY;
  const left = percentOf(row.start, range);
  const right = percentOf(row.end, range);
  const dots: LaneDot[] = [];
  for (const finding of row.findings) {
    const instant = instantOf(finding.at);
    if (instant === null) continue;
    dots.push({left: percentOf(instant, range), mark: findingMark(finding), at: instant});
  }
  return {left, width: Math.max(0, right - left), dots};
}

/** The findings in the order the attention list reads them: severity first (failed, degraded,
 *  redirected, repeated, then a delivery step), then the moment they happened. A finding with no
 *  instant sorts last within its rank; the sort is stable, so the wire's merge order survives a tie. */
export function findingsForDisplay(overview: RunOverview | null): readonly RunFinding[] {
  if (overview === null) return [];
  return [...overview.findings].sort((one, other) => {
    const bySeverity = severityRank(one) - severityRank(other);
    if (bySeverity !== 0) return bySeverity;
    const oneAt = instantOf(one.at) ?? Number.POSITIVE_INFINITY;
    const otherAt = instantOf(other.at) ?? Number.POSITIVE_INFINITY;
    return oneAt === otherAt ? 0 : oneAt - otherAt;
  });
}

/** The exact counts the attention section states, and the findings it lists. */
interface AttentionCounts {
  readonly failed: number;
  readonly degraded: number;
  readonly redirect: number;
  readonly repeated: number;
  /** How many delivery checks the Run stated, from the Run's own kind totals. */
  readonly delivery: number;
}

export interface AttentionSummary {
  readonly counts: AttentionCounts;
  readonly findings: readonly RunFinding[];
  /** How many findings the whole Run states, which can exceed the capped list. */
  readonly total: number;
  /** How many findings the wire's cap left out of `findings`. */
  readonly truncated: number;
}

const NO_ATTENTION: AttentionSummary = {
  counts: {failed: 0, degraded: 0, redirect: 0, repeated: 0, delivery: 0},
  findings: [],
  total: 0,
  truncated: 0,
};

/** What the Run asks a reader to look at: the exact per-mark counts beside the capped findings,
 *  sorted for reading. A Run with none of either is a confident empty list, not a blank section. */
export function attentionSummary(overview: RunOverview | null): AttentionSummary {
  if (overview === null) return NO_ATTENTION;
  const marks = overview.totals.marks;
  const findings = findingsForDisplay(overview);
  const total = Math.max(overview.findingsTotal, findings.length);
  return {
    counts: {
      failed: marks.failed,
      degraded: marks.degraded,
      redirect: marks.redirect,
      repeated: marks.repeated,
      delivery: overview.totals.kinds.delivery,
    },
    findings,
    total,
    truncated: total - findings.length,
  };
}

/** One Artifact as the delivery card lists it: the file, its record, and whether it is readable. */
interface DeliveryArtifactState {
  readonly artifact: AnswerPresentation['artifacts'][number];
  readonly attestation: ArtifactAttestation | null;
  readonly available: boolean;
}

/** How many Artifacts each attestation status holds; `none` is an Artifact the server published no
 *  record for, which is an honest bucket rather than a fourth claim. */
interface DeliveryAttestationCounts {
  readonly attested: number;
  readonly unattested: number;
  readonly unknown: number;
  readonly none: number;
}

/** The delivery facts one succeeded Run states: what it cited, what it published, the publication
 *  outcome, and the one attestation record it carries. */
export interface DeliverySummary {
  readonly sources: number;
  readonly artifacts: readonly DeliveryArtifactState[];
  readonly attestation: DeliveryAttestationCounts;
  /** How many published Artifacts a reader cannot open. */
  readonly unavailable: number;
  /** The publication outcome's own status, and the issues it states. */
  readonly outcome: AnswerPresentation['artifactOutcome']['status'];
  readonly issues: AnswerPresentation['artifactOutcome']['issues'];
  /** The one record the Run's Artifacts carry, for the card's whole-record line; null when the
   *  server published none. */
  readonly record: ArtifactAttestation | null;
}

/** What a Run delivered, from the Answer it published; null when it published no Answer at all. */
export function deliveryStates(presentation: AnswerPresentation | null): DeliverySummary | null {
  if (presentation === null) return null;
  let attested = 0;
  let unattested = 0;
  let unknown = 0;
  let none = 0;
  let unavailable = 0;
  let record: ArtifactAttestation | null = null;
  // A presentation from a hand-built surface can state only the parts it has; the card draws what
  // it was given rather than taking the whole panel down for an absent list.
  const artifacts = (presentation.artifacts ?? []).map((artifact): DeliveryArtifactState => {
    const attestation = artifact.attestation;
    switch (attestation?.status) {
      case 'attested': attested += 1; break;
      case 'unattested': unattested += 1; break;
      case 'unknown': unknown += 1; break;
      default: none += 1;
    }
    const available = artifact.status === 'available';
    if (!available) unavailable += 1;
    // The record is the same on every Artifact of one Run; the first readable one names it.
    if (record === null && available && attestation !== null) record = attestation;
    return {artifact, attestation, available};
  });
  const outcome = presentation.artifactOutcome;
  return {
    sources: presentation.sources?.length ?? 0,
    artifacts,
    attestation: {attested, unattested, unknown, none},
    unavailable,
    outcome: outcome?.status ?? 'complete',
    issues: outcome?.issues ?? [],
    record,
  };
}
