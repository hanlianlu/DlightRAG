// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Run panel's view models: the arithmetic the sections draw, tested without a page.
 *
 *  The cases are the ones the wire can actually state: a Run with no steps, an agent the overview
 *  names but the roster page has not, a finding for a Session past the roster page, an absent token
 *  record, a capped findings list, and clocks that do not all agree. */

import assert from 'node:assert/strict';
import test from 'node:test';
import type {
  AgentChildStatus,
  AgentRunStatus,
  AnswerPresentation,
  RunFinding,
  RunOverview,
} from '../api/conversations.ts';
import {
  attentionSummary,
  barSegments,
  deliveryStates,
  findingsForDisplay,
  laneGeometry,
  laneRange,
  ledgerRows,
  runDurationMs,
  runIsRunning,
  runTokens,
  severityRank,
  KIND_ORDER,
} from './run-overview.ts';

const RUN = 'run-1';

/** One child's roster row: every field a row states, with the two a test reads set per case. */
function child(
  id: string,
  extra: Partial<AgentChildStatus & {childSessionId: string}> = {},
): AgentChildStatus & {childSessionId: string} {
  return {
    childSessionId: id,
    status: 'succeeded',
    objective: `objective ${id}`,
    modelRole: 'query',
    usage: null,
    operationId: null,
    operationSequence: null,
    operationStatus: null,
    cancellationOrigin: null,
    summary: null,
    resultHandles: [],
    startedAt: null,
    finishedAt: null,
    pendingQuestions: 0,
    resultSequence: null,
    feedback: null,
    ...extra,
  };
}

/** One finding as the overview states it; a test sets the mark and moment it reads. */
function finding(extra: Partial<RunFinding> = {}): RunFinding {
  return {
    agent: 'main',
    sequence: 2,
    kind: 'tool',
    at: '2026-10-05T12:00:15Z',
    name: 'read',
    subject: 'northwind-msa.pdf',
    outcome: 'failed',
    marks: ['failed'],
    excerpt: 'The file could not be read.',
    ...extra,
  };
}

const FINDINGS: readonly RunFinding[] = [
  finding({agent: 'child-1', sequence: 7, at: '2026-10-05T12:00:40Z', marks: ['repeated']}),
  finding({agent: 'main', sequence: 2, at: '2026-10-05T12:00:15Z', marks: ['failed', 'repeated']}),
  finding({
    agent: 'child-1', sequence: 5, kind: 'delivery', at: '2026-10-05T12:00:30Z',
    name: null, subject: null, outcome: null, marks: [], excerpt: 'The record names no verifier.',
  }),
];

/** One overview as the read states it, with the whole Run's facts a section reads. */
function overview(extra: Partial<RunOverview> = {}): RunOverview {
  return {
    runId: RUN,
    totals: {
      steps: 6,
      kinds: {thinking: 1, tool: 2, say: 1, answer: 1, instruction: 0, delivery: 1},
      decisions: 2,
      marks: {failed: 1, degraded: 1, repeated: 1, redirect: 0},
      toolTimeMs: 1200,
    },
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 3, toolCalls: 1,
        marks: {failed: 1, degraded: 0, repeated: 1, redirect: 0},
        toolTimeMs: 800,
        firstAt: '2026-10-05T12:00:00Z', lastAt: '2026-10-05T12:00:30Z',
      },
      {
        agent: 'child-1', label: 'Read the filing', order: 1, steps: 3, toolCalls: 1,
        marks: {failed: 0, degraded: 1, repeated: 0, redirect: 0},
        toolTimeMs: 400,
        firstAt: '2026-10-05T12:00:10Z', lastAt: '2026-10-05T12:00:50Z',
      },
    ],
    findings: [...FINDINGS],
    findingsTotal: 3,
    usage: {
      usageDetails: {total_tokens: 1000},
      childUsageDetails: {total_tokens: 500},
      inclusiveUsageDetails: {total_tokens: 1500},
    },
    ...extra,
  };
}

const run = (extra: Partial<AgentRunStatus> = {}): AgentRunStatus => ({
  status: 'succeeded',
  startedAt: '2026-10-05T11:59:00Z',
  finishedAt: '2026-10-05T12:01:00Z',
  ...extra,
});

const NOW = Date.parse('2026-10-05T12:02:00Z');

test('the kind bar holds the six kinds in product order, zero counts left out', () => {
  const segments = barSegments(overview(), 'kind');
  assert.deepEqual(segments.map((segment) => segment.key), ['thinking', 'tool', 'say', 'answer', 'delivery']);
  assert.deepEqual(segments.map((segment) => segment.count), [1, 2, 1, 1, 1]);
  const whole = segments.reduce((total, segment) => total + segment.share, 0);
  assert.ok(Math.abs(whole - 1) < 1e-9, 'the visible segments are the whole bar');
  assert.equal(segments[1]!.share, 1 / 3);
  assert.equal(KIND_ORDER.length, 6);
});

test('the agent bar holds main first, then roster order, and leaves a zero-step agent out', () => {
  const segments = barSegments(overview({
    agents: [
      { ...overview().agents[1]!, order: 1 },
      { ...overview().agents[0]!, order: 0 },
      {
        agent: 'child-2', label: 'Still starting', order: 2, steps: 0, toolCalls: 0,
        marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, toolTimeMs: 0,
        firstAt: null, lastAt: null,
      },
    ],
  }), 'agent');
  assert.deepEqual(segments.map((segment) => segment.key), ['main', 'child-1']);
  assert.deepEqual(segments.map((segment) => segment.share), [0.5, 0.5]);
});

test('a Run with no steps draws no bar, in either dimension', () => {
  const empty = overview({
    totals: {
      steps: 0,
      kinds: {thinking: 0, tool: 0, say: 0, answer: 0, instruction: 0, delivery: 0},
      decisions: 0, marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, toolTimeMs: 0,
    },
    agents: [],
    findings: [],
    findingsTotal: 0,
    usage: null,
  });
  assert.deepEqual(barSegments(empty, 'kind'), []);
  assert.deepEqual(barSegments(empty, 'agent'), []);
  assert.deepEqual(barSegments(null, 'kind'), []);
});

test('the token triple is what the record states, and a missing detail is null, never zero', () => {
  assert.deepEqual(runTokens(overview()), {total: 1500, main: 1000, children: 500});
  assert.deepEqual(runTokens(null), {total: null, main: null, children: null});
  assert.deepEqual(runTokens(overview({usage: null})), {total: null, main: null, children: null});
  // A Fast Run stores only its root counters: the other two are unknown, not the root's value.
  assert.deepEqual(
    runTokens(overview({usage: {usageDetails: {total_tokens: 900}} as unknown as RunOverview['usage']})),
    {total: null, main: 900, children: null},
  );
  assert.deepEqual(
    runTokens(overview({
      usage: {usageDetails: {prompt_tokens: 200, completion_tokens: 100}} as unknown as RunOverview['usage'],
    })),
    {total: null, main: null, children: null},
  );
});

test('the Run clock prefers the roster row, otherwise every agent\'s steps, and runs to now', () => {
  assert.deepEqual(
    laneRange(overview(), run(), NOW),
    {start: Date.parse('2026-10-05T11:59:00Z'), end: Date.parse('2026-10-05T12:01:00Z')},
  );
  // No roster row: the agents' first and last instants cover it.
  assert.deepEqual(
    laneRange(overview(), null, NOW),
    {start: Date.parse('2026-10-05T12:00:00Z'), end: Date.parse('2026-10-05T12:00:50Z')},
  );
  // A Run still going ends at the render's clock, and a row that started later is not its start.
  assert.deepEqual(
    laneRange(overview(), run({status: 'running', finishedAt: null}), NOW),
    {start: Date.parse('2026-10-05T11:59:00Z'), end: NOW},
  );
  assert.equal(runDurationMs(overview(), run(), NOW), 120_000);
  assert.equal(runDurationMs(overview(), run({status: 'running', finishedAt: null}), NOW), 180_000);
  const empty = overview({
    totals: {
      steps: 0,
      kinds: {thinking: 0, tool: 0, say: 0, answer: 0, instruction: 0, delivery: 0},
      decisions: 0, marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, toolTimeMs: 0,
    },
    agents: [], findings: [], findingsTotal: 0, usage: null,
  });
  assert.equal(laneRange(empty, null, NOW), null);
  assert.equal(runDurationMs(empty, null, NOW), null);
  assert.equal(runDurationMs(overview({agents: [], findings: [], findingsTotal: 0}), null, NOW), null);
});

test('a finished clock is not trusted to run backwards', () => {
  const range = laneRange(overview({ agents: [] }), run({
    startedAt: '2026-10-05T12:10:00Z',
    finishedAt: '2026-10-05T12:00:00Z',
  }), NOW);
  assert.deepEqual(range, {
    start: Date.parse('2026-10-05T12:10:00Z'),
    end: Date.parse('2026-10-05T12:10:00Z'),
  });
  assert.equal(runDurationMs(overview({ agents: [] }), run({
    startedAt: '2026-10-05T12:10:00Z',
    finishedAt: '2026-10-05T12:00:00Z',
  }), NOW), 0);
});

test('running is every state but the three settled ones', () => {
  assert.equal(runIsRunning('running'), true);
  assert.equal(runIsRunning('queued'), true);
  assert.equal(runIsRunning('succeeded'), false);
  assert.equal(runIsRunning('failed'), false);
  assert.equal(runIsRunning('cancelled'), false);
  assert.equal(runIsRunning(null), false);
});

test('an agent\'s span is placed on the Run\'s clock as percentages of it', () => {
  const rows = ledgerRows(overview(), [], child('main'), run(), NOW);
  const main = rows.find((row) => row.agent === 'main')!;
  const range = laneRange(overview(), run(), NOW)!;
  const geometry = laneGeometry(main, range);
  // The Run's clock is 11:59:00 to 12:01:00; main acts 12:00:00 to 12:00:30.
  assert.equal(geometry.left, 60 / 120 * 100);
  assert.equal(geometry.width, 30 / 120 * 100);
  // One finding, and the most severe of the marks it states speaks for it.
  assert.deepEqual(geometry.dots.map((dot) => dot.mark), ['failed']);
  assert.equal(geometry.dots[0]!.left, 75 / 120 * 100);
  // No clock, or an agent that has not acted: no geometry rather than one from zero.
  assert.deepEqual(laneGeometry(main, null), {left: 0, width: 0, dots: []});
  const starting = ledgerRows(overview({
    agents: [{
      agent: 'child-2', label: 'Still starting', order: 1, steps: 0, toolCalls: 0,
      marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, toolTimeMs: 0,
      firstAt: null, lastAt: null,
    }],
    findings: [],
  }), [], child('main'), run(), NOW);
  const newborn = starting.find((row) => row.agent === 'child-2')!;
  assert.deepEqual(laneGeometry(newborn, range), {left: 0, width: 0, dots: []});
});

test('the ledger fuses rollups with the roster: status, clock, tokens, and the running edge', () => {
  const rows = ledgerRows(
    overview(),
    [child('child-1', {
      status: 'running', usage: {total_tokens: 500},
      startedAt: '2026-10-05T12:00:10Z', finishedAt: null,
    })],
    child('main', {status: 'running', usage: {total_tokens: 1000}}),
    run({status: 'running', finishedAt: null}),
    NOW,
  );
  assert.deepEqual(rows.map((row) => row.agent), ['main', 'child-1']);
  const main = rows[0]!;
  assert.equal(main.main, true);
  assert.equal(main.status, 'running');
  assert.equal(main.tokens, 1000);
  assert.equal(main.steps, 3);
  assert.equal(main.toolCalls, 1);
  assert.deepEqual(main.marks, {failed: 1, degraded: 0, repeated: 1, redirect: 0});
  assert.equal(main.markTotal, 2);
  // It is still going: its span starts at its first step and ends at the render's clock.
  assert.equal(main.durationMs, NOW - Date.parse('2026-10-05T12:00:00Z'));
  const girl = rows[1]!;
  assert.equal(girl.label, 'Read the filing');
  assert.equal(girl.status, 'running');
  // The running agent's lane reaches the render's clock, not its last step.
  assert.equal(girl.end, NOW);
  assert.deepEqual(girl.findings.map((found) => found.sequence), [7, 5]);
});

test('the ledger adds a roster row before the child writes a step, and after the overview did not name it', () => {
  const rows = ledgerRows(
    overview(),
    [child('child-1'), child('child-9', {status: 'running', objective: 'Late child'})],
    child('main'),
    run(),
    NOW,
  );
  assert.deepEqual(rows.map((row) => row.agent), ['main', 'child-1', 'child-9']);
  const late = rows[2]!;
  assert.equal(late.steps, 0);
  assert.equal(late.toolCalls, 0);
  assert.equal(late.status, 'running');
  assert.equal(late.label, 'Late child');
  assert.equal(late.durationMs, null, 'a row with no clock states no duration');
});

test('a Run with no overview still draws its main agent row from the lead, and no others', () => {
  const rows = ledgerRows(null, [], child('main', {status: 'running'}), run(), NOW);
  assert.deepEqual(rows.map((row) => row.agent), ['main']);
  assert.equal(rows[0]!.steps, 0);
  assert.equal(rows[0]!.status, 'running');
  assert.deepEqual(rows[0]!.marks, {failed: 0, degraded: 0, repeated: 0, redirect: 0});
  assert.deepEqual(ledgerRows(null, [], null, null, NOW), []);
});

test('findings sort failed, degraded, redirected, repeated, then delivery, oldest first inside a rank', () => {
  const ordered = findingsForDisplay(overview({
    findings: [
      finding({agent: 'child-1', sequence: 7, at: '2026-10-05T12:00:40Z', marks: ['repeated']}),
      finding({agent: 'child-1', sequence: 5, kind: 'delivery', at: '2026-10-05T12:00:30Z', marks: []}),
      finding({agent: 'main', sequence: 2, at: '2026-10-05T12:00:15Z', marks: ['failed']}),
      finding({agent: 'main', sequence: 3, at: '2026-10-05T12:00:05Z', marks: ['failed']}),
      finding({agent: 'main', sequence: 4, at: '2026-10-05T12:00:20Z', marks: ['degraded']}),
      finding({agent: 'main', sequence: 5, at: '2026-10-05T12:00:25Z', marks: ['redirect']}),
    ],
  }));
  assert.deepEqual(ordered.map((found) => found.sequence), [3, 2, 4, 5, 7, 5]);
  assert.deepEqual(ordered.map(severityRank), [0, 0, 1, 2, 3, 4]);
  assert.deepEqual(
    ordered.map((found) => found.kind),
    ['tool', 'tool', 'tool', 'tool', 'tool', 'delivery'],
  );
  // No instant sorts last inside its rank, and the wire order survives an equal instant.
  const undated = findingsForDisplay(overview({
    findings: [
      finding({sequence: 9, at: null, marks: ['failed']}),
      finding({sequence: 8, at: null, marks: ['failed']}),
    ],
  }));
  assert.deepEqual(undated.map((found) => found.sequence), [9, 8]);
  assert.deepEqual(findingsForDisplay(null), []);
});

test('the attention counts are the Run\'s exact mark counts, and a capped list states its total', () => {
  const full = attentionSummary(overview());
  assert.deepEqual(full.counts, {failed: 1, degraded: 1, repeated: 1, redirect: 0, delivery: 1});
  assert.equal(full.findings.length, 3);
  assert.equal(full.total, 3);
  assert.equal(full.truncated, 0);

  const capped = attentionSummary(overview({
    findings: FINDINGS.slice(0, 2),
    findingsTotal: 40,
  }));
  assert.equal(capped.findings.length, 2);
  assert.equal(capped.total, 40);
  assert.equal(capped.truncated, 38);

  const empty = attentionSummary(overview({
    totals: {
      steps: 0,
      kinds: {thinking: 0, tool: 0, say: 0, answer: 0, instruction: 0, delivery: 0},
      decisions: 0, marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0}, toolTimeMs: 0,
    },
    findings: [], findingsTotal: 0,
  }));
  assert.deepEqual(empty.findings, []);
  assert.deepEqual(empty.counts, {failed: 0, degraded: 0, repeated: 0, redirect: 0, delivery: 0});
  assert.deepEqual(attentionSummary(null), empty);
});

test('a finding for a Session past the roster page is still listed, with no row invented for it', () => {
  const overviewOf = overview({
    findings: [finding({agent: 'child-9', sequence: 4, marks: ['failed']})],
    findingsTotal: 1,
  });
  assert.deepEqual(findingsForDisplay(overviewOf).map((found) => found.agent), ['child-9']);
  const rows = ledgerRows(overviewOf, [child('child-1')], child('main'), run(), NOW);
  assert.deepEqual(rows.map((row) => row.agent), ['main', 'child-1']);
  assert.deepEqual(attentionSummary(overviewOf).counts.failed, 1);
});

test('the delivery card states what the Answer published, with its record and buckets', () => {
  const presentation = {
    answerText: 'Answer.',
    parts: [],
    sources: [
      {id: '1', title: 'a.pdf', sourceUrl: null, downloadUrl: null, chunks: []},
      {id: '2', title: 'b.pdf', sourceUrl: null, downloadUrl: null, chunks: []},
    ],
    linkCards: [],
    evidenceImages: [],
    artifacts: [
      artifact('file-1', 'attested', 'available'),
      artifact('file-2', 'unattested', 'available'),
      artifact('file-3', 'unknown', 'available'),
      artifact('file-4', null, 'unavailable', {description: 'The bytes were never stored.'}),
    ],
    artifactOutcome: {
      status: 'partial',
      issues: [{kind: 'missing_bytes', description: 'One file could not be published.', resourceId: 'file-4'}],
    },
  } as unknown as AnswerPresentation;
  const delivery = deliveryStates(presentation)!;
  assert.equal(delivery.sources, 2);
  assert.equal(delivery.artifacts.length, 4);
  assert.deepEqual(delivery.attestation, {attested: 1, unattested: 1, unknown: 1, none: 1});
  assert.equal(delivery.unavailable, 1);
  assert.equal(delivery.outcome, 'partial');
  assert.deepEqual(delivery.issues.map((issue) => issue.resourceId), ['file-4']);
  assert.equal(delivery.record?.status, 'attested', 'the record is the first readable Artifact\'s');
  assert.equal(deliveryStates(null), null);
});

test('an Answer with no Artifacts is a delivery of sources only', () => {
  const delivery = deliveryStates({
    answerText: 'Answer.', parts: [], sources: [], linkCards: [], evidenceImages: [],
    artifacts: [],
    artifactOutcome: {status: 'complete', issues: []},
  } as unknown as AnswerPresentation)!;
  assert.deepEqual(delivery.attestation, {attested: 0, unattested: 0, unknown: 0, none: 0});
  assert.equal(delivery.record, null);
  assert.equal(delivery.outcome, 'complete');
});

/** One published Artifact, with the attestation record its card states. */
function artifact(
  resourceId: string,
  status: 'attested' | 'unattested' | 'unknown' | null,
  availability: 'available' | 'unavailable',
  issue: {description: string} | null = null,
): Record<string, unknown> {
  return {
    resource_id: resourceId,
    media_type: 'text/markdown',
    label: resourceId,
    filename: `${resourceId}.md`,
    byte_size: 12,
    digest: 'abc',
    presentation: 'markdown',
    status: availability,
    uri: `artifact://${resourceId}`,
    width: null,
    height: null,
    data_url: null,
    download_url: null,
    presentation_url: null,
    issue: issue === null ? null : {kind: 'missing_bytes', ...issue, resource_id: resourceId},
    attestation: status === null ? null : {
      status,
      skill: status === 'attested' ? 'research-report' : null,
      verifier: status === 'attested' ? 'verify.py' : null,
      reason: status === 'unattested' ? 'no_skill_loaded' : null,
      basis: null,
      required_reads_missing: null,
    },
  };
}
