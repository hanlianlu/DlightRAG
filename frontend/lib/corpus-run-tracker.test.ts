// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import type {WebCorpusRunReceipt, WebCorpusRunStatus} from '../api/corpus-runs.ts';
import {ApiError} from '../api/wire.ts';
import {CorpusRunTracker, type TrackedCorpusRun} from './corpus-run-tracker.ts';

function receipt(runId = 'run-1'): WebCorpusRunReceipt {
  return {
    runId,
    runKind: 'corpus_mutation',
    lane: 'corpus_mutation',
    status: 'queued',
    statusUrl: `/web/api/corpus-runs/${runId}`,
    eventsUrl: `/web/api/corpus-runs/${runId}/events`,
    cancelUrl: `/web/api/corpus-runs/${runId}`,
    resumeUrl: `/web/api/corpus-runs/${runId}/resume`,
    workspace: 'finance',
    fileCount: 3,
  };
}

function status(
  status: WebCorpusRunStatus['status'],
  extra: Partial<WebCorpusRunStatus> = {},
  runId = 'run-1',
): WebCorpusRunStatus {
  const {workspace: _workspace, fileCount: _fileCount, ...base} = receipt(runId);
  return {
    ...base,
    status,
    result: null,
    phase: null,
    errorKind: null,
    errorMessage: null,
    repairReason: null,
    repairRemedy: null,
    ...extra,
  };
}

const waiting = () => status('running', {
  phase: 'waiting_for_repair',
  repairReason: 'Upstream state is ambiguous.',
});

/** Let pending promise jobs and zero-delay timers run. */
async function settle(turns = 5): Promise<void> {
  for (let turn = 0; turn < turns; turn += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

function harness(answers: Array<WebCorpusRunStatus | Error>, resumes: Array<WebCorpusRunStatus | Error> = []) {
  const reads: string[] = [];
  const settled: TrackedCorpusRun[] = [];
  const lost: unknown[] = [];
  let changes = 0;
  const tracker = new CorpusRunTracker({
    onChange: () => { changes += 1; },
    onSettled: (run) => { settled.push(run); },
    onLost: (error) => { lost.push(error); },
    pollIntervalMs: 0,
    getStatus: async (url) => {
      reads.push(url);
      const answer = answers.shift() ?? status('running');
      if (answer instanceof Error) throw answer;
      return answer;
    },
    resume: async () => {
      const answer = resumes.shift() ?? status('queued');
      if (answer instanceof Error) throw answer;
      return answer;
    },
  });
  return {tracker, reads, settled, lost, changes: () => changes};
}

test('a followed Run is read at once, polled while active, and settled once with receipt facts', async () => {
  const {tracker, reads, settled, changes} = harness([status('running'), status('succeeded')]);

  tracker.follow(receipt());
  assert.equal(tracker.run?.status, 'queued');
  assert.equal(tracker.active, true);
  await settle();

  assert.deepEqual(reads, ['/web/api/corpus-runs/run-1', '/web/api/corpus-runs/run-1']);
  assert.equal(settled.length, 1);
  assert.equal(settled[0]?.status, 'succeeded');
  assert.equal(settled[0]?.workspace, 'finance');
  assert.equal(settled[0]?.fileCount, 3);
  // The settled Run stays observable; nothing is read after it.
  assert.equal(tracker.run?.status, 'succeeded');
  assert.equal(tracker.active, false);
  assert.ok(changes() >= 3);
  await settle();
  assert.equal(reads.length, 2);
});

test('a repair wait parks the Run until an explicit resume, then reading continues', async () => {
  const {tracker, reads, settled} = harness([waiting(), status('succeeded')], [status('queued')]);

  tracker.follow(receipt());
  await settle();
  assert.equal(tracker.waitingForRepair, true);
  assert.equal(tracker.run?.repairReason, 'Upstream state is ambiguous.');
  tracker.wake();
  await settle();
  assert.equal(reads.length, 1, 'a parked Run is not read again');

  const resumed = tracker.resume();
  assert.equal(tracker.resuming, true);
  assert.equal(await tracker.resume(), 'stale', 'one resume at a time');
  assert.equal(await resumed, 'accepted');
  assert.equal(tracker.resuming, false);
  assert.equal(tracker.waitingForRepair, false);
  assert.equal(settled.length, 0, 'acceptance is reported before the Run settles');
  await settle();

  assert.equal(settled[0]?.status, 'succeeded');
  assert.equal(reads.length, 2);
});

test('a failed resume leaves the Run parked and retryable', async () => {
  const {tracker} = harness([waiting()], [new ApiError(503, {detail: 'Writer unavailable'})]);
  tracker.follow(receipt());
  await settle();

  assert.equal(await tracker.resume(), 'failed');
  assert.equal(tracker.waitingForRepair, true);
  assert.equal(tracker.resuming, false);
  assert.equal(await tracker.resume(), 'accepted');
  tracker.clear();
});

test('a refused status read stops tracking; any other failure is retried', async () => {
  // The status route answers 404 for a Run it no longer shows this reader.
  const refusal = new ApiError(404);
  const {tracker, reads, lost, settled} = harness([
    new TypeError('network down'),
    new ApiError(503, {errorType: 'unavailable'}),
    new ApiError(410, {errorType: 'not_found'}),
    refusal,
  ]);

  tracker.follow(receipt());
  await settle(12);

  assert.equal(reads.length, 4);
  assert.deepEqual(lost, [refusal]);
  assert.equal(tracker.run, null);
  assert.equal(settled.length, 0);
  tracker.wake();
  await settle();
  assert.equal(reads.length, 4);
});

test('pause stops reading until wake; a newer Run and clear drop late answers', async () => {
  let release!: (value: WebCorpusRunStatus) => void;
  const reads: string[] = [];
  const settled: TrackedCorpusRun[] = [];
  const tracker = new CorpusRunTracker({
    onChange: () => {},
    onSettled: (run) => { settled.push(run); },
    onLost: () => {},
    pollIntervalMs: 0,
    getStatus: (url) => {
      reads.push(url);
      return new Promise((resolve) => { release = resolve; });
    },
  });

  tracker.follow(receipt('run-old'));
  const lateOld = release;
  tracker.follow(receipt('run-new'));
  lateOld(status('succeeded', {}, 'run-old'));
  await settle();
  assert.equal(settled.length, 0, 'an answer for the replaced Run is dropped');
  assert.equal(tracker.run?.runId, 'run-new');

  tracker.pause();
  release(status('succeeded', {}, 'run-new'));
  await settle();
  assert.equal(settled.length, 0, 'an answer that arrives after pause is dropped');

  tracker.wake();
  tracker.wake();
  assert.equal(reads.length, 3, 'wake reads once');
  tracker.clear();
  release(status('succeeded', {}, 'run-new'));
  await settle();
  assert.equal(settled.length, 0);
  assert.equal(tracker.run, null);
});
