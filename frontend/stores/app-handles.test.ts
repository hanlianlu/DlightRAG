// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';
import type {AnswerSubmissionAdapter} from '../api/answer-submission.ts';
import type {AnswerJudgement} from '../api/feedback.ts';
import type {AttachmentLease} from './attachment-store.ts';
import {createAppHandles, productionHandles} from './app-handles.ts';
import {judgementKey} from './kept-judgements.ts';
import type {TracesSource} from './traces-source.ts';

const intent = {
  submissionId: 'submission-1',
  conversationId: 'c1',
  query: 'Question',
  workspaces: ['default'],
  mode: null,
};

/** A submission that never answers, so it is still under way when the bag is reset. */
const silentAdapter: AnswerSubmissionAdapter = {
  submit: () => new Promise(() => undefined),
  lookup: () => new Promise(() => undefined),
};

function lease(): AttachmentLease & {discarded: number} {
  const value = {
    items: [], discarded: 0, settled: false,
    accept() { value.settled = true; },
    restore() { value.settled = true; },
    discard() { value.discarded += 1; value.settled = true; },
  };
  return value;
}

const source = (runId: string): TracesSource => ({
  runId,
  mainAgent: () => ({childSessionId: 'main', status: 'succeeded'}),
  presentation: () => null,
  page: async () => ({children: [], nextCursor: null}),
  observe: async () => { throw new Error('unused'); },
  activity: async () => ({messages: [], nextBefore: null, running: false}),
  control: async () => { throw new Error('unused'); },
  reply: async () => { throw new Error('unused'); },
}) as unknown as TracesSource;

test('a bag holds every shared store, and no two bags share one', () => {
  const one = createAppHandles();
  const other = createAppHandles();
  for (const name of [
    'conversations', 'workspaces', 'ingest', 'attachments',
    'answerEventCursors', 'submissions', 'judgements', 'traces', 'memoryNotices', 'runActivity',
  ] as const) {
    assert.ok(one[name], `${name} is missing from the bag`);
    assert.notEqual(one[name], other[name], `${name} is shared between two bags`);
  }
  assert.equal(productionHandles(), productionHandles(), 'the page has one bag');
  assert.notEqual(productionHandles(), one);
});

test('resetting a bag forgets what the reader sent, rated and read, and leaves the other bag alone', async () => {
  const page = createAppHandles();
  const other = createAppHandles();
  const key = judgementKey('run-1', 'main', 7);
  const given: AnswerJudgement = {value: 'up', reasons: [], note: null};
  const sources = {page: source('run-1'), other: source('run-1')};
  for (const [bag, held] of [[page, sources.page], [other, sources.other]] as const) {
    bag.judgements.keep(key, given);
    assert.equal(bag.traces.sourceFor('run-1', () => held).runId, 'run-1');
  }
  const held = lease();
  assert.ok(page.submissions.start(intent, held, silentAdapter));
  assert.equal(page.submissions.list().length, 1);
  const remembered = {
    memoryId: 'memory-1', kind: 'fact', body: 'Lives in Munich.', action: 'add', changeId: 'change-1', writer: 'agent',
  };
  const fetched = globalThis.fetch;
  globalThis.fetch = (async () => Response.json({changes: [remembered].map((change) => ({
    memory_id: change.memoryId, kind: change.kind, body: change.body, action: change.action,
    change_id: change.changeId, writer: change.writer,
  }))})) as typeof fetch;
  try {
    for (const bag of [page, other]) {
      bag.memoryNotices.ensure('run-1');
      await new Promise((resolve) => setTimeout(resolve, 0));
      assert.equal(bag.memoryNotices.changes('run-1').length, 1);
    }
  } finally {
    globalThis.fetch = fetched;
  }

  page.reset();

  assert.equal(page.judgements.current(key, null), null, 'a rating outlived the sign-out');
  let rebuilt = false;
  page.traces.sourceFor('run-1', () => { rebuilt = true; return sources.page; });
  assert.equal(rebuilt, true, 'a Run still held the pages its last reader read');
  assert.equal(page.submissions.list().length, 0, 'a submission outlived the sign-out');
  assert.equal(held.discarded, 1, 'the files it held were not given back');
  assert.equal(page.memoryNotices.changes('run-1').length, 0, 'a notice outlived the sign-out');
  assert.equal(other.memoryNotices.changes('run-1').length, 1, 'the reset reached the notices of another bag');

  assert.deepEqual(other.judgements.current(key, null), given, 'the reset reached another bag');
  let reused = true;
  other.traces.sourceFor('run-1', () => { reused = false; return sources.other; });
  assert.equal(reused, true, 'the reset reached the pages of another bag');
});
