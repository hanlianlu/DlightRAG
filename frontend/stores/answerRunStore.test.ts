// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';
import {AnswerRunStore} from './answerRunStore.ts';

function ids(): () => string {
  let next = 0;
  return () => `id-${++next}`;
}

test('an identical unfinished payload keeps its submission id', () => {
  const store = new AnswerRunStore(ids());

  assert.equal(store.getOrCreateSubmissionId('c1', 'fp'), 'id-1');
  assert.equal(store.getOrCreateSubmissionId('c1', 'fp'), 'id-1');
});

test('a changed payload becomes a new submission', () => {
  const store = new AnswerRunStore(ids());

  store.getOrCreateSubmissionId('c1', 'fp-a');

  assert.equal(store.getOrCreateSubmissionId('c1', 'fp-b'), 'id-2');
});

test('each conversation carries its own submission', () => {
  const store = new AnswerRunStore(ids());

  assert.equal(store.getOrCreateSubmissionId('c1', 'fp'), 'id-1');
  assert.equal(store.getOrCreateSubmissionId('c2', 'fp'), 'id-2');
});

test('the accepted run is bound to the conversation that submitted it', () => {
  const store = new AnswerRunStore(ids());
  store.getOrCreateSubmissionId('c1', 'fp');

  store.attachRun('c1', 'run-1');

  assert.equal(store.runId('c1'), 'run-1');
  assert.equal(store.lastSequence('c1', 'run-1'), 0);
});

test('the resume cursor only ever moves forward', () => {
  const store = new AnswerRunStore(ids());
  store.getOrCreateSubmissionId('c1', 'fp');
  store.attachRun('c1', 'run-1');

  store.recordSequence('c1', 'run-1', 4);
  store.recordSequence('c1', 'run-1', 2);

  assert.equal(store.lastSequence('c1', 'run-1'), 4);
});

test('a cursor from another run is never resumed from', () => {
  const store = new AnswerRunStore(ids());
  store.getOrCreateSubmissionId('c1', 'fp');
  store.attachRun('c1', 'run-1');
  store.recordSequence('c1', 'run-1', 7);

  assert.equal(store.lastSequence('c1', 'run-2'), 0);
  store.recordSequence('c1', 'run-2', 9);
  assert.equal(store.lastSequence('c1', 'run-1'), 7);
});

test('rebinding a different run restarts its cursor', () => {
  const store = new AnswerRunStore(ids());
  store.getOrCreateSubmissionId('c1', 'fp');
  store.attachRun('c1', 'run-1');
  store.recordSequence('c1', 'run-1', 5);

  store.attachRun('c1', 'run-2');

  assert.equal(store.lastSequence('c1', 'run-2'), 0);
});

test('a run discovered from history is tracked without a submission', () => {
  const store = new AnswerRunStore(ids());

  store.trackRun('c1', 'run-9');

  assert.equal(store.runId('c1'), 'run-9');
  assert.equal(store.lastSequence('c1', 'run-9'), 0);
});

test('tracking the run already followed keeps its cursor', () => {
  const store = new AnswerRunStore(ids());
  store.trackRun('c1', 'run-9');
  store.recordSequence('c1', 'run-9', 3);

  store.trackRun('c1', 'run-9');

  assert.equal(store.lastSequence('c1', 'run-9'), 3);
});

test('clearing a conversation forgets its run', () => {
  const store = new AnswerRunStore(ids());
  store.trackRun('c1', 'run-9');

  store.clear('c1');

  assert.equal(store.runId('c1'), null);
});
