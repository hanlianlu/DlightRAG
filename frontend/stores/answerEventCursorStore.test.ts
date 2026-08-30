// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';
import {AnswerEventCursorStore} from './answerEventCursorStore.ts';

test('the resume cursor only moves forward for its tracked run', () => {
  const store = new AnswerEventCursorStore();
  store.trackRun('c1', 'run-1');
  store.recordSequence('c1', 'run-1', 7);
  store.recordSequence('c1', 'run-1', 3);
  store.recordSequence('c1', 'run-other', 10);

  assert.equal(store.lastSequence('c1', 'run-1'), 7);
  assert.equal(store.lastSequence('c1', 'run-other'), 0);
});

test('tracking a different run resets the conversation cursor', () => {
  const store = new AnswerEventCursorStore();
  store.trackRun('c1', 'run-1');
  store.recordSequence('c1', 'run-1', 7);

  store.trackRun('c1', 'run-2');

  assert.equal(store.runId('c1'), 'run-2');
  assert.equal(store.lastSequence('c1', 'run-2'), 0);
});

test('clear forgets one conversation cursor', () => {
  const store = new AnswerEventCursorStore();
  store.trackRun('c1', 'run-1');
  store.clear('c1');
  assert.equal(store.runId('c1'), null);
});
