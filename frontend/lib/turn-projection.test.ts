// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';

import type {ChatTurnView} from './chat-views.ts';
import {ANSWER_PHASE_LABELS, answerPhaseLabel, applyAnswerEvent} from './turn-projection.ts';

function turn(overrides: Partial<ChatTurnView> = {}): ChatTurnView {
  return {
    id: 't1',
    userText: 'q',
    userAttachments: [],
    runId: 'r1',
    state: 'pending',
    streamText: '',
    presentation: null,
    usage: {},
    evidence: {},
    error: '',
    progress: '',
    liveStatus: '',
    sawChildren: false,
    cancelRequested: false,
    steeringMessages: [],
    toolRows: [],
    ...overrides,
  };
}

test('phase labels map every server phase and reject unknowns', () => {
  assert.equal(answerPhaseLabel('searching'), 'Searching knowledge base...');
  assert.equal(answerPhaseLabel('bogus'), null);
  assert.ok(Object.keys(ANSWER_PHASE_LABELS).length >= 5);
});

test('tokens accumulate within a batch and settle the stream state', () => {
  let view = applyAnswerEvent(turn(), {kind: 'token', text: 'He'}, 1000);
  view = applyAnswerEvent(view, {kind: 'token', text: 'llo'}, 1000);
  assert.equal(view.streamText, 'Hello');
  assert.equal(view.state, 'streaming');
  assert.equal(view.error, '');
});

test('reset clears the stream back to pending', () => {
  const streamed = turn({state: 'streaming', streamText: 'abc', progress: 'x', error: ''});
  const view = applyAnswerEvent(streamed, {kind: 'reset'}, 1000);
  assert.deepEqual(
    {state: view.state, streamText: view.streamText, progress: view.progress},
    {state: 'pending', streamText: '', progress: ''},
  );
});

test('progress applies known phases and ignores unknown ones', () => {
  const base = turn();
  const known = applyAnswerEvent(base, {kind: 'progress', payload: {phase: 'searching'}}, 1000);
  assert.ok(known.progress.includes('Searching'));
  const ignored = applyAnswerEvent(known, {kind: 'progress', payload: {phase: 'bogus'}}, 1000);
  assert.equal(ignored, known);
});

test('memory events never change the view', () => {
  const base = turn();
  assert.equal(
    applyAnswerEvent(base, {
      kind: 'memory',
      operation: {
        operation: 'remember', outcome: 'changed', changeId: 'change-1',
        intentId: null, body: '', live: true,
      },
    }, 1000),
    base,
  );
});

test('tool events drive the trace and sawChildren', () => {
  let view = applyAnswerEvent(turn(), {
    kind: 'tool',
    eventType: 'tool_start',
    payload: {tool_name: 'spawn_agent', call_id: 'c1', tool_label: 'Child agent'},
  }, 4000);
  assert.equal(view.sawChildren, true);
  assert.equal(view.toolRows[0].startedAt, 4000);
  assert.equal(view.toolRows[0].label, 'Child agent');
  assert.equal(view.progress, 'Child agent');
  view = applyAnswerEvent(view, {
    kind: 'tool',
    eventType: 'tool_progress',
    payload: {tool_name: 'spawn_agent', call_id: 'c1', object_label: 'review'},
  }, 1000);
  assert.ok(view.progress.includes('review'));
  view = applyAnswerEvent(turn(), {
    kind: 'tool',
    eventType: 'tool_start',
    payload: {tool_name: 'wait_subagent', call_id: 'c2'},
  }, 1000);
  assert.equal(view.sawChildren, true);
});

test('errors fail the turn', () => {
  const view = applyAnswerEvent(turn(), {
    kind: 'error',
    payload: {kind: 'run_abandoned', message: 'Run could not be recovered.'},
  }, 1000);
  assert.equal(view.state, 'failed');
  assert.equal(view.error, 'Run could not be recovered.');
});

test('done settles succeeded, cancelled, and malformed payloads', () => {
  const succeeded = applyAnswerEvent(turn(), {
    kind: 'done',
    payload: {
      status: 'succeeded',
      presentation: {answer_text: 'answer', sources: []},
      usage: {tokens: 1},
      evidence: {sources: 2},
    },
  }, 1000);
  assert.equal(succeeded.state, 'succeeded');
  assert.equal(succeeded.streamText, 'answer');
  assert.equal(succeeded.evidence.sources, 2);

  const cancelled = applyAnswerEvent(turn(), {kind: 'done', payload: {status: 'cancelled'}}, 1000);
  assert.equal(cancelled.state, 'cancelled');

  const malformed = applyAnswerEvent(turn(), {kind: 'done', payload: {status: 'running'}}, 1000);
  assert.equal(malformed.state, 'failed');
});
