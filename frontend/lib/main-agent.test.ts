// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import type {AnswerPresentation} from '../api/conversations.ts';
import type {ChatTurnView} from './chat-views.ts';
import {MAIN_AGENT, mainAgentStatus} from './main-agent.ts';

function turn(extra: Partial<ChatTurnView> = {}): ChatTurnView {
  return {
    id: 'turn-1', userText: 'What changed?', userAttachments: [], runId: 'run-1', state: 'succeeded',
    streamText: '', presentation: null, usage: {}, error: '', progress: '', liveStatus: '',
    sawChildren: false, cancelRequested: false, steeringMessages: [], toolRows: [], ...extra,
  };
}

const answer = {
  answerText: '  Clause 9.2 caps it.  ',
  sources: [
    {id: '1', title: 'northwind-msa.pdf', sourceUrl: null, downloadUrl: null, chunks: []},
    {id: '2', title: 'amendment-3.pdf', sourceUrl: null, downloadUrl: null, chunks: []},
  ],
} as unknown as AnswerPresentation;

test('the main agent is a status row: the question is its objective, the answer its result, the cited sources its Evidence', () => {
  const row = mainAgentStatus(turn({presentation: answer, usage: {usage_details: {total_tokens: 9400}}}));

  assert.equal(row.childSessionId, MAIN_AGENT);
  assert.equal(row.status, 'succeeded');
  assert.equal(row.objective, 'What changed?');
  assert.equal(row.summary, 'Clause 9.2 caps it.');
  assert.deepEqual(row.resultHandles, ['[1] northwind-msa.pdf', '[2] amendment-3.pdf']);
  assert.deepEqual(row.usage, {total_tokens: 9400});
});

test('a Run that is still going is running, whichever way its stream stands', () => {
  for (const state of ['pending', 'streaming', 'retryable'] as const) {
    assert.equal(mainAgentStatus(turn({state})).status, 'running');
  }
  assert.equal(mainAgentStatus(undefined).status, 'running');
});

test('a Run that failed says why in place of an answer, and one that was cancelled says nothing', () => {
  assert.equal(mainAgentStatus(turn({state: 'failed', error: 'Service error.'})).summary, 'Service error.');
  const stopped = mainAgentStatus(turn({state: 'cancelled'}));
  assert.equal(stopped.status, 'cancelled');
  assert.equal(stopped.summary, null);
});

test('a token total counts only when it is a finite number', () => {
  assert.equal(mainAgentStatus(turn()).usage, null);
  assert.equal(mainAgentStatus(turn({usage: {usage_details: {total_tokens: 'many'}}})).usage, null);
  assert.equal(mainAgentStatus(turn({usage: {usage_details: {total_tokens: Number.NaN}}})).usage, null);
});
