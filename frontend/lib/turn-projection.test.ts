// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';

import type {ChatTurnView} from './chat-views.ts';
import {
  ANSWER_PHASE_LABELS,
  answerPhaseLabel,
  applyAnswerEvent,
  isChildToolEvent,
} from './turn-projection.ts';

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

// A succeeded done frame as the server sends it: the history's presentation wire,
// every unset field an explicit null.
const doneFrame = {
  status: 'succeeded',
  presentation: {
    video_links: [],
    answer_text: 'Revenue increased [1].',
    parts: [{
      type: 'markdown',
      text: 'Revenue increased [1].',
      html: '<p>Revenue increased <cite class="citation-badge" data-ref="1" role="button" tabindex="0" title="Report" aria-label="Source 1">1</cite>.</p>\n',
      artifact: null,
      evidence_image: null,
      inline: false,
      slot: null,
    }],
    sources: [{
      id: '1',
      title: 'Report',
      source_url: null,
      download_url: '/web/api/files/raw/report?workspace=default',
      chunks: [{
        chunk_idx: 1,
        page_number: 3,
        content_html: '<p>Revenue grew eleven percent.</p>\n',
        image_url: null,
        thumbnail_url: null,
      }],
    }],
    evidence_images: [],
    link_cards: [],
    artifacts: [],
    artifact_outcome: {status: 'complete', issues: []},
  },
  usage: {usage_details: {total_tokens: 42}},
};

test('done settles succeeded, cancelled, and malformed payloads', () => {
  const succeeded = applyAnswerEvent(turn(), {kind: 'done', payload: doneFrame}, 1000);
  assert.equal(succeeded.state, 'succeeded');
  assert.equal(succeeded.streamText, 'Revenue increased [1].');
  assert.deepEqual(succeeded.presentation, {
    answerText: 'Revenue increased [1].',
    parts: [{
      type: 'markdown',
      text: 'Revenue increased [1].',
      html: doneFrame.presentation.parts[0].html,
      artifact: null,
      evidenceImage: null,
      inline: false,
    }],
    sources: [{
      id: '1',
      title: 'Report',
      sourceUrl: null,
      downloadUrl: '/web/api/files/raw/report?workspace=default',
      chunks: [{
        chunkIdx: 1,
        pageNumber: 3,
        contentHtml: '<p>Revenue grew eleven percent.</p>\n',
        imageUrl: null,
        thumbnailUrl: null,
      }],
    }],
    linkCards: [],
    evidenceImages: [],
    artifacts: [],
    artifactOutcome: {status: 'complete', issues: []},
  });
  assert.deepEqual(succeeded.usage, {usage_details: {total_tokens: 42}});

  const cancelled = applyAnswerEvent(turn(), {kind: 'done', payload: {status: 'cancelled'}}, 1000);
  assert.equal(cancelled.state, 'cancelled');

  const malformed = applyAnswerEvent(turn(), {kind: 'done', payload: {status: 'running'}}, 1000);
  assert.equal(malformed.state, 'failed');

  // A presentation that breaks the contract is a service error, never a half-read answer.
  const {artifact_outcome: _outcome, ...incomplete} = doneFrame.presentation;
  const violated = applyAnswerEvent(turn(), {
    kind: 'done',
    payload: {...doneFrame, presentation: incomplete},
  }, 1000);
  assert.equal(violated.state, 'failed');
  assert.equal(violated.error, 'Service error. Please try again.');
  assert.equal(violated.presentation, null);
});

test('only child-agent tool events count as child activity', () => {
  const tool = (name: unknown) => ({kind: 'tool' as const, eventType: 'tool_progress' as const, payload: {tool_name: name}});
  assert.equal(isChildToolEvent(tool('wait_subagent')), true);
  assert.equal(isChildToolEvent(tool('spawn_agent')), true);
  assert.equal(isChildToolEvent(tool('search_corpus')), false);
  assert.equal(isChildToolEvent(tool(42)), false);
  assert.equal(isChildToolEvent({kind: 'token', text: 'spawn_agent'}), false);
  assert.equal(isChildToolEvent({kind: 'tool', eventType: 'tool_end', payload: null}), false);
});
