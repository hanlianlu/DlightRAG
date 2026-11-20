// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';

import {MAX_PROCESS_ENTRIES, type ChatTurnView, type ProcessEntry} from './chat-views.ts';
import {
  ANSWER_PHASE_LABELS,
  answerPhaseLabel,
  applyAnswerEvent,
} from './turn-projection.ts';

function turn(overrides: Partial<ChatTurnView> = {}): ChatTurnView {
  return {
    id: 't1',
    userText: 'q',
    userAttachments: [],
    runId: 'r1',
    state: 'pending',
    streamText: '',
    processEntries: [],
    presentation: null,
    usage: {},
    error: null,
    activity: null,
    cancelRequested: false,
    steeringMessages: [],
    toolRows: [],
    answerSequence: null,
    feedback: null,
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
  assert.equal(view.error, null);
});

test('reset clears the stream back to pending', () => {
  const streamed = turn({
    state: 'streaming',
    streamText: 'abc',
    processEntries: [],
    activity: {kind: 'tool'},
    error: null,
  });
  const view = applyAnswerEvent(streamed, {kind: 'reset'}, 1000);
  assert.deepEqual(
    {state: view.state, streamText: view.streamText, activity: view.activity},
    {state: 'pending', streamText: '', activity: null},
  );
});

test('progress applies known phases and ignores unknown ones', () => {
  const base = turn();
  const known = applyAnswerEvent(base, {kind: 'progress', payload: {phase: 'searching'}}, 1000);
  // The view keeps the source and the id; the component applies the words, so the
  // line follows a language change.
  assert.deepEqual(known.activity, {kind: 'phase', phase: 'searching'});
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

test('tool events drive the trace', () => {
  let view = applyAnswerEvent(turn(), {
    kind: 'tool',
    eventType: 'tool_start',
    payload: {tool_name: 'spawn_agent', call_id: 'c1', tool_label: 'Child agent'},
  }, 4000);
  assert.equal(view.toolRows[0].startedAt, 4000);
  assert.equal(view.toolRows[0].label, 'Child agent');
  assert.deepEqual(view.activity, {kind: 'tool'});
  view = applyAnswerEvent(view, {
    kind: 'tool',
    eventType: 'tool_progress',
    payload: {tool_name: 'spawn_agent', call_id: 'c1', object_label: 'review'},
  }, 1000);
  assert.equal(view.toolRows[0].object, 'review');
});

test('errors fail the turn', () => {
  const view = applyAnswerEvent(turn(), {
    kind: 'error',
    payload: {kind: 'run_abandoned', message: 'Run could not be recovered.'},
  }, 1000);
  assert.equal(view.state, 'failed');
  // Server words carry no catalog id; the view holds them verbatim.
  assert.deepEqual(view.error, {source: 'Run could not be recovered.', id: null});
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
  assert.deepEqual(violated.error, {
    source: 'Service error. Please try again.',
    id: 'errors.service',
  });
  assert.equal(violated.presentation, null);
});

test('reasoning grows the one open process fold and starts a pending turn streaming', () => {
  let view = applyAnswerEvent(turn(), {kind: 'reasoning', text: 'Let me '}, 1000);
  view = applyAnswerEvent(view, {kind: 'reasoning', text: 'check the filing.'}, 1000);
  assert.deepEqual(view.processEntries, [
    {kind: 'thinking', text: 'Let me check the filing.', open: true},
  ]);
  assert.equal(view.state, 'streaming');

  // The answer's own words do not disturb the fold the process trace holds.
  view = applyAnswerEvent(view, {kind: 'token', text: 'The fee '}, 1000);
  view = applyAnswerEvent(view, {kind: 'token', text: 'was fixed.'}, 1000);
  assert.deepEqual(view.processEntries, [
    {kind: 'thinking', text: 'Let me check the filing.', open: true},
  ]);
  assert.equal(view.streamText, 'The fee was fixed.');
});

test('a tool call shuts the open fold and joins its row to the one merged trace', () => {
  const search = {
    kind: 'tool', eventType: 'tool_start', payload: {tool_name: 'search_web', call_id: 'c1'},
  } as const;
  let view = applyAnswerEvent(turn(), {kind: 'reasoning', text: 'Search first.'}, 1000);
  view = applyAnswerEvent(view, search, 1000);
  assert.deepEqual(view.processEntries.map((entry) => entry.kind), ['thinking', 'tool']);
  assert.deepEqual(view.processEntries[0], {kind: 'thinking', text: 'Search first.', open: false});
  // The Tool entry holds the row object the trace's own state machine holds.
  assert.equal(
    view.processEntries[1]?.kind === 'tool' ? view.processEntries[1].row : null,
    view.toolRows[0],
  );

  // A progress event updates that row in place: the trace never holds the same call twice.
  view = applyAnswerEvent(view, {
    kind: 'tool', eventType: 'tool_progress',
    payload: {tool_name: 'search_web', call_id: 'c1', object_label: 'MSA 9.2'},
  }, 1000);
  assert.equal(view.processEntries.length, 2);
  assert.equal(
    view.processEntries[1]?.kind === 'tool' ? view.processEntries[1].row.object : '',
    'MSA 9.2',
  );

  // Reasoning after a call opens a new fold after the Tool entries; the call that follows shuts it.
  view = applyAnswerEvent(view, {kind: 'reasoning', text: 'Read it next.'}, 1000);
  view = applyAnswerEvent(view, {
    kind: 'tool', eventType: 'tool_start', payload: {tool_name: 'read', call_id: 'c2'},
  }, 1000);
  assert.deepEqual(view.processEntries.map((entry) => entry.kind), ['thinking', 'tool', 'thinking', 'tool']);
  assert.deepEqual(
    view.processEntries.flatMap((entry) => (entry.kind === 'thinking' ? [entry.text, entry.open] : [])),
    ['Search first.', false, 'Read it next.', false],
  );
});

test('the folds and the rows keep one growing history, so the block can scroll back through it', () => {
  const reason = (text: string) => ({kind: 'reasoning', text} as const);
  const call = (id: string) => ({
    kind: 'tool', eventType: 'tool_start', payload: {tool_name: 'read', call_id: id},
  } as const);
  let view = turn();
  // think a, call 1, think b, call 2, think c: five entries, which is what the block shows at rest.
  for (const event of [reason('a'), call('c1'), reason('b'), call('c2'), reason('c')]) {
    view = applyAnswerEvent(view, event, 1000);
  }
  assert.equal(view.processEntries.length, 5);

  // A sixth and seventh entry join the history rather than rolling the first steps out: the block
  // scrolls them, and the newest five are what its viewport holds.
  view = applyAnswerEvent(view, call('c3'), 1000);
  view = applyAnswerEvent(view, reason('d'), 1000);
  assert.deepEqual(
    view.processEntries.map((entry) => entry.kind),
    ['thinking', 'tool', 'thinking', 'tool', 'thinking', 'tool', 'thinking'],
  );
  assert.deepEqual(
    view.processEntries.flatMap((entry) => (entry.kind === 'thinking' ? [entry.text] : [])),
    ['a', 'b', 'c', 'd'],
  );
});

test('a run longer than the history bound keeps its newest steps and its whole count', () => {
  const call = (id: string) => ({
    kind: 'tool', eventType: 'tool_start', payload: {tool_name: 'read', call_id: id},
  } as const);
  let view = turn();
  for (let index = 0; index < MAX_PROCESS_ENTRIES + 10; index += 1) {
    view = applyAnswerEvent(view, call(`c${index}`), 1000);
  }

  assert.equal(view.processEntries.length, MAX_PROCESS_ENTRIES);
  const first = view.processEntries[0];
  assert.equal(
    first?.kind === 'tool' ? first.row.callId : null,
    'c10',
    'the oldest ten steps are the ones the bound drops',
  );
  const newest = view.processEntries.at(-1);
  assert.equal(
    newest?.kind === 'tool' ? newest.row.ordinal : null,
    MAX_PROCESS_ENTRIES + 10,
    'the count outlives the entries the bound drops',
  );
});

test('a terminal frame clears the process trace: the settled reading surface is silent', () => {
  const streaming = (): ChatTurnView => turn({
    state: 'streaming',
    streamText: 'The answer so far',
    processEntries: [{kind: 'thinking', text: 'An open thought', open: true}],
  });
  const cleared = (view: ChatTurnView): readonly ProcessEntry[] => view.processEntries;

  assert.deepEqual(cleared(applyAnswerEvent(streaming(), {kind: 'reset'}, 1000)), []);
  assert.deepEqual(cleared(applyAnswerEvent(streaming(), {
    kind: 'error',
    payload: {kind: 'run_abandoned', message: 'Run could not be recovered.'},
  }, 1000)), []);
  assert.deepEqual(cleared(applyAnswerEvent(streaming(), {
    kind: 'done',
    payload: {status: 'cancelled'},
  }, 1000)), []);
  assert.deepEqual(cleared(applyAnswerEvent(streaming(), {kind: 'done', payload: doneFrame}, 1000)), []);
});
