// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {RUN_AGENT} from '../lib/agent-id.ts';
import type {ActivityMessage} from '../api/conversations.ts';
import {joinActivity, projectActivity, type ActivityStep} from './activity.ts';

const OBJECTIVE = 'Check the key evidence: the termination clause and the penalty cap';

/** The messages of one page, numbered from `from` in the order given. */
function numbered(from: number, messages: Omit<ActivityMessage, 'sequence'>[]): ActivityMessage[] {
  return messages.map((message, index) => ({...message, sequence: from + index}));
}

/** A message written at a moment, as the wire states one; the clock the timed fixtures use. */
function at(moment: string, message: Omit<ActivityMessage, 'sequence' | 'at'>): Omit<ActivityMessage, 'sequence'> {
  return {...message, at: moment};
}

const T0 = Date.parse('2026-10-05T12:00:00Z');

/** A moment this many seconds after the fixtures' clock, in the form the wire carries. */
function ago(seconds: number): string {
  return new Date(T0 + seconds * 1000).toISOString();
}

function user(content: string): Omit<ActivityMessage, 'sequence'> {
  return {role: 'user', content, agent: RUN_AGENT, agentLabel: null, snippet: '', toolCalls: [], toolCallId: '', name: '', isError: false, at: null, reasoning: ''};
}

function assistant(
  content: string,
  ...calls: [id: string, name: string, subject?: string][]
): Omit<ActivityMessage, 'sequence'> {
  return {
    role: 'assistant', content, agent: RUN_AGENT, agentLabel: null, snippet: '',
    toolCalls: calls.map(([id, name, subject = '']) => ({id, name, subject})),
    toolCallId: '', name: '', isError: false, at: null, reasoning: '',
  };
}

function result(id: string, name: string, content: string, isError = false): Omit<ActivityMessage, 'sequence'> {
  return {role: 'tool', content, agent: RUN_AGENT, agentLabel: null, snippet: '', toolCalls: [], toolCallId: id, name, isError, at: null, reasoning: ''};
}

const running = {objective: OBJECTIVE, running: true, controls: []};
const settled = {...running, running: false};

/** The times the tool steps of a projection state, in order. */
function toolTimes(steps: readonly ActivityStep[]): (number | null)[] {
  return steps.flatMap((step) => (step.kind === 'tool' ? [step.durationMs] : []));
}

test('each call of a batch pairs with the result that carries its id, whatever order they come back in', () => {
  const steps = projectActivity(numbered(1, [
    user(OBJECTIVE),
    assistant('I will look in two places.', ['a', 'search_knowledge_base', 'MSA §9.2 penalty cap'], ['b', 'read']),
    result('b', 'read', 'northwind-msa.pdf, pages 8 to 10', true),
    result('a', 'search_knowledge_base', '12 results\nTop: MSA §9.2'),
  ]), settled);

  assert.deepEqual(steps, [
    {kind: 'say', key: 'say:2', text: 'I will look in two places.'},
    {
      kind: 'tool', key: 'tool:a', verb: 'Searching the knowledge base', state: 'done',
      subject: 'MSA §9.2 penalty cap', excerpt: '12 results', full: '12 results\nTop: MSA §9.2', durationMs: null,
    },
    {
      kind: 'tool', key: 'tool:b', verb: 'Reading a document', state: 'failed',
      subject: '', excerpt: 'northwind-msa.pdf, pages 8 to 10', full: 'northwind-msa.pdf, pages 8 to 10', durationMs: null,
    },
  ]);
});

test('a tool step carries the bounded subject the call reported, and none when the wire has none', () => {
  const steps = projectActivity(numbered(1, [
    user(OBJECTIVE),
    assistant('Looking.', ['a', 'search_web', 'volvo 2026 tariff exposure'], ['b', 'read']),
  ]), running);

  assert.deepEqual(
    steps.flatMap((step) => (step.kind === 'tool' ? [step.subject] : [])),
    ['volvo 2026 tariff exposure', ''],
  );
});

test('a tool step states how long the call took, from the entry that made it to the result that settled it', () => {
  const steps = projectActivity(numbered(1, [
    at(ago(0), assistant('Looking.', ['a', 'read'])),
    at(ago(2.5), result('a', 'read', 'ok')),
  ]), {...settled, now: T0 + 60_000});

  assert.deepEqual(toolTimes(steps), [2500]);
});

test('a call in flight is timed to the clock the render holds; one with no result and no clock states no time', () => {
  const messages = numbered(1, [at(ago(0), assistant('Looking.', ['a', 'read']))]);
  const times = (context: Parameters<typeof projectActivity>[1]) =>
    toolTimes(projectActivity(messages, context));

  assert.deepEqual(times({...running, now: T0 + 8000}), [8000]);
  // A render that holds no clock cannot time a call, and neither can a settled agent with no result.
  assert.deepEqual(times(running), [null]);
  assert.deepEqual(times({...settled, now: T0 + 8000}), [null]);
});

test('a step the wire gave no clock for states no time, however long it ran', () => {
  const steps = projectActivity(numbered(1, [
    at(ago(0), assistant('Looking.', ['a', 'read'])),
    result('a', 'read', 'ok'),
  ]), {...settled, now: T0 + 60_000});

  assert.deepEqual(toolTimes(steps), [null]);
});

test('calls of one batch all run at once; once the agent has settled, a call without a result was lost', () => {
  const transcript = numbered(1, [
    assistant('', ['a', 'read'], ['b', 'read'], ['c', 'search_web']),
    result('a', 'read', 'one'),
  ]);

  assert.deepEqual(
    projectActivity(transcript, running).map((step) => step.kind === 'tool' && step.state),
    ['done', 'running', 'running'],
  );
  assert.deepEqual(
    projectActivity(transcript, settled).map((step) => step.kind === 'tool' && [step.state, step.excerpt, step.full]),
    [['done', 'one', 'one'], ['failed', '', ''], ['failed', '', '']],
  );
});

test('the objective is the agent\'s task and any other user message is an instruction it was given', () => {
  const steer = 'User steer: only contracts signed after 2020';
  // The first operation starts the transcript, a continuation or steer follows it.
  assert.deepEqual(projectActivity(numbered(1, [user(OBJECTIVE), assistant('On it.'), user(steer)]), running), [
    {kind: 'say', key: 'say:2', text: 'On it.'},
    {kind: 'instruction', key: 'told:3', sender: '', text: steer},
  ]);
  // A page may start after the objective, and the steer is still shown.
  assert.deepEqual(projectActivity(numbered(7, [assistant('Working.'), user(steer)]), running), [
    {kind: 'say', key: 'say:7', text: 'Working.'},
    {kind: 'instruction', key: 'told:8', sender: '', text: steer},
  ]);
  // An agent with no objective on the wire cannot tell its task from an instruction, and drops nothing.
  assert.deepEqual(
    projectActivity(numbered(1, [user(OBJECTIVE)]), {...running, objective: ''}).map((step) => step.kind),
    ['instruction'],
  );
});

test('a steer a control record claims shows as its sender wrote it, and any other user message unlabeled', () => {
  const steer = 'User steer: only contracts signed after 2020';
  const controls = [{origin: 'user', content: 'only contracts signed after 2020'}];

  assert.deepEqual(projectActivity(numbered(1, [user(steer), user('Parent steer: other')]), {...running, controls}), [
    {kind: 'instruction', key: 'told:1', sender: 'user', text: 'only contracts signed after 2020'},
    {kind: 'instruction', key: 'told:2', sender: '', text: 'Parent steer: other'},
  ]);
});

test('a result stands on one line of at most 120 characters, and the whole of it stays available', () => {
  const [step] = projectActivity(numbered(1, [
    assistant('', ['a', 'search_web']),
    result('a', 'search_web', `\n\n   Found   ${'passage '.repeat(40)}\nsecond line`),
  ]), settled);

  assert.ok(step?.kind === 'tool');
  assert.equal(step.excerpt.length, 120);
  assert.ok(step.excerpt.startsWith('Found passage passage'));
  assert.ok(step.excerpt.endsWith('…'));
  assert.ok(step.full.endsWith('second line'));

  // A cut counts characters, so it never leaves half of an astral one behind.
  const [astral] = projectActivity(numbered(1, [
    assistant('', ['a', 'search_web']),
    result('a', 'search_web', '𠮷'.repeat(130)),
  ]), settled);
  assert.ok(astral?.kind === 'tool');
  assert.equal(Array.from(astral.excerpt).length, 120);
  assert.equal(astral.excerpt, `${'𠮷'.repeat(119)}…`);
});

test('a tool step speaks the main trace\'s verb, and an unknown tool is named by its prettified name', () => {
  const steps = projectActivity(numbered(1, [
    assistant('', ['a', 'ask_parent'], ['b', '']),
    result('a', 'ask_parent', 'Yes, include them.'),
    result('b', 'read', 'ok'),
  ]), settled);

  // The result's own tool name stands in when the call arrived unnamed.
  assert.deepEqual(steps.map((step) => step.kind === 'tool' && step.verb), ['Ask Parent', 'Reading a document']);
});

test('a step keeps its key when older messages are read in front of it', () => {
  const page = numbered(5, [assistant('Latest.', ['a', 'read']), result('a', 'read', 'ok')]);
  const keys = (messages: ActivityMessage[]) => projectActivity(messages, settled).map((step) => step.key);

  assert.deepEqual(keys(page), ['say:5', 'tool:a']);
  assert.deepEqual(keys([...numbered(1, [user('Earlier.')]), ...page]).slice(-2), ['say:5', 'tool:a']);
});

test('a newest page that reaches back to what is shown replaces its own range and keeps the older messages', () => {
  const shown = numbered(1, [user('a'), user('b'), user('c'), user('d')]);
  const newest = numbered(3, [user('c'), user('d'), user('e')]);

  assert.deepEqual(joinActivity(shown, newest)?.map((message) => message.content), ['a', 'b', 'c', 'd', 'e']);
});

test('a newest page that starts after what is shown cannot join it, since sequences have holes', () => {
  const shown = numbered(1, [user('a'), user('b')]);

  assert.equal(joinActivity(shown, numbered(3, [user('x')])), null);
});

test('nothing shown yet takes the page whole, and an empty page leaves what is shown', () => {
  const page = numbered(1, [user('a')]);

  assert.deepEqual(joinActivity([], page), page);
  assert.deepEqual(joinActivity(page, []), page);
});

test('a pane narrowed to a search shows every message the words were found in', () => {
  const steps = projectActivity(numbered(1, [
    user('the branch question'),
    result('a', 'search_knowledge_base', 'a branch of the tree'),
    assistant('another branch', ['b', 'read']),
  ]), {...settled, terms: ['branch']});
  assert.deepEqual(
    steps.map((step) => step.kind),
    ['say', 'say', 'say'],
    'a word found in a tool result is a match like any other, and is shown like one',
  );
  assert.deepEqual(
    steps.map((step) => (step.kind === 'say' ? step.text : '')),
    ['the branch question', 'a branch of the tree', 'another branch'],
  );
});

test('ends the list on what the agent concluded, and says it once', () => {
  const messages = numbered(1, [assistant('the last word')]);
  const steps = projectActivity(messages, {objective: '', running: false, controls: [], brief: 'the last word'});
  assert.deepEqual(steps.map((step) => step.kind), ['brief']);
  assert.equal(steps[0]?.kind === 'brief' ? steps[0].text : '', 'the last word');
});

test('leaves a page without a conclusion alone', () => {
  const messages = numbered(1, [assistant('work')]);
  const steps = projectActivity(messages, {objective: '', running: false, controls: []});
  assert.deepEqual(steps.map((step) => step.kind), ['say']);
});

test('a delivery warning is a neutral step, not an instruction from the reader', () => {
  const steps = projectActivity(numbered(1, [
    user(OBJECTIVE),
    {
      ...user('not read: references/gate-audit.md (before delivery)'),
      role: 'delivery',
    },
  ]), settled);

  assert.deepEqual(steps, [
    {
      kind: 'delivery',
      key: 'delivery:2',
      text: 'not read: references/gate-audit.md (before delivery)',
    },
  ]);
});

test('a say step carries the reasoning the message showed, trimmed, and leaves an empty one off', () => {
  const withReasoning = (message: Omit<ActivityMessage, 'sequence'>, reasoning: string) =>
    ({...message, reasoning});
  const steps = projectActivity(numbered(1, [
    user(OBJECTIVE),
    withReasoning(assistant('I will look.'), '\n  Weighing the amendment.  '),
    assistant('No thought on record.'),
    withReasoning(assistant('Blank.'), '   '),
  ]), running);

  assert.deepEqual(steps, [
    {kind: 'say', key: 'say:2', text: 'I will look.', reasoning: 'Weighing the amendment.'},
    {kind: 'say', key: 'say:3', text: 'No thought on record.'},
    {kind: 'say', key: 'say:4', text: 'Blank.'},
  ]);
});

test('an assistant turn that only chose a tool states the reasoning that chose it, before the call', () => {
  const withReasoning = (message: Omit<ActivityMessage, 'sequence'>, reasoning: string) =>
    ({...message, reasoning});
  const steps = projectActivity(numbered(1, [
    user(OBJECTIVE),
    // The tool-selection turn: reasoning and a call, but no words of its own.
    withReasoning(assistant('', ['a', 'search_web']), '  The filing is what to search.  '),
    result('a', 'search_web', '12 results'),
    // A turn with words keeps its reasoning on the say step, where it explains those words.
    withReasoning(assistant('Found it.', ['b', 'read']), 'Reading the excerpt now.'),
    result('b', 'read', 'ok'),
  ]), settled);

  assert.deepEqual(steps, [
    {
      kind: 'thinking', key: 'thinking:2', text: 'The filing is what to search.',
    },
    {
      kind: 'tool', key: 'tool:a', verb: 'Searching the web', state: 'done',
      subject: '', excerpt: '12 results', full: '12 results', durationMs: null,
    },
    {
      kind: 'say', key: 'say:4', text: 'Found it.', reasoning: 'Reading the excerpt now.',
    },
    {
      kind: 'tool', key: 'tool:b', verb: 'Reading a document', state: 'done',
      subject: '', excerpt: 'ok', full: 'ok', durationMs: null,
    },
  ]);
  // Whitespace is not words: a turn whose content is blank is still a turn that only chose.
  const blank = projectActivity(numbered(1, [
    withReasoning(assistant('   ', ['a', 'read']), 'Reading the file.'),
  ]), settled);
  assert.deepEqual(blank.map((step) => step.kind), ['thinking', 'tool']);
});

test('the conclusion keeps the reasoning that produced the words it folds', () => {
  const withReasoning = (message: Omit<ActivityMessage, 'sequence'>, reasoning: string) =>
    ({...message, reasoning});
  const final = projectActivity(numbered(1, [
    withReasoning(assistant('the last word'), 'Weighing the evidence one last time.'),
  ]), {objective: '', running: false, controls: [], brief: ' the last word '});

  assert.deepEqual(final, [{
    kind: 'brief', key: 'brief', text: 'the last word', reasoning: 'Weighing the evidence one last time.',
  }]);

  // A conclusion the page does not hold as a say step has no reasoning to carry.
  const elsewhere = projectActivity(numbered(1, [assistant('work')]), {
    objective: '', running: false, controls: [], brief: 'the last word',
  });
  assert.deepEqual(elsewhere.map((step) => step.kind), ['say', 'brief']);
});
