// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import type {ActivityMessage} from '../api/conversations.ts';
import {joinActivity, projectActivity} from './activity.ts';

const OBJECTIVE = 'Check the key evidence: the termination clause and the penalty cap';

/** The messages of one page, numbered from `from` in the order given. */
function numbered(from: number, messages: Omit<ActivityMessage, 'sequence'>[]): ActivityMessage[] {
  return messages.map((message, index) => ({...message, sequence: from + index}));
}

function user(content: string): Omit<ActivityMessage, 'sequence'> {
  return {role: 'user', content, toolCalls: [], toolCallId: '', name: '', isError: false};
}

function assistant(content: string, ...calls: [id: string, name: string][]): Omit<ActivityMessage, 'sequence'> {
  return {
    role: 'assistant', content, toolCalls: calls.map(([id, name]) => ({id, name})),
    toolCallId: '', name: '', isError: false,
  };
}

function result(id: string, name: string, content: string, isError = false): Omit<ActivityMessage, 'sequence'> {
  return {role: 'tool', content, toolCalls: [], toolCallId: id, name, isError};
}

const running = {objective: OBJECTIVE, running: true, controls: []};
const settled = {...running, running: false};

test('each call of a batch pairs with the result that carries its id, whatever order they come back in', () => {
  const steps = projectActivity(numbered(1, [
    user(OBJECTIVE),
    assistant('I will look in two places.', ['a', 'search_knowledge_base'], ['b', 'read']),
    result('b', 'read', 'northwind-msa.pdf, pages 8 to 10', true),
    result('a', 'search_knowledge_base', '12 results\nTop: MSA §9.2'),
  ]), settled);

  assert.deepEqual(steps, [
    {kind: 'say', key: 'say:2', text: 'I will look in two places.'},
    {
      kind: 'tool', key: 'tool:a', verb: 'Searching the knowledge base', state: 'done',
      excerpt: '12 results', full: '12 results\nTop: MSA §9.2',
    },
    {
      kind: 'tool', key: 'tool:b', verb: 'Reading a document', state: 'failed',
      excerpt: 'northwind-msa.pdf, pages 8 to 10', full: 'northwind-msa.pdf, pages 8 to 10',
    },
  ]);
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
