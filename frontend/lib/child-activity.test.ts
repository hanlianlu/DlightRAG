// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import type {ChildTranscriptMessage} from '../api/conversations.ts';
import {projectActivity} from './child-activity.ts';

const OBJECTIVE = 'Check the key evidence: the termination clause and the penalty cap';

function user(content: string): ChildTranscriptMessage {
  return {role: 'user', content, toolCalls: [], toolCallId: '', name: '', isError: false};
}

function assistant(content: string, ...calls: [id: string, name: string][]): ChildTranscriptMessage {
  return {
    role: 'assistant', content, toolCalls: calls.map(([id, name]) => ({id, name})),
    toolCallId: '', name: '', isError: false,
  };
}

function result(id: string, name: string, content: string, isError = false): ChildTranscriptMessage {
  return {role: 'tool', content, toolCalls: [], toolCallId: id, name, isError};
}

const running = {objective: OBJECTIVE, childRunning: true};
const settled = {objective: OBJECTIVE, childRunning: false};

test('each call of a batch pairs with the result that carries its id, whatever order they come back in', () => {
  const steps = projectActivity([
    user(OBJECTIVE),
    assistant('I will look in two places.', ['a', 'search_knowledge_base'], ['b', 'read']),
    result('b', 'read', 'northwind-msa.pdf, pages 8 to 10', true),
    result('a', 'search_knowledge_base', '12 results\nTop: MSA §9.2'),
  ], settled);

  assert.deepEqual(steps, [
    {kind: 'say', key: 'say:1', text: 'I will look in two places.'},
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

test('calls of one batch all run at once; once the child has settled, a call without a result was lost', () => {
  const transcript = [
    assistant('', ['a', 'read'], ['b', 'read'], ['c', 'search_web']),
    result('a', 'read', 'one'),
  ];

  assert.deepEqual(
    projectActivity(transcript, running).map((step) => step.kind === 'tool' && step.state),
    ['done', 'running', 'running'],
  );
  assert.deepEqual(
    projectActivity(transcript, settled).map((step) => step.kind === 'tool' && [step.state, step.excerpt, step.full]),
    [['done', 'one', 'one'], ['failed', '', ''], ['failed', '', '']],
  );
});

test('the objective is the child\'s task and any other user message is an instruction it was given', () => {
  const steer = 'User steer: only contracts signed after 2020';
  // The first operation starts the transcript, a continuation or steer follows it.
  assert.deepEqual(projectActivity([user(OBJECTIVE), assistant('On it.'), user(steer)], running), [
    {kind: 'say', key: 'say:1', text: 'On it.'},
    {kind: 'instruction', key: 'told:2', text: steer},
  ]);
  // The tail is the latest messages only, so the objective may be gone and the steer is still shown.
  assert.deepEqual(projectActivity([assistant('Working.'), user(steer)], running), [
    {kind: 'say', key: 'say:0', text: 'Working.'},
    {kind: 'instruction', key: 'told:1', text: steer},
  ]);
  // A child with no objective on the wire cannot tell its task from an instruction, and drops nothing.
  assert.deepEqual(
    projectActivity([user(OBJECTIVE)], {objective: '', childRunning: true}).map((step) => step.kind),
    ['instruction'],
  );
});

test('a result stands on one line of at most 120 characters, and the whole of it stays available', () => {
  const [step] = projectActivity([
    assistant('', ['a', 'search_web']),
    result('a', 'search_web', `\n\n   Found   ${'passage '.repeat(40)}\nsecond line`),
  ], settled);

  assert.ok(step?.kind === 'tool');
  assert.equal(step.excerpt.length, 120);
  assert.ok(step.excerpt.startsWith('Found passage passage'));
  assert.ok(step.excerpt.endsWith('…'));
  assert.ok(step.full.endsWith('second line'));

  // A cut counts characters, so it never leaves half of an astral one behind.
  const [astral] = projectActivity([
    assistant('', ['a', 'search_web']),
    result('a', 'search_web', '𠮷'.repeat(130)),
  ], settled);
  assert.ok(astral?.kind === 'tool');
  assert.equal(Array.from(astral.excerpt).length, 120);
  assert.equal(astral.excerpt, `${'𠮷'.repeat(119)}…`);
});

test('a tool step speaks the main trace\'s verb, and an unknown tool is named by its prettified name', () => {
  const steps = projectActivity([
    assistant('', ['a', 'ask_parent'], ['b', '']),
    result('a', 'ask_parent', 'Yes, include them.'),
    result('b', 'read', 'ok'),
  ], settled);

  // The result's own tool name stands in when the call arrived unnamed.
  assert.deepEqual(steps.map((step) => step.kind === 'tool' && step.verb), ['Ask Parent', 'Reading a document']);
});
