// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import type {ConversationCommand} from '../api/conversations.ts';
import {commandResultText, commandSuggestions, findWebCommand, goalCommandErrorText} from './web-commands.ts';

function command(overrides: Partial<ConversationCommand> = {}): ConversationCommand {
  return {
    commandId: 'command-1',
    sequence: 2,
    name: 'goal',
    input: '',
    result: 'The canonical server result.',
    resultData: null,
    tone: 'success',
    createdAt: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

function goalState(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    objective: 'finish the sealed model',
    phase: 'blocked',
    max_rounds: 7,
    rounds_started: 3,
    blocked_reason: 'the goal reached its round cap',
    ...overrides,
  };
}

function goalStatus(goal: unknown): ConversationCommand {
  return command({resultData: {kind: 'goal_status', goal}});
}

test('reads a complete command line into its command and arguments', () => {
  assert.equal(findWebCommand('/goal')?.command.name, 'goal');
  assert.equal(findWebCommand('/goal')?.args, '');
  assert.deepEqual(findWebCommand('/goal cap 7')?.args, 'cap 7');
  assert.deepEqual(findWebCommand('  /goal pause  ')?.args, 'pause');
  assert.equal(findWebCommand('/goalkeeper'), null);
  assert.equal(findWebCommand('/skill:tdd'), null);
  assert.equal(findWebCommand('goal'), null);
});

test('suggests commands only while the reader types a bare command word', () => {
  assert.deepEqual(commandSuggestions('/').map((command) => command.name), ['goal']);
  assert.deepEqual(commandSuggestions('/go').map((command) => command.name), ['goal']);
  assert.deepEqual(commandSuggestions('/goal').map((command) => command.name), ['goal']);
  assert.deepEqual(commandSuggestions('/goal '), []);
  assert.deepEqual(commandSuggestions('/goal cap'), []);
  assert.deepEqual(commandSuggestions('/x'), []);
  assert.deepEqual(commandSuggestions('hello'), []);
});

test('renders a structured goal state as one compact system line, not the canonical result', () => {
  assert.equal(
    commandResultText(goalStatus(goalState()))(),
    'Blocked goal · Round: 3/7',
  );
  assert.equal(
    commandResultText(goalStatus(goalState({phase: 'paused', blocked_reason: null})))(),
    'Paused goal · Round: 3/7 · Paused. The goal round in progress was cancelled; no next round will start. Your own running turn is not stopped.',
  );
});

test('names a goal phase the way the goal bar does', () => {
  const phase = (overrides: Record<string, unknown>): string => (
    commandResultText(goalStatus(goalState({blocked_reason: null, ...overrides})))()
      .split(' · ')[0]
  );
  assert.equal(phase({phase: 'active'}), 'Ongoing goal');
  assert.equal(phase({phase: 'paused'}), 'Paused goal');
  assert.equal(phase({phase: 'blocked'}), 'Blocked goal');
  assert.equal(phase({phase: 'complete'}), 'Complete goal');
});

test('renders the goal command’s outcomes that carry no state', () => {
  assert.equal(
    commandResultText(goalStatus(null))(),
    'No goal in this conversation. /goal <objective> creates one.',
  );
  assert.equal(
    commandResultText(command({resultData: {kind: 'goal_cleared'}}))(),
    'Goal cleared.',
  );
  assert.equal(
    commandResultText(command({resultData: {kind: 'goal_invalid', reason: 'cap'}}))(),
    'Add a positive number: /goal cap 5',
  );
  assert.equal(
    commandResultText(command({resultData: {kind: 'goal_invalid', reason: 'objective'}}))(),
    'Add an objective: /goal <objective>',
  );
  assert.equal(
    commandResultText(command({resultData: {
      kind: 'goal_error',
      code: 'goal_invalid_transition',
      message: 'pause needs an active goal; this one is complete',
    }}))(),
    'The goal cannot make that change now. pause needs an active goal; this one is complete',
    'the known code localizes and the concrete server sentence is still stated',
  );
  assert.equal(
    commandResultText(command({resultData: {
      kind: 'goal_error',
      code: 'goal_invalid_transition',
    }}))(),
    'The goal cannot make that change now.',
    'a known code without a sentence still localizes',
  );
  assert.equal(
    commandResultText(command({resultData: {
      kind: 'goal_error',
      code: 'a_later_domain_reason',
      message: 'the goal is sealed for this release',
    }}))(),
    'the goal is sealed for this release',
    'an unknown code keeps the server sentence',
  );
  assert.equal(
    commandResultText(command({resultData: {kind: 'goal_error', code: 'goal_conflict'}}))(),
    'The canonical server result.',
    'a payload without a sentence falls back to the server\'s canonical result',
  );
});

test('localizes a goal refusal by its stable code, and knows no unknown code', () => {
  assert.deepEqual(goalCommandErrorText('goal_not_found'), {
    source: 'This conversation carries no goal.',
    id: 'goalCommand.error.goalNotFound',
  });
  assert.deepEqual(goalCommandErrorText('goal_invalid_transition'), {
    source: 'The goal cannot make that change now.',
    id: 'goalCommand.error.goalInvalidTransition',
  });
  assert.equal(
    goalCommandErrorText('a_reason_from_a_later_domain'),
    null,
    'an unknown code keeps the server sentence in the caller',
  );
  assert.equal(goalCommandErrorText(null), null);
});

test('falls back to the canonical result for result_data this client cannot read', () => {
  const fallback = command().result;
  const unreadable: ConversationCommand[] = [
    command(),
    command({resultData: {kind: 'goal_fact_from_a_later_server'}}),
    command({resultData: {kind: 'goal_invalid', reason: 'a_later_reason'}}),
    command({resultData: {kind: 'goal_error'}}),
    goalStatus({objective: 'only some of the facts'}),
    goalStatus(goalState({phase: 'a_later_phase'})),
    goalStatus(goalState({max_rounds: 'seven'})),
    goalStatus(goalState({blocked_reason: 4})),
    // A command that is not the goal command does not read goal-shaped data.
    command({name: 'another-command', resultData: {kind: 'goal_cleared'}}),
  ];
  for (const unreadableCommand of unreadable) {
    assert.equal(commandResultText(unreadableCommand)(), fallback);
  }
});
