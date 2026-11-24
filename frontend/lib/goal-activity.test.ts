// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {goalWasWritten} from './goal-activity.ts';
import type {AnswerRunEvent} from './run-controller.ts';

/** One settled Tool frame, as the answer stream projects it. */
function settled(toolName: string, outcome = 'succeeded'): AnswerRunEvent {
  return {
    kind: 'tool',
    eventType: 'tool_end',
    payload: {tool_name: toolName, call_id: 'c1', outcome},
  };
}

test('a settled Goal write is the signal to read the Goal again', () => {
  assert.equal(goalWasWritten([settled('create_goal')]), true);
  assert.equal(goalWasWritten([settled('update_goal')]), true);
  // A write among other Tool traffic still counts.
  assert.equal(goalWasWritten([settled('search_web'), settled('create_goal')]), true);
});

test('a Goal read, a failed write, and any other Tool ask for no read', () => {
  assert.equal(goalWasWritten([settled('get_goal')]), false);
  assert.equal(goalWasWritten([settled('create_goal', 'failed')]), false);
  assert.equal(goalWasWritten([settled('search_web')]), false);
  assert.equal(goalWasWritten([]), false);
});

test('only a settled Tool frame counts, not a start, a progress or prose', () => {
  assert.equal(
    goalWasWritten([{kind: 'tool', eventType: 'tool_start', payload: {tool_name: 'create_goal'}}]),
    false,
  );
  assert.equal(
    goalWasWritten([
      {kind: 'tool', eventType: 'tool_progress', payload: {tool_name: 'create_goal'}},
    ]),
    false,
  );
  assert.equal(goalWasWritten([{kind: 'token', text: 'create_goal'}]), false);
  assert.equal(goalWasWritten([{kind: 'tool', eventType: 'tool_end', payload: null}]), false);
});
