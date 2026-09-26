// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {parseMemoryOperationEvent} from './memory.ts';

test('a streamed memory settlement keeps its change id, so Undo can be offered', () => {
  // The exact field names the browser event projection sends.
  const event = parseMemoryOperationEvent({
    operation: 'remember',
    outcome: 'changed',
    change_id: 'change-1',
    intent_id: 'intent-1',
    body: 'Use concise answers',
    memory_ids: ['memory-1'],
    session_id: 'session-1',
    live: true,
  });

  assert.deepEqual(event, {
    operation: 'remember',
    outcome: 'changed',
    changeId: 'change-1',
    intentId: 'intent-1',
    body: 'Use concise answers',
    live: true,
  });
});

test('a memory settlement this client cannot interpret is dropped, not guessed', () => {
  assert.equal(parseMemoryOperationEvent({operation: 'remember'}), null);
  assert.equal(parseMemoryOperationEvent({operation: 'rewrite', outcome: 'changed'}), null);
  assert.equal(parseMemoryOperationEvent('remember'), null);
});
