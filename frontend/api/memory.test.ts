// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {listRunMemoryChanges, parseMemoryOperationEvent} from './memory.ts';

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

test('a Run\'s memory changes are read from its own route and keep the change that undoes each', async () => {
  const original = globalThis.fetch;
  const asked: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    asked.push(String(input));
    return Response.json({
      changes: [
        {
          memory_id: 'memory-1',
          kind: 'preference',
          body: 'Tables first.',
          action: 'add',
          change_id: 'change-1',
          writer: 'agent',
        },
        {
          memory_id: 'memory-2',
          kind: 'fact',
          body: 'Lives in Munich.',
          action: 'update',
          change_id: 'change-2',
          writer: 'conversation',
        },
      ],
    });
  }) as typeof fetch;
  try {
    const changes = await listRunMemoryChanges('run/1');

    assert.deepEqual(asked, ['/web/api/memory/runs/run%2F1/changes']);
    assert.deepEqual(changes, [
      {
        memoryId: 'memory-1',
        kind: 'preference',
        body: 'Tables first.',
        action: 'add',
        changeId: 'change-1',
        writer: 'agent',
      },
      {
        memoryId: 'memory-2',
        kind: 'fact',
        body: 'Lives in Munich.',
        action: 'update',
        changeId: 'change-2',
        writer: 'conversation',
      },
    ]);
  } finally {
    globalThis.fetch = original;
  }
});

test('a change this client cannot interpret fails the read instead of being guessed', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = (async () => Response.json({
    changes: [{memory_id: 'm', kind: 'preference', body: 'x', action: 'rewrite', change_id: 'c', writer: 'agent'}],
  })) as typeof fetch;
  try {
    await assert.rejects(listRunMemoryChanges('run-1'));
  } finally {
    globalThis.fetch = original;
  }
});
