// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {
  continueAnswerRun,
  getAnswerRunChildren,
  listConversations,
  steerAnswerRun,
} from './conversations.ts';

const originalFetch = globalThis.fetch;
const originalDocument = globalThis.document;

test.beforeEach(() => {
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: {cookie: ''},
  });
});

test.afterEach(() => {
  globalThis.fetch = originalFetch;
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: originalDocument,
  });
});

test('conversation pages use the bounded route and encode an opaque continuation', async () => {
  const seen: string[] = [];
  globalThis.fetch = async (input) => {
    seen.push(String(input));
    return new Response(JSON.stringify({items: [], next_cursor: null}), {
      headers: {'Content-Type': 'application/json'},
    });
  };

  assert.deepEqual(await listConversations(), {items: [], next_cursor: null});
  await listConversations('facts/signature + padding');

  assert.deepEqual(seen, [
    '/web/api/conversations',
    '/web/api/conversations?cursor=facts%2Fsignature%20%2B%20padding',
  ]);
});

test('continuation posts one submission id to the selected branch operation', async () => {
  const seen: Request[] = [];
  globalThis.fetch = async (input, init) => {
    seen.push(new Request(new URL(String(input), 'http://localhost'), init));
    return new Response(JSON.stringify({
      run_id: 'child-run',
      status: 'queued',
      status_url: '/status',
      events_url: '/events',
      cancel_url: '/cancel',
      conversation: {
        conversation_id: 'conversation-2',
        title: null,
        created_at: '2026-08-23T00:00:00Z',
        updated_at: '2026-08-23T00:00:00Z',
      },
    }), {status: 202, headers: {'Content-Type': 'application/json'}});
  };

  const result = await continueAnswerRun('parent/run', 'fork', 'branch', 'submission-1');

  const request = seen[0]!;
  assert.equal(request.url, 'http://localhost/web/api/answer/parent%2Frun/fork');
  assert.equal(request.method, 'POST');
  assert.deepEqual(await request.json(), {content: 'branch', submission_id: 'submission-1'});
  assert.equal(result.conversation.conversation_id, 'conversation-2');
});

test('steer and child roster use their shared run routes', async () => {
  const paths: string[] = [];
  globalThis.fetch = async (input) => {
    const request = new Request(new URL(String(input), 'http://localhost'));
    paths.push(new URL(request.url).pathname);
    if (request.url.endsWith('/children')) {
      return new Response(JSON.stringify({children: [{child_session_id: 'child-1', status: 'running'}]}));
    }
    return new Response(JSON.stringify({run_id: 'run-1', control_sequence: 1}), {status: 202});
  };

  await steerAnswerRun('run-1', 'focus');
  const children = await getAnswerRunChildren('run-1');

  assert.deepEqual(paths, [
    '/web/api/answer/run-1/steer',
    '/web/api/answer/run-1/children',
  ]);
  assert.equal(children[0]?.status, 'running');
});
