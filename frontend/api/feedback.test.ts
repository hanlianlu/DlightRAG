// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {deleteAnswerFeedback, putAnswerFeedback} from './feedback.ts';
import {ApiError} from './wire.ts';

const originalFetch = globalThis.fetch;
const originalDocument = globalThis.document;

test.beforeEach(() => {
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: {cookie: 'dlightrag_web_csrf=token-1'},
  });
});

test.afterEach(() => {
  globalThis.fetch = originalFetch;
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: originalDocument,
  });
});

test('a rating is put whole to the message it names, and the answer is the judgement that was kept', async () => {
  const seen: Request[] = [];
  globalThis.fetch = async (input, init) => {
    seen.push(new Request(new URL(String(input), 'http://localhost'), init));
    return Response.json({
      run_id: 'run/1', agent: 'child-1', sequence: 12, value: 'down',
      reasons: ['unsupported', 'off_topic'], note: 'cites nothing', updated_at: '2026-10-09T12:00:00Z',
    });
  };

  const kept = await putAnswerFeedback('run/1', 'child-1', 12, {
    value: 'down', reasons: ['unsupported', 'off_topic'], note: 'cites nothing',
  });

  assert.equal(seen[0]?.method, 'PUT');
  assert.equal(seen[0]?.url, 'http://localhost/web/api/answer/run%2F1/feedback');
  assert.equal(seen[0]?.headers.get('content-type'), 'application/json');
  assert.equal(seen[0]?.headers.get('x-csrf-token'), 'token-1');
  assert.deepEqual(await seen[0]?.json(), {
    agent: 'child-1', sequence: 12, value: 'down', reasons: ['unsupported', 'off_topic'], note: 'cites nothing',
  });
  // The page needs the judgement and not the echo of the message or when it was given.
  assert.deepEqual(kept, {value: 'down', reasons: ['unsupported', 'off_topic'], note: 'cites nothing'});
});

test('a rating the server refuses throws the refusal with its status and the reason it gave', async () => {
  globalThis.fetch = async () => Response.json(
    {detail: 'Answer feedback is turned off for you', error_type: 'auth'},
    {status: 403},
  );

  await assert.rejects(
    putAnswerFeedback('run-1', 'main', 3, {value: 'up', reasons: [], note: null}),
    (error: unknown) => error instanceof ApiError
      && error.status === 403
      && error.errorType === 'auth'
      && error.errorKind === null
      && error.detail === 'Answer feedback is turned off for you',
  );
});

test('a judgement is taken back by naming its message in the query', async () => {
  const seen: Request[] = [];
  globalThis.fetch = async (input, init) => {
    seen.push(new Request(new URL(String(input), 'http://localhost'), init));
    return new Response(null, {status: 204});
  };

  await deleteAnswerFeedback('run/1', 'child-1', 12);

  assert.equal(seen[0]?.method, 'DELETE');
  assert.equal(seen[0]?.url, 'http://localhost/web/api/answer/run%2F1/feedback?agent=child-1&sequence=12');
  assert.equal(seen[0]?.headers.get('x-csrf-token'), 'token-1');
});

test('taking a judgement back from a Run the server does not know is a refusal', async () => {
  globalThis.fetch = async () => Response.json({detail: 'Run not found', error_type: 'not_found'}, {status: 404});

  await assert.rejects(
    deleteAnswerFeedback('run-1', 'main', 3),
    (error: unknown) => error instanceof ApiError && error.status === 404,
  );
});
