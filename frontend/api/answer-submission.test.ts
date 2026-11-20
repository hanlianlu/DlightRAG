// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {BrowserAnswerSubmissionAdapter, type AnswerSubmissionIntent} from './answer-submission.ts';
import {AnswerSubmissionError} from './web-command-error.ts';
import type {AcceptedAnswer} from './conversations.ts';

const intent: AnswerSubmissionIntent = {
  submissionId: '11111111-1111-4111-8111-111111111111',
  conversationId: 'conversation/1',
  query: 'Original question',
  workspaces: ['alpha', 'beta'],
  mode: 'research',
};

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

const accepted: AcceptedAnswer = {
  kind: 'accepted',
  conversation: {
    conversationId: 'conversation-1',
    title: 'Answer',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
  },
  turn: {
    turnId: 'turn-1',
    turnNumber: 1,
    answerRunId: 'run-1',
    submissionId: intent.submissionId,
    status: 'queued',
    cancelRequested: false,
    userText: intent.query,
    assistantText: '',
    userAttachments: [],
    steeringMessages: [],
    presentation: null,
    usage: {},
    errorKind: null,
    errorMessage: null,
    childCount: 0,
    answerSequence: null,
    feedback: null,
    createdAt: '2026-01-01T00:00:00Z',
  },
};

// What the server actually puts on the wire; the adapter must translate it.
const acceptedWire = {
  kind: 'accepted',
  conversation: {
    conversation_id: 'conversation-1',
    title: 'Answer',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  },
  turn: {
    turn_id: 'turn-1',
    turn_number: 1,
    answer_run_id: 'run-1',
    submission_id: intent.submissionId,
    status: 'queued',
    cancel_requested: false,
    user_text: intent.query,
    assistant_text: '',
    user_attachments: [],
    steering_messages: [],
    presentation: null,
    usage: {},
    error_kind: null,
    error_message: null,
    created_at: '2026-01-01T00:00:00Z',
  },
};

test('browser answer command posts the immutable intent and leased files', async () => {
  let request!: Request;
  const adapter = new BrowserAnswerSubmissionAdapter();
  globalThis.fetch = async (input, init) => {
    request = new Request(new URL(String(input), 'http://localhost'), init);
    return new Response(JSON.stringify(acceptedWire), {
      status: 202,
      headers: {'Content-Type': 'application/json'},
    });
  };

  const result = await adapter.submit(
    intent,
    [new File(['notes'], 'notes.md', {type: 'text/markdown'})],
    new AbortController().signal,
  );

  assert.deepEqual(result, accepted);
  assert.equal(request.method, 'POST');
  assert.equal(new URL(request.url).pathname, '/web/api/answer');
  const body = await request.formData();
  assert.equal(body.get('query'), intent.query);
  assert.equal(body.get('workspaces'), JSON.stringify(intent.workspaces));
  assert.equal(body.get('conversation_id'), intent.conversationId);
  assert.equal(body.get('submission_id'), intent.submissionId);
  assert.equal(body.get('mode'), intent.mode);
  assert.equal((body.get('attachments') as File).name, 'notes.md');
  assert.equal(body.get('delivery'), null);
});

test('a send that arrives while a Run is live carries its delivery and reads the receipt', async () => {
  let payload: string | null = null;
  const adapter = new BrowserAnswerSubmissionAdapter();
  globalThis.fetch = async (_input, init) => {
    payload = typeof init?.body === 'string' ? init.body : null;
    return new Response(JSON.stringify({
      kind: 'queue',
      message: {
        message_id: 'message-1',
        content: 'One more thing',
        position: 1,
        created_at: '2026-01-01T00:00:00Z',
      },
    }), {
      status: 202,
      headers: {'Content-Type': 'application/json'},
    });
  };

  const result = await adapter.submit(
    {...intent, delivery: 'queue'},
    [],
    new AbortController().signal,
  );

  assert.deepEqual(result, {
    kind: 'queue',
    message: {
      messageId: 'message-1',
      content: 'One more thing',
      position: 1,
      createdAt: '2026-01-01T00:00:00Z',
    },
  });
  assert.deepEqual(JSON.parse(payload ?? ''), {
    query: intent.query,
    workspaces: intent.workspaces,
    conversation_id: intent.conversationId,
    submission_id: intent.submissionId,
    mode: intent.mode,
    delivery: 'queue',
  });

  globalThis.fetch = async () => new Response(JSON.stringify({
    kind: 'steer',
    run_id: 'run-1',
    control_sequence: 3,
  }), {status: 202, headers: {'Content-Type': 'application/json'}});
  assert.deepEqual(
    await adapter.submit({...intent, delivery: 'steer'}, [], new AbortController().signal),
    {kind: 'steer', runId: 'run-1', controlSequence: 3},
  );
});

test('browser answer command preserves typed failures and marks transport failures ambiguous', async () => {
  const typed = new BrowserAnswerSubmissionAdapter();
  globalThis.fetch = async () => new Response(JSON.stringify({
    kind: 'submission_conflict',
    message: 'Submission already has a different intent',
  }), {
    status: 409,
    headers: {'Content-Type': 'application/json'},
  });
  await assert.rejects(
    typed.submit(intent, [], new AbortController().signal),
    (error: unknown) => error instanceof AnswerSubmissionError
      && error.status === 409
      && error.kind === 'submission_conflict'
      && error.message === 'Submission already has a different intent',
  );

  const offline = new BrowserAnswerSubmissionAdapter();
  globalThis.fetch = async () => {
    throw new TypeError('Failed to fetch');
  };
  await assert.rejects(
    offline.submit(intent, [], new AbortController().signal),
    (error: unknown) => error instanceof AnswerSubmissionError
      && error.status === 0
      && error.kind === 'ambiguous',
  );
});

test('a rejected input keeps the server reason and its stable error kind', async () => {
  const adapter = new BrowserAnswerSubmissionAdapter();
  globalThis.fetch = async () => new Response(JSON.stringify({
    kind: 'invalid_request',
    message: 'This request needs a resource capability that no answer mode can provide.',
    error_kind: 'unsupported_resource_capability',
  }), {
    status: 422,
    headers: {'Content-Type': 'application/json'},
  });
  await assert.rejects(
    adapter.submit(intent, [], new AbortController().signal),
    (error: unknown) => error instanceof AnswerSubmissionError
      && error.status === 422
      && error.kind === 'invalid_request'
      && error.errorKind === 'unsupported_resource_capability'
      && error.message.startsWith('This request needs'),
  );

  globalThis.fetch = async () => new Response('upstream failure', {status: 502});
  await assert.rejects(
    adapter.submit(intent, [], new AbortController().signal),
    // No server reason: the UI shows its localized fallback, not English copy.
    (error: unknown) => error instanceof AnswerSubmissionError
      && error.kind === 'ambiguous'
      && error.errorKind === null
      && error.message === '',
  );
});

test('submission lookup is owner route shaped and distinguishes absent from accepted', async () => {
  const seen: string[] = [];
  const responses = [
    new Response(null, {status: 404}),
    new Response(JSON.stringify(acceptedWire), {
      status: 200,
      headers: {'Content-Type': 'application/json'},
    }),
  ];
  const adapter = new BrowserAnswerSubmissionAdapter();
  globalThis.fetch = async (input) => {
    seen.push(String(input));
    return responses.shift()!;
  };

  assert.equal(await adapter.lookup(intent.submissionId, new AbortController().signal), null);
  assert.deepEqual(
    await adapter.lookup(intent.submissionId, new AbortController().signal),
    accepted,
  );
  assert.deepEqual(seen, [
    `/web/api/answer-submissions/${intent.submissionId}`,
    `/web/api/answer-submissions/${intent.submissionId}`,
  ]);
});

test('a lookup answer that cannot be read is an unknown outcome, as an unreadable submit answer is', async () => {
  const adapter = new BrowserAnswerSubmissionAdapter();
  for (const body of ['<html>half a page', JSON.stringify({turn: 'not an accepted answer'})]) {
    globalThis.fetch = async () => new Response(body, {status: 200});
    await assert.rejects(
      adapter.lookup(intent.submissionId, new AbortController().signal),
      (error: unknown) => error instanceof AnswerSubmissionError
        && error.kind === 'ambiguous'
        && error.message === '',
      `${body} was taken for an answer`,
    );
  }
});

test('browser answer command carries the reader\'s own limit as a retryable refusal with its kind', async () => {
  const adapter = new BrowserAnswerSubmissionAdapter();
  globalThis.fetch = async () => new Response(JSON.stringify({
    kind: 'service_unavailable',
    message: 'You have as many answers in progress as you may run at once.',
    error_kind: 'active_run_limit',
  }), {
    status: 429,
    headers: {'Content-Type': 'application/json'},
  });

  await assert.rejects(
    adapter.submit(intent, [], new AbortController().signal),
    (error: unknown) => error instanceof AnswerSubmissionError
      && error.status === 429
      && error.kind === 'service_unavailable'
      && error.errorKind === 'active_run_limit',
  );
});
