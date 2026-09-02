// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {
  AnswerSubmissionError,
  BrowserAnswerSubmissionAdapter,
  type AnswerSubmissionIntent,
} from './answer-submission.ts';
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
    presentation: null,
    usage: {},
    evidence: {},
    errorKind: null,
    errorMessage: null,
    createdAt: '2026-01-01T00:00:00Z',
  },
};

// What the server actually puts on the wire; the adapter must translate it.
const acceptedWire = {
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
    presentation: null,
    usage: {},
    evidence: {},
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
