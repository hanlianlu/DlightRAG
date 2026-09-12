// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ChildControlRejectedError,
  continueAnswerRun,
  controlAnswerChild,
  getAnswerRunChild,
  getAnswerRunChildren,
  getAnswerRunChildrenPage,
  getConversationHistory,
  listConversations,
  replyAnswerChild,
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

  assert.deepEqual(await listConversations(), {items: [], nextCursor: null});
  await listConversations('facts/signature + padding');

  assert.deepEqual(seen, [
    '/web/api/conversations',
    '/web/api/conversations?cursor=facts%2Fsignature%20%2B%20padding',
  ]);
});

test('conversation titles preserve JavaScript-sensitive separators as inert data', async () => {
  const title = 'line\u2028separator\u2029</script><script>globalThis.__conversationTitleRan=true</script>';
  const runtime = globalThis as typeof globalThis & {__conversationTitleRan?: boolean};
  delete runtime.__conversationTitleRan;
  globalThis.fetch = async () => new Response(JSON.stringify({
    items: [{
      conversation_id: 'conversation-sensitive',
      title,
      created_at: '2026-08-23T00:00:00Z',
      updated_at: '2026-08-23T00:00:00Z',
    }],
    next_cursor: null,
  }), {headers: {'Content-Type': 'application/json'}});

  const page = await listConversations();

  assert.equal(page.items[0]?.title, title);
  assert.equal(runtime.__conversationTitleRan, undefined);
});

test('history pages encode cursor and limit, normalize rollback payloads, and pass abort', async () => {
  let seenUrl = '';
  let seenSignal: AbortSignal | null | undefined;
  globalThis.fetch = async (input, init) => {
    seenUrl = new URL(String(input), 'http://localhost').toString();
    seenSignal = init?.signal;
    return new Response(JSON.stringify({
      conversation: {
        conversation_id: 'conversation-1', title: null,
        created_at: '2026-08-23T00:00:00Z', updated_at: '2026-08-23T00:00:00Z',
      },
      turns: [],
    }), {headers: {'Content-Type': 'application/json'}});
  };
  const controller = new AbortController();

  const result = await getConversationHistory(
    'conversation/1', 'opaque cursor+', 25, controller.signal,
  );

  assert.equal(
    seenUrl,
    'http://localhost/web/api/conversations/conversation%2F1/history?cursor=opaque+cursor%2B&limit=25',
  );
  assert.equal(seenSignal, controller.signal);
  assert.equal(result.nextCursor, null);
});

test('continuation posts one submission id to the selected branch operation', async () => {
  const seen: Request[] = [];
  globalThis.fetch = async (input, init) => {
    seen.push(new Request(new URL(String(input), 'http://localhost'), init));
    return new Response(JSON.stringify({
      conversation: {
        conversation_id: 'conversation-2',
        title: null,
        created_at: '2026-08-23T00:00:00Z',
        updated_at: '2026-08-23T00:00:00Z',
      },
      turn: {
        turn_id: 'child-turn',
        turn_number: 1,
        answer_run_id: 'child-run',
        submission_id: 'submission-1',
        status: 'queued',
        cancel_requested: false,
        user_text: 'branch',
        assistant_text: '',
        user_attachments: [],
        presentation: null,
        usage: {},
        evidence: {},
        error_kind: null,
        error_message: null,
        created_at: '2026-08-23T00:00:00Z',
      },
    }), {status: 202, headers: {'Content-Type': 'application/json'}});
  };

  const result = await continueAnswerRun('parent/run', 'fork', 'branch', 'submission-1');

  const request = seen[0]!;
  assert.equal(request.url, 'http://localhost/web/api/answer/parent%2Frun/fork');
  assert.equal(request.method, 'POST');
  assert.deepEqual(await request.json(), {content: 'branch', submission_id: 'submission-1'});
  assert.equal(result.conversation.conversationId, 'conversation-2');
});

test('steer and child roster use their Answer-specific routes', async () => {
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

test('child controls and replies carry durable submission identity', async () => {
  const requests: Request[] = [];
  globalThis.fetch = async (input, init) => {
    const request = new Request(new URL(String(input), 'http://localhost'), init);
    requests.push(request);
    const reply = request.url.includes('child-guidance');
    return new Response(JSON.stringify(reply ? {
      run_id: 'run-1', request_id: 'request-1', action: 'reply', outcome: 'replied',
    } : {
      run_id: 'run-1', child_session_id: 'child/1', action: 'steer', outcome: 'queued',
      operation_id: 'operation-1', operation_sequence: 1, control_sequence: 7,
      consumed_at: null,
    }), {status: 202, headers: {'Content-Type': 'application/json'}});
  };

  const control = await controlAnswerChild(
    'run-1', 'child/1', 'steer', 'focus', 'submission-control', false,
  );
  const reply = await replyAnswerChild(
    'run-1', 'request-1', 'use report', 'submission-reply',
  );

  assert.equal(requests[0]?.headers.get('Idempotency-Key'), 'submission-control');
  assert.equal(requests[1]?.headers.get('Idempotency-Key'), 'submission-reply');
  assert.equal(
    new URL(requests[0]!.url).pathname,
    '/web/api/answer/run-1/children/child%2F1/control',
  );
  assert.deepEqual(await requests[0]!.clone().json(), {
    action: 'steer', content: 'focus', reauthorize_user_cancelled: false,
  });
  assert.equal(control.controlSequence, 7);
  assert.equal(reply.requestId, 'request-1');
});

test('child roster pages encode the opaque cursor and normalize the continuation', async () => {
  const requests: string[] = [];
  globalThis.fetch = async (input) => {
    const request = new Request(new URL(String(input), 'http://localhost'));
    requests.push(request.url);
    if (request.url.includes('cursor')) {
      return new Response(JSON.stringify({
        children: [{child_session_id: 'child-2', status: 'succeeded'}],
      }));
    }
    return new Response(JSON.stringify({
      children: [{child_session_id: 'child-1', status: 'running'}],
      next_cursor: 'opaque-token',
    }));
  };

  const first = await getAnswerRunChildrenPage('run-1');
  assert.deepEqual(first.children.map((child) => child.childSessionId), ['child-1']);
  assert.equal(first.nextCursor, 'opaque-token');
  assert.equal(requests[0], 'http://localhost/web/api/answer/run-1/children');

  const older = await getAnswerRunChildrenPage('run-1', 'opaque-token');
  assert.deepEqual(older.children.map((child) => child.childSessionId), ['child-2']);
  assert.equal(older.nextCursor, null);
  assert.equal(
    requests[1],
    'http://localhost/web/api/answer/run-1/children?cursor=opaque-token',
  );
});

test('child observation normalizes transcript, controls, and questions', async () => {
  globalThis.fetch = async () => new Response(JSON.stringify({
    run_id: 'run-1',
    child: {child_session_id: 'child-1', status: 'running', result_handles: ['ev-1']},
    transcript: [{role: 'user', content: 'inspect'}],
    controls: [{
      control_sequence: 3, kind: 'steer', content: 'focus', origin: 'user',
      consumed: false, consumed_at: null,
    }],
    questions: [{request_id: 'req-1', question: 'Which source?', status: 'pending'}],
    result: {status: 'running', summary: 'working', handles: ['ev-1']},
  }));

  const observation = await getAnswerRunChild('run-1', 'child/1');

  assert.equal(observation.child.childSessionId, 'child-1');
  assert.deepEqual(observation.child.resultHandles, ['ev-1']);
  assert.equal(observation.transcript[0]?.content, 'inspect');
  assert.equal(observation.controls[0]?.consumed, false);
  assert.equal(observation.questions[0]?.requestId, 'req-1');
  assert.deepEqual(observation.result?.handles, ['ev-1']);
});

test('child control 409 surfaces the explicit terminal outcome', async () => {
  globalThis.fetch = async () => new Response(JSON.stringify({detail: 'terminal_child'}), {
    status: 409,
    headers: {'Content-Type': 'application/json'},
  });

  await assert.rejects(
    () => controlAnswerChild('run-1', 'child-1', 'steer', 'focus', 'submission-late'),
    (error: unknown) => {
      assert.ok(error instanceof ChildControlRejectedError);
      assert.equal(error.outcome, 'terminal_child');
      assert.equal(error.status, 409);
      return true;
    },
  );
});
