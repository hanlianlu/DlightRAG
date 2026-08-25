// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';
import type {ConversationTurn} from '../api/conversations.ts';
import {RunController} from './run_controller.ts';
import {answerRunStore} from '../stores/answerRunStore.ts';

function eventResponse(body: string): Response {
  return new Response(body, {
    status: 200,
    headers: {'Content-Type': 'text/event-stream'},
  });
}

function stored(status: ConversationTurn['status']): ConversationTurn {
  return {
    turn_id: 'turn-1',
    turn_number: 1,
    answer_run_id: 'run-1',
    submission_id: 'submission-1',
    status,
    cancel_requested: false,
    user_text: 'Question',
    assistant_text: '',
    user_attachments: [],
    presentation: null,
    usage: {},
    evidence: {},
    error_kind: null,
    error_message: null,
    created_at: '2026-01-01T00:00:00Z',
  };
}

test('RunController resumes SSE from the last durable sequence without replaying events', async () => {
  const conversationId = 'run-controller-replay';
  const runId = 'run-replay';
  const headers: string[] = [];
  const responses = [
    eventResponse('id: 1\nevent: token\ndata: "Hello"\n\n'),
    eventResponse([
      'id: 1\nevent: token\ndata: "duplicate"\n\n',
      'id: 2\nevent: done\ndata: {"status":"cancelled","presentation":null}\n\n',
    ].join('')),
  ];
  answerRunStore.trackRun(conversationId, runId);
  const controller = new RunController({
    reconnectDelayMs: 0,
    fetch: (async (_input, init) => {
      headers.push(new Headers(init?.headers).get('Last-Event-ID') ?? '');
      return responses.shift()!;
    }) as typeof fetch,
  });
  assert.ok(controller.beginFollow(runId, false));
  const events: string[] = [];

  const result = await controller.follow(conversationId, runId, (type, data) => {
    events.push(`${type}:${data}`);
  });

  assert.equal(result.kind, 'terminal');
  assert.deepEqual(headers, ['', '1']);
  assert.deepEqual(events, [
    'token:"Hello"',
    'done:{"status":"cancelled","presentation":null}',
  ]);
  assert.equal(answerRunStore.lastSequence(conversationId, runId), 2);
  controller.finish(runId);
  answerRunStore.clear(conversationId);
});

test('RunController settles an exhausted stream from the authoritative run row', async () => {
  const conversationId = 'run-controller-settle';
  const runId = 'run-settle';
  answerRunStore.trackRun(conversationId, runId);
  const controller = new RunController({
    maxReconnectAttempts: 0,
    reconnectDelayMs: 0,
    fetch: (async () => eventResponse('')) as typeof fetch,
    getRun: async () => ({...stored('running'), answer_run_id: runId}),
  });
  controller.beginFollow(runId, false);

  const result = await controller.follow(conversationId, runId, () => {});

  assert.equal(result.kind, 'retryable');
  assert.equal(result.kind === 'retryable' ? result.stored.status : '', 'running');
  controller.finish(runId);
  answerRunStore.clear(conversationId);
});

test('RunController aborts run-owned commands when its lifecycle finishes', () => {
  const controller = new RunController();
  controller.beginFollow('run-command', false);
  const signal = controller.signalFor('run-command');

  assert.ok(signal);
  assert.equal(controller.signalFor('another-run'), null);
  controller.finish('run-command');

  assert.equal(signal.aborted, true);
  assert.equal(controller.signalFor('run-command'), null);
});

test('RunController preserves a durable cancellation request when reattaching', () => {
  const controller = new RunController();

  controller.beginFollow('run-stopping', true);

  assert.equal(controller.active, true);
  assert.equal(controller.stopping, true);
  controller.detach();
});

test('RunController distinguishes explicit cancel from detach and owns both aborts', async () => {
  let cancelCalls = 0;
  let cancelAborted = false;
  let releaseCancel!: () => void;
  const controller = new RunController({
    cancelRun: async (_runId, signal) => {
      cancelCalls += 1;
      await new Promise<void>((resolve) => {
        releaseCancel = resolve;
        signal?.addEventListener('abort', () => {
          cancelAborted = true;
          resolve();
        }, {once: true});
      });
      return stored('cancelled');
    },
  });
  controller.beginFollow('run-1', false);
  const cancel = controller.cancel();
  assert.equal(controller.stopping, true);
  assert.equal(cancelCalls, 1);

  controller.detach();
  assert.equal(controller.active, false);
  assert.equal(cancelAborted, false, 'detaching the reader must not undo the durable cancel request');
  controller.disconnect();
  assert.equal(cancelAborted, true);
  releaseCancel();
  await cancel;
});

test('RunController keeps an earlier durable cancellation independent from a later Stop', async () => {
  const requests = new Map<string, {
    signal: AbortSignal;
    reject: (reason: Error) => void;
  }>();
  const controller = new RunController({
    cancelRun: (runId, signal) => new Promise<ConversationTurn>((_resolve, reject) => {
      assert.ok(signal);
      requests.set(runId, {signal, reject});
      signal.addEventListener(
        'abort',
        () => reject(new DOMException('Aborted', 'AbortError')),
        {once: true},
      );
    }),
  });

  controller.beginFollow('run-a', false);
  const cancelA = controller.cancel();
  controller.detach();
  controller.beginFollow('run-b', false);
  const cancelB = controller.cancel();

  assert.equal(requests.get('run-a')?.signal.aborted, false);
  assert.equal(requests.get('run-b')?.signal.aborted, false);
  requests.get('run-a')?.reject(new Error('temporary failure'));
  await cancelA;
  assert.equal(controller.stopping, true, 'run A failure must not clear run B stopping state');

  controller.disconnect();
  assert.equal(requests.get('run-b')?.signal.aborted, true);
  await cancelB;
});
