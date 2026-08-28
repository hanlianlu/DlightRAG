// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';
import type {ConversationTurn} from '../api/conversations.ts';
import {RunController, type AnswerRunEvent} from './run_controller.ts';
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

class TestFrames {
  readonly callbacks = new Map<number, () => void>();
  #nextId = 1;

  readonly schedule = (callback: () => void): number => {
    const id = this.#nextId;
    this.#nextId += 1;
    this.callbacks.set(id, callback);
    return id;
  };

  readonly cancel = (handle: unknown): void => {
    this.callbacks.delete(handle as number);
  };

  runAll(): void {
    const callbacks = [...this.callbacks.values()];
    this.callbacks.clear();
    callbacks.forEach((callback) => callback());
  }
}

function chunkedEventResponse(
  chunks: readonly string[],
  beforeChunk?: (index: number) => void,
): Response {
  const encoder = new TextEncoder();
  let index = 0;
  return new Response(new ReadableStream<Uint8Array>({
    pull(controller): void {
      if (index === chunks.length) {
        controller.close();
        return;
      }
      beforeChunk?.(index);
      controller.enqueue(encoder.encode(chunks[index]));
      index += 1;
    },
  }), {
    status: 200,
    headers: {'Content-Type': 'text/event-stream'},
  });
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
  const events: AnswerRunEvent[] = [];

  const result = await controller.follow(conversationId, runId, (batch) => {
    events.push(...batch);
  });

  assert.equal(result.kind, 'terminal');
  assert.deepEqual(headers, ['', '1']);
  assert.deepEqual(events, [
    {kind: 'token', text: 'Hello'},
    {kind: 'done', payload: {status: 'cancelled', presentation: null}},
  ]);
  assert.equal(answerRunStore.lastSequence(conversationId, runId), 2);
  controller.finish(runId);
  answerRunStore.clear(conversationId);
});

test('RunController frame-batches 2,000 streamed tokens without losing order or durability', async () => {
  const conversationId = 'run-controller-batching';
  const runId = 'run-batching';
  const frames = new TestFrames();
  const expected = Array.from(
    {length: 2_000},
    (_, index) => String.fromCharCode(97 + (index % 26)),
  ).join('');
  const chunks = Array.from({length: 2_000}, (_, index) => (
    `id: ${index + 1}\nevent: token\ndata: ${JSON.stringify(expected[index])}\n\n`
  ));
  chunks.push(
    'id: 2001\nevent: done\ndata: {"status":"cancelled","presentation":null}\n\n',
  );
  answerRunStore.trackRun(conversationId, runId);
  const controller = new RunController({
    scheduleFrame: frames.schedule,
    cancelFrame: frames.cancel,
    fetch: (async () => chunkedEventResponse(chunks, (index) => {
      if (index > 0 && index % 500 === 0) frames.runAll();
    })) as typeof fetch,
  });
  controller.beginFollow(runId, false);
  const batches: Array<readonly AnswerRunEvent[]> = [];
  let terminalDelivered = false;

  const result = await controller.follow(conversationId, runId, (batch) => {
    batches.push(batch);
    if (batch.some((event) => event.kind === 'done')) terminalDelivered = true;
  });

  const events = batches.flat();
  const text = events
    .filter((event): event is Extract<AnswerRunEvent, {kind: 'token'}> => event.kind === 'token')
    .map((event) => event.text)
    .join('');
  assert.equal(result.kind, 'terminal');
  assert.equal(terminalDelivered, true, 'terminal delivery must happen before follow resolves');
  assert.equal(text, expected);
  assert.equal(answerRunStore.lastSequence(conversationId, runId), 2_001);
  assert.equal(batches.length, 6, 'four controlled frames plus token and terminal flushes');
  assert.ok(
    events.filter((event) => event.kind === 'token').length <= 7,
    'contiguous token parts should be joined once per delivered frame',
  );
  assert.deepEqual(events.at(-1), {
    kind: 'done',
    payload: {status: 'cancelled', presentation: null},
  });
  assert.equal(frames.callbacks.size, 0);
  controller.finish(runId);
  answerRunStore.clear(conversationId);
});

test('RunController preserves token-reset-token and progress/tool boundaries', async () => {
  const conversationId = 'run-controller-order';
  const runId = 'run-order';
  const frames = new TestFrames();
  answerRunStore.trackRun(conversationId, runId);
  const controller = new RunController({
    scheduleFrame: frames.schedule,
    cancelFrame: frames.cancel,
    fetch: (async () => eventResponse([
      'id: 1\nevent: token\ndata: "before"\n\n',
      'id: 2\nevent: reset\ndata: {}\n\n',
      'id: 3\nevent: token\ndata: "after"\n\n',
      'id: 4\nevent: progress\ndata: {"phase":"planning"}\n\n',
      'id: 5\nevent: token\ndata: " tail"\n\n',
      'id: 6\nevent: tool_start\ndata: {"tool_name":"spawn_agent"}\n\n',
      'id: 7\nevent: done\ndata: {"status":"cancelled","presentation":null}\n\n',
    ].join(''))) as typeof fetch,
  });
  controller.beginFollow(runId, false);
  const events: AnswerRunEvent[] = [];

  await controller.follow(conversationId, runId, (batch) => events.push(...batch));

  assert.deepEqual(events, [
    {kind: 'token', text: 'before'},
    {kind: 'reset'},
    {kind: 'token', text: 'after'},
    {kind: 'progress', payload: {phase: 'planning'}},
    {kind: 'token', text: ' tail'},
    {kind: 'tool', eventType: 'tool_start', payload: {tool_name: 'spawn_agent'}},
    {kind: 'done', payload: {status: 'cancelled', presentation: null}},
  ]);
  assert.equal(frames.callbacks.size, 0);
  controller.finish(runId);
  answerRunStore.clear(conversationId);
});

test('RunController ignores unknown wire events after durably advancing their sequence', async () => {
  const conversationId = 'run-controller-unknown';
  const runId = 'run-unknown';
  const headers: string[] = [];
  const responses = [
    eventResponse('id: 1\nevent: future_event\ndata: {"value":1}\n\n'),
    eventResponse('id: 2\nevent: done\ndata: {"status":"cancelled","presentation":null}\n\n'),
  ];
  answerRunStore.trackRun(conversationId, runId);
  const controller = new RunController({
    reconnectDelayMs: 0,
    fetch: (async (_input, init) => {
      headers.push(new Headers(init?.headers).get('Last-Event-ID') ?? '');
      return responses.shift()!;
    }) as typeof fetch,
  });
  controller.beginFollow(runId, false);
  const events: AnswerRunEvent[] = [];

  await controller.follow(conversationId, runId, (batch) => events.push(...batch));

  assert.deepEqual(headers, ['', '1']);
  assert.deepEqual(events, [
    {kind: 'done', payload: {status: 'cancelled', presentation: null}},
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

test('RunController flushes a pending frame at stream end and stream failure', async () => {
  for (const mode of ['end', 'failure'] as const) {
    const conversationId = `run-controller-flush-${mode}`;
    const runId = `run-flush-${mode}`;
    const frames = new TestFrames();
    const encoder = new TextEncoder();
    let pulled = false;
    const response = mode === 'end'
      ? eventResponse('id: 1\nevent: token\ndata: "delivered"\n\n')
      : new Response(new ReadableStream<Uint8Array>({
          pull(controller): void {
            if (!pulled) {
              pulled = true;
              controller.enqueue(encoder.encode(
                'id: 1\nevent: token\ndata: "delivered"\n\n',
              ));
              return;
            }
            controller.error(new Error('stream failed'));
          },
        }), {status: 200, headers: {'Content-Type': 'text/event-stream'}});
    answerRunStore.trackRun(conversationId, runId);
    const controller = new RunController({
      maxReconnectAttempts: 0,
      reconnectDelayMs: 0,
      scheduleFrame: frames.schedule,
      cancelFrame: frames.cancel,
      fetch: (async () => response) as typeof fetch,
      getRun: async () => ({...stored('running'), answer_run_id: runId}),
    });
    controller.beginFollow(runId, false);
    const events: AnswerRunEvent[] = [];

    const result = await controller.follow(
      conversationId,
      runId,
      (batch) => events.push(...batch),
    );

    assert.equal(result.kind, 'retryable');
    assert.deepEqual(events, [{kind: 'token', text: 'delivered'}]);
    assert.equal(answerRunStore.lastSequence(conversationId, runId), 1);
    assert.equal(frames.callbacks.size, 0);
    controller.finish(runId);
    answerRunStore.clear(conversationId);
  }
});

test('RunController detach synchronously flushes accepted events and cancels its reader frame', async () => {
  const conversationId = 'run-controller-detach-flush';
  const runId = 'run-detach-flush';
  const frames = new TestFrames();
  const encoder = new TextEncoder();
  answerRunStore.trackRun(conversationId, runId);
  const controller = new RunController({
    scheduleFrame: frames.schedule,
    cancelFrame: frames.cancel,
    fetch: (async () => new Response(new ReadableStream<Uint8Array>({
      start(stream): void {
        stream.enqueue(encoder.encode('id: 1\nevent: token\ndata: "accepted"\n\n'));
      },
    }), {status: 200, headers: {'Content-Type': 'text/event-stream'}})) as typeof fetch,
  });
  controller.beginFollow(runId, false);
  const events: AnswerRunEvent[] = [];
  const following = controller.follow(
    conversationId,
    runId,
    (batch) => events.push(...batch),
  );
  while (answerRunStore.lastSequence(conversationId, runId) < 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  assert.equal(frames.callbacks.size, 1);

  controller.detach();

  assert.deepEqual(events, [{kind: 'token', text: 'accepted'}]);
  assert.equal(frames.callbacks.size, 0);
  assert.equal((await following).kind, 'aborted');
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
