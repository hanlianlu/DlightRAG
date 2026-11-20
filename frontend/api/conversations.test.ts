// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {AnswerSubmissionError} from './web-command-error.ts';
import {ApiError} from './wire.ts';
import {
  getAnswerActivityPage,
  forkAnswerRun,
  getAnswerRunChild,
  getAnswerRunChildrenPage,
  getArtifactTextAt,
  getConversationHistory,
  createConversation,
  listConversations,
  parseAnswerPresentation,
  readConversationQueue,
  editQueuedMessage,
  removeQueuedMessage,
  reorderConversationQueue,
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

test('a new conversation posts an empty creation and reads the summary the server kept', async () => {
  let seen: {url: string; method: string | undefined; body: string | undefined} | null = null;
  globalThis.fetch = async (input, init) => {
    seen = {
      url: String(input),
      method: init?.method,
      body: init?.body === undefined ? undefined : String(init.body),
    };
    return new Response(JSON.stringify({
      conversation_id: 'conversation-new',
      title: null,
      created_at: '2026-10-25T00:00:00Z',
      updated_at: '2026-10-25T00:00:00Z',
    }), {headers: {'Content-Type': 'application/json'}});
  };

  const summary = await createConversation();

  assert.equal(summary.conversationId, 'conversation-new');
  assert.deepEqual(seen, {
    url: '/web/api/conversations',
    method: 'POST',
    body: '{}',
  });
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
      commands: [],
      next_cursor: null,
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

test('a stored turn reports how many children its Run started, and none when the server says nothing', async () => {
  const turn = (count: Record<string, number>) => ({
    turn_id: 'turn-1', turn_number: 1, answer_run_id: 'run-1', submission_id: 'submission-1',
    status: 'succeeded', cancel_requested: false, user_text: 'question', assistant_text: '',
    user_attachments: [], steering_messages: [], presentation: null, usage: {}, error_kind: null,
    error_message: null, created_at: '2026-08-23T00:00:00Z', ...count,
  });
  globalThis.fetch = async () => new Response(JSON.stringify({
    conversation: {
      conversation_id: 'conversation-1', title: null,
      created_at: '2026-08-23T00:00:00Z', updated_at: '2026-08-23T00:00:00Z',
    },
    turns: [turn({child_count: 3}), turn({})],
    commands: [],
    next_cursor: null,
  }), {headers: {'Content-Type': 'application/json'}});

  const history = await getConversationHistory('conversation-1');

  assert.deepEqual(history.turns.map((stored) => stored.childCount), [3, 0]);
});

test('a stored turn names the message the reader judges and the judgement they gave, and none when the server says nothing', async () => {
  const turn = (extra: Record<string, unknown>) => ({
    turn_id: 'turn-1', turn_number: 1, answer_run_id: 'run-1', submission_id: 'submission-1',
    status: 'succeeded', cancel_requested: false, user_text: 'question', assistant_text: '',
    user_attachments: [], steering_messages: [], presentation: null, usage: {}, error_kind: null,
    error_message: null, created_at: '2026-08-23T00:00:00Z', ...extra,
  });
  globalThis.fetch = async () => new Response(JSON.stringify({
    conversation: {
      conversation_id: 'conversation-1', title: null,
      created_at: '2026-08-23T00:00:00Z', updated_at: '2026-08-23T00:00:00Z',
    },
    turns: [
      turn({
        answer_sequence: 7,
        feedback: {value: 'down', reasons: ['unsupported', 'off_topic'], note: 'cites nothing'},
      }),
      turn({answer_sequence: 4, feedback: {value: 'up', reasons: [], note: null}}),
      turn({answer_sequence: 3, feedback: null}),
      turn({answer_sequence: null, feedback: null}),
      // A turn that states neither.
      turn({}),
    ],
    commands: [],
    next_cursor: null,
  }), {headers: {'Content-Type': 'application/json'}});

  const history = await getConversationHistory('conversation-1');

  assert.deepEqual(history.turns.map((stored) => [stored.answerSequence, stored.feedback]), [
    [7, {value: 'down', reasons: ['unsupported', 'off_topic'], note: 'cites nothing'}],
    [4, {value: 'up', reasons: [], note: null}],
    [3, null],
    [null, null],
    [null, null],
  ]);
});

test('continuation posts one submission id to the selected branch operation', async () => {
  const seen: Request[] = [];
  globalThis.fetch = async (input, init) => {
    seen.push(new Request(new URL(String(input), 'http://localhost'), init));
    return new Response(JSON.stringify({
      kind: 'accepted',
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
        steering_messages: [],
        presentation: null,
        usage: {},
        error_kind: null,
        error_message: null,
        created_at: '2026-08-23T00:00:00Z',
      },
    }), {status: 202, headers: {'Content-Type': 'application/json'}});
  };

  const result = await forkAnswerRun('parent/run', 'branch', 'submission-1');

  const request = seen[0]!;
  assert.equal(request.url, 'http://localhost/web/api/answer/parent%2Frun/fork');
  assert.equal(request.method, 'POST');
  assert.deepEqual(await request.json(), {content: 'branch', submission_id: 'submission-1'});
  assert.equal(result.conversation.conversationId, 'conversation-2');
});

test('child roster uses its Answer-specific route', async () => {
  const paths: string[] = [];
  globalThis.fetch = async (input) => {
    const request = new Request(new URL(String(input), 'http://localhost'));
    paths.push(new URL(request.url).pathname);
    return new Response(JSON.stringify({
      children: [{child_session_id: 'child-1', status: 'running'}],
      next_cursor: null, run_status: 'running', run_started_at: null, run_finished_at: null,
    }));
  };

  const page = await getAnswerRunChildrenPage('run-1');

  assert.deepEqual(paths, ['/web/api/answer/run-1/children']);
  assert.equal(page.children[0]?.status, 'running');
});

test('the conversation queue reads, rewrites, reorders and drops its held messages', async () => {
  const wireMessage = (id: string, position: number) => ({
    message_id: id,
    content: `word ${id}`,
    position,
    created_at: '2026-01-01T00:00:00Z',
  });
  const requests: string[] = [];
  globalThis.fetch = async (input, init) => {
    const request = new Request(new URL(String(input), 'http://localhost'), init);
    requests.push(`${request.method} ${request.url}`);
    if (request.method === 'DELETE') return new Response(null, {status: 204});
    if (request.method === 'PUT') {
      const order = (await request.clone().json() as {message_ids: string[]}).message_ids;
      return new Response(JSON.stringify({
        items: order.map((id, index) => wireMessage(id, index + 1)),
      }));
    }
    if (request.method === 'PATCH') {
      return new Response(JSON.stringify({message: wireMessage('message-b', 1)}));
    }
    return new Response(JSON.stringify({
      items: [wireMessage('message-a', 1), wireMessage('message-b', 2)],
    }));
  };

  const queue = await readConversationQueue('conversation/1');
  const rewritten = await editQueuedMessage('conversation/1', 'message-b', 'changed words');
  const reordered = await reorderConversationQueue('conversation/1', ['message-b', 'message-a']);
  await removeQueuedMessage('conversation/1', 'message-a');

  assert.deepEqual(queue.map((message) => message.messageId), ['message-a', 'message-b']);
  assert.equal(rewritten.content, 'word message-b');
  assert.deepEqual(reordered.map((message) => message.messageId), ['message-b', 'message-a']);
  assert.deepEqual(requests, [
    'GET http://localhost/web/api/conversations/conversation%2F1/queue',
    'PATCH http://localhost/web/api/conversations/conversation%2F1/queue/message-b',
    'PUT http://localhost/web/api/conversations/conversation%2F1/queue/order',
    'DELETE http://localhost/web/api/conversations/conversation%2F1/queue/message-a',
  ]);
});

test('child roster pages encode the opaque cursor and normalize the continuation', async () => {
  const requests: string[] = [];
  globalThis.fetch = async (input) => {
    const request = new Request(new URL(String(input), 'http://localhost'));
    requests.push(request.url);
    if (request.url.includes('cursor')) {
      return new Response(JSON.stringify({
        children: [{child_session_id: 'child-2', status: 'succeeded'}],
        next_cursor: null,
        run_status: 'succeeded',
        run_started_at: null,
        run_finished_at: null,
      }));
    }
    return new Response(JSON.stringify({
      children: [{
        child_session_id: 'child-1', status: 'running',
        started_at: '2026-10-05T12:00:00Z', finished_at: null, pending_questions: 2,
      }],
      next_cursor: 'opaque-token',
      run_status: 'running',
      run_started_at: '2026-10-05T12:00:00Z',
      run_finished_at: null,
    }));
  };

  const first = await getAnswerRunChildrenPage('run-1');
  assert.deepEqual(first.children.map((child) => child.childSessionId), ['child-1']);
  assert.equal(first.nextCursor, 'opaque-token');
  assert.deepEqual(first.run, {
    status: 'running', startedAt: '2026-10-05T12:00:00Z', finishedAt: null,
  });
  assert.equal(requests[0], 'http://localhost/web/api/answer/run-1/children');
  assert.deepEqual(
    [first.children[0]?.startedAt, first.children[0]?.finishedAt, first.children[0]?.pendingQuestions],
    ['2026-10-05T12:00:00Z', null, 2],
  );

  const older = await getAnswerRunChildrenPage('run-1', 'opaque-token');
  assert.deepEqual(older.children.map((child) => child.childSessionId), ['child-2']);
  assert.equal(older.nextCursor, null);
  assert.deepEqual(older.run, {status: 'succeeded', startedAt: null, finishedAt: null});
  // A server that sends no timing or question count still reads: nothing known, nothing waiting.
  assert.deepEqual(
    [older.children[0]?.startedAt, older.children[0]?.finishedAt, older.children[0]?.pendingQuestions],
    [null, null, 0],
  );
  assert.equal(
    requests[1],
    'http://localhost/web/api/answer/run-1/children?cursor=opaque-token',
  );
});

test('child observation normalizes controls, questions, and the result', async () => {
  let requested = '';
  globalThis.fetch = async (input) => {
    requested = String(input);
    return new Response(JSON.stringify({
      run_id: 'run-1',
      child: {child_session_id: 'child-1', status: 'running', result_handles: ['ev-1']},
      controls: [{
        control_sequence: 3, kind: 'steer', content: 'focus', origin: 'user',
        consumed: false, consumed_at: null,
      }],
      questions: [{request_id: 'req-1', question: 'Which source?', status: 'pending'}],
      result: {status: 'running', summary: 'working', handles: ['ev-1']},
    }));
  };

  const observation = await getAnswerRunChild('run-1', 'child/1');

  assert.equal(requested, '/web/api/answer/run-1/children/child%2F1');
  assert.equal(observation.child.childSessionId, 'child-1');
  assert.deepEqual(observation.child.resultHandles, ['ev-1']);
  assert.equal(observation.controls[0]?.consumed, false);
  assert.equal(observation.questions[0]?.requestId, 'req-1');
  assert.deepEqual(observation.result?.handles, ['ev-1']);
});

test('a child states the message it answered with and the judgement the reader gave it, on its row and in its observation', async () => {
  const judged = {value: 'down', reasons: ['too_verbose'], note: null};
  globalThis.fetch = async (input) => new Response(JSON.stringify(
    String(input).endsWith('/children')
      ? {
        children: [
          {child_session_id: 'child-1', status: 'succeeded', result_sequence: 12, feedback: judged},
          {child_session_id: 'child-2', status: 'succeeded', result_sequence: 5, feedback: null},
          {child_session_id: 'child-3', status: 'running', result_sequence: null},
          // A row that states neither.
          {child_session_id: 'child-4', status: 'succeeded'},
        ],
        next_cursor: null,
        run_status: 'succeeded',
        run_started_at: null,
        run_finished_at: null,
      }
      : {
        run_id: 'run-1',
        child: {child_session_id: 'child-1', status: 'succeeded', result_sequence: 12, feedback: judged},
        controls: [],
        questions: [],
      },
  ));

  const page = await getAnswerRunChildrenPage('run-1');
  assert.deepEqual(page.children.map((child) => [child.resultSequence, child.feedback]), [
    [12, judged],
    [5, null],
    [null, null],
    [null, null],
  ]);
  const observed = await getAnswerRunChild('run-1', 'child-1');
  assert.deepEqual([observed.child.resultSequence, observed.child.feedback], [12, judged]);
});

test('an activity page names the agent and the cursor it reads, and hands back the cursor of the page before it', async () => {
  const requests: string[] = [];
  globalThis.fetch = async (input) => {
    requests.push(String(input));
    return new Response(JSON.stringify({
      messages: [
        {sequence: 7, role: 'user', content: 'inspect', at: '2026-10-05T12:00:00.000000Z'},
        {
          sequence: 9,
          role: 'assistant',
          content: '',
          tool_calls: [
            {id: 'call-1', name: 'search_knowledge_base', subject: 'volvo 2026 tariff exposure'},
            {},
          ],
        },
      ],
      next_before: 7,
      running: true,
    }));
  };

  const main = await getAnswerActivityPage('run-1', null);
  await getAnswerActivityPage('run/1', 'child/1', '7');

  assert.deepEqual(requests, [
    '/web/api/answer/run-1/transcript',
    '/web/api/answer/run%2F1/transcript?child=child%2F1&before=7',
  ]);
  assert.equal(main.nextCursor, '7');
  assert.equal(main.running, true);
  assert.equal(main.messages[0]?.sequence, 7);
  assert.equal(main.messages[0]?.at, '2026-10-05T12:00:00.000000Z');
  // A message from a server that states no clock, or a search hit, carries no moment at all.
  assert.equal(main.messages[1]?.at, null);
  assert.deepEqual(main.messages[1]?.toolCalls, [
    {id: 'call-1', name: 'search_knowledge_base', subject: 'volvo 2026 tariff exposure'},
    {id: '', name: '', subject: ''},
  ]);
});

test('the oldest activity page has no cursor', async () => {
  globalThis.fetch = async () => new Response(JSON.stringify({messages: [], next_before: null, running: false}));

  assert.equal((await getAnswerActivityPage('run-1', null)).nextCursor, null);
});

test('a rejected continuation keeps the server reason and its stable kind', async () => {
  globalThis.fetch = async () => new Response(JSON.stringify({
    kind: 'invalid_request',
    message: 'Current model does not support image input.',
    error_kind: 'CURRENT_IMAGES_UNSUPPORTED',
  }), {status: 422, headers: {'Content-Type': 'application/json'}});

  await assert.rejects(
    forkAnswerRun('parent-run', 'branch', 'submission-2'),
    (error: unknown) => error instanceof AnswerSubmissionError
      && error.status === 422
      && error.kind === 'invalid_request'
      && error.errorKind === 'CURRENT_IMAGES_UNSUPPORTED',
  );
});

test('an Artifact carries the attestation the server derived, and one derived none carries none', () => {
  const artifact = (extra: Record<string, unknown>) => ({
    resource_id: 'artifact-report',
    media_type: 'text/markdown',
    label: 'Report',
    filename: 'report.md',
    byte_size: 20,
    digest: 'a'.repeat(64),
    presentation: 'markdown',
    status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/artifact-report',
    width: null,
    height: null,
    data_url: null,
    download_url: null,
    presentation_url: null,
    issue: null,
    attestation: null,
    ...extra,
  });
  const presentation = (wire: Record<string, unknown>) => parseAnswerPresentation({
    answer_text: '',
    parts: [{type: 'artifact', text: '', html: '', artifact: wire, evidence_image: null, inline: true}],
    sources: [],
    evidence_images: [],
    artifacts: [wire],
    artifact_outcome: {status: 'complete', issues: []},
  });
  const attestationOf = (wire: Record<string, unknown>) =>
    presentation(artifact(wire))?.artifacts[0]?.attestation;

  // The record travels whole, beside the Artifact it judges, in the camelCase the browser model keeps.
  assert.deepEqual(
    attestationOf({
      attestation: {
        status: 'attested',
        skill: 'sealed-research-brief',
        verifier: 'scripts/validate_model.py',
        reason: null,
        basis: 'named_in_a_recorded_command',
      },
    }),
    {
      status: 'attested',
      skill: 'sealed-research-brief',
      verifier: 'scripts/validate_model.py',
      reason: null,
      basis: 'named_in_a_recorded_command',
      // A payload that states no reads reads as "the records could not settle it".
      requiredReadsMissing: null,
    },
  );

  // The reads a Run left unread travel whole, an empty list stays distinct from null, and a value
  // this client cannot read is the honest "cannot be told" rather than a reason to drop the Answer.
  const withReads = (required_reads_missing: unknown) =>
    attestationOf({
      attestation: {
        status: 'unattested', skill: 'interactive-html', verifier: null,
        reason: 'skill_has_no_verifier_contract', basis: null, required_reads_missing,
      },
    })?.requiredReadsMissing;
  const missing = {
    skill: 'interactive-html',
    paths: ['references/example-brief.html', 'references/example-dashboard.html'],
    when: 'before writing the fragment',
  };
  assert.deepEqual(withReads([missing]), [missing]);
  assert.deepEqual(withReads([]), []);
  assert.equal(withReads(null), null);
  assert.equal(withReads({skill: 'not a list'}), null);
  assert.equal(withReads([{skill: 'interactive-html', paths: 'references/x.md', when: 'x'}]), null);

  // A server that derived no record reads as "not derived", which is not a verdict and never
  // becomes one here.
  assert.equal(attestationOf({attestation: null}), null);

  // The reason vocabulary is the server's. A token this client has not met travels as itself, so
  // the badge can name it rather than take the whole Answer down.
  assert.equal(
    attestationOf({
      attestation: {
        status: 'unattested', skill: null, verifier: null,
        reason: 'a_reason_from_the_future', basis: null,
      },
    })?.reason,
    'a_reason_from_the_future',
  );

  // A status the contract does not name reads as "cannot be told": the honest answer about a
  // vocabulary this client cannot judge, and one the Artifact still parses past.
  assert.equal(
    attestationOf({
      attestation: {status: 'attested_by_a_human', skill: null, verifier: null, reason: null, basis: null},
    })?.status,
    'unknown',
  );
});

test('an artifact\'s text is read from this origin, a preview asks for its first bytes only, and a refusal is typed', async () => {
  const originalWindow = globalThis.window;
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {location: {origin: 'https://agent.example'}},
  });
  try {
    const seen: Array<{url: string; range: string | null}> = [];
    globalThis.fetch = async (input, init) => {
      seen.push({url: String(input), range: new Headers(init?.headers).get('Range')});
      return new Response('# Report');
    };

    assert.equal(await getArtifactTextAt('https://agent.example/web/api/answer/run-1/artifacts/a?x=1'), '# Report');
    assert.equal(
      await getArtifactTextAt('/web/api/answer/run-1/artifacts/a', {firstBytes: 1024}),
      '# Report',
    );
    assert.deepEqual(seen, [
      {url: '/web/api/answer/run-1/artifacts/a?x=1', range: null},
      {url: '/web/api/answer/run-1/artifacts/a', range: 'bytes=0-1023'},
    ]);

    await assert.rejects(
      getArtifactTextAt('https://elsewhere.example/web/api/answer/run-1/artifacts/a'),
      /Untrusted artifact URL/,
    );
    assert.equal(seen.length, 2, 'a route on another origin was fetched');

    globalThis.fetch = async () => Response.json(
      {detail: 'Artifact expired', error_type: 'not_found'},
      {status: 404},
    );
    await assert.rejects(
      getArtifactTextAt('/web/api/answer/run-1/artifacts/a'),
      (error: unknown) => error instanceof ApiError && error.status === 404 && error.detail === 'Artifact expired',
    );
  } finally {
    Object.defineProperty(globalThis, 'window', {configurable: true, value: originalWindow});
  }
});
