// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What the Child agents tests share: one Run's children as the server puts them on the wire, a
 *  `fetch` that answers the Run's routes, and the source the Shell hands the dock, built on the
 *  product's own api calls. */

import {
  controlAnswerChild,
  getAnswerActivityPage,
  getAnswerRunChild,
  getAnswerRunChildrenPage,
  replyAnswerChild,
} from '../api/conversations.ts';
import type {ChatTurnView} from '../lib/chat-views.ts';
import {mainAgentStatus} from '../lib/main-agent.ts';
import type {TracesSource} from '../ui/inspector-traces.ts';

/** The clock the fixtures are drawn against: a test that shows elapsed times fixes `Date.now` to it. */
export const NOW = Date.parse('2026-10-05T12:10:00Z');

export function ago(seconds: number): string {
  return new Date(NOW - seconds * 1000).toISOString();
}

/** One child's status row, roster and observation alike. */
export function row(id: string, status = 'succeeded', extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    child_session_id: id, status, objective: `objective ${id}`, model_role: 'query', usage: null,
    operation_id: `op-${id}`, operation_sequence: 1, operation_status: status,
    cancellation_origin: null, summary: null, result_handles: [],
    started_at: null, finished_at: null, pending_questions: 0, ...extra,
  };
}

/** One page of the roster; `runStatus` is where the Run stands, left out as an older server leaves it out. */
export function roster(
  children: Record<string, unknown>[],
  nextCursor: string | null = null,
  runStatus: string | null = null,
): Response {
  return Response.json({
    run_id: 'run-1', children, next_cursor: nextCursor, ...(runStatus === null ? {} : {run_status: runStatus}),
  });
}

/** What each child's latest observation fixture said of its transcript. The transcript route serves it as one
 *  page, so a fixture can still describe a child in one place. */
const transcripts = new Map<string, {messages: Record<string, unknown>[]; running: boolean}>();

/** `transcript` in `extra` is the child's transcript, oldest first; it reaches the page route, not the wire
 *  of the observation. */
export function observation(child: Record<string, unknown>, extra: Record<string, unknown> = {}): Response {
  const {transcript, ...rest} = extra;
  transcripts.set(String(child.child_session_id), {
    messages: (transcript as Record<string, unknown>[] | undefined) ?? [],
    running: child.status === 'running',
  });
  return Response.json({run_id: 'run-1', child, controls: [], questions: [], result: null, ...rest});
}

export function receipt(action: string, outcome: string, extra: Record<string, unknown> = {}): Response {
  return Response.json({
    run_id: 'run-1', child_session_id: 'a', action, outcome, operation_id: 'op-a',
    operation_sequence: 1, control_sequence: 1, consumed_at: null, ...extra,
  }, {status: 202});
}

/** A command the server refuses, naming the outcome it met. */
export function refusal(detail: string, status = 409): Response {
  return Response.json({detail}, {status});
}

/** A question to the parent that still has four minutes to wait. */
export function question(requestId: string, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    request_id: requestId, question: `Question ${requestId}?`, status: 'pending', reply: null,
    reply_origin: null, expires_at: new Date(NOW + 4 * 60_000 - 1000).toISOString(),
    created_at: ago(60), ...extra,
  };
}

interface Served {
  method: string;
  path: string;
  search: string;
  body: Record<string, unknown> | undefined;
  headers: Headers;
}

interface Routes {
  page?: (cursor: string | null) => Response | Promise<Response>;
  observe?: (id: string) => Response | Promise<Response>;
  /** One page of an agent's transcript; the main agent when `agent` is null. Answering nothing leaves the
   *  page to the child's observation fixture, which answers with its whole transcript. */
  transcript?: (agent: string | null, before: string | null) => Response | Promise<Response> | undefined;
  control?: (id: string, body: Record<string, unknown>) => Response | Promise<Response>;
  reply?: (requestId: string, body: Record<string, unknown>) => Response | Promise<Response>;
}

/** One page of a transcript as the server puts it on the wire; messages are numbered from 1 unless they
 *  carry a sequence. */
export function activityPage(
  messages: Record<string, unknown>[],
  {nextBefore = null, running = false}: {nextBefore?: number | null; running?: boolean} = {},
): Response {
  return Response.json({
    messages: messages.map((message, index) => ({sequence: index + 1, ...message})),
    next_before: nextBefore,
    running,
  });
}

/** Answer the requests of Run `run-1` from `routes`, and keep the ones that came. The caller
 *  restores `window.fetch`. */
export function serve(routes: Routes): Served[] {
  const requests: Served[] = [];
  transcripts.clear();
  window.fetch = async (input, init) => {
    const url = new URL(String(input), window.location.origin);
    const body = typeof init?.body === 'string' ? JSON.parse(init.body) : undefined;
    requests.push({
      method: init?.method ?? 'GET', path: url.pathname, search: url.search, body,
      headers: new Headers(init?.headers),
    });
    const base = '/web/api/answer/run-1';
    const control = new RegExp(`^${base}/children/([^/]+)/control$`).exec(url.pathname);
    const reply = new RegExp(`^${base}/child-guidance/([^/]+)/reply$`).exec(url.pathname);
    const child = new RegExp(`^${base}/children/([^/]+)$`).exec(url.pathname);
    if (url.pathname === `${base}/children` && routes.page) {
      return routes.page(url.searchParams.get('cursor'));
    }
    if (url.pathname === `${base}/transcript`) {
      const agent = url.searchParams.get('child');
      const custom = routes.transcript?.(agent, url.searchParams.get('before'));
      if (custom !== undefined) return custom;
      const observed = agent === null ? undefined : transcripts.get(agent);
      return activityPage(observed?.messages ?? [], {running: observed?.running ?? false});
    }
    if (control && routes.control) return routes.control(decodeURIComponent(control[1]!), body ?? {});
    if (reply && routes.reply) return routes.reply(decodeURIComponent(reply[1]!), body ?? {});
    if (child && routes.observe) return routes.observe(decodeURIComponent(child[1]!));
    return new Response('unexpected request', {status: 500});
  };
  return requests;
}

/** The turn a Run answers, as the chat holds it: settled, with the question that was asked and no answer
 *  yet unless a test gives one. */
export function answeredTurn(extra: Partial<ChatTurnView> = {}): ChatTurnView {
  return {
    id: 'turn-1', userText: 'What changed?', userAttachments: [], runId: 'run-1', state: 'succeeded',
    streamText: '', presentation: null, usage: {}, error: '', progress: '', liveStatus: '',
    sawChildren: false, cancelRequested: false, steeringMessages: [], toolRows: [], ...extra,
  };
}

/** What the Shell hands the dock for one Run: the product's api calls, with the arguments the dock
 *  passes them kept for the test. The Run's turn is what the main agent's row is drawn from: a test that
 *  leaves it out has a main agent that is still running, with no answer. */
export function sourceFor(runId = 'run-1', turn?: ChatTurnView) {
  const controls: Parameters<TracesSource['control']>[] = [];
  const replies: Parameters<TracesSource['reply']>[] = [];
  const source: TracesSource = {
    runId,
    mainAgent: () => mainAgentStatus(turn),
    presentation: () => turn?.presentation ?? null,
    page: (cursor, signal) => getAnswerRunChildrenPage(runId, cursor, signal),
    observe: (id, signal) => getAnswerRunChild(runId, id, signal),
    activity: (agent, cursor, signal) => getAnswerActivityPage(runId, agent, cursor, signal),
    control: (id, action, content, reauthorize, operationId, signal) => {
      controls.push([id, action, content, reauthorize, operationId, signal]);
      return controlAnswerChild(runId, id, action, content, crypto.randomUUID(), reauthorize, signal);
    },
    reply: (requestId, content, signal) => {
      replies.push([requestId, content, signal]);
      return replyAnswerChild(runId, requestId, content, crypto.randomUUID(), signal);
    },
  };
  return {source, controls, replies};
}
