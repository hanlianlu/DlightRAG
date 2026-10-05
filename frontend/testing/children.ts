// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What the Child agents tests share: one Run's children as the server puts them on the wire, a
 *  `fetch` that answers the Run's routes, and the source the Shell hands the dock, built on the
 *  product's own api calls. */

import {
  controlAnswerChild,
  getAnswerRunChild,
  getAnswerRunChildrenPage,
  replyAnswerChild,
} from '../api/conversations.ts';
import type {ChildrenSource} from '../ui/inspector-children.ts';

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

export function roster(children: Record<string, unknown>[], nextCursor: string | null = null): Response {
  return Response.json({run_id: 'run-1', children, next_cursor: nextCursor});
}

export function observation(child: Record<string, unknown>, extra: Record<string, unknown> = {}): Response {
  return Response.json({
    run_id: 'run-1', child, transcript: [], controls: [], questions: [], result: null, ...extra,
  });
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
  control?: (id: string, body: Record<string, unknown>) => Response | Promise<Response>;
  reply?: (requestId: string, body: Record<string, unknown>) => Response | Promise<Response>;
}

/** Answer the requests of Run `run-1` from `routes`, and keep the ones that came. The caller
 *  restores `window.fetch`. */
export function serve(routes: Routes): Served[] {
  const requests: Served[] = [];
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
    if (control && routes.control) return routes.control(decodeURIComponent(control[1]!), body ?? {});
    if (reply && routes.reply) return routes.reply(decodeURIComponent(reply[1]!), body ?? {});
    if (child && routes.observe) return routes.observe(decodeURIComponent(child[1]!));
    return new Response('unexpected request', {status: 500});
  };
  return requests;
}

/** What the Shell hands the dock for one Run: the product's api calls, with the arguments the dock
 *  passes them kept for the test. */
export function sourceFor(runId = 'run-1') {
  const controls: Parameters<ChildrenSource['control']>[] = [];
  const replies: Parameters<ChildrenSource['reply']>[] = [];
  const source: ChildrenSource = {
    runId,
    page: (cursor, signal) => getAnswerRunChildrenPage(runId, cursor, signal),
    observe: (id, signal) => getAnswerRunChild(runId, id, signal),
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
