// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What the Child agents tests share: one Run's children as the server puts them on the wire, a
 *  `fetch` that answers the Run's routes, and the source the Shell hands the dock, built on the
 *  product's own api calls. */

import type {ChatTurnView} from '../lib/chat-views.ts';
import {runTracesSource, type TracesSource} from '../stores/traces-source.ts';

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

/** One page of the roster; `runStatus` is where the Run stands and `clock` the stamps of its own row. */
export function roster(
  children: Record<string, unknown>[],
  nextCursor: string | null = null,
  runStatus = 'running',
  clock: {startedAt?: string | null; finishedAt?: string | null} = {},
): Response {
  return Response.json({
    run_id: 'run-1', children, next_cursor: nextCursor,
    // The Run's own row, stated the way each child row below it is.
    run_status: runStatus,
    run_started_at: clock.startedAt ?? null,
    run_finished_at: clock.finishedAt ?? null,
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
  transcript?: (
    agent: string | null,
    before: string | null,
    since: number | null,
  ) => Response | Promise<Response> | undefined;
  /** The messages of every agent that hold the reader's words; answering nothing is an empty search. */
  search?: (text: string) => Response | Promise<Response> | undefined;
  /** The Run's whole-Run overview; answering nothing is a Run that states no steps yet. */
  overview?: () => Response | Promise<Response> | undefined;
  /** One page of the Run-wide step search; answering nothing is an empty page of matches. */
  stepSearch?: (text: string, before: string | null) => Response | Promise<Response> | undefined;
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

/** One search answer as the route states it: the messages holding the words, each naming its agent. */
export function searchPage(
  messages: Record<string, unknown>[],
  extra: Record<string, unknown> = {},
): Response {
  return Response.json({
    messages: messages.map((message, index) => ({sequence: index + 1, ...message})),
    terms: [],
    total: messages.length,
    match: 'all',
    ...extra,
  });
}

/** One Run's whole-Run overview as the route states it: its totals, its agents in roster order, its
 *  capped findings, and the token record it states. The default Run is one main agent, three steps
 *  and one failed Tool call; a test overrides the facts it reads. */
export function overview(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    run_id: 'run-1',
    totals: {
      steps: 3,
      kinds: {thinking: 0, tool: 2, say: 0, answer: 1, instruction: 0, delivery: 0},
      decisions: 1,
      marks: {failed: 1, degraded: 0, repeated: 0, redirect: 0},
      tool_time_ms: 1200,
    },
    agents: [
      {
        agent: 'main', label: 'main', order: 0, steps: 3, tool_calls: 2,
        marks: {failed: 1, degraded: 0, repeated: 0, redirect: 0},
        tool_time_ms: 1200,
        first_at: '2026-10-05T12:00:00Z', last_at: '2026-10-05T12:01:00Z',
      },
    ],
    findings: [
      {
        agent: 'main', sequence: 2, kind: 'tool', at: '2026-10-05T12:00:20Z',
        name: 'read', subject: 'northwind-msa.pdf', outcome: 'failed',
        marks: ['failed'], excerpt: 'The file could not be read.',
      },
    ],
    findings_total: 1,
    usage: {
      usage_details: {total_tokens: 9000},
      child_usage_details: {total_tokens: 0},
      inclusive_usage_details: {total_tokens: 9000},
    },
    ...extra,
  };
}

/** An overview for a Run that states nothing yet: the read that lands before the first step. */
function emptyOverview(runId: string): Record<string, unknown> {
  return overview({
    run_id: runId,
    totals: {
      steps: 0,
      kinds: {thinking: 0, tool: 0, say: 0, answer: 0, instruction: 0, delivery: 0},
      decisions: 0,
      marks: {failed: 0, degraded: 0, repeated: 0, redirect: 0},
      tool_time_ms: 0,
    },
    agents: [],
    findings: [],
    findings_total: 0,
    usage: null,
  });
}

/** One page of the Run-wide step search as its route states it: the matching steps, oldest first
 *  within the page, and the cursor the page before it is read from. A step states only the fields a
 *  test reads. `sequence` identifies the row, and the thinking steps the route excludes never appear. */
export function stepSearchPage(
  steps: Record<string, unknown>[],
  extra: Record<string, unknown> = {},
): Response {
  return Response.json({
    run_id: 'run-1',
    steps: steps.map((step, index) => ({
      agent: 'main',
      agent_label: 'main',
      at: '2026-10-20T10:00:00Z',
      kind: 'tool',
      sequence: index + 1,
      name: 'read',
      subject: null,
      outcome: 'succeeded',
      duration_ms: null,
      excerpt: '',
      marks: [],
      ...step,
    })),
    next_before: null,
    matched: steps.length,
    terms: [],
    match: 'all',
    ...extra,
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
    const child = new RegExp(`^${base}/children/([^/]+)$`).exec(url.pathname);
    if (url.pathname === `${base}/children` && routes.page) {
      return routes.page(url.searchParams.get('cursor'));
    }
    const overviewRoute = /^\/web\/api\/answer\/([^/]+)\/overview$/.exec(url.pathname);
    if (overviewRoute) {
      const called = routes.overview?.();
      return called ?? Response.json(emptyOverview(decodeURIComponent(overviewRoute[1]!)));
    }
    const stepSearchRoute = /^\/web\/api\/answer\/([^/]+)\/steps\/search$/.exec(url.pathname);
    if (stepSearchRoute) {
      const called = routes.stepSearch?.(url.searchParams.get('q') ?? '', url.searchParams.get('before'));
      return called ?? stepSearchPage([], {run_id: decodeURIComponent(stepSearchRoute[1]!)});
    }
    const searchRoute = new RegExp(`^${base}/transcript/search$`).exec(url.pathname);
    if (searchRoute) {
      const served = routes.search?.(url.searchParams.get('q') ?? '');
      return served ?? searchPage([]);
    }
    if (url.pathname === `${base}/transcript`) {
      const agent = url.searchParams.get('child');
      const rawSince = url.searchParams.get('since');
      const since = rawSince === null ? null : Number(rawSince);
      const custom = routes.transcript?.(agent, url.searchParams.get('before'), since);
      if (custom !== undefined) return custom;
      const observed = agent === null ? undefined : transcripts.get(agent);
      const held = observed?.messages ?? [];
      // A delta answers with what the reader does not hold yet, exactly as the route does.
      const answered = since === null
        ? held
        : held.filter((message) => Number(message.sequence ?? 0) > since);
      return activityPage(answered, {running: observed?.running ?? false});
    }
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
    streamText: '', processEntries: [],
    presentation: null, usage: {}, error: null, activity: null,
    cancelRequested: false, steeringMessages: [], toolRows: [], answerSequence: null, feedback: null, ...extra,
  };
}

/** What the Shell hands the dock for one Run: the product's own source, with no search, which the
 *  fixtures do not serve. The Run's turn is what the main agent's row is drawn from: a test that leaves
 *  it out has a main agent that is still running, with no answer. */
export function sourceFor(runId = 'run-1', turn?: ChatTurnView) {
  const source: TracesSource = runTracesSource(runId, {turn: () => turn});
  return {source};
}
