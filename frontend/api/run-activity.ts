// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The reader's recent conversation-linked Answer runs, as the Web reports them.

 * One bounded page, newest first: the page polls it to say which conversation is running and
 * which has an answer the reader has not opened yet. Read-only: nothing here accepts an action.
 * The wire format and the item's fields are frozen in docs/research/background-callbacks/p0-design.md.
 * `goalRound` states that the Run is a goal round's continuation, which is activity like any other
 * but not a completion the page should announce; the unread marker stays.
 */

import * as v from 'valibot';
import {parseWire} from './wire.ts';

/** The statuses a Run passes through, as the server names them. */
const RUN_STATUSES = ['queued', 'running', 'succeeded', 'failed', 'cancelled'] as const;

const runActivity = v.pipe(
  v.object({
    run_id: v.string(),
    conversation_id: v.string(),
    conversation_title: v.nullable(v.string()),
    status: v.picklist(RUN_STATUSES),
    phase: v.nullable(v.string()),
    created_at: v.string(),
    started_at: v.nullable(v.string()),
    finished_at: v.nullable(v.string()),
    goal_round: v.boolean(),
  }),
  v.transform((wire) => ({
    runId: wire.run_id,
    conversationId: wire.conversation_id,
    conversationTitle: wire.conversation_title,
    status: wire.status,
    phase: wire.phase,
    createdAt: wire.created_at,
    startedAt: wire.started_at,
    finishedAt: wire.finished_at,
    goalRound: wire.goal_round,
  })),
);
export type RunActivity = v.InferOutput<typeof runActivity>;
export type RunStatus = RunActivity['status'];

const runActivityPage = v.pipe(
  v.object({items: v.array(runActivity)}),
  v.transform((wire) => ({items: wire.items})),
);
export type RunActivityPage = v.InferOutput<typeof runActivityPage>;

/** Where the reader's Run activity is read. */
const RUN_ACTIVITY_PATH = '/web/api/run-activity';

/** The newest runs one read asks for: the route's default page. */
export const RUN_ACTIVITY_LIMIT = 30;

/** Whether the Run is still under way. */
export function runActive(status: RunStatus): boolean {
  return status === 'queued' || status === 'running';
}

/** Whether the Run has reached a status it will not leave. */
export function runTerminal(status: RunStatus): boolean {
  return status === 'succeeded' || status === 'failed' || status === 'cancelled';
}

export async function listRunActivity(
  limit = RUN_ACTIVITY_LIMIT,
  signal?: AbortSignal,
): Promise<RunActivityPage> {
  const response = await fetch(`${RUN_ACTIVITY_PATH}?limit=${limit}`, {signal});
  return parseWire(response, runActivityPage);
}
