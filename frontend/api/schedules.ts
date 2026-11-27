// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Web API client for the owner's Schedules: the work the clock starts (ADR 0103). */

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {apiError, parseWire, timestamp} from './wire.ts';

export type ScheduleMode = 'research' | 'fast';
export type ScheduleTarget = 'new_conversation' | 'same_conversation';
export type ScheduleState = 'active' | 'paused' | 'completed';

const scheduleSchema = v.pipe(
  v.object({
    schedule_id: v.string(),
    name: v.string(),
    prompt: v.string(),
    mode: v.picklist(['research', 'fast']),
    workspaces: v.array(v.string()),
    target: v.picklist(['new_conversation', 'same_conversation']),
    conversation_id: v.nullable(v.string()),
    zone: v.string(),
    // A recurrence is `cron`, a single time is `at`; the other is null.
    cron: v.nullable(v.string()),
    at: v.nullable(timestamp),
    max_tokens_per_day: v.number(),
    goal_max_rounds: v.nullable(v.number()),
    state: v.picklist(['active', 'paused', 'completed']),
    pause_reason: v.nullable(v.string()),
    next_fire_at: v.nullable(timestamp),
  }),
  v.transform((w) => ({
    scheduleId: w.schedule_id,
    name: w.name,
    prompt: w.prompt,
    mode: w.mode,
    workspaces: w.workspaces,
    target: w.target,
    conversationId: w.conversation_id,
    zone: w.zone,
    cron: w.cron,
    at: w.at,
    maxTokensPerDay: w.max_tokens_per_day,
    goalMaxRounds: w.goal_max_rounds,
    state: w.state,
    pauseReason: w.pause_reason,
    nextFireAt: w.next_fire_at,
  })),
);
export type Schedule = v.InferOutput<typeof scheduleSchema>;

const scheduleList = v.pipe(
  v.object({schedules: v.array(scheduleSchema), default_max_tokens_per_day: v.number()}),
  v.transform((w) => ({schedules: w.schedules, defaultMaxTokensPerDay: w.default_max_tokens_per_day})),
);

const fireSchema = v.pipe(
  v.object({
    fire_id: v.string(),
    slot: timestamp,
    state: v.picklist([
      'pending',
      'started',
      'succeeded',
      'failed',
      'cancelled',
      'refused',
      'skipped_busy',
      'skipped_limit',
      'missed',
    ]),
    reason: v.nullable(v.string()),
    conversation_id: v.nullable(v.string()),
    missed_count: v.nullable(v.number()),
    manual: v.boolean(),
    last_error: v.nullable(v.string()),
  }),
  v.transform((w) => ({
    fireId: w.fire_id,
    slot: w.slot,
    state: w.state,
    conversationId: w.conversation_id,
    missedCount: w.missed_count,
    manual: w.manual,
    // A pending Fire has no reason yet; what stopped its latest attempt is the reason to show.
    reason: w.reason ?? w.last_error,
  })),
);
export type ScheduleFire = v.InferOutput<typeof fireSchema>;

const firesPage = v.pipe(
  v.object({fires: v.array(fireSchema), next_cursor: v.nullable(v.string())}),
  v.transform((w) => ({items: w.fires, nextCursor: w.next_cursor})),
);

const preview = v.pipe(
  v.object({slots: v.array(v.object({at: timestamp}))}),
  v.transform((w) => w.slots.map((slot) => slot.at)),
);

/** When a Schedule fires: a recurrence (five cron fields) or one instant, read in an IANA zone. `at` carries
 *  its UTC offset. */
interface CadenceFields {
  readonly zone: string;
  readonly cron?: string;
  readonly at?: string;
}

export interface NewSchedule extends CadenceFields {
  readonly name: string;
  readonly prompt: string;
  readonly mode: ScheduleMode;
  readonly workspaces: readonly string[];
  readonly target: ScheduleTarget;
  readonly maxTokensPerDay: number;
  /** Present when each fire starts a goal whose objective is the prompt. */
  readonly goalMaxRounds?: number;
}

/** The parts of a Schedule to change; what is left out stays. A new `cron` or `at` replaces the cadence. */
export interface ScheduleChanges {
  readonly name?: string;
  readonly prompt?: string;
  readonly mode?: ScheduleMode;
  readonly workspaces?: readonly string[];
  readonly zone?: string;
  readonly cron?: string;
  readonly at?: string;
  readonly maxTokensPerDay?: number;
  readonly goalMaxRounds?: number;
}

const BASE = '/web/api/schedules';

function wireBody(fields: NewSchedule | ScheduleChanges): Record<string, unknown> {
  const {maxTokensPerDay, goalMaxRounds, ...rest} = fields;
  return {...rest, max_tokens_per_day: maxTokensPerDay, goal_max_rounds: goalMaxRounds};
}

function send(method: string, path: string, body: unknown, signal?: AbortSignal): Promise<Response> {
  return fetch(path, {
    method,
    headers: csrfHeaders(body === undefined ? undefined : 'application/json'),
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
}

function path(scheduleId: string, ...rest: string[]): string {
  return [BASE, encodeURIComponent(scheduleId), ...rest].join('/');
}

export async function listSchedules(signal?: AbortSignal): Promise<v.InferOutput<typeof scheduleList>> {
  return parseWire(await fetch(BASE, {signal}), scheduleList);
}

export async function createSchedule(fields: NewSchedule, signal?: AbortSignal): Promise<Schedule> {
  return parseWire(await send('POST', BASE, wireBody(fields), signal), scheduleSchema);
}

export async function updateSchedule(
  scheduleId: string,
  changes: ScheduleChanges,
  signal?: AbortSignal,
): Promise<Schedule> {
  return parseWire(await send('PATCH', path(scheduleId), wireBody(changes), signal), scheduleSchema);
}

/** Delete a Schedule and its Fire history; a 404 means it is already gone. */
export async function deleteSchedule(scheduleId: string, signal?: AbortSignal): Promise<void> {
  const response = await send('DELETE', path(scheduleId), undefined, signal);
  if (!response.ok) throw await apiError(response);
}

export async function pauseSchedule(scheduleId: string, signal?: AbortSignal): Promise<Schedule> {
  return parseWire(await send('POST', path(scheduleId, 'pause'), undefined, signal), scheduleSchema);
}

export async function resumeSchedule(scheduleId: string, signal?: AbortSignal): Promise<Schedule> {
  return parseWire(await send('POST', path(scheduleId, 'resume'), undefined, signal), scheduleSchema);
}

/** Fire once now, under every rule a clock fire meets; the Fire says what became of it. */
export async function runScheduleNow(scheduleId: string, signal?: AbortSignal): Promise<ScheduleFire> {
  return parseWire(await send('POST', path(scheduleId, 'run'), undefined, signal), fireSchema);
}

/** One page of a Schedule's Fires, newest slot first. */
export async function listScheduleFires(
  scheduleId: string,
  cursor: string | null,
  signal?: AbortSignal,
): Promise<v.InferOutput<typeof firesPage>> {
  const query = new URLSearchParams({limit: '20'});
  if (cursor) query.set('cursor', cursor);
  return parseWire(await fetch(`${path(scheduleId, 'fires')}?${query}`, {signal}), firesPage);
}

/** The next slots of a cadence as saving it would read them, or the refusal saving it would meet (an
 *  `ApiError` whose `detail` is the reason). Nothing is stored. */
export async function previewCadence(cadence: CadenceFields, signal?: AbortSignal): Promise<readonly Date[]> {
  return parseWire(await send('POST', `${BASE}/preview`, cadence, signal), preview);
}
