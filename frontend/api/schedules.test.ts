// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import test from 'node:test';
import {
  createSchedule,
  deleteSchedule,
  listScheduleFires,
  listSchedules,
  pauseSchedule,
  previewCadence,
  runScheduleNow,
  updateSchedule,
} from './schedules.ts';
import {ApiError} from './wire.ts';

const originalFetch = globalThis.fetch;
const originalDocument = globalThis.document;
test.beforeEach(() => {
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: {cookie: 'dlightrag_web_csrf=csrf-token'},
  });
});
test.afterEach(() => {
  globalThis.fetch = originalFetch;
  Object.defineProperty(globalThis, 'document', {configurable: true, value: originalDocument});
});

const wireSchedule = {
  schedule_id: 'schedule/1',
  name: 'Weekly review',
  prompt: 'Summarize the week.',
  mode: 'research',
  workspaces: ['alpha'],
  target: 'new_conversation',
  conversation_id: null,
  zone: 'Europe/Stockholm',
  cron: '0 8 * * 1',
  at: null,
  max_tokens_per_day: 2_000_000,
  goal_max_rounds: null,
  state: 'active',
  pause_reason: null,
  changed_by: 'owner',
  next_fire_at: '2026-12-14T07:00:00.000000Z',
  created_at: '2026-12-07T10:00:20.000000Z',
  updated_at: '2026-12-07T10:00:20.000000Z',
};

interface Sent {
  url: string;
  init: RequestInit | undefined;
}

function answering(reply: unknown, status = 200): Sent[] {
  const sent: Sent[] = [];
  globalThis.fetch = async (url, init) => {
    sent.push({url: String(url), init});
    return Response.json(reply, {status});
  };
  return sent;
}

test('the list keeps its times as dates, its recurrence or single time as stated, and carries the default ceiling', async () => {
  answering({
    schedules: [wireSchedule, {...wireSchedule, schedule_id: 's2', cron: null, at: '2099-06-01T06:00:00.000000Z', state: 'paused', pause_reason: 'paused by its owner', changed_by: 'agent'}],
    default_max_tokens_per_day: 1_500_000,
  });

  const {schedules, defaultMaxTokensPerDay} = await listSchedules();

  assert.equal(defaultMaxTokensPerDay, 1_500_000);
  assert.deepEqual(schedules[0], {
    scheduleId: 'schedule/1',
    name: 'Weekly review',
    prompt: 'Summarize the week.',
    mode: 'research',
    workspaces: ['alpha'],
    target: 'new_conversation',
    conversationId: null,
    zone: 'Europe/Stockholm',
    cron: '0 8 * * 1',
    at: null,
    maxTokensPerDay: 2_000_000,
    goalMaxRounds: null,
    state: 'active',
    pauseReason: null,
    changedBy: 'owner',
    nextFireAt: new Date('2026-12-14T07:00:00Z'),
  });
  assert.equal(schedules[1].cron, null);
  assert.deepEqual(schedules[1].at, new Date('2099-06-01T06:00:00Z'));
  assert.equal(schedules[1].pauseReason, 'paused by its owner');
  assert.equal(schedules[1].changedBy, 'agent');
});

test('a Schedule this client cannot interpret fails the read instead of being guessed', async () => {
  answering({schedules: [{...wireSchedule, state: 'sleeping'}], default_max_tokens_per_day: 1});
  await assert.rejects(listSchedules(), ApiError);
  answering({schedules: [{...wireSchedule, next_fire_at: 'Monday'}], default_max_tokens_per_day: 1});
  await assert.rejects(listSchedules(), ApiError);
  answering({schedules: [{...wireSchedule, changed_by: 'someone'}], default_max_tokens_per_day: 1});
  await assert.rejects(listSchedules(), ApiError);
});

test('a new Schedule is one POST of snake_case JSON with the CSRF token, and answers the Schedule', async () => {
  const sent = answering(wireSchedule, 201);

  const created = await createSchedule({
    name: 'Weekly review',
    prompt: 'Summarize the week.',
    mode: 'research',
    workspaces: ['alpha'],
    target: 'same_conversation',
    zone: 'Europe/Stockholm',
    cron: '0 8 * * 1',
    maxTokensPerDay: 500,
    goalMaxRounds: 4,
  });

  assert.equal(created.scheduleId, 'schedule/1');
  assert.equal(sent[0].url, '/web/api/schedules');
  assert.equal(sent[0].init?.method, 'POST');
  const headers = new Headers(sent[0].init?.headers);
  assert.equal(headers.get('X-CSRF-Token'), 'csrf-token');
  assert.equal(headers.get('Content-Type'), 'application/json');
  assert.deepEqual(JSON.parse(String(sent[0].init?.body)), {
    name: 'Weekly review',
    prompt: 'Summarize the week.',
    mode: 'research',
    workspaces: ['alpha'],
    target: 'same_conversation',
    zone: 'Europe/Stockholm',
    cron: '0 8 * * 1',
    max_tokens_per_day: 500,
    goal_max_rounds: 4,
  });
});

test('an edit names only what changed, and an id that needs escaping is escaped', async () => {
  const sent = answering(wireSchedule);

  await updateSchedule('schedule/1', {name: 'Monday review', maxTokensPerDay: 10});

  assert.equal(sent[0].url, '/web/api/schedules/schedule%2F1');
  assert.equal(sent[0].init?.method, 'PATCH');
  assert.deepEqual(JSON.parse(String(sent[0].init?.body)), {name: 'Monday review', max_tokens_per_day: 10});
});

test('pausing and deleting are bodiless writes that carry the CSRF token, and a refusal keeps the server\'s reason', async () => {
  const sent = answering(wireSchedule);
  await pauseSchedule('s1');
  assert.deepEqual([sent[0].url, sent[0].init?.method, sent[0].init?.body], ['/web/api/schedules/s1/pause', 'POST', undefined]);
  assert.equal(new Headers(sent[0].init?.headers).get('X-CSRF-Token'), 'csrf-token');

  globalThis.fetch = async () => new Response(null, {status: 204});
  await deleteSchedule('s1');

  globalThis.fetch = async () => Response.json(
    {detail: 'a completed Schedule has nothing left to pause', error_type: 'conflict'},
    {status: 409},
  );
  await assert.rejects(pauseSchedule('s1'), (error: unknown) => (
    error instanceof ApiError && error.status === 409 && error.detail === 'a completed Schedule has nothing left to pause'
  ));
  globalThis.fetch = async () => Response.json({detail: 'Schedule not found', error_type: 'not_found'}, {status: 404});
  await assert.rejects(deleteSchedule('s1'), (error: unknown) => error instanceof ApiError && error.status === 404);
});

test('running now answers the Fire, whose state says what became of it', async () => {
  const sent = answering({
    fire_id: 'f1',
    schedule_id: 's1',
    slot: '2026-12-07T10:00:20.000000Z',
    state: 'refused',
    reason: 'its owner can no longer read any of its workspaces',
    run_id: null,
    conversation_id: null,
    missed_count: null,
    last_slot: null,
    manual: true,
    last_error: null,
  });

  const fire = await runScheduleNow('s1');

  assert.equal(sent[0].url, '/web/api/schedules/s1/run');
  assert.deepEqual(fire, {
    fireId: 'f1',
    slot: new Date('2026-12-07T10:00:20Z'),
    state: 'refused',
    reason: 'its owner can no longer read any of its workspaces',
    conversationId: null,
    missedCount: null,
    manual: true,
  });
});

test('a page of Fires is asked for by its cursor and answers the cursor of the next', async () => {
  const sent = answering({
    fires: [{
      fire_id: 'f2',
      slot: '2026-12-14T07:00:00.000000Z',
      state: 'succeeded',
      reason: null,
      conversation_id: 'c1',
      missed_count: null,
      manual: false,
      last_error: null,
    }],
    next_cursor: '2026-12-14T07:00:00.000000Z',
  });

  const page = await listScheduleFires('s1', '2026-12-21T07:00:00.000000Z');

  assert.equal(sent[0].url, '/web/api/schedules/s1/fires?limit=20&cursor=2026-12-21T07%3A00%3A00.000000Z');
  assert.equal(page.items[0].conversationId, 'c1');
  assert.equal(page.nextCursor, '2026-12-14T07:00:00.000000Z');
});

test('a preview answers the slots as dates, and a cadence saving would refuse is the refusal with its reason', async () => {
  const sent = answering({
    zone: 'UTC',
    slots: [
      {at: '2026-12-08T08:00:00.000000Z', local: '2026-12-08T08:00:00+00:00'},
      {at: '2026-12-15T08:00:00.000000Z', local: '2026-12-15T08:00:00+00:00'},
    ],
  });

  const slots = await previewCadence({zone: 'UTC', cron: '0 8 * * 2'});

  assert.deepEqual(slots, [new Date('2026-12-08T08:00:00Z'), new Date('2026-12-15T08:00:00Z')]);
  assert.deepEqual(JSON.parse(String(sent[0].init?.body)), {zone: 'UTC', cron: '0 8 * * 2'});

  globalThis.fetch = async () => Response.json(
    {detail: 'slots may be at least 60 minutes apart', error_type: 'validation'},
    {status: 422},
  );
  await assert.rejects(previewCadence({zone: 'UTC', cron: '* * * * *'}), (error: unknown) => (
    error instanceof ApiError && error.status === 422 && error.detail === 'slots may be at least 60 minutes apart'
  ));
});
