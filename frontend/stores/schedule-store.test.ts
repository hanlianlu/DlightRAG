// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import type {Schedule, ScheduleFire} from '../api/schedules.ts';
import {ApiError} from '../api/wire.ts';
import {type ScheduleApi, ScheduleStore} from './schedule-store.ts';

function schedule(id: string, extra: Partial<Schedule> = {}): Schedule {
  return {
    scheduleId: id, name: id, prompt: 'Ask.', mode: 'research', workspaces: ['default'], target: 'new_conversation',
    conversationId: null, zone: 'UTC', cron: '0 8 * * 1', at: null, maxTokensPerDay: 1000, goalMaxRounds: null,
    state: 'active', pauseReason: null, nextFireAt: null, ...extra,
  };
}

function fire(state: ScheduleFire['state']): ScheduleFire {
  return {fireId: 'f', slot: new Date(0), state, conversationId: null, missedCount: null, manual: true, reason: null};
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((yes) => { resolve = yes; });
  return {promise, resolve};
}

function api(overrides: Partial<ScheduleApi> = {}): ScheduleApi {
  return {
    list: async () => ({schedules: [schedule('a'), schedule('b')], defaultMaxTokensPerDay: 5000}),
    create: async () => schedule('c'),
    update: async (id) => schedule(id, {name: 'renamed'}),
    remove: async () => {},
    pause: async (id) => schedule(id, {state: 'paused'}),
    resume: async (id) => schedule(id),
    run: async () => fire('started'),
    ...overrides,
  };
}

test('the list is read once, with the ceiling a new Schedule starts with', async () => {
  let reads = 0;
  const store = new ScheduleStore(api({list: async () => { reads += 1; return {schedules: [schedule('a')], defaultMaxTokensPerDay: 5000}; }}));
  const states: string[] = [];
  store.subscribe(() => { states.push(store.listState); });

  await store.load();
  await store.load();

  assert.equal(reads, 1);
  assert.deepEqual([store.listState, store.defaultCeiling, store.schedules.map((row) => row.scheduleId)], ['ready', 5000, ['a']]);
  assert.deepEqual(states, ['loading', 'ready']);
});

test('a list that cannot be read says so and is read again on the next ask', async () => {
  let answers = 0;
  const store = new ScheduleStore(api({list: async () => {
    answers += 1;
    if (answers === 1) throw new ApiError(503);
    return {schedules: [schedule('a')], defaultMaxTokensPerDay: 1};
  }}));

  await store.load();
  assert.equal(store.listState, 'error');
  await store.load();

  assert.deepEqual([store.listState, store.schedules.length], ['ready', 1]);
});

test('a refresh that fails leaves the list as it was read', async () => {
  let fail = false;
  const store = new ScheduleStore(api({list: async () => {
    if (fail) throw new ApiError(503);
    return {schedules: [schedule('a')], defaultMaxTokensPerDay: 1};
  }}));
  await store.load();
  fail = true;

  await store.refresh();

  assert.deepEqual([store.listState, store.schedules.length], ['ready', 1]);
});

test('a command\'s reply replaces its own row only, and says how it ended', async () => {
  const store = new ScheduleStore(api());
  await store.load();

  const outcome = await store.pause('a');

  assert.equal(outcome.kind, 'ok');
  assert.deepEqual(store.schedules.map((row) => [row.scheduleId, row.state]), [['a', 'paused'], ['b', 'active']]);
});

test('a reply for a Schedule that has gone does not bring it back', async () => {
  const slow = deferred<Schedule>();
  const store = new ScheduleStore(api({pause: () => slow.promise}));
  await store.load();

  const pausing = store.pause('a');
  store.drop('a');
  slow.resolve(schedule('a', {state: 'paused'}));
  await pausing;

  assert.deepEqual(store.schedules.map((row) => row.scheduleId), ['b']);
});

test('a Schedule the server no longer has is dropped, and a refusal carries the server\'s reason', async () => {
  const store = new ScheduleStore(api({
    pause: async () => { throw new ApiError(404, {detail: 'Schedule not found'}); },
    update: async () => { throw new ApiError(422, {detail: 'slots may be at least 60 minutes apart'}); },
    resume: async () => { throw new Error('offline'); },
  }));
  await store.load();

  assert.deepEqual(await store.update('b', {name: 'x'}), {kind: 'failed', reason: 'slots may be at least 60 minutes apart'});
  assert.deepEqual(await store.resume('b'), {kind: 'failed', reason: null});
  assert.deepEqual(await store.pause('a'), {kind: 'gone'});
  assert.deepEqual(store.schedules.map((row) => row.scheduleId), ['b']);
});

test('a second command on a Schedule waits for none: it is not sent', async () => {
  const slow = deferred<Schedule>();
  let sent = 0;
  const store = new ScheduleStore(api({pause: () => { sent += 1; return slow.promise; }}));
  await store.load();

  const first = store.pause('a');
  assert.equal(store.isPending('a'), true);
  assert.deepEqual(await store.pause('a'), {kind: 'busy'});
  assert.equal(store.isPending('b'), false);
  slow.resolve(schedule('a', {state: 'paused'}));
  await first;

  assert.equal(sent, 1);
  assert.equal(store.isPending('a'), false);
});

test('running a Schedule reads the list again, because a run can pause it', async () => {
  let reads = 0;
  const store = new ScheduleStore(api({list: async () => {
    reads += 1;
    return {schedules: [schedule('a', reads > 1 ? {state: 'paused'} : {})], defaultMaxTokensPerDay: 1};
  }}));
  await store.load();

  const outcome = await store.run('a');

  assert.equal(outcome.kind, 'ok');
  assert.equal(reads, 2);
  assert.equal(store.get('a')?.state, 'paused');
});

test('creating adds the Schedule, and deleting removes it', async () => {
  const store = new ScheduleStore(api());
  await store.load();

  const created = await store.create({
    name: 'c', prompt: 'Ask.', mode: 'research', workspaces: ['default'], target: 'new_conversation',
    maxTokensPerDay: 1, zone: 'UTC', cron: '0 8 * * 1',
  });
  assert.equal(created.kind, 'ok');
  assert.deepEqual(store.schedules.map((row) => row.scheduleId), ['a', 'b', 'c']);

  assert.deepEqual(await store.remove('a'), {kind: 'ok', value: null});
  assert.deepEqual(store.schedules.map((row) => row.scheduleId), ['b', 'c']);
});

test('forgetting empties the store, and an answer still on its way is dropped', async () => {
  const slow = deferred<{schedules: Schedule[]; defaultMaxTokensPerDay: number}>();
  const store = new ScheduleStore(api({list: () => slow.promise}));
  const loading = store.load();

  store.forget();
  slow.resolve({schedules: [schedule('a')], defaultMaxTokensPerDay: 1});
  await loading;

  assert.deepEqual([store.listState, store.schedules.length], ['idle', 0]);
});
