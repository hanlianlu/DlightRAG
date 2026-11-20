// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import type {ActivityPage, ChildObservation} from '../api/conversations.ts';
import {KeptTraces} from './child-observation-cache.ts';
import type {TracesSource} from './traces-source.ts';

function observation(childSessionId: string, status: string, words = 10): ChildObservation {
  return {
    run_id: 'run-1',
    child: {childSessionId, status},
    transcript: [{role: 'assistant', content: 'x'.repeat(words)}],
    controls: [],
    questions: [],
    result: null,
  } as unknown as ChildObservation;
}

function countingSource(reads: Record<string, number>, status = 'succeeded'): TracesSource {
  return {
    runId: 'run-1',
    mainAgent: () => ({childSessionId: 'main'}),
    presentation: () => null,
    page: async () => ({children: [], next_cursor: null}),
    observe: async (childSessionId: string) => {
      reads[childSessionId] = (reads[childSessionId] ?? 0) + 1;
      return observation(childSessionId, status);
    },
    activity: async (agent: string | null) => {
      reads[`page:${agent ?? 'main'}`] = (reads[`page:${agent ?? 'main'}`] ?? 0) + 1;
      return {messages: [{role: 'assistant', content: 'y'.repeat(10)}], next_before: null, running: false} as unknown as ActivityPage;
    },
    control: async () => ({}) as never,
    reply: async () => ({}) as never,
  } as unknown as TracesSource;
}

const signal = new AbortController().signal;

/** A Run's source as the page keeps it, with room for as much as `budgetBytes` of pages. */
function cachedSource(source: TracesSource, budgetBytes?: number) {
  return new KeptTraces().sourceFor('run-1', () => source, budgetBytes);
}

test('a settled child is read once, however often the reader returns to it', async () => {
  const reads: Record<string, number> = {};
  const source = cachedSource(countingSource(reads));

  await source.observe('child-a', signal);
  await source.observe('child-a', signal);
  await source.observe('child-a', signal);

  assert.equal(reads['child-a'], 1);
});

test('a child still running is never kept: every look reads it again', async () => {
  const reads: Record<string, number> = {};
  const source = cachedSource(countingSource(reads, 'running'));

  await source.observe('child-a', signal);
  await source.observe('child-a', signal);

  assert.equal(reads['child-a'], 2);
});

test('two readers asking at once share one read', async () => {
  const reads: Record<string, number> = {};
  const source = cachedSource(countingSource(reads));

  await Promise.all([source.observe('child-a', signal), source.observe('child-a', signal)]);

  assert.equal(reads['child-a'], 1);
});

test('warming a child reads it before anyone asks', async () => {
  const reads: Record<string, number> = {};
  const source = cachedSource(countingSource(reads));

  source.warm('child-b');
  await new Promise((resolve) => setTimeout(resolve, 0));
  await source.observe('child-b', signal);

  assert.equal(reads['child-b'], 1);
});

test('the budget evicts the least recently read child, never the one just read', async () => {
  const reads: Record<string, number> = {};
  const source = cachedSource(countingSource(reads), 900);

  await source.observe('child-a', signal);
  await source.observe('child-b', signal);
  await source.observe('child-c', signal); // over budget: child-a goes
  await source.observe('child-a', signal);

  assert.equal(reads['child-a'], 2, 'the evicted child is read again');
  assert.equal(reads['child-c'], 1, 'the child read last is still kept');
});

test('a transcript page is kept too: the heavy read is what a reader comes back for', async () => {
  const reads: Record<string, number> = {};
  const source = cachedSource(countingSource(reads));

  await source.activity('child-a', null, signal);
  await source.activity('child-a', null, signal);

  assert.equal(reads['page:child-a'], 1);
});

test('the page on screen survives a budget a prefetch of another child blows', async () => {
  const reads: Record<string, number> = {};
  const source = cachedSource(countingSource(reads), 1200);

  await source.observe('child-a', signal); // the reader is looking at a
  source.warm('child-b'); // the pointer visits b, which does not become what is on screen
  await new Promise((resolve) => setTimeout(resolve, 0));
  await source.observe('child-b', signal);
  await source.observe('child-a', signal); // back to what was on screen

  assert.equal(reads['child-a'], 1, 'the child on screen was kept');
});

test('signing out reaches the pages a dock still holds, and the read it still had coming', async () => {
  const reads: Record<string, number> = {};
  const kept = new KeptTraces();
  const held = kept.sourceFor('run-1', () => countingSource(reads));

  await held.observe('child-a', signal);
  await held.activity('child-a', null, signal);
  await held.observe('child-a', signal);
  await held.activity('child-a', null, signal);
  assert.deepEqual(
    {child: reads['child-a'], page: reads['page:child-a']},
    {child: 1, page: 1},
    'a held source answers from what it kept',
  );

  kept.forget();

  await held.observe('child-a', signal);
  await held.activity('child-a', null, signal);
  assert.deepEqual(
    {child: reads['child-a'], page: reads['page:child-a']},
    {child: 2, page: 2},
    'a source that predates the sign-out served the last reader its pages',
  );

  // A read still in the air when the sign-out lands belongs to the reader who left.
  const land: Array<() => void> = [];
  const slow = kept.sourceFor('run-slow', () => ({
    ...countingSource(reads),
    observe: (childSessionId: string) =>
      new Promise((resolve) => {
        reads[childSessionId] = (reads[childSessionId] ?? 0) + 1;
        land.push(() => resolve(observation(childSessionId, 'succeeded')));
      }),
  } as unknown as TracesSource));

  const leftBehind = slow.observe('child-b', signal);
  kept.forget();
  land.shift()!();
  await leftBehind;

  const next = slow.observe('child-b', signal);
  assert.equal(reads['child-b'], 2, 'the read in flight when the sign-out landed stayed with the reader who left');
  land.shift()!();
  await next;
});

test('a Run keeps one source however often its dock opens, and only the last few Runs keep any', () => {
  const kept = new KeptTraces();
  let built = 0;
  const build = (): TracesSource => { built += 1; return countingSource({}); };

  const first = kept.sourceFor('run-1', build);
  assert.equal(kept.sourceFor('run-1', build), first, 'reopening the dock built another source');
  assert.equal(built, 1);

  kept.sourceFor('run-2', build);
  kept.sourceFor('run-3', build);
  kept.sourceFor('run-4', build); // the fourth Run pushes out the one read longest ago: run-1
  assert.equal(built, 4);
  assert.notEqual(kept.sourceFor('run-1', build), first, 'the oldest Run still held its source');
  assert.equal(built, 5);
});
