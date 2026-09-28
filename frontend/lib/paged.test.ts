// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {KeysetPager} from './paged.ts';

interface Page {
  items: string[];
  nextCursor: string | null;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return {promise, resolve, reject};
}

/** A pager whose every request waits for the test to answer it. */
function harness() {
  const requests: Array<{
    cursor: string | null;
    signal: AbortSignal;
    answer: ReturnType<typeof deferred<Page>>;
  }> = [];
  let notified = 0;
  const pager = new KeysetPager<Page>((cursor, signal) => {
    const answer = deferred<Page>();
    requests.push({cursor, signal, answer});
    return answer.promise;
  }, () => { notified += 1; });
  return {pager, requests, notified: () => notified};
}

test('start loads the first page and anchors the cursor for the next one', async () => {
  const {pager, requests, notified} = harness();
  const pages: string[][] = [];

  const first = pager.start((page) => { pages.push(page.items); });
  assert.equal(pager.state, 'loading');
  assert.equal(pager.hasOlder, false);
  requests[0]!.answer.resolve({items: ['c', 'b'], nextCursor: 'after-b'});
  await first;
  assert.deepEqual(requests.map(({cursor}) => cursor), [null]);
  assert.equal(pager.state, 'idle');
  assert.equal(pager.hasOlder, true);
  assert.equal(pager.outcome, null, 'a first page is not a next-page outcome');

  const next = pager.loadNext((page) => { pages.push(page.items); });
  assert.equal(pager.loadNext(() => { pages.push(['duplicate']); }), next, 'one flight at a time');
  requests[1]!.answer.resolve({items: ['a'], nextCursor: null});
  await next;

  assert.deepEqual(requests.map(({cursor}) => cursor), [null, 'after-b']);
  assert.deepEqual(pages, [['c', 'b'], ['a']]);
  assert.equal(pager.hasOlder, false);
  assert.equal(pager.outcome, 'loaded');
  assert.equal(notified(), 4, 'each load reports its start and its end');
  await pager.loadNext(() => { pages.push(['past the end']); });
  assert.equal(requests.length, 2);
});

test('a newer start aborts and drops the page it replaces', async () => {
  const {pager, requests} = harness();
  const pages: string[][] = [];
  pager.reset('older');
  const older = pager.loadNext((page) => { pages.push(page.items); });

  const restarted = pager.start((page) => { pages.push(page.items); });
  assert.equal(requests[0]!.signal.aborted, true);
  requests[0]!.answer.resolve({items: ['stale'], nextCursor: 'stale-cursor'});
  requests[1]!.answer.resolve({items: ['fresh'], nextCursor: null});
  await Promise.all([older, restarted]);

  assert.deepEqual(pages, [['fresh']]);
  assert.equal(pager.hasOlder, false);
  assert.equal(pager.outcome, null);
});

test('a failed next page keeps its cursor for a retry and reports the error', async () => {
  const {pager, requests} = harness();
  const errors: unknown[] = [];
  const pages: string[][] = [];
  pager.reset('older');

  const failing = pager.loadNext(() => {}, (error) => { errors.push(error); });
  const failure = new Error('unavailable');
  requests[0]!.answer.reject(failure);
  await failing;
  assert.equal(pager.state, 'error');
  assert.equal(pager.outcome, 'failed');
  assert.equal(pager.hasOlder, true);
  assert.deepEqual(errors, [failure]);

  const retry = pager.loadNext((page) => { pages.push(page.items); });
  assert.equal(pager.outcome, null, 'a retry in flight is no longer a failure');
  assert.equal(requests[1]!.cursor, 'older');
  requests[1]!.answer.resolve({items: ['old'], nextCursor: null});
  await retry;
  assert.deepEqual(pages, [['old']]);
  assert.equal(pager.state, 'idle');
});

test('a failed first page reports its error without a next-page outcome', async () => {
  const {pager, requests} = harness();
  let failed = false;
  const starting = pager.start(() => {}, () => { failed = true; });
  requests[0]!.answer.reject(new Error('down'));
  await starting;

  assert.equal(failed, true);
  assert.equal(pager.state, 'error');
  assert.equal(pager.outcome, null);
  assert.equal(pager.hasOlder, false);
});

test('cancel drops the flight silently but keeps the cursor; reset re-anchors it', async () => {
  const {pager, requests, notified} = harness();
  const pages: string[][] = [];
  pager.reset('older');

  const cancelled = pager.loadNext((page) => { pages.push(page.items); });
  const before = notified();
  pager.cancel();
  assert.equal(notified(), before);
  assert.equal(requests[0]!.signal.aborted, true);
  assert.equal(pager.state, 'idle');
  requests[0]!.answer.reject(new DOMException('Aborted', 'AbortError'));
  await cancelled;
  assert.equal(pager.hasOlder, true);
  assert.equal(pager.state, 'idle');
  assert.equal(notified(), before, 'a dropped flight reports nothing');

  pager.reset(null);
  assert.equal(pager.hasOlder, false);
  pager.reset('elsewhere');
  const next = pager.loadNext((page) => { pages.push(page.items); });
  assert.equal(requests[1]!.cursor, 'elsewhere');
  requests[1]!.answer.resolve({items: ['kept'], nextCursor: null});
  await next;
  assert.deepEqual(pages, [['kept']]);
});
