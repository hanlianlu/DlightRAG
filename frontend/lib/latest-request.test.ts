// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';
import {LatestRequest} from './latest-request.ts';

test('starting a request ends the one before it, and only the latest is current', () => {
  const latest = new LatestRequest();
  assert.equal(latest.busy, false);

  const first = latest.start();
  assert.equal(latest.busy, true);
  assert.equal(first.aborted, false);
  assert.equal(latest.isCurrent(first), true);

  const second = latest.start();
  assert.equal(first.aborted, true, 'the request before was not ended');
  assert.equal(second.aborted, false);
  assert.equal(latest.isCurrent(first), false);
  assert.equal(latest.isCurrent(second), true);
  assert.notEqual(first, second);
});

test('a request that finishes leaves none under way, and a replaced one settles nothing', () => {
  const latest = new LatestRequest();
  const first = latest.start();
  const second = latest.start();

  assert.equal(latest.finish(first), false, 'a replaced request settled for the latest');
  assert.equal(latest.busy, true);
  assert.equal(latest.isCurrent(second), true);

  assert.equal(latest.finish(second), true);
  assert.equal(latest.busy, false);
  assert.equal(latest.isCurrent(second), false);
  assert.equal(second.aborted, false, 'finishing is not aborting: the request ended by itself');
  assert.equal(latest.finish(second), false, 'a request finished twice');
});

test('cancelling ends the request under way and starts none; with none it does nothing', () => {
  const latest = new LatestRequest();
  latest.cancel();
  assert.equal(latest.busy, false);

  const signal = latest.start();
  latest.cancel();
  assert.equal(signal.aborted, true);
  assert.equal(latest.isCurrent(signal), false);
  assert.equal(latest.busy, false);
  assert.equal(latest.finish(signal), false);

  const next = latest.start();
  assert.equal(latest.isCurrent(next), true, 'a request after a cancel is a fresh one');
});

test('a signal from elsewhere is never current', () => {
  const latest = new LatestRequest();
  const stranger = new AbortController().signal;
  assert.equal(latest.isCurrent(stranger), false);
  const own = latest.start();
  assert.equal(latest.isCurrent(stranger), false);
  assert.equal(latest.isCurrent(own), true);
  assert.equal(new LatestRequest().isCurrent(own), false, 'another instance claimed a request');
});

test('an answer that lands late drops itself: the stale call sees it is no longer the latest', async () => {
  const latest = new LatestRequest();
  const shown: string[] = [];
  const answers = new Map<string, (value: string) => void>();
  const ask = async (name: string): Promise<void> => {
    const signal = latest.start();
    const value = await new Promise<string>((resolve) => { answers.set(name, resolve); });
    if (!latest.isCurrent(signal)) return;
    shown.push(value);
    latest.finish(signal);
  };

  const first = ask('first');
  const second = ask('second');
  answers.get('second')!('second answer');
  await second;
  answers.get('first')!('first answer');
  await first;

  assert.deepEqual(shown, ['second answer']);
  assert.equal(latest.busy, false);
});
