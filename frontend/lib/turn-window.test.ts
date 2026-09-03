// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';

import {
  TURN_PLACEHOLDER_MIN_PX,
  TURN_WINDOW_PAD,
  turnIsLive,
  visibleTurnWindow,
} from './turn-window.ts';

test('short conversations stay fully mounted', () => {
  const window = visibleTurnWindow({
    count: 5,
    heights: [100, 100, 100, 100, 100],
    scrollTop: 0,
    viewportHeight: 400,
    alwaysOn: new Set(),
  });
  assert.deepEqual(window, {start: 0, end: 4});
});

test('pads the visible range and always keeps live turns', () => {
  const count = TURN_WINDOW_PAD * 2 + 30;
  const heights = Array.from({length: count}, () => 100);
  const window = visibleTurnWindow({
    count,
    heights,
    scrollTop: 0,
    viewportHeight: 200,
    alwaysOn: new Set([count - 1]),
  });
  assert.equal(window.start, 0);
  assert.ok(window.end >= TURN_WINDOW_PAD);
  assert.equal(window.end, count - 1);
});

test('uses placeholder height when a turn has not been measured', () => {
  const window = visibleTurnWindow({
    count: 80,
    heights: [],
    scrollTop: TURN_PLACEHOLDER_MIN_PX * 40,
    viewportHeight: TURN_PLACEHOLDER_MIN_PX * 2,
    alwaysOn: new Set(),
  });
  assert.ok(window.start >= 40 - TURN_WINDOW_PAD - 2);
  assert.ok(window.start <= 40);
  assert.ok(window.end >= 42);
});

test('live states stay mounted', () => {
  assert.equal(turnIsLive('streaming'), true);
  assert.equal(turnIsLive('succeeded'), false);
});
