// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {roomBelow} from './window-fit.ts';

test('the room is what the page has left below the thing standing in it', () => {
  assert.equal(roomBelow({height: 874, top: 189, scrollerTop: 0, paddingBottom: 24}), 661);
});

test('scrolling the page does not change the answer', () => {
  // Scrolling moves everything inside the page up by the same amount, both edges together, and the
  // answer holds still. A window whose height moved with the scroll would grow as a reader read it.
  const atTop = roomBelow({height: 874, top: 189, scrollerTop: 0, paddingBottom: 24});
  const scrolled = roomBelow({height: 874, top: -211, scrollerTop: -400, paddingBottom: 24});
  assert.equal(scrolled, atTop);
});

test('a page with no room left says so, and the caller decides what to do', () => {
  assert.ok(roomBelow({height: 430, top: 2038, scrollerTop: 0, paddingBottom: 0}) < 0);
});
