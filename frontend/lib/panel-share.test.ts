// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {PANEL_SHARES, shareWidth} from './panel-share.ts';

test('a grain opens at its share of the split, rounded', () => {
  assert.equal(shareWidth('conversation', 1800), 324);
  assert.equal(shareWidth('inspector', 1475), PANEL_SHARES.inspector.max);
  assert.equal(shareWidth('traces', 1475), 738);
  assert.equal(shareWidth('canvas', 900), 378);
});

test('each grain holds its own minimum and maximum', () => {
  assert.equal(shareWidth('conversation', 600), PANEL_SHARES.conversation.min);
  assert.equal(shareWidth('conversation', 9000), PANEL_SHARES.conversation.max);
  assert.equal(shareWidth('inspector', 400), PANEL_SHARES.inspector.min);
  assert.equal(shareWidth('inspector', 9000), PANEL_SHARES.inspector.max);
  assert.equal(shareWidth('traces', 400), PANEL_SHARES.traces.min);
  assert.equal(shareWidth('traces', 9000), PANEL_SHARES.traces.max);
  assert.equal(shareWidth('canvas', 400), PANEL_SHARES.canvas.min);
  assert.equal(shareWidth('canvas', 9000), PANEL_SHARES.canvas.max);
});

test('the shares are the ones ADR 0068 states', () => {
  assert.deepEqual(PANEL_SHARES.conversation, {fraction: 0.18, min: 240, max: 360});
  assert.deepEqual(PANEL_SHARES.inspector, {fraction: 0.38, min: 320, max: 560});
  assert.deepEqual(PANEL_SHARES.traces, {fraction: 0.5, min: 420, max: 960});
  assert.deepEqual(PANEL_SHARES.canvas, {fraction: 0.42, min: 320, max: 720});
});
