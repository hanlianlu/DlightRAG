// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';

import {
  AGENT_EFFORT_STORAGE_KEY,
  displayedAgentEffort,
  isAgentEffort,
  offeredLevels,
  storeAgentEffort,
  storedAgentEffort,
} from './agent-effort.ts';

const offer = {levels: ['low', 'high', 'max'], default: 'high'} as const;

/** Node has no Web Storage; the module reads the same browser store the UI does. */
const entries = new Map<string, string>();
Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: {
    getItem: (key: string) => entries.get(key) ?? null,
    setItem: (key: string, value: string) => void entries.set(key, value),
    removeItem: (key: string) => void entries.delete(key),
  },
});

test('only the three known efforts are levels', () => {
  assert.deepEqual(['low', 'high', 'max'].filter(isAgentEffort), ['low', 'high', 'max']);
  for (const value of ['xhigh', 'medium', '', null, undefined, 1, {}]) {
    assert.equal(isAgentEffort(value), false);
  }
});

test('a deployment offers only the levels it still supports', () => {
  assert.deepEqual(offeredLevels(offer), ['low', 'high', 'max']);
  assert.deepEqual(offeredLevels({levels: ['high', 'xhigh'] as never, default: null}), ['high']);
  assert.deepEqual(offeredLevels({levels: [], default: null}), []);
});

test('a stored choice counts only while the deployment still offers it', () => {
  localStorage.removeItem(AGENT_EFFORT_STORAGE_KEY);
  assert.equal(storedAgentEffort(offer), null);
  storeAgentEffort('max');
  assert.equal(storedAgentEffort(offer), 'max');
  // Another surface, or a level this deployment dropped, never leaks into a run.
  localStorage.setItem(AGENT_EFFORT_STORAGE_KEY, 'xhigh');
  assert.equal(storedAgentEffort(offer), null);
  localStorage.setItem(AGENT_EFFORT_STORAGE_KEY, 'low');
  assert.equal(storedAgentEffort({levels: ['high'], default: null}), null);
  storeAgentEffort(null);
  assert.equal(localStorage.getItem(AGENT_EFFORT_STORAGE_KEY), null);
});

test('the shown level is the stored choice, else the deployment default', () => {
  localStorage.removeItem(AGENT_EFFORT_STORAGE_KEY);
  assert.equal(displayedAgentEffort(offer), 'high');
  assert.equal(displayedAgentEffort({levels: ['low', 'high', 'max'], default: null}), null);
  storeAgentEffort('low');
  assert.equal(displayedAgentEffort(offer), 'low');
  storeAgentEffort(null);
});
