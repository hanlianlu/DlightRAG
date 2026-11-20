// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock: the browser channel masks what ADR 0039 calls private, and no page depends on it. */

import assert from 'node:assert/strict';
import test from 'node:test';
import {
  initializeObservability,
  replayEnabledForSession,
  type ObservabilityConfig,
  type ReplayMemory,
} from './observability.ts';

const CONFIG: ObservabilityConfig = {
  url: 'http://127.0.0.1:4318',
  apiKey: 'ingestion-key',
  service: 'dlightrag-web',
  replaySampleRate: 1,
};

/** Let a fired-and-forgotten start reach its end, so an unhandled rejection would surface here. */
function settle(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

test('no browser channel starts no SDK', async () => {
  let loads = 0;
  initializeObservability(null, () => {
    loads += 1;
    return Promise.resolve({default: {init: () => {}}});
  });

  await settle();
  assert.equal(loads, 0);
});

test('an SDK that will not load is swallowed', async () => {
  initializeObservability(CONFIG, () => {
    throw new Error('blocked by an ad blocker');
  });
  initializeObservability(CONFIG, async () => {
    throw new Error('blocked by an ad blocker');
  });
  initializeObservability(CONFIG, () =>
    Promise.resolve({default: {init: () => {throw new Error('no collector')}}}),
  );

  await settle();
});

test('the loaded SDK is initialised with the masking the ADR requires, and no body capture', async () => {
  const calls: unknown[] = [];
  initializeObservability(CONFIG, () =>
    Promise.resolve({default: {init: (options) => calls.push(options)}}),
  );

  await settle();
  assert.deepEqual(calls, [{
    url: CONFIG.url,
    apiKey: CONFIG.apiKey,
    service: CONFIG.service,
    tracePropagationTargets: [/\/web\/api/, /\/api\//],
    maskAllInputs: true,
    maskAllText: true,
    consoleCapture: false,
    advancedNetworkCapture: false,
    recordCanvas: false,
    disableReplay: false,
  }]);
});

test('a session the coin flip passed over reports errors and RUM without replay', async () => {
  const calls: Array<Record<string, unknown>> = [];
  initializeObservability({...CONFIG, replaySampleRate: 0}, () =>
    Promise.resolve({default: {init: (options) => calls.push(options)}}),
  );

  await settle();
  assert.equal(calls.length, 1, 'the SDK is still initialised: sampling is not an off state');
  assert.equal(calls[0]?.disableReplay, true);
  // The rest of the configuration — masking included — is untouched by the flip.
  assert.equal(calls[0]?.maskAllText, true);
  assert.equal(calls[0]?.maskAllInputs, true);
  assert.deepEqual(calls[0]?.tracePropagationTargets, [/\/web\/api/, /\/api\//]);
});

test('every session is recorded at a rate of one', async () => {
  const calls: Array<Record<string, unknown>> = [];
  initializeObservability(CONFIG, () =>
    Promise.resolve({default: {init: (options) => calls.push(options)}}),
  );

  await settle();
  assert.equal(calls[0]?.disableReplay, false);
});

function memory(): {stored: Map<string, string>; store: ReplayMemory} {
  const stored = new Map<string, string>();
  return {
    stored,
    store: {
      read: () => stored.get('decision') ?? null,
      write: (value) => stored.set('decision', value),
    },
  };
}

test('the replay gate decides once per session and remembers it', () => {
  const sampled = memory();
  assert.equal(replayEnabledForSession(0.5, sampled.store, () => 0.2), true);
  // The same session keeps its decision whatever the next flip would have said.
  assert.equal(replayEnabledForSession(0.5, sampled.store, () => 0.9), true);

  const passedOver = memory();
  assert.equal(replayEnabledForSession(0.5, passedOver.store, () => 0.8), false);
  assert.equal(replayEnabledForSession(0.5, passedOver.store, () => 0.1), false);
  assert.equal(passedOver.stored.get('decision'), '0');
});

test('the gate is certain at both ends and survives a storage that refuses', () => {
  assert.equal(replayEnabledForSession(1, memory().store, () => 0.999), true);
  assert.equal(replayEnabledForSession(0, memory().store, () => 0.001), false);

  const refuses: ReplayMemory = {
    read: () => {
      throw new Error('sessionStorage is blocked');
    },
    write: () => {
      throw new Error('sessionStorage is blocked');
    },
  };
  // A blocked storage costs the decision its memory, never the page.
  assert.equal(replayEnabledForSession(1, refuses, () => 0.999), true);
  assert.equal(replayEnabledForSession(0, refuses, () => 0.001), false);
  assert.equal(replayEnabledForSession(0.5, refuses, () => 0.1), true);
});

