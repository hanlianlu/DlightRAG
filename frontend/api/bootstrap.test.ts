// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {BootstrapApiError, getWebBootstrap} from './bootstrap.ts';

const originalFetch = globalThis.fetch;

test.afterEach(() => {
  globalThis.fetch = originalFetch;
});

test('bootstrap rejects a non-success response through its typed error', async () => {
  globalThis.fetch = async () => new Response('', {status: 503});

  await assert.rejects(
    getWebBootstrap(),
    (error: unknown) => error instanceof BootstrapApiError && error.status === 503,
  );
});

test('bootstrap rejects malformed success JSON through its typed error', async () => {
  globalThis.fetch = async () => new Response('<html>', {
    status: 200,
    headers: {'Content-Type': 'text/html'},
  });

  await assert.rejects(
    getWebBootstrap(),
    (error: unknown) => error instanceof BootstrapApiError && error.status === 200,
  );
});
