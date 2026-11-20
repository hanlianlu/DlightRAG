// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {getWebBootstrap} from './bootstrap.ts';
import {ApiError} from './wire.ts';

const originalFetch = globalThis.fetch;

test.afterEach(() => {
  globalThis.fetch = originalFetch;
});

test('bootstrap rejects a non-success response through its typed error', async () => {
  globalThis.fetch = async () => new Response('', {status: 503});

  await assert.rejects(
    getWebBootstrap(),
    (error: unknown) => error instanceof ApiError && error.status === 503,
  );
});

test('bootstrap rejects malformed success JSON through its typed error', async () => {
  globalThis.fetch = async () => new Response('<html>', {
    status: 200,
    headers: {'Content-Type': 'text/html'},
  });

  await assert.rejects(
    getWebBootstrap(),
    (error: unknown) => error instanceof ApiError && error.status === 200,
  );
});

test('bootstrap v6 requires the agent effort offer', async () => {
  const fixture = {
    contract_version: 6, workspaces: [], primary_workspace: '', active_workspaces: [],
    answer_attachments: {count_limit: 0, image_max_bytes: 1, document_max_bytes: 1,
      extensions: [], image_capability: 'unknown', image_limit: 0, accept: ''},
    active_html_preview_enabled: false,
    agent_effort: {levels: ['low', 'high', 'max'], default: 'high'},
    answer_feedback_enabled: true,
  };
  globalThis.fetch = async () => Response.json(fixture);
  const bootstrap = await getWebBootstrap();
  assert.deepEqual(bootstrap.agentEffort, {levels: ['low', 'high', 'max'], default: 'high'});
  globalThis.fetch = async () => Response.json({...fixture, contract_version: 5});
  await assert.rejects(getWebBootstrap(), ApiError);
  const {agent_effort: _offer, ...missing} = fixture;
  globalThis.fetch = async () => Response.json(missing);
  await assert.rejects(getWebBootstrap(), ApiError);
  // A deployment that names no level, or one the three-level control cannot
  // name, reports no default rather than a level it would not run.
  globalThis.fetch = async () => Response.json({...fixture, agent_effort: {levels: ['low', 'high', 'max']}});
  assert.equal((await getWebBootstrap()).agentEffort.default, null);
  globalThis.fetch = async () => Response.json({...fixture, agent_effort: {levels: ['low'], default: null}});
  assert.deepEqual((await getWebBootstrap()).agentEffort, {levels: ['low'], default: null});
});

test('bootstrap v6 states whether the reader is offered the thumbs under an answer', async () => {
  const fixture = {
    contract_version: 6, workspaces: [], primary_workspace: '', active_workspaces: [],
    answer_attachments: {count_limit: 0, image_max_bytes: 1, document_max_bytes: 1,
      extensions: [], image_capability: 'unknown', image_limit: 0, accept: ''},
    active_html_preview_enabled: false,
    agent_effort: {levels: ['low'], default: null},
  };
  globalThis.fetch = async () => Response.json({...fixture, answer_feedback_enabled: true});
  assert.equal((await getWebBootstrap()).answerFeedbackEnabled, true);
  globalThis.fetch = async () => Response.json({...fixture, answer_feedback_enabled: false});
  assert.equal((await getWebBootstrap()).answerFeedbackEnabled, false);
  // The offer is never assumed: a bootstrap that does not say is not one this page can read.
  globalThis.fetch = async () => Response.json(fixture);
  await assert.rejects(getWebBootstrap(), ApiError);
});

test('bootstrap carries the browser channel only where the server configured both its halves', async () => {
  const fixture = {
    contract_version: 6, workspaces: [], primary_workspace: '', active_workspaces: [],
    answer_attachments: {count_limit: 0, image_max_bytes: 1, document_max_bytes: 1,
      extensions: [], image_capability: 'unknown', image_limit: 0, accept: ''},
    active_html_preview_enabled: false,
    agent_effort: {levels: ['low'], default: null},
    answer_feedback_enabled: true,
  };
  globalThis.fetch = async () => Response.json(fixture);
  // The server omits the object when it has no channel to offer, and the page starts no SDK.
  assert.equal((await getWebBootstrap()).observability, null);
  const observability = {url: 'http://127.0.0.1:4318', apiKey: 'key-1', service: 'dlightrag-web'};
  globalThis.fetch = async () => Response.json({...fixture, observability});
  // A server that does not send the replay rate is one every session is recorded on.
  assert.deepEqual((await getWebBootstrap()).observability, {...observability, replaySampleRate: 1});
  globalThis.fetch = async () =>
    Response.json({...fixture, observability: {...observability, replay_sample_rate: 0.25}});
  assert.equal((await getWebBootstrap()).observability?.replaySampleRate, 0.25);
});
