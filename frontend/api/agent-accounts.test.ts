// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict';
import test from 'node:test';
import {getAgentAccounts, removeAgentAccount, setAgentAccountRegistration} from './agent-accounts.ts';
import {ApiError, onSignedOut} from './wire.ts';

const originalFetch = globalThis.fetch;
const originalDocument = globalThis.document;
test.beforeEach(() => {
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: {cookie: 'dlightrag_web_csrf=csrf-token'},
  });
});
test.afterEach(() => {
  globalThis.fetch = originalFetch;
  Object.defineProperty(globalThis, 'document', {configurable: true, value: originalDocument});
});

const wireAccount = {
  site: 'discourse.org',
  email: 'agent@example.test',
  username: 'dlr20261004k7q2',
  created_at: '2026-10-04T15:12:30Z',
  last_used_at: '2026-10-04T15:13:01Z',
};
const wireView = {
  available: true,
  registration: {allowed: true, enabled: false},
  accounts: [wireAccount],
};

test('the view keeps its dates as dates and its absent identity or sign-in as null', async () => {
  globalThis.fetch = async () => Response.json({
    ...wireView,
    accounts: [wireAccount, {...wireAccount, site: 'ycombinator.com', email: null, username: null, last_used_at: null}],
  });
  const {available, registration, accounts} = await getAgentAccounts();
  assert.equal(available, true);
  assert.deepEqual(registration, {allowed: true, enabled: false});
  assert.deepEqual(accounts[0], {
    site: 'discourse.org',
    email: 'agent@example.test',
    username: 'dlr20261004k7q2',
    createdAt: new Date('2026-10-04T15:12:30Z'),
    lastUsedAt: new Date('2026-10-04T15:13:01Z'),
  });
  assert.deepEqual(accounts[1], {
    site: 'ycombinator.com', email: null, username: null,
    createdAt: new Date('2026-10-04T15:12:30Z'), lastUsedAt: null,
  });
});

test('a reply that carries any field beyond the contract is refused, so no secret is ever held', async () => {
  const leaks: unknown[] = [
    {...wireView, accounts: [{...wireAccount, password: 'must-not-enter-ui'}]},
    {...wireView, accounts: [{...wireAccount, envelope: 'sealed', key_id: 'k1', account_id: 'a1'}]},
    {...wireView, secret: 'must-not-enter-ui'},
    {...wireView, registration: {...wireView.registration, password: 'must-not-enter-ui'}},
  ];
  for (const reply of leaks) {
    globalThis.fetch = async () => Response.json(reply);
    await assert.rejects(getAgentAccounts(), ApiError);
  }
});

test('a missing field or a malformed timestamp is refused as one unreadable reply', async () => {
  globalThis.fetch = async () => Response.json({...wireView, accounts: [{...wireAccount, created_at: 'Oct 4'}]});
  await assert.rejects(getAgentAccounts(), ApiError);
  globalThis.fetch = async () => Response.json({available: true, accounts: []});
  await assert.rejects(getAgentAccounts(), ApiError);
});

test('the sign-up switch is a PUT of one boolean with the CSRF token, and answers the fresh view', async () => {
  const requests: Array<{url: string; init: RequestInit | undefined}> = [];
  globalThis.fetch = async (url, init) => {
    requests.push({url: String(url), init});
    return Response.json({...wireView, registration: {allowed: true, enabled: true}});
  };
  const view = await setAgentAccountRegistration(true);
  assert.equal(view.registration.enabled, true);
  assert.equal(requests.length, 1);
  assert.equal(requests[0]!.url, '/web/api/agent-accounts/settings');
  assert.equal(requests[0]!.init?.method, 'PUT');
  assert.deepEqual(JSON.parse(String(requests[0]!.init?.body)), {registration_enabled: true});
  assert.deepEqual(requests[0]!.init?.headers, {'Content-Type': 'application/json', 'X-CSRF-Token': 'csrf-token'});
});

test('removing an account names its site in the path, sends no body, and answers the fresh view', async () => {
  const requests: Array<{url: string; init: RequestInit | undefined}> = [];
  globalThis.fetch = async (url, init) => {
    requests.push({url: String(url), init});
    return Response.json({...wireView, accounts: []});
  };
  const view = await removeAgentAccount('discourse.org');
  assert.deepEqual(view.accounts, []);
  assert.equal(requests[0]!.url, '/web/api/agent-accounts/discourse.org');
  assert.equal(requests[0]!.init?.method, 'DELETE');
  assert.equal(requests[0]!.init?.body, undefined);
  assert.deepEqual(requests[0]!.init?.headers, {'X-CSRF-Token': 'csrf-token'});
});

test('a refusal answers the general envelope: a 404 is told apart and a 401 signs the page out', async () => {
  // The envelope the routes answer with: a detail, an error type, and a null error kind.
  globalThis.fetch = async () => Response.json(
    {detail: 'No account for this website', error_type: 'not_found', error_kind: null},
    {status: 404},
  );
  await assert.rejects(removeAgentAccount('gone.example'), (error: unknown) => error instanceof ApiError
    && error.status === 404
    && error.errorType === 'not_found'
    && error.errorKind === null
    && error.detail === 'No account for this website');

  let signedOut = 0;
  const stop = onSignedOut(() => { signedOut += 1; });
  globalThis.fetch = async () => Response.json({detail: 'Not authenticated', error_type: 'auth'}, {status: 401});
  await assert.rejects(getAgentAccounts(), (error: unknown) => error instanceof ApiError && error.status === 401);
  stop();
  assert.equal(signedOut, 1);
});

test('a malformed site (422) and a refused CSRF check (403) are refusals of the same kind, never a view', async () => {
  globalThis.fetch = async () => Response.json(
    {detail: 'Invalid website', error_type: 'validation', error_kind: null},
    {status: 422},
  );
  await assert.rejects(removeAgentAccount('Not A Host'), (error: unknown) => error instanceof ApiError
    && error.status === 422
    && error.errorType === 'validation');

  globalThis.fetch = async () => Response.json({detail: 'CSRF check failed', error_type: 'auth', error_kind: null}, {status: 403});
  await assert.rejects(setAgentAccountRegistration(false), (error: unknown) => error instanceof ApiError
    && error.status === 403);
});
