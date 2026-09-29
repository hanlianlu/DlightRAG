// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import * as v from 'valibot';
import {webCommandError} from './web-command-error.ts';
import {ApiError, apiError, apiErrorFromBody, onSignedOut, parseWire} from './wire.ts';

const schema = v.pipe(
  v.object({run_id: v.string()}),
  v.transform((wire) => ({runId: wire.run_id})),
);

test('the general envelope becomes one typed error', async () => {
  const error = await apiError(Response.json(
    {detail: 'Image input is not supported.', error_type: 'validation', error_kind: 'CURRENT_IMAGES_UNSUPPORTED'},
    {status: 422},
  ));

  assert.ok(error instanceof ApiError);
  assert.equal(error.status, 422);
  assert.equal(error.detail, 'Image input is not supported.');
  assert.equal(error.errorType, 'validation');
  assert.equal(error.errorKind, 'CURRENT_IMAGES_UNSUPPORTED');
});

test('a body that is not the envelope keeps only its status: no type is read into it', async () => {
  const responses = [
    new Response('<html>Bad gateway</html>', {status: 502}),
    new Response('Cross-origin request rejected', {status: 403}),
    new Response('', {status: 410}),
    new Response('', {status: 429}),
    // FastAPI's own request validation answers a list, not a reason.
    Response.json({detail: [{loc: ['body'], msg: 'field required'}]}, {status: 422}),
  ];
  for (const response of responses) {
    const error = await apiError(response);
    assert.equal(error.status, response.status);
    assert.equal(error.detail, null);
    assert.equal(error.errorType, null);
    assert.equal(error.errorKind, null);
  }
  // A type outside the vocabulary is no type; the reason still stands.
  const unknown = await apiError(Response.json({detail: 'Gone.', error_type: 'mystery'}, {status: 404}));
  assert.equal(unknown.errorType, null);
  assert.equal(unknown.detail, 'Gone.');
  assert.equal((await apiError(Response.json({detail: '   '}, {status: 409}))).detail, null);
});

test('a body already read parses the same way', () => {
  const error = apiErrorFromBody(403, {detail: 'Access denied', error_type: 'auth'});
  assert.equal(error.status, 403);
  assert.equal(error.detail, 'Access denied');
  assert.equal(error.errorType, 'auth');
  assert.equal(apiErrorFromBody(500, null).errorType, null);
});

test('parseWire translates a success and refuses an unreadable one with its status', async () => {
  assert.deepEqual(await parseWire(Response.json({run_id: 'run-1'}), schema), {runId: 'run-1'});

  for (const body of ['{truncated', JSON.stringify({run: 'run-1'})]) {
    await assert.rejects(
      parseWire(new Response(body, {status: 202}), schema),
      (error: unknown) => error instanceof ApiError
        && error.status === 202
        && error.detail === null,
    );
  }
});

test('parseWire hands a refusal to the route family parser it is given', async () => {
  const refusal = Response.json({detail: 'Workspace not found', error_type: 'not_found'}, {status: 404});
  await assert.rejects(
    parseWire(refusal, schema),
    (error: unknown) => error instanceof ApiError && error.detail === 'Workspace not found',
  );

  class Distinct extends Error {}
  await assert.rejects(
    parseWire(new Response('', {status: 409}), schema, async () => new Distinct()),
    Distinct,
  );
});

test('every refusal reader signs the page out on a 401, and only on a 401', async () => {
  let signedOut = 0;
  const stop = onSignedOut(() => { signedOut += 1; });
  const expired = () => Response.json({detail: 'Token expired', error_type: 'auth'}, {status: 401});
  try {
    assert.equal((await apiError(expired())).status, 401);
    assert.equal(signedOut, 1);
    await assert.rejects(parseWire(expired(), schema), ApiError);
    assert.equal(signedOut, 2);
    // The answer commands meet a sign-in refusal in the general envelope too.
    assert.equal((await webCommandError(expired())).status, 401);
    assert.equal(signedOut, 3);

    for (const status of [400, 403, 404, 409, 422, 500, 503]) {
      await apiError(Response.json({detail: 'Refused.', error_type: 'auth'}, {status}));
      await webCommandError(Response.json({kind: 'scope_forbidden', message: 'No.'}, {status}));
    }
    assert.equal(signedOut, 3);
  } finally {
    stop();
  }
  await apiError(expired());
  assert.equal(signedOut, 3, 'a listener that stopped hears nothing');
});
