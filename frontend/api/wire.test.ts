// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import * as v from 'valibot';
import {ApiError, apiError, parseWire} from './wire.ts';

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

test('a body that is not the envelope keeps the status and the type it implies', async () => {
  const cases: Array<[Response, string]> = [
    [new Response('<html>Bad gateway</html>', {status: 502}), 'internal'],
    [new Response('', {status: 403}), 'auth'],
    [new Response('', {status: 410}), 'not_found'],
    [new Response('', {status: 412}), 'conflict'],
    [new Response('', {status: 429}), 'unavailable'],
    // FastAPI's own request validation answers a list, not a reason.
    [Response.json({detail: [{loc: ['body'], msg: 'field required'}]}, {status: 422}), 'validation'],
    // An unknown type is not trusted over the status.
    [Response.json({detail: 'Gone.', error_type: 'mystery'}, {status: 404}), 'not_found'],
  ];
  for (const [response, errorType] of cases) {
    const error = await apiError(response);
    assert.equal(error.status, response.status);
    assert.equal(error.errorType, errorType);
    assert.equal(error.errorKind, null);
  }
  assert.equal((await apiError(Response.json({detail: '   '}, {status: 409}))).detail, null);
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
