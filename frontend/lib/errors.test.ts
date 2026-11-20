// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {ApiError, apiError as refusal} from '../api/wire.ts';
import {refusalText} from '../lib/errors.ts';
import {localizedText} from '../lib/localized-text.ts';

/** The words a reader would see, which is what these cases are about. */
function words(error: unknown, fallback: string): string {
  return localizedText(refusalText(error, {source: fallback, id: null}));
}

test('API refusals show the server reason, and the caller localizes everything else', () => {
  for (const [status, errorType] of [[422, 'validation'], [409, 'conflict'], [503, 'unavailable']] as const) {
    const refused = new ApiError(status, {detail: 'Corpus writes are paused.', errorType});
    assert.equal(words(refused, 'Upload failed.'), 'Corpus writes are paused.');
  }
  for (const errorType of ['not_found', 'configuration', 'internal'] as const) {
    const refused = new ApiError(404, {detail: 'Workspace not found', errorType});
    assert.equal(words(refused, 'Upload failed.'), 'Workspace not found', 'every non-auth type');
  }
  assert.equal(words(new ApiError(502), 'Upload failed.'), 'Upload failed.');
  assert.equal(words(new TypeError('network down'), 'Upload failed.'), 'Upload failed.');
});

test('a refusal naming a known error kind gets that kind\'s copy, id and all', () => {
  const refused = new ApiError(422, {
    detail: 'Current model does not support image input. [CURRENT_IMAGES_UNSUPPORTED]',
    errorType: 'validation',
    errorKind: 'CURRENT_IMAGES_UNSUPPORTED',
  });
  assert.deepEqual(refusalText(refused, {source: 'Upload failed.', id: null}), {
    source: 'Current model does not support image input. Use a vision-capable model or remove images.',
    id: 'errors.kind.CURRENT_IMAGES_UNSUPPORTED',
  });
  const unmapped = new ApiError(422, {detail: 'Server said why.', errorKind: 'not_a_known_kind'});
  assert.equal(words(unmapped, 'Upload failed.'), 'Server said why.');
});

test('a bare 403 is not read as an authorization refusal', async () => {
  // A 403 whose body is not the envelope, such as a proxy's, names no type.
  const bare = await refusal(new Response('Forbidden', {status: 403}));
  assert.equal(bare.errorType, null);
  assert.equal(words(bare, 'Failed to create workspace'), 'Failed to create workspace');
});

test('a cross-origin refusal says the origin could not be verified, not that access is denied', async () => {
  const guarded = await refusal(Response.json(
    {detail: 'Cross-origin request rejected', error_type: 'auth', error_kind: 'cross_origin_rejected'},
    {status: 403},
  ));
  assert.equal(
    words(guarded, 'Failed to create workspace'),
    'This request was blocked because its origin could not be verified. Reload the page and try again.',
  );
});

test('an authorization refusal is explained, never echoed', () => {
  const denied = new ApiError(403, {
    detail: 'Access denied for action=workspace.create workspace=finance',
    errorType: 'auth',
  });
  assert.equal(words(denied, 'Failed to create workspace'), 'You do not have permission to do that.');
  const expired = new ApiError(401, {detail: 'Not authenticated', errorType: 'auth'});
  assert.equal(
    words(expired, 'Failed to create workspace'),
    'Your session has ended. Sign in again to continue.',
  );
});
