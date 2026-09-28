// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {ApiError} from '../api/wire.ts';
import {answerErrorMessage, apiErrorMessage} from '../lib/errors.ts';

test('answer errors reject non-object payloads', () => {
  assert.equal(answerErrorMessage('Document parsing failed.'), 'Service error. Please try again.');
});

test('answer errors accept structured message payloads', () => {
  assert.equal(
    answerErrorMessage({message: 'Could not read report.pdf.', error_kind: 'PARSE_FAILED'}),
    'Could not read report.pdf.',
  );
});

test('answer errors use the fallback for empty messages', () => {
  assert.equal(answerErrorMessage(''), 'Service error. Please try again.');
  assert.equal(answerErrorMessage({message: '   '}, 'Unavailable.'), 'Unavailable.');
});

test('answer errors do not expose malformed payload fields', () => {
  assert.equal(
    answerErrorMessage({detail: 'raw provider failure', error: {secret: 'token'}}),
    'Service error. Please try again.',
  );
  assert.equal(answerErrorMessage(null), 'Service error. Please try again.');
});

test('API refusals show the server reason, and the caller localizes everything else', () => {
  for (const [status, errorType] of [[422, 'validation'], [409, 'conflict'], [503, 'unavailable']] as const) {
    const refused = new ApiError(status, {detail: 'Corpus writes are paused.', errorType});
    assert.equal(apiErrorMessage(refused, 'Upload failed.'), 'Corpus writes are paused.');
  }
  assert.equal(apiErrorMessage(new ApiError(502), 'Upload failed.'), 'Upload failed.');
  assert.equal(apiErrorMessage(new TypeError('network down'), 'Upload failed.'), 'Upload failed.');
});

test('an authorization refusal is explained, never echoed', () => {
  const denied = new ApiError(403, {
    detail: 'Access denied for action=workspace.create workspace=finance',
    errorType: 'auth',
  });
  assert.equal(apiErrorMessage(denied, 'Failed to create workspace'), 'You do not have permission to do that.');
  const expired = new ApiError(401, {detail: 'Not authenticated', errorType: 'auth'});
  assert.equal(
    apiErrorMessage(expired, 'Failed to create workspace'),
    'Your session has ended. Sign in again to continue.',
  );
});
