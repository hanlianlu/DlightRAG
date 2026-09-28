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
  const refused = new ApiError(503, {detail: 'Corpus writes are paused.', errorType: 'unavailable'});
  assert.equal(apiErrorMessage(refused, 'Upload failed.'), 'Corpus writes are paused.');
  assert.equal(apiErrorMessage(new ApiError(502), 'Upload failed.'), 'Upload failed.');
  assert.equal(apiErrorMessage(new TypeError('network down'), 'Upload failed.'), 'Upload failed.');
});
