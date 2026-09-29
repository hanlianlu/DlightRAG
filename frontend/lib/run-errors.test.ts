// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {answerErrorMessage, localizedErrorKind, localizedStoredRunError} from './run-errors.ts';

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

test('known kinds localize and unknown or inherited names do not', () => {
  assert.equal(
    localizedErrorKind('unsupported_resource_capability'),
    'This request needs a resource capability that no answer mode can provide.',
  );
  assert.equal(localizedErrorKind('UNSUPPORTED_RESOURCE_CAPABILITY'), null);
  assert.equal(localizedErrorKind('constructor'), null);
  assert.equal(localizedErrorKind(null), null);
});

test('a stored failure prefers kind copy, then the server message', () => {
  assert.equal(
    localizedStoredRunError('ANSWER_RESOURCE_INVALID', 'raw'),
    'An answer attachment or link could not be admitted safely.',
  );
  assert.equal(localizedStoredRunError('run_execution_failed', 'Server said why.'), 'Server said why.');
});
