// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {answerErrorText, errorKindText, runErrorText, storedRunErrorText} from './run-errors.ts';

const SERVICE_ERROR = {source: 'Service error. Please try again.', id: 'errors.service'};

test('answer errors reject non-object payloads', () => {
  assert.deepEqual(answerErrorText('Document parsing failed.'), SERVICE_ERROR);
});

test('answer errors accept structured message payloads', () => {
  assert.deepEqual(
    answerErrorText({message: 'Could not read report.pdf.', error_kind: 'PARSE_FAILED'}),
    {source: 'Could not read report.pdf.', id: null},
  );
});

test('answer errors use the fallback for empty messages', () => {
  assert.deepEqual(answerErrorText(''), SERVICE_ERROR);
  assert.deepEqual(
    answerErrorText({message: '   '}, {source: 'Unavailable.', id: null}),
    {source: 'Unavailable.', id: null},
  );
});

test('answer errors do not expose malformed payload fields', () => {
  assert.deepEqual(
    answerErrorText({detail: 'raw provider failure', error: {secret: 'token'}}),
    SERVICE_ERROR,
  );
  assert.deepEqual(answerErrorText(null), SERVICE_ERROR);
});

test('known kinds carry their catalog id and unknown or inherited names do not', () => {
  assert.deepEqual(errorKindText('unsupported_resource_capability'), {
    source: 'This request needs a resource capability that no answer mode can provide.',
    id: 'errors.kind.unsupported_resource_capability',
  });
  assert.equal(errorKindText('UNSUPPORTED_RESOURCE_CAPABILITY'), null);
  assert.equal(errorKindText('constructor'), null);
  assert.equal(errorKindText(null), null);
});

test('a live failure prefers kind copy, then the server message', () => {
  assert.deepEqual(runErrorText({kind: 'ANSWER_RESOURCE_INVALID', message: 'raw'}), {
    source: 'An answer attachment or link could not be admitted safely.',
    id: 'errors.kind.ANSWER_RESOURCE_INVALID',
  });
  assert.deepEqual(runErrorText({message: 'Server said why.'}), {
    source: 'Server said why.',
    id: null,
  });
});

test('a stored failure prefers kind copy, then the server message', () => {
  assert.deepEqual(storedRunErrorText('ANSWER_RESOURCE_INVALID', 'raw'), {
    source: 'An answer attachment or link could not be admitted safely.',
    id: 'errors.kind.ANSWER_RESOURCE_INVALID',
  });
  assert.deepEqual(storedRunErrorText('run_execution_failed', 'Server said why.'), {
    source: 'Server said why.',
    id: null,
  });
});
