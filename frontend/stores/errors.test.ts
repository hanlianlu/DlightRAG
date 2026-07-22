// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {answerErrorMessage, notifyAnswerWarning} from '../lib/errors.ts';

test('answer errors accept legacy string payloads', () => {
  assert.equal(answerErrorMessage('Document parsing failed.'), 'Document parsing failed.');
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

test('answer warnings call the callback with the typed public message', () => {
  const messages: string[] = [];

  notifyAnswerWarning(
    {
      code: 'HISTORICAL_DOCUMENT_PARSE_FAILED',
      filename: 'history-report.pdf',
      message: 'Could not read history-report.pdf. The answer will continue without it.',
    },
    (message) => messages.push(message),
  );

  assert.deepEqual(messages, [
    'Could not read history-report.pdf. The answer will continue without it.',
  ]);
});

test('answer warnings use a nonfatal fallback for malformed payloads', () => {
  const messages: string[] = [];

  notifyAnswerWarning(
    {detail: 'raw parser exception', error: {secret: 'token'}},
    (message) => messages.push(message),
  );

  assert.deepEqual(messages, [
    'A document could not be used. The answer will continue without it.',
  ]);
});