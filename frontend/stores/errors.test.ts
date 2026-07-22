// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {
  answerErrorMessage,
  answerWarningMessage,
  notifyAnswerWarning,
} from '../lib/errors.ts';
import type {WarningPayload} from '../lib/errors.ts';

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

test('answer warnings accept a typed aggregate payload', () => {
  const payload: WarningPayload = {
    message: 'Could not read history-report.pdf. The answer will continue without it.',
    documents: [
      {
        code: 'HISTORICAL_DOCUMENT_PARSE_FAILED',
        filename: 'history-report.pdf',
        message: 'Could not read history-report.pdf. The answer will continue without it.',
      },
    ],
  };

  assert.equal(answerWarningMessage(payload), payload.message);
});

test('answer warnings reject malformed document details', () => {
  const malformedPayloads: unknown[] = [
    null,
    {message: 'Forged warning', documents: []},
    {message: 'Forged warning', documents: 'not-an-array'},
    {
      message: 'Forged warning',
      documents: [{code: 'UNKNOWN', filename: 'report.pdf', message: 'Unavailable'}],
    },
    {
      message: 'Forged warning',
      documents: [
        {
          code: 'HISTORICAL_DOCUMENT_UNAVAILABLE',
          filename: '',
          message: 'Unavailable',
        },
      ],
    },
  ];

  for (const payload of malformedPayloads) {
    assert.equal(
      answerWarningMessage(payload),
      'A document could not be used. The answer will continue without it.',
    );
  }
});

test('answer warnings invoke the callback once per aggregate event', () => {
  let callCount = 0;
  const messages: string[] = [];
  const payload: WarningPayload = {
    message: '2 referenced documents could not be used. The answer will continue without them.',
    documents: [
      {
        code: 'HISTORICAL_DOCUMENT_PARSE_FAILED',
        filename: 'report.pdf',
        message: 'Could not read report.pdf.',
      },
      {
        code: 'HISTORICAL_DOCUMENT_UNAVAILABLE',
        filename: 'appendix.pdf',
        message: 'appendix.pdf is no longer available.',
      },
    ],
  };

  notifyAnswerWarning(
    payload,
    (message) => {
      callCount += 1;
      messages.push(message);
    },
  );

  assert.equal(callCount, 1);
  assert.deepEqual(messages, [payload.message]);
});
