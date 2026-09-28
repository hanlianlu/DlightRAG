// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {localizedErrorKind, localizedStoredRunError} from './run-errors.ts';

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
