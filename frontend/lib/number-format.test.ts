// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {formatCompactCount} from './number-format.ts';

test('token totals are compacted to the reader unit', () => {
  assert.equal(formatCompactCount(12344750, 'en'), '12.3M');
  assert.equal(formatCompactCount(9400, 'en'), '9.4K');
  assert.equal(formatCompactCount(999, 'en'), '999');
  assert.equal(formatCompactCount(1000, 'en'), '1K');
  assert.equal(formatCompactCount(0, 'en'), '0');
});

test('a reader of another locale gets that locale unit', () => {
  assert.equal(formatCompactCount(12344750, 'zh-CN'), '1234.5万');
});

test('a count that is not a finite number reads as nothing', () => {
  assert.equal(formatCompactCount(Number.NaN, 'en'), '');
  assert.equal(formatCompactCount(Number.POSITIVE_INFINITY, 'en'), '');
});
