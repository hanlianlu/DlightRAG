// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {formatFileSize} from './file-size.ts';

test('attachment sizes preserve compact whole and fractional units', () => {
  assert.equal(formatFileSize(2 * 1024), '2 KB');
  assert.equal(formatFileSize(1536), '1.5 KB');
  assert.equal(formatFileSize(12 * 1024), '12 KB');
  assert.equal(formatFileSize(2 * 1024 * 1024), '2 MB');
  assert.equal(formatFileSize(3 * 1024 * 1024 * 1024), '3 GB');
});
