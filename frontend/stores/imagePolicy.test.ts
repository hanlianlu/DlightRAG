// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {acceptsImageUpload} from '../ui/image_policy.ts';

const policy = {maxCurrentImages: 3, maxUploadBytes: 15 * 1024 * 1024};

test('browser image admission accepts the exact configured byte limit', () => {
  assert.equal(
    acceptsImageUpload(
      {type: 'image/png', size: policy.maxUploadBytes},
      0,
      policy,
    ),
    true,
  );
});

test('browser image admission rejects bytes or count above configured limits', () => {
  assert.equal(
    acceptsImageUpload(
      {type: 'image/png', size: policy.maxUploadBytes + 1},
      0,
      policy,
    ),
    false,
  );
  assert.equal(
    acceptsImageUpload({type: 'image/png', size: 1}, policy.maxCurrentImages, policy),
    false,
  );
});
