// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock: the Shell may query Feature custom elements, not internals. */

import {readFileSync} from 'node:fs';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const APP = join(dirname(fileURLToPath(import.meta.url)), 'app.ts');

const FEATURE_TAGS = new Set([
  'dl-artifact-canvas',
  'dl-chat-feature',
  'dl-children-roster',
  'dl-continuation-dialog',
  'dl-conversation-sidebar',
  'dl-image-lightbox',
  'dl-inspector',
  'dl-settings-dialog',
  'dl-toast-region',
]);

test('shell querySelector targets are Feature custom elements', () => {
  const source = readFileSync(APP, 'utf8');
  const selectors = [
    ...source.matchAll(/querySelector(?:All)?(?:<[^>]+>)?\('([^']+)'\)/g),
  ].map((match) => match[1]);
  assert.ok(selectors.length > 0, 'dl-app has no querySelector calls to lock');
  const illegal = selectors.filter((selector) => (
    !FEATURE_TAGS.has(selector)
    || selector.includes('.')
    || selector.includes('#')
    || selector.includes(' ')
  ));
  assert.deepEqual(illegal, []);
});
