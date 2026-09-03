// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock: the Shell may query Feature custom elements, not internals. */

import {readdirSync, readFileSync} from 'node:fs';
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

const STORE_SINGLETONS = [
  'conversationStore',
  'workspaceStore',
  'ingestStore',
  'attachmentStore',
  'answerEventCursorStore',
];

test('features do not import store singleton values', () => {
  const dir = dirname(APP);
  const offenders: string[] = [];
  for (const name of readdirSync(dir)) {
    if (!name.endsWith('.ts') || name.endsWith('.test.ts')) continue;
    const source = readFileSync(join(dir, name), 'utf8');
    const importRe = new RegExp(String.raw`^import[\s\S]*?from '\.\./stores/[^']+';`, 'gm');
    for (const match of source.matchAll(importRe)) {
      const block = match[0];
      if (block.includes(' type ') && !block.includes('{')) continue;
      for (const singleton of STORE_SINGLETONS) {
        if (new RegExp(`\\b${singleton}\\b`).test(block) && !block.includes(`type ${singleton}`)) {
          offenders.push(`${name}: ${singleton}`);
        }
      }
    }
  }
  assert.deepEqual(offenders, []);
});
