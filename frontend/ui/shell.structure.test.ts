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
  'dl-continuation-dialog',
  'dl-conversation-sidebar',
  'dl-image-lightbox',
  'dl-inspector',
  'dl-settings-dialog',
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
  assert.equal(source.includes('document.getElementById'), false, 'the Shell reaches Features by tag, never by id');
});

const SHARED_STORES = [
  'ConversationStore',
  'WorkspaceStore',
  'IngestStore',
  'AttachmentStore',
  'AnswerEventCursorStore',
  'AnswerSubmissionRegistry',
  'KeptJudgements',
  'KeptTraces',
  'RunActivityStore',
];

test('createAppHandles is the only constructor of the shared stores', () => {
  const frontend = join(dirname(APP), '..');
  const offenders: string[] = [];
  for (const directory of ['api', 'lib', 'stores', 'ui']) {
    for (const name of readdirSync(join(frontend, directory))) {
      if (!name.endsWith('.ts') || name.endsWith('.test.ts')) continue;
      if (directory === 'stores' && name === 'app-handles.ts') continue;
      const source = readFileSync(join(frontend, directory, name), 'utf8');
      for (const store of SHARED_STORES) {
        if (new RegExp(`\\bnew ${store}\\(`).test(source)) offenders.push(`${directory}/${name}: ${store}`);
      }
    }
  }
  assert.deepEqual(offenders, []);
  const handles = readFileSync(join(frontend, 'stores', 'app-handles.ts'), 'utf8');
  for (const store of SHARED_STORES) assert.match(handles, new RegExp(`\\bnew ${store}\\(`));
});

test('no store module holds an instance of its own: state a module keeps is the bag\'s', () => {
  const frontend = join(dirname(APP), '..');
  const offenders: string[] = [];
  for (const name of readdirSync(join(frontend, 'stores'))) {
    if (!name.endsWith('.ts') || name.endsWith('.test.ts') || name === 'app-handles.ts') continue;
    const source = readFileSync(join(frontend, 'stores', name), 'utf8');
    // A top-level `const x = new Thing(` / `let x` / `export const x = new Thing(`: state that outlives
    // any bag and that no sign-out reset reaches.
    for (const match of source.matchAll(/^(?:export )?(?:let [a-zA-Z_]\w*|const [a-zA-Z_]\w* = new [A-Z]\w*\()/gm)) {
      offenders.push(`stores/${name}: ${match[0].trim()}`);
    }
  }
  assert.deepEqual(offenders, []);
});
