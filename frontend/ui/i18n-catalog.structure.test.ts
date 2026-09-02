// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on the localization catalogs.

 * Every statically declared `id:` on a msg() call in ui/ and lib/ must exist
 * in the zh catalog, so a renamed or newly added message cannot silently fall
 * back to English under a non-English locale. Dynamically derived ids
 * (template-literal ids like `chatFeature.phase.${phase}`) are outside the
 * lock and stay covered by the browser locale tests.
 */

import {readdirSync, readFileSync, statSync} from 'node:fs';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const FRONTEND_DIR = join(dirname(fileURLToPath(import.meta.url)), '..');

function sourceFiles(): string[] {
  const files: string[] = [];
  for (const relative of ['ui', 'lib']) {
    const dir = join(FRONTEND_DIR, relative);
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      if (statSync(path).isFile() && name.endsWith('.ts') && !name.endsWith('.test.ts')) {
        files.push(path);
      }
    }
  }
  return files;
}

function declaredIds(): Set<string> {
  const ids = new Set<string>();
  for (const file of sourceFiles()) {
    const source = readFileSync(file, 'utf8');
    for (const match of source.matchAll(/\bid:\s*'([A-Za-z][\w.]*)'/g)) {
      ids.add(match[1]);
    }
  }
  return ids;
}

test('every declared msg id exists in the zh catalog', () => {
  const catalog = readFileSync(
    join(FRONTEND_DIR, 'i18n', 'locales', 'zh.ts'),
    'utf8',
  );
  const catalogKeys = new Set(
    [...catalog.matchAll(/^\s*'([A-Za-z][\w.]*)':/gm)].map((match) => match[1]),
  );
  const missing = [...declaredIds()].filter((id) => !catalogKeys.has(id));
  assert.deepEqual(missing, []);
});
