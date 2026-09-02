// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on module file naming.

 * Every source file in the frontend is kebab-case (the convention of Lit's own
 * repositories and modern scaffolds), so a stray camelCase or snake_case file
 * cannot reintroduce the split this rule retired.
 */

import {readdirSync, statSync} from 'node:fs';
import {dirname, join, relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const FRONTEND_DIR = dirname(fileURLToPath(import.meta.url));
const SKIP_DIRS = new Set(['node_modules']);
const KEBAB_NAME = /^\.?[a-z0-9][a-z0-9-]*(\.[a-z0-9]+)+$/;

function sourceFiles(): string[] {
  const files: string[] = [];
  const walk = (dir: string): void => {
    for (const name of readdirSync(dir)) {
      if (SKIP_DIRS.has(name)) continue;
      const path = join(dir, name);
      if (statSync(path).isDirectory()) walk(path);
      else if (/\.(ts|mjs|css|html|json)$/.test(name)) files.push(path);
    }
  };
  walk(FRONTEND_DIR);
  return files;
}

test('every frontend source file is kebab-case', () => {
  const offenders = sourceFiles()
    .map((path) => relative(FRONTEND_DIR, path))
    .filter((path) => !KEBAB_NAME.test(path.split('/').pop() ?? ''));
  assert.deepEqual(offenders, []);
});
