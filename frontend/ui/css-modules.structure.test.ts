// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock: Feature CSS Modules keep every rule in `@layer features` (ADR 0003).
 *
 * An unlayered rule outranks every layered one whatever its specificity, so a
 * rule left outside the block would win silently.
 */

import {readdirSync, readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const STYLES = join(dirname(fileURLToPath(import.meta.url)), '..', 'styles');

/** Preludes of a stylesheet's top-level statements, comments removed. */
function topLevelStatements(css: string): string[] {
  const source = css.replace(/\/\*[\s\S]*?\*\//g, '');
  const statements: string[] = [];
  let depth = 0;
  let start = 0;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (char === '{') {
      if (depth === 0) statements.push(source.slice(start, index));
      depth += 1;
    } else if (char === '}') {
      depth -= 1;
      if (depth === 0) start = index + 1;
    } else if (char === ';' && depth === 0) {
      statements.push(source.slice(start, index));
      start = index + 1;
    }
  }
  statements.push(source.slice(start));
  return statements.map((statement) => statement.trim().replace(/\s+/g, ' ')).filter(Boolean);
}

test('every CSS Module rule sits in @layer features', () => {
  const modules = readdirSync(STYLES).filter((name) => name.endsWith('.module.css'));
  assert.ok(modules.length > 0, 'no CSS Modules to lock');
  const outside = modules.flatMap((name) => (
    topLevelStatements(readFileSync(join(STYLES, name), 'utf8'))
      .filter((statement) => statement !== '@layer features')
      .map((statement) => `${name}: ${statement}`)
  ));
  assert.deepEqual(outside, []);
});
