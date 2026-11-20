// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Motion contract: a pulse is defined once, and a looping animation says what it does under reduced motion. */

import assert from 'node:assert/strict';
import {readdirSync, readFileSync} from 'node:fs';
import {basename, join, relative} from 'node:path';
import {test} from 'node:test';
import {fileURLToPath} from 'node:url';

const FRONTEND_DIR = fileURLToPath(new URL('..', import.meta.url));

function cssFilesUnder(directory: string): string[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return cssFilesUnder(path);
    return entry.name.endsWith('.css') ? [path] : [];
  });
}

const cssFiles = ['styles', 'design-system'].flatMap((directory) => cssFilesUnder(join(FRONTEND_DIR, directory)));

function withoutComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, (comment) => comment.replace(/[^\n]/g, ' '));
}

test('a pulse has one definition in the design system, and no module names a keyframes it lacks', () => {
  const definitions = new Map<string, string[]>();
  for (const path of cssFiles) {
    for (const match of withoutComments(readFileSync(path, 'utf8')).matchAll(/@keyframes\s+([\w-]+)/g)) {
      definitions.set(match[1], [...(definitions.get(match[1]) ?? []), relative(FRONTEND_DIR, path)]);
    }
  }
  assert.deepEqual(
    [...definitions].filter(([name]) => /pulse/.test(name)),
    [['dl-pulse', ['design-system/primitives/primitives.css']]],
  );

  // A CSS Module renames every animation name it uses, even one it does not define; global() keeps the
  // design system's keyframes under its own name. A bare name that the module does not define would be
  // renamed to nothing and the animation would never run, and so would a global() that names no keyframes
  // of the design system.
  const shared = new Set([...definitions]
    .filter(([, files]) => files.some((file) => file.startsWith('design-system/')))
    .map(([name]) => name));
  const keywords = new Set([
    'none', 'infinite', 'normal', 'reverse', 'alternate', 'alternate-reverse', 'forwards', 'backwards', 'both',
    'running', 'paused', 'ease', 'ease-in', 'ease-out', 'ease-in-out', 'linear', 'step-start', 'step-end',
    'initial', 'inherit', 'unset', 'revert', 'revert-layer',
  ]);
  for (const path of cssFiles.filter((file) => file.endsWith('.module.css'))) {
    const source = withoutComments(readFileSync(path, 'utf8'));
    const own = new Set([...source.matchAll(/@keyframes\s+([\w-]+)/g)].map((match) => match[1]));
    for (const match of source.matchAll(/(?<![\w-])animation(?:-name)?\s*:\s*([^;]+);/g)) {
      for (const [, name] of match[1].matchAll(/\bglobal\(\s*([\w-]+)\s*\)/g)) {
        assert.ok(shared.has(name), `${basename(path)}: global(${name}) names no keyframes of the design system`);
      }
      const outsideFunctions = match[1].replace(/[\w-]+\([^()]*(?:\([^()]*\)[^()]*)*\)/g, ' ');
      for (const name of outsideFunctions.split(/[\s,]+/).filter((word) => /^[a-z_][\w-]*$/i.test(word))) {
        assert.ok(
          keywords.has(name) || own.has(name),
          `${basename(path)}: animation "${name}" is not defined here; name a shared keyframes as global(${name})`,
        );
      }
    }
  }
});

test('every file with a looping animation states what it does under reduced motion', () => {
  for (const path of cssFiles) {
    const source = withoutComments(readFileSync(path, 'utf8'));
    if (!/animation(?:-iteration-count)?\s*:[^;]*\binfinite\b/.test(source)) continue;
    assert.match(source, /@media \(prefers-reduced-motion: reduce\)/,
      `${relative(FRONTEND_DIR, path)} loops without a reduced-motion rule`);
  }
});
