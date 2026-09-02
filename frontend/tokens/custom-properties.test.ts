// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {readdirSync, readFileSync} from 'node:fs';
import {basename, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import {DESIGN_SYSTEM_TOKEN_GROUPS} from './design-system-tokens.ts';

const FRONTEND_DIR = fileURLToPath(new URL('..', import.meta.url));
const CSS_DIRS = ['styles', 'design-system'].map((directory) => join(FRONTEND_DIR, directory));

interface CssSource {
  path: string;
  source: string;
}

function cssSourcesIn(directory: string): CssSource[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return cssSourcesIn(path);
    return entry.name.endsWith('.css') ? [{path, source: readFileSync(path, 'utf8')}] : [];
  });
}

function cssSources(): CssSource[] {
  return CSS_DIRS.flatMap(cssSourcesIn);
}

function withoutComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, (comment) => comment.replace(/[^\n]/g, ' '));
}

function declaredCustomProperties(source: string): Set<string> {
  return new Set([...withoutComments(source).matchAll(/(?<![\w-])(--[\w-]+)\s*:/g)]
    .map((match) => match[1]));
}

function sharedDeclarations(sources: readonly CssSource[]): Set<string> {
  return new Set(sources
    .filter(({path}) => path.includes('/design-system/foundations/')
      || basename(path) === 'global.css')
    .flatMap(({source}) => [...declaredCustomProperties(source)]));
}

function undefinedCustomProperties(sources: readonly CssSource[]): string[] {
  const shared = sharedDeclarations(sources);
  const errors: string[] = [];
  for (const {path, source} of sources) {
    const local = declaredCustomProperties(source);
    const uncommented = withoutComments(source);
    for (const match of uncommented.matchAll(/var\(\s*(--[\w-]+)\s*(,)?/g)) {
      const [, property, fallback] = match;
      if (shared.has(property) || local.has(property) || fallback) continue;
      const line = uncommented.slice(0, match.index).split('\n').length;
      errors.push(`${basename(path)}:${line}: ${property}`);
    }
  }
  return errors;
}

test('every CSS custom-property use is shared, source-local, or explicitly fallbacked', () => {
  assert.deepEqual(undefinedCustomProperties(cssSources()), []);
});

test('the design-system token specimen references only declared shared tokens', () => {
  const declared = sharedDeclarations(cssSources());
  const referenced = DESIGN_SYSTEM_TOKEN_GROUPS.flatMap(([primary, related]) => [
    primary,
    ...related,
  ]);
  assert.deepEqual(
    referenced.filter((name) => !declared.has(`--${name}`)),
    [],
  );
});

test('the CSS custom-property contract permits explicit fallback syntax', () => {
  const fixture = [{
    path: 'fallback.css',
    source: '.fallback { color: var(--optional-embed-color, currentcolor); }',
  }];
  assert.deepEqual(undefinedCustomProperties(fixture), []);
});

test('comments and declarations in unrelated component files cannot hide an undefined use', () => {
  const fixture = [
    {path: 'component.css', source: '.local { --component-only: red; }'},
    {
      path: 'consumer.css',
      source: '/* --comment-only: red; */\n.consumer { color: var(--component-only); }',
    },
  ];
  assert.deepEqual(undefinedCustomProperties(fixture), [
    'consumer.css:2: --component-only',
  ]);
});
