// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on which layer may import which.
 *
 * The product's modules sit in layers, and an import runs one way: toward the layer that knows less of
 * the product. `api` is the wire and imports nothing of the product; `lib` is what the layers above share
 * and may use `api` and the design system's primitives, which know no product at all; `stores` hold state
 * and may use `api` and `lib`; `ui` composes everything. A layer that reaches
 * back (a store that imports a Feature, a wire module that imports a store) lets the one that knows less
 * depend on the one that knows more, and the layers stop meaning anything. An import counts the same
 * whether it carries a value or only a type: a type is a dependency too.
 *
 * The matrix is documented in docs/architecture.md (Web Frontend Ownership); this is its test.
 */

import assert from 'node:assert/strict';
import {readdirSync, readFileSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';

const FRONTEND = dirname(fileURLToPath(import.meta.url));

/** Which layers each layer may import from. A layer always may import itself. */
const ALLOWED: Readonly<Record<string, readonly string[]>> = {
  // The wire: no module of the product, except the pure helpers of `lib` named below.
  api: [],
  lib: ['api', 'design-system'],
  stores: ['api', 'lib'],
  // Language loading reads the stored choice and the language list, and nothing above it.
  i18n: ['lib'],
  // Primitives know no product at all.
  'design-system': [],
  ui: ['api', 'lib', 'stores', 'i18n', 'design-system'],
};

/** Directories whose modules this test does not constrain: test support, token tooling, build scripts. */
const UNCONSTRAINED = new Set(['testing', 'tokens', 'scripts', 'styles', 'node_modules']);

function productionSources(directory: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(directory, {withFileTypes: true})) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules' || entry.name === 'testing') continue;
      files.push(...productionSources(path));
    } else if (
      entry.name.endsWith('.ts')
      && !entry.name.endsWith('.test.ts')
      && !entry.name.endsWith('.d.ts')
    ) {
      files.push(path);
    }
  }
  return files;
}

/** Every module a source file imports, as paths relative to the frontend root: static imports of either
 *  kind, re-exports, side-effect imports, and dynamic `import()`. */
function importsOf(file: string): string[] {
  const source = readFileSync(file, 'utf8');
  const specifiers = [
    ...source.matchAll(/^[ \t]*(?:import|export)\b[^'";]*?\bfrom\s*['"](\.[^'"]+)['"]/gm),
    ...source.matchAll(/^[ \t]*import\s*['"](\.[^'"]+)['"]/gm),
    ...source.matchAll(/\bimport\(\s*['"](\.[^'"]+)['"]\s*\)/g),
  ].map((match) => match[1]!);
  return [...new Set(specifiers)]
    .filter((specifier) => specifier.endsWith('.ts'))
    .map((specifier) => relative(FRONTEND, resolve(dirname(file), specifier)));
}

const sources = new Map<string, string[]>();
for (const entry of readdirSync(FRONTEND, {withFileTypes: true})) {
  if (!entry.isDirectory() || UNCONSTRAINED.has(entry.name)) continue;
  for (const file of productionSources(join(FRONTEND, entry.name))) {
    sources.set(relative(FRONTEND, file), importsOf(file));
  }
}

const layerOf = (path: string): string => path.split('/')[0]!;

test('every directory of product modules is placed in the layer matrix', () => {
  const unplaced = [...new Set([...sources.keys()].map(layerOf))].filter((layer) => !(layer in ALLOWED));
  assert.deepEqual(unplaced, [], 'place the new directory in ALLOWED (and in docs/architecture.md)');
  assert.ok(sources.size > 100, 'the product modules were not found');
});

/** A `lib` module that stays within `lib`: it imports no other layer, so a layer below it may use it
 *  without being used back. */
function isPureHelper(path: string, seen: Set<string> = new Set()): boolean {
  if (layerOf(path) !== 'lib') return false;
  if (seen.has(path)) return true;
  seen.add(path);
  return (sources.get(path) ?? []).every((imported) => isPureHelper(imported, seen));
}

test('imports run one way: a layer imports only from the layers the matrix names', () => {
  const offenders: string[] = [];
  for (const [path, imports] of sources) {
    const layer = layerOf(path);
    const allowed = ALLOWED[layer];
    if (!allowed) continue;
    for (const imported of imports) {
      const target = layerOf(imported);
      if (target === layer || UNCONSTRAINED.has(target)) continue;
      if (allowed.includes(target)) continue;
      // The wire may use the pure helpers of `lib` (who an agent is, the agent efforts): they import
      // nothing of the product, so using them does not make the wire depend on what is above it.
      if (layer === 'api' && isPureHelper(imported)) continue;
      offenders.push(`${path} -> ${imported}`);
    }
  }
  assert.deepEqual(offenders, [], 'a layer imported from one above it, or from one it may not know');
});

/** The groups of modules that import one another, found as strongly connected components. */
function cycles(): string[][] {
  const index = new Map<string, number>();
  const low = new Map<string, number>();
  const onStack = new Set<string>();
  const stack: string[] = [];
  const found: string[][] = [];
  let counter = 0;
  const visit = (path: string): void => {
    index.set(path, counter);
    low.set(path, counter);
    counter += 1;
    stack.push(path);
    onStack.add(path);
    for (const imported of sources.get(path) ?? []) {
      if (!sources.has(imported)) continue;
      if (!index.has(imported)) {
        visit(imported);
        low.set(path, Math.min(low.get(path)!, low.get(imported)!));
      } else if (onStack.has(imported)) {
        low.set(path, Math.min(low.get(path)!, index.get(imported)!));
      }
    }
    if (low.get(path) !== index.get(path)) return;
    const group: string[] = [];
    for (let member = stack.pop()!; ; member = stack.pop()!) {
      onStack.delete(member);
      group.push(member);
      if (member === path) break;
    }
    if (group.length > 1 || (sources.get(path) ?? []).includes(path)) found.push(group.sort());
  };
  for (const path of sources.keys()) if (!index.has(path)) visit(path);
  return found;
}

test('no two modules import one another, directly or through others', () => {
  assert.deepEqual(cycles().map((group) => group.join(' <-> ')), []);
});

test('the matrix finds what it is meant to find', () => {
  // A check that never fails is not a check: it must see the edges it forbids and the ones it allows.
  const edges = (from: string, to: string): number => [...sources].filter(
    ([path, imports]) => layerOf(path) === from && imports.some((imported) => layerOf(imported) === to),
  ).length;
  assert.ok(edges('ui', 'stores') > 0, 'ui imports stores');
  assert.ok(edges('stores', 'api') > 0, 'stores import api');
  assert.ok(edges('lib', 'api') > 0, 'lib imports api');
  assert.ok(edges('api', 'lib') > 0, 'api uses the pure helpers of lib');
  assert.equal(isPureHelper('lib/agent-id.ts'), true);
  assert.equal(isPureHelper('lib/errors.ts'), false, 'errors.ts imports api, so api may not use it');
  assert.equal(isPureHelper('ui/app.ts'), false);
  assert.ok(importsOf(join(FRONTEND, 'ui/app.ts')).includes('stores/app-handles.ts'));
  assert.deepEqual(importsOf(join(FRONTEND, 'lib/agent-id.ts')), []);
});
