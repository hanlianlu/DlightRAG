// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {readdirSync, readFileSync} from 'node:fs';
import {basename, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';

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

test('the design-system token specimen references only declared shared tokens', () => {
  const declared = sharedDeclarations(cssSources());
  const page = readFileSync(join(FRONTEND_DIR, 'ui', 'design-system.ts'), 'utf8');
  const list = page.match(/const TOKEN_NAMES = \[([^\]]+)\]/)?.[1];
  assert.ok(list, 'the specimen page states the tokens it shows');
  const referenced = [...list.matchAll(/'([\w-]+)'/g)].map((match) => match[1]);
  assert.ok(referenced.length > 0);
  assert.deepEqual(
    referenced.filter((name) => !declared.has(`--${name}`)),
    [],
  );
});

/** Every `var(--name, fallback)` in a source, with the fallback read to its closing parenthesis. */
function varFallbacks(source: string): Array<[name: string, fallback: string]> {
  const found: Array<[string, string]> = [];
  for (const match of source.matchAll(/var\(\s*(--[\w-]+)\s*,/g)) {
    const start = match.index + match[0].length;
    let depth = 1;
    let end = start;
    for (; end < source.length && depth > 0; end += 1) {
      if (source[end] === '(') depth += 1;
      else if (source[end] === ')') depth -= 1;
    }
    found.push([match[1], source.slice(start, end - 1).trim()]);
  }
  return found;
}

/** What each foundation token holds on `:root` before a theme, media query or forced-colors rule restates it. */
function rootTokenValues(): Map<string, string> {
  const directory = join(FRONTEND_DIR, 'design-system/foundations');
  const values = new Map<string, string>();
  for (const name of readdirSync(directory).filter((file) => file.endsWith('.css')).sort()) {
    const source = withoutComments(readFileSync(join(directory, name), 'utf8'));
    const lightAt = source.indexOf(":root[data-color-mode='light']");
    const defaults = (lightAt < 0 ? source : source.slice(0, lightAt))
      .replace(/@media[^{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}/g, '');
    for (const match of defaults.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
      values.set(match[1], match[2].trim());
    }
  }
  return values;
}

function resolveTokenValue(value: string, tokens: ReadonlyMap<string, string>): string {
  let resolved = value;
  for (let step = 0; step < 8 && /var\(--[\w-]+\)/.test(resolved); step += 1) {
    resolved = resolved.replace(/var\((--[\w-]+)\)/g, (reference, name: string) => tokens.get(name) ?? reference);
  }
  return resolved.toLowerCase().replace(/\s+/g, ' ');
}

/** Properties an element reads as inputs of its own rather than foundation tokens, each with the default the
 *  design-system README documents for it. A page sets them on the element; nothing else declares them. */
const ELEMENT_INPUT_DEFAULTS = new Map([
  ['--split-start-layer', '0'],
  ['--split-end-layer', '0'],
]);

function elementSources(): Array<[file: string, source: string]> {
  const directory = join(FRONTEND_DIR, 'design-system/elements');
  return readdirSync(directory)
    .filter((file) => file.endsWith('.ts') && !file.endsWith('.test.ts'))
    .sort()
    .map((file) => [file, readFileSync(join(directory, file), 'utf8')]);
}

test('every shadow element\'s var() fallbacks restate the token\'s :root value', () => {
  const tokens = rootTokenValues();
  const withFallbacks = new Set<string>();
  for (const [file, source] of elementSources()) {
    for (const [name, fallback] of varFallbacks(source)) {
      withFallbacks.add(file);
      const normalized = fallback.toLowerCase().replace(/\s+/g, ' ');
      const input = ELEMENT_INPUT_DEFAULTS.get(name);
      if (input !== undefined) {
        assert.equal(normalized, input, `${file}: the fallback of ${name} must be its documented default`);
        continue;
      }
      const value = tokens.get(name);
      assert.notEqual(value, undefined, `${file}: ${name} is neither a foundation token nor a listed element input`);
      assert.equal(
        normalized,
        resolveTokenValue(value ?? '', tokens),
        `${file}: the fallback of ${name} must be the token's own value`,
      );
    }
  }
  // The two elements that style a shadow tree; a rename or a move must not leave the test checking nothing.
  assert.deepEqual([...withFallbacks].sort(), ['icon-button.ts', 'split-layout.ts']);
});
