// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {readdirSync, readFileSync} from 'node:fs';
import {join, relative} from 'node:path';
import test from 'node:test';

const frontend = process.cwd();

function filesUnder(directory: string, suffixes: readonly string[]): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(directory, {withFileTypes: true})) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...filesUnder(path, suffixes));
    else if (suffixes.some((suffix) => entry.name.endsWith(suffix))) files.push(path);
  }
  return files;
}

function source(path: string): string {
  return readFileSync(path, 'utf8');
}

const designSystemSources = filesUnder(join(frontend, 'design-system'), ['.ts'])
  .filter((path) => !path.endsWith('.test.ts'));
const featureSources = filesUnder(join(frontend, 'ui'), ['.ts'])
  .filter((path) => !path.endsWith('.test.ts'));
const governedSources = [
  ...featureSources,
  ...filesUnder(join(frontend, 'styles'), ['.css']),
  ...designSystemSources,
];
const RAW_SVG_ALLOWLIST = new Set([
  'ui/mathjax.ts', // Self-hosted MathJax owns its generated SVG output.
  'ui/mermaid.ts', // Mermaid returns sanitized content SVG, never a control icon.
]);

test('design-system sources import only Lit and one another', () => {
  // Product layers are locked by layers.structure.test.ts; this is the third-party half.
  for (const path of designSystemSources) {
    const specifiers = [...source(path).matchAll(/(?:\bfrom\s*|\bimport\s*\(?\s*)['"]([^'"]+)['"]/g)]
      .map((match) => match[1]!);
    for (const specifier of specifiers) {
      assert.ok(
        specifier.startsWith('.') || specifier === 'lit' || specifier.startsWith('lit/'),
        `${relative(frontend, path)} imports ${specifier}, which the design system may not depend on`,
      );
    }
  }
});

test('features consume only the public design-system entries', () => {
  const deepImport = /(?:from\s+)?['"]\.\.\/design-system\/(?!index\.(?:ts|css)['"])[^'"]+['"]/;
  for (const path of featureSources) {
    assert.equal(
      deepImport.test(source(path)),
      false,
      `${relative(frontend, path)} bypasses design-system/index.ts`,
    );
  }
});

test('visually hidden text wears the design system\'s dl-sr-only', () => {
  for (const path of governedSources) {
    assert.equal(/(?<![\w-])sr-only\b/.test(source(path)), false,
      `${relative(frontend, path)} uses sr-only; use the design system's dl-sr-only`);
  }
});

/** The text characters that stand in for an icon: a close cross, a play triangle, a chevron, a dot, a check. */
const GLYPH = /[✕×▶‹›●✓]|\\27(?:13|15)/u;

/** A multiplication sign between two values -- `${width}×${height}`, `1920×1080` -- is arithmetic written
 *  in words, not an icon, so the glyph rule skips it. An optional space may stand on either side of the
 *  sign. A lone sign (`<button>×</button>`, `content: '×'`) has no value on one side and stays caught. */
const MULTIPLICATION = /(?<=[\d}] ?)×(?= ?[\d$])/gu;

function hasGlyph(text: string): boolean {
  return GLYPH.test(text.replace(MULTIPLICATION, ' '));
}

test('the glyph rule allows a multiplication sign between values and catches a lone one', () => {
  assert.equal(hasGlyph(`<span>\${artifact.width}×\${artifact.height}</span>`), false);
  assert.equal(hasGlyph('const size = "1920×1080";'), false);
  assert.equal(hasGlyph('const size = "1920 × 1080";'), false);
  assert.equal(hasGlyph('<button aria-label="Close">×</button>'), true);
  assert.equal(hasGlyph("content: '×';"), true);
  assert.equal(hasGlyph(`<span>\${count}×</span>`), true);
  assert.equal(hasGlyph(`<span>×\${count}</span>`), true);
  assert.equal(hasGlyph('<span>✕</span>'), true);
});

test('feature UI has no raw SVG or text-glyph icons', () => {
  for (const path of featureSources) {
    const content = source(path);
    const projectPath = relative(frontend, path);
    if (!RAW_SVG_ALLOWLIST.has(projectPath)) {
      assert.equal(/<svg\b/i.test(content), false,
        `${projectPath} must use icon() instead of raw SVG`);
    }
    assert.equal(hasGlyph(content), false,
      `${relative(frontend, path)} must use icon() instead of a text glyph`);
  }
  for (const path of filesUnder(join(frontend, 'styles'), ['.css'])) {
    assert.equal(hasGlyph(source(path)), false,
      `${relative(frontend, path)} must not synthesize a glyph icon in CSS`);
  }
});

test('design-system elements have no import-time registration side effect', () => {
  for (const path of designSystemSources) {
    assert.equal(source(path).includes('customElements.define('), false,
      `${relative(frontend, path)} registers an element during import`);
  }
  assert.match(source(join(frontend, 'design-system/elements/define.ts')), /registry\.define/);
});

test('shipped sources never import the browser-test helpers', () => {
  const shipped = ['api', 'design-system', 'i18n', 'lib', 'stores', 'ui']
    .flatMap((directory) => filesUnder(join(frontend, directory), ['.ts']))
    .filter((path) => !path.endsWith('.test.ts'));
  for (const path of shipped) {
    assert.doesNotMatch(source(path), /from\s+['"][^'"]*\/testing\//,
      `${relative(frontend, path)} imports a test helper`);
  }
});
