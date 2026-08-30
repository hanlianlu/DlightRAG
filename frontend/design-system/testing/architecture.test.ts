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

test('design-system source graph cannot import product infrastructure', () => {
  const forbidden = [
    '/api/',
    '/stores/',
    '/router',
    '/events/',
    'xstate',
    'mermaid',
    'dompurify',
    'messageIds',
  ];
  for (const path of designSystemSources) {
    const content = source(path).toLowerCase();
    for (const needle of forbidden.map((value) => value.toLowerCase())) {
      assert.equal(
        content.includes(needle),
        false,
        `${relative(frontend, path)} crosses the design-system boundary via ${needle}`,
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

test('legacy class namespace and Web Awesome are absent', () => {
  const packageJson = source(join(frontend, 'package.json'));
  assert.equal(packageJson.includes('@awesome.me/webawesome'), false);
  for (const path of governedSources) {
    const content = source(path);
    assert.equal(content.includes('ui-'), false, `${relative(frontend, path)} retains .ui-*`);
    assert.equal(/<\/?wa-|@awesome\.me\/webawesome|--wa-/.test(content), false,
      `${relative(frontend, path)} retains Web Awesome`);
  }
});

test('feature UI has no raw SVG or text-glyph icons', () => {
  const glyph = /[✕×▶‹›●✓]|\\27(?:13|15)/u;
  for (const path of featureSources) {
    const content = source(path);
    const projectPath = relative(frontend, path);
    if (!RAW_SVG_ALLOWLIST.has(projectPath)) {
      assert.equal(/<svg\b/i.test(content), false,
        `${projectPath} must use icon() instead of raw SVG`);
    }
    assert.equal(glyph.test(content), false,
      `${relative(frontend, path)} must use icon() instead of a text glyph`);
  }
  for (const path of filesUnder(join(frontend, 'styles'), ['.css'])) {
    assert.equal(glyph.test(source(path)), false,
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
