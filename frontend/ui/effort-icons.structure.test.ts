// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on the effort ladder's own presentation.

 * Each offered effort must keep a glyph of its own: the same lucide dial with the
 * needle at that level's stop, so changing the level changes what the control
 * shows. A shared glyph, or a lost dial, is the regression this guards. The row
 * that shows those glyphs must keep a stable three-column shape, so a level's
 * label never crowds the Default marker beside it.
 */

import {readFileSync} from 'node:fs';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import assert from 'node:assert/strict';

import {ICON_REGISTRY} from '../design-system/icons/registry.generated.ts';

const FRONTEND_DIR = join(dirname(fileURLToPath(import.meta.url)), '..');
const LEVELS = ['low', 'high', 'max'] as const;
const DIAL = 'M3.34 19a10 10 0 1 1 17.32 0';

function paths(name: string): string[] {
  const definition = ICON_REGISTRY[name as keyof typeof ICON_REGISTRY];
  assert.ok(definition, `icon registry is missing ${name}`);
  return definition.nodes.filter(([tag]) => tag === 'path')
    .map(([, attributes]) => String((attributes as {d?: string}).d ?? ''));
}

test('every offered effort renders the dial with its own needle stop', () => {
  const needles = LEVELS.map((level) => {
    const all = paths(`effort-${level}`);
    assert.ok(all.includes(DIAL), `effort-${level} must keep the lucide dial`);
    const needle = all.filter((d) => d !== DIAL);
    assert.equal(needle.length, 1, `effort-${level} must carry exactly one needle`);
    return needle[0];
  });

  assert.equal(new Set(needles).size, LEVELS.length, 'each level must own a distinct stop');
});

/** The one rule whose selector is exactly `selector`, never a shared list's member. */
function ruleMaybe(css: string, selector: string): string | null {
  const source = css.replace(/\/\*[\s\S]*?\*\//g, '');
  for (const match of source.matchAll(/(?:^|\n)([^{}]+)\{([^}]*)\}/g)) {
    const names = match[1].split(',').map((part) => part.trim());
    if (names.length === 1 && names[0] === selector) return match[2];
  }
  return null;
}

function ruleFor(css: string, selector: string): string {
  const rule = ruleMaybe(css, selector);
  assert.ok(rule !== null, `no rule declares ${selector}`);
  return rule;
}

test('a level row keeps its Default marker beside the label it qualifies', () => {
  const css = readFileSync(join(FRONTEND_DIR, 'styles', 'layout.css'), 'utf8');
  const row = ruleFor(css, '.composer-effort-menu button');

  // Content-sized tracks only: a growing or auto track would push the marker away
  // from its label, which is a qualifier rather than a selection column.
  assert.match(
    row,
    /grid-template-columns:\s*var\(--size-icon-md\)\s+max-content\s+max-content;/,
  );
  assert.doesNotMatch(row, /grid-template-columns:[^;]*(?:1fr|\bauto\b)/);
  // The row shares the mode picker's menu width instead of widening its own.
  const dedicated = ruleMaybe(css, '.composer-effort-menu');
  assert.equal(
    dedicated !== null && /min-width/.test(dedicated),
    false,
    'the effort menu must not widen itself past the shared default',
  );
});
