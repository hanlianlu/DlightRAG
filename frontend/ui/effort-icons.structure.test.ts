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

function ruleFor(css: string, selector: string): string {
  const source = css.replace(/\/\*[\s\S]*?\*\//g, '');
  for (const match of source.matchAll(/(?:^|\n)([^{}]+)\{([^}]*)\}/g)) {
    const names = match[1].split(',').map((part) => part.trim());
    if (names.length === 1 && names[0] === selector) return match[2];
  }
  assert.fail(`no rule declares ${selector}`);
}

test('a level row keeps room for its glyph, its label, and the Default marker', () => {
  const css = readFileSync(join(FRONTEND_DIR, 'styles', 'layout.css'), 'utf8');

  const menu = ruleFor(css, '.composer-effort-menu');
  assert.match(menu, /min-width:\s*(?!7\.5rem)[^;]+;/, 'the menu must outgrow the shared default');
  // The label column grows but never shrinks below its own text, so the label and
  // the trailing Default marker cannot crowd each other.
  assert.match(
    ruleFor(css, '.composer-effort-menu button'),
    /grid-template-columns:\s*[^;]*minmax\(max-content,\s*1fr\)[^;]*;/,
  );
});
