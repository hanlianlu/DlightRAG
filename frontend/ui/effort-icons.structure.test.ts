// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on the effort ladder's own geometry.

 * Each offered effort must keep a glyph of its own: the same lucide dial with the
 * needle at that level's stop, so changing the level changes what the control
 * shows. A shared glyph, or a lost dial, is the regression this guards.
 */

import {test} from 'node:test';
import assert from 'node:assert/strict';

import {ICON_REGISTRY} from '../design-system/icons/registry.generated.ts';

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
