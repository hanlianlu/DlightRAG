// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on the semantic icon registry's paint and provenance.

 * A solid glyph is one that paints a fill and strokes nothing. Pouring ink into a stroke drawing is how
 * the thumbs lost the cuff that separates the sleeve from the hand: the divider was still drawn, in the
 * same colour as the body it sat on, and both `-filled` entries left `currentColor` on both sides of one
 * shape. A glyph whose source does not ship it is this product's own drawing, so it declares itself a
 * `dlightrag` entry derived from the shape it keeps — the rule the effort ladder already followed, and
 * the one the thumbs did not. `scripts/generate-icons.mjs` holds the same line at the door: it refuses
 * authored geometry or authored paint under a source's label, and refuses a derived entry whose base it
 * cannot find or whose base shape the entry no longer draws.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {ICON_REGISTRY, type IconDefinition} from '../design-system/icons/registry.generated.ts';

const ENTRIES = Object.entries(ICON_REGISTRY) as [string, IconDefinition][];
const OUTLINES = ['thumbs-up', 'thumbs-down'] as const;
const THUMBS = OUTLINES.flatMap((outline) => [outline, `${outline}-filled`] as const);

function paths(definition: IconDefinition): string[] {
  return definition.nodes
    .filter(([tag]) => tag === 'path')
    .map(([, attributes]) => String(attributes.d ?? ''));
}

/** How many subpaths a `d` closes: one outline, or an outline and the hole it keeps. */
function closedSubpaths(d: string): number {
  return (d.match(/[Zz]/g) ?? []).length;
}

function longestPath(definition: IconDefinition): string | undefined {
  // The base's principal shape is its longest path: lucide's gauge draws its needle before the dial, so
  // the one shape a ladder derives from is the dial it has to keep.
  return paths(definition).sort((left, right) => right.length - left.length)[0];
}

test('a solid glyph paints its fill and strokes nothing', () => {
  for (const [name, definition] of ENTRIES) {
    if (definition.fill === 'none') continue;
    assert.equal(
      definition.stroke,
      'none',
      `${name} paints a fill, so its outline must not be stroked over it`,
    );
  }
});

test('a solid twin declares the glyph it was drawn from', () => {
  // Lucide draws with strokes and ships no solid variants of its own, so a name that claims a filled state
  // for one of its glyphs claims authored drawing with it, wherever the geometry came from.
  for (const [name, definition] of ENTRIES) {
    if (!name.endsWith('-filled') || definition.source !== 'lucide') continue;
    assert.fail(`${name} claims a solid from a stroke-only source; it belongs to this product`);
  }
});

test('a glyph drawn from another declares the shape it was drawn from', () => {
  // A state ladder's levels share the base's principal shape, so the drawing itself says which glyph an
  // entry is a variant of — more than a name can, and enough to catch a level that forgot to declare one.
  for (const [name, definition] of ENTRIES) {
    if (definition.source !== 'dlightrag') continue;
    const held = ENTRIES
      .filter(([other]) => other !== name)
      .filter(([, base]) => {
        const basePath = longestPath(base);
        return basePath !== undefined && basePath.length > 0
          && paths(definition).some((d) => d.includes(basePath));
      })
      .map(([, base]) => `${base.source}/${base.sourceName}`);
    if (held.length === 0) continue;
    assert.ok(
      held.includes(definition.derivedFrom ?? ''),
      `${name} draws the shape ${held.join(' or ')} holds, so it must derive from it`,
    );
  }
});

test('a derived glyph declares itself, and keeps the shape it derives from', () => {
  for (const [name, definition] of ENTRIES) {
    if (definition.derivedFrom === null) continue;
    assert.equal(
      definition.source,
      'dlightrag',
      `${name} is authored here, so its source is this product`,
    );
    // The package holds shapes this selection need not select — `stop` derives from lucide's square and
    // selects it under no other name — so the shape is checked where the selection does hold the base.
    const base = ENTRIES.find(([, each]) => `${each.source}/${each.sourceName}` === definition.derivedFrom);
    if (base === undefined) continue;
    const basePath = longestPath(base[1]);
    assert.ok(
      basePath !== undefined && basePath.length > 0,
      `${definition.derivedFrom} draws no path to keep`,
    );
    assert.ok(
      paths(definition).some((d) => d.includes(basePath)),
      `${name} no longer draws the ${definition.derivedFrom} shape it derives from`,
    );
  }
});

test('each solid thumb is its outline plus a hole, and the four stay distinct', () => {
  for (const outline of OUTLINES) {
    const filledName = `${outline}-filled` as const;
    const filled = ICON_REGISTRY[filledName];
    assert.equal(filled.derivedFrom, `lucide/${ICON_REGISTRY[outline].sourceName}`);
    const [outlinePath] = paths(ICON_REGISTRY[outline]);
    const [filledPath] = paths(filled);
    // The solid is one shape, so the hole is the hole of that `d` rather than a second path drawn behind
    // it. It keeps the outline exactly and closes one subpath more, which is the cuff as a hole a reader
    // sees at the 16 CSS pixels the answer row uses, not a divider stroked over the body just filled.
    assert.equal(paths(filled).length, 1, `${filledName} must be drawn by one path`);
    assert.ok(filledPath.startsWith(outlinePath), `${filledName} must keep the ${outline} shape`);
    assert.equal(
      closedSubpaths(filledPath),
      closedSubpaths(outlinePath) + 1,
      `${filledName} must close exactly one more subpath than ${outline}`,
    );
    assert.equal(
      filled.nodes[0][1]['fill-rule'],
      'evenodd',
      `${filledName} must rule its hole even-odd, so the hole does not depend on how it is wound`,
    );
  }
  for (let left = 0; left < THUMBS.length; left += 1) {
    for (let right = left + 1; right < THUMBS.length; right += 1) {
      assert.notEqual(
        paths(ICON_REGISTRY[THUMBS[left]])[0],
        paths(ICON_REGISTRY[THUMBS[right]])[0],
        `${THUMBS[left]} and ${THUMBS[right]} must not share one shape`,
      );
    }
  }
});
