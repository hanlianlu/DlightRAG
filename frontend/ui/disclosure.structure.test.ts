// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on the disclosure affordance and the References collapse.
 *
 * The MCP card and group chevrons used to live in a flexible grid track: their
 * box stretched with the panel (605px inside a 720px row), the glyph was drawn at
 * the box's left edge, and rotation pivoted on that box centre, so the icon
 * drifted by half the track whenever the panel resized or the group toggled.
 * Both affordances now pin one fixed glyph column to the row end.
 *
 * The References list shows five rows, and three on a narrow answer column. The
 * control is the list's next sibling and renders only while the collapsed list
 * hides a row, so both numbers live in one stylesheet.
 */

import {readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

function read(path: string): string {
  return readFileSync(join(ROOT, path), 'utf8');
}

/** Return one rule's declarations, matched at a line start so a descendant
 * selector such as `.cardToggle[aria-expanded] .chevron` cannot satisfy it. */
function rule(css: string, selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = new RegExp(`(?:^|\\n)\\s*${escaped} \\{([^}]*)\\}`, 'm').exec(css);
  assert.ok(match, `${selector} rule is missing`);
  return match[1];
}

/** Return the declarations of the one rule whose selector list is exactly these.
 * Comments are stripped first, because a grouped selector may follow one. */
function sharedRule(css: string, selectors: string[]): string {
  const source = css.replace(/\/\*[\s\S]*?\*\//g, '');
  for (const match of source.matchAll(/(?:^|\n)([^{}]+)\{([^}]*)\}/g)) {
    const names = match[1].split(',').map((part) => part.trim());
    if (names.length === selectors.length && selectors.every((s) => names.includes(s))) {
      return match[2];
    }
  }
  assert.fail(`no rule declares exactly ${selectors.join(', ')}`);
}

/** Return the body of the first block whose header text is exactly this, so an
 * assertion can prove a rule sits *inside* a query rather than merely nearby. */
function block(css: string, header: string): string {
  const start = css.indexOf(header);
  assert.ok(start >= 0, header + ' is missing');
  const open = css.indexOf('{', start);
  assert.ok(open > start, header + ' has no body');
  let depth = 0;
  for (let index = open; index < css.length; index += 1) {
    if (css[index] === '{') depth += 1;
    else if (css[index] === '}') {
      depth -= 1;
      if (depth === 0) return css.slice(open + 1, index);
    }
  }
  assert.fail(header + ' block is not closed');
}

test('MCP disclosures pin one fixed glyph column to the row end', () => {
  const css = read('styles/settings-connections.module.css');
  const disclosure = sharedRule(css, ['.groupChevron', '.chevron']);
  assert.match(disclosure, /flex-shrink: 0/);
  assert.match(disclosure, /margin-inline-start: auto/);
  assert.match(disclosure, /transition: transform var\(--duration-control\)/);
  // A width here is what stretched the box across the row before; the box
  // must stay glyph-sized so rotation pivots on the glyph.
  assert.doesNotMatch(disclosure, /width:/);
  assert.match(rule(css, '.groupRow'), /display: flex/);
  assert.doesNotMatch(rule(css, '.groupRow'), /grid-template-columns/);
  assert.match(rule(css, '.cardToggle'), /display: flex/);
  assert.doesNotMatch(rule(css, '.cardToggle'), /grid-template-columns/);
  assert.match(css, /\.groupRow\[aria-expanded='true'\] \.groupChevron \{/);
  assert.match(css, /\.cardToggle\[aria-expanded='true'\] \.chevron \{/);
});

test('MCP disclosures use the shared disclosure icon without inline rotation', () => {
  const source = read('ui/settings-connections.ts');
  assert.equal((source.match(/icon\('disclosure'/g) ?? []).length, 2);
  assert.doesNotMatch(source, /rotate\(/);
});

test('References keep each threshold and gate inside its own query', () => {
  const css = read('styles/answer-presentation.module.css');
  const wide = block(css, '@media (width > 640px)');
  const narrow = block(css, '@media (width <= 640px)');
  assert.match(wide, /\.answer-reference-list:not\(\.expanded\) \.answer-ref-item:nth-child\(n \+ 6\)/);
  assert.match(wide, /:has\(\.answer-reference-list \.answer-ref-item:nth-child\(6\)\)/);
  assert.match(narrow, /\.answer-reference-list:not\(\.expanded\) \.answer-ref-item:nth-child\(n \+ 4\)/);
  assert.match(narrow, /:has\(\.answer-reference-list \.answer-ref-item:nth-child\(4\)\)/);
  // An expanded list always keeps its control, so a resize cannot hide the node
  // the user is standing on.
  assert.match(wide, /:has\(\.answer-reference-list\.expanded\),/);
  assert.match(narrow, /:has\(\.answer-reference-list\.expanded\),/);
  // The five-row rules must never apply to a narrow answer column: a
  // viewport-wide gate would hide the control while row four is collapsed away.
  const outsideQueries = css.replace(wide, '').replace(narrow, '');
  assert.doesNotMatch(outsideQueries, /nth-child\(n \+ 6\)/);
});

test('References control shares the reference id gutter', () => {
  const css = read('styles/answer-presentation.module.css');
  const gutter = sharedRule(css, ['.answer-ref-id', '.answer-references-toggle-icon']);
  assert.match(gutter, /flex-shrink: 0/);
  assert.match(gutter, /min-width: 24px/);
  assert.match(sharedRule(css, ['.answer-references-toggle-icon']), /justify-content: flex-start/);
  // The svg carries the rotation, so the pivot stays on the glyph while the
  // column keeps the gutter width.
  assert.match(css, /\.answer-references-toggle\[aria-expanded='true'\] \.answer-references-toggle-icon svg \{/);
  assert.match(read('ui/answer-presentation.ts'), /answer-references-toggle-icon/);
  assert.doesNotMatch(read('ui/answer-presentation.ts'), /answer-references-show-all/);
});
