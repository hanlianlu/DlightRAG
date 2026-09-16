// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on hover and keyboard emphasis for the two controls that had
 *  none: the Settings language rows and the composer's answer-mode switcher.
 *
 * Both follow the same house emphasis — `--color-bg-hover` on the row, a
 * `--color-control-ring` outline for keyboard focus — so a later edit cannot
 * quietly drop one of them back to a bare trigger.
 */

import {readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

/** Return the declarations of the rule whose selector list is exactly these. */
function rule(css: string, ...selectors: string[]): string {
  const source = css.replace(/\/\*[\s\S]*?\*\//g, '');
  for (const match of source.matchAll(/(?:^|\n)([^{}]+)\{([^}]*)\}/g)) {
    const names = match[1].split(',').map((part) => part.trim());
    if (names.length === selectors.length && selectors.every((name) => names.includes(name))) {
      return match[2];
    }
  }
  assert.fail(`no rule declares exactly ${selectors.join(', ')}`);
}

test('the design-system row emphasises hover and keyboard focus', () => {
  const css = readFileSync(join(ROOT, 'design-system', 'primitives', 'primitives.css'), 'utf8');

  const hover = rule(css, '.dl-dialog-checkbox:hover', '.dl-dialog-checkbox:focus-within');
  assert.match(hover, /background:\s*var\(--color-bg-hover\)/);
  assert.match(hover, /color:\s*var\(--color-text-primary\)/);

  const ring = rule(css, 'label.dl-dialog-checkbox:has(input:focus-visible)');
  assert.match(ring, /outline:\s*2px solid var\(--color-control-ring\)/);

  const row = rule(css, '.dl-dialog-checkbox');
  assert.match(row, /margin-inline:\s*calc\(-1 \* var\(--space-tight\)\)/, 'the tint insets');
  assert.match(row, /padding-inline:\s*var\(--space-tight\)/);
});

test('both composer pickers emphasise their triggers, their rows, and the attach control', () => {
  const css = readFileSync(join(ROOT, 'styles', 'layout.css'), 'utf8');

  // The effort control reuses the mode switcher's rules, so the two stay one
  // visual definition; a new picker must join them rather than restate them.
  assert.match(
    rule(css, '.composer-mode-trigger:hover', '.composer-effort-trigger:hover'),
    /background:\s*var\(--color-bg-hover\)/,
  );
  assert.match(
    rule(css, '.composer-mode-trigger:focus-visible', '.composer-effort-trigger:focus-visible'),
    /outline:\s*2px solid var\(--color-control-ring\)/,
  );
  assert.match(
    rule(
      css,
      '.composer-mode-menu button:hover',
      '.composer-mode-menu button:focus-visible',
      '.composer-effort-menu button:hover',
      '.composer-effort-menu button:focus-visible',
    ),
    /background:\s*var\(--color-bg-hover\)/,
  );
  assert.match(
    rule(css, '.composer-mode-menu button:focus-visible', '.composer-effort-menu button:focus-visible'),
    /outline:\s*2px solid var\(--color-control-ring\)/,
  );

  const chosen = rule(css, ".composer-mode-menu button[aria-checked='true']", ".composer-effort-menu button[aria-checked='true']");
  const chosenHover = rule(
    css,
    ".composer-mode-menu button[aria-checked='true']:hover",
    ".composer-effort-menu button[aria-checked='true']:hover",
  );
  assert.match(chosen, /background:\s*var\(--color-bg-elevated\)/);
  assert.match(chosenHover, /background:\s*var\(--color-bg-elevated\)/, 'the chosen row keeps it');

  assert.match(rule(css, '.composer-plus:hover'), /background:\s*var\(--color-bg-hover\)/);
});
