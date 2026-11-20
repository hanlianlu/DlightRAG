// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {
  parseThemePreference,
  resolveColorMode,
  THEME_STORAGE_KEY,
} from './theme.ts';

test('parseThemePreference accepts valid values', () => {
  assert.equal(parseThemePreference('system'), 'system');
  assert.equal(parseThemePreference('light'), 'light');
  assert.equal(parseThemePreference('dark'), 'dark');
});

test('parseThemePreference returns system for missing values', () => {
  assert.equal(parseThemePreference(null), 'system');
});

test('parseThemePreference returns system for invalid values', () => {
  assert.equal(parseThemePreference(''), 'system');
  assert.equal(parseThemePreference('auto'), 'system');
  assert.equal(parseThemePreference('LIGHT'), 'system');
});

test('resolveColorMode honors explicit preference values', () => {
  assert.equal(resolveColorMode('light', true), 'light');
  assert.equal(resolveColorMode('light', false), 'light');
  assert.equal(resolveColorMode('dark', true), 'dark');
  assert.equal(resolveColorMode('dark', false), 'dark');
});

test('resolveColorMode maps system preference from media preference', () => {
  assert.equal(resolveColorMode('system', true), 'dark');
  assert.equal(resolveColorMode('system', false), 'light');
});

/** `theme-init.ts` is the classic script the page runs before first paint, so it cannot import this
 *  module: it carries its own copy of the storage key and of the preference parsing, and this test
 *  holds that copy to the module's behaviour. */
test('the pre-paint script resolves a stored preference as the theme module does', async () => {
  const cases: Array<[stored: string | null, prefersDark: boolean]> = [
    [null, true], [null, false], ['system', false], ['light', true], ['dark', false], ['LIGHT', true], ['auto', false],
  ];
  try {
    for (const [index, [stored, prefersDark]] of cases.entries()) {
      const root = {dataset: {} as Record<string, string>, style: {} as Record<string, string>};
      const removed: string[] = [];
      Object.assign(globalThis, {
        document: {documentElement: root},
        window: {
          localStorage: {
            getItem: (key: string) => (key === THEME_STORAGE_KEY ? stored : null),
            removeItem: (key: string) => { removed.push(key); },
          },
          matchMedia: () => ({matches: prefersDark}),
        },
      });
      await import(`../theme-init.ts?case=${index}`);

      const preference = parseThemePreference(stored);
      assert.equal(root.dataset.theme, preference);
      assert.equal(root.dataset.colorMode, resolveColorMode(preference, prefersDark));
      // A value outside the accepted set is dropped from storage, under the key the module writes.
      const dropped = stored !== null && stored !== preference;
      assert.deepEqual(removed, dropped ? [THEME_STORAGE_KEY] : []);
    }
  } finally {
    Reflect.deleteProperty(globalThis, 'window');
    Reflect.deleteProperty(globalThis, 'document');
  }
});