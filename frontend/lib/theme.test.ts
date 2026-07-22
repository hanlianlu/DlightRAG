// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {
  parseThemePreference,
  resolveColorMode,
  THEME_STORAGE_KEY,
} from './theme.ts';

test('theme storage key is stable', () => {
  assert.equal(THEME_STORAGE_KEY, 'dlightrag-theme');
});

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