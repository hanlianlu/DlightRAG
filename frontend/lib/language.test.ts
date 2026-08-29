// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  LANGUAGE_STORAGE_KEY,
  parseLanguagePreference,
  resolveLocale,
} from './language.ts';

test('preferences outside the accepted set resolve to auto', () => {
  assert.equal(parseLanguagePreference(null), 'auto');
  assert.equal(parseLanguagePreference(undefined), 'auto');
  assert.equal(parseLanguagePreference(''), 'auto');
  assert.equal(parseLanguagePreference('fr'), 'auto');
  assert.equal(parseLanguagePreference('AUTO'), 'auto');
  assert.equal(parseLanguagePreference('en'), 'en');
  assert.equal(parseLanguagePreference('zh'), 'zh');
});

test('auto resolution follows the browser language', () => {
  assert.equal(resolveLocale('auto', 'zh-CN'), 'zh');
  assert.equal(resolveLocale('auto', 'zh'), 'zh');
  assert.equal(resolveLocale('auto', 'en-US'), 'en');
  assert.equal(resolveLocale('auto', 'fr'), 'en');
  assert.equal(resolveLocale('auto', ''), 'en');
});

test('explicit preferences win over the browser language', () => {
  assert.equal(resolveLocale('en', 'zh-CN'), 'en');
  assert.equal(resolveLocale('zh', 'en-US'), 'zh');
});

test('storage key is stable', () => {
  assert.equal(LANGUAGE_STORAGE_KEY, 'dlightrag-lang');
});
