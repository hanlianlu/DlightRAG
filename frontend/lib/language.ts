// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Language preference parsing and locale resolution, independent of the DOM. */

export const LANGUAGE_STORAGE_KEY = 'dlightrag-lang';

export type LanguagePreference = 'auto' | 'en' | 'zh';
export type Locale = 'en' | 'zh';

export function parseLanguagePreference(value: string | null | undefined): LanguagePreference {
  return value === 'en' || value === 'zh' ? value : 'auto';
}

/** `auto` follows the browser language; explicit choices win unchanged. */
export function resolveLocale(preference: LanguagePreference, navigatorLanguage: string): Locale {
  if (preference === 'en' || preference === 'zh') return preference;
  const language = navigatorLanguage || '';
  return language.toLowerCase().startsWith('zh') ? 'zh' : 'en';
}
