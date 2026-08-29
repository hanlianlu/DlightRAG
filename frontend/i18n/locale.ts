// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Browser language wiring: lit-localize runtime, persistence, and `<html lang>`. */

import {configureLocalization} from '@lit/localize';
import {
  LANGUAGE_STORAGE_KEY,
  parseLanguagePreference,
  resolveLocale,
  type LanguagePreference,
} from '../lib/language.ts';

export const {getLocale, setLocale} = configureLocalization({
  sourceLocale: 'en',
  targetLocales: ['zh'],
  loadLocale: async (locale: string) => await import(`./locales/${locale}.ts`),
});

/**
 * Message id convention: every `msg()` call carries an explicit dotted id
 * (`{id: 'feature.purpose'}`) so catalogs stay hand-authorable and greppable
 * without extraction tooling.
 */

function readPreference(): LanguagePreference {
  try {
    return parseLanguagePreference(window.localStorage.getItem(LANGUAGE_STORAGE_KEY));
  } catch {
    return 'auto';
  }
}

function writePreference(preference: LanguagePreference): void {
  try {
    if (preference === 'auto') window.localStorage.removeItem(LANGUAGE_STORAGE_KEY);
    else window.localStorage.setItem(LANGUAGE_STORAGE_KEY, preference);
  } catch {
    // The choice remains active for this page when storage is blocked.
  }
}

function applyDocumentLanguage(locale: string): void {
  document.documentElement.lang = locale;
}

export function currentLanguagePreference(): LanguagePreference {
  return readPreference();
}

/** Apply one language preference: persist, resolve, sync `<html lang>`, and load. */
export async function setLanguagePreference(preference: LanguagePreference): Promise<void> {
  writePreference(preference);
  const locale = resolveLocale(preference, window.navigator.language);
  applyDocumentLanguage(locale);
  if (locale !== getLocale()) {
    try {
      await setLocale(locale);
    } catch {
      // Keep the current locale when a catalog fails to load.
    }
  }
}

/** Resolve the stored preference once per page before the first Lit render. */
export async function initializeLanguagePreference(): Promise<void> {
  await setLanguagePreference(readPreference());
}
