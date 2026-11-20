// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Browser language wiring: lit-localize runtime, persistence, and `<html lang>`. */

import {configureLocalization} from '@lit/localize';
import {
  LANGUAGE_STORAGE_KEY,
  parseLanguagePreference,
  resolveLocale,
  type LanguagePreference,
} from '../lib/language.ts';
import {readStored, writeStored} from '../lib/storage.ts';

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
  return parseLanguagePreference(readStored(LANGUAGE_STORAGE_KEY));
}

function writePreference(preference: LanguagePreference): void {
  writeStored(LANGUAGE_STORAGE_KEY, preference === 'auto' ? null : preference);
}

/** The BCP 47 tag `<html lang>` carries. Chinese names its script, so a reader whose system prefers
 *  Traditional glyphs is not shown them for Simplified text. */
function documentLanguage(locale: string): string {
  return locale === 'zh' ? 'zh-Hans' : locale;
}

function applyDocumentLanguage(locale: string): void {
  document.documentElement.lang = documentLanguage(locale);
}

export function currentLanguagePreference(): LanguagePreference {
  return readPreference();
}

/**
 * Make `locale` the active language and point `<html lang>` at the one that is active once that
 * settles. The tag moves with the catalog, never ahead of it, and stays where it was when the
 * catalog fails to load, so it never names a language the page is not speaking. `install` is
 * injectable so a test can hold a load or fail one.
 */
export async function activateLocale(
  locale: string,
  install: (locale: string) => Promise<void> = setLocale,
): Promise<void> {
  try {
    await install(locale);
  } catch {
    // Keep the current locale when a catalog fails to load.
  }
  applyDocumentLanguage(getLocale());
}

/** Apply one language preference: persist, resolve, load, and sync `<html lang>`. */
export async function setLanguagePreference(preference: LanguagePreference): Promise<void> {
  writePreference(preference);
  await activateLocale(resolveLocale(preference, window.navigator.language));
}

/** Resolve the stored preference once per page before the first Lit render. */
export async function initializeLanguagePreference(): Promise<void> {
  await setLanguagePreference(readPreference());
}
