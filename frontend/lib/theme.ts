// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export const THEME_STORAGE_KEY = 'dlightrag-theme';

export type ThemePreference = 'system' | 'light' | 'dark';
export type ColorMode = 'light' | 'dark';

export function parseThemePreference(value: string | null): ThemePreference {
  if (value === 'light' || value === 'dark' || value === 'system') {
    return value;
  }
  return 'system';
}

export function resolveColorMode(
  preference: ThemePreference,
  prefersDark: boolean,
): ColorMode {
  if (preference === 'light' || preference === 'dark') {
    return preference;
  }
  return prefersDark ? 'dark' : 'light';
}

/** One preference change touched by every settings surface. */
export const THEME_CHANGE_EVENT = 'dlightrag-theme-change';

export function applyThemePreference(preference: ThemePreference): void {
  const root = document.documentElement;
  const colorMode = resolveColorMode(preference, window.matchMedia('(prefers-color-scheme: dark)').matches);
  root.setAttribute('data-theme', preference);
  root.setAttribute('data-color-mode', colorMode);
  root.style.colorScheme = colorMode;
  try {
    if (preference === 'system') {
      window.localStorage.removeItem(THEME_STORAGE_KEY);
    } else {
      window.localStorage.setItem(THEME_STORAGE_KEY, preference);
    }
  } catch (_error) {
    // Ignore unavailable or blocked storage.
  }
  window.dispatchEvent(
    new CustomEvent<ThemePreference>(THEME_CHANGE_EVENT, {detail: preference}),
  );
}