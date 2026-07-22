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