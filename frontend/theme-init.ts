// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

(function initializeThemeBeforePaint() {
  const root = document.documentElement;
  const storageKey = 'dlightrag-theme';
  let preference: 'system' | 'light' | 'dark' = 'system';
  let storedPreference: string | null = null;

  try {
    storedPreference = window.localStorage.getItem(storageKey);
  } catch {
    storedPreference = null;
  }

  if (
    storedPreference === 'system'
    || storedPreference === 'light'
    || storedPreference === 'dark'
  ) {
    preference = storedPreference;
  } else if (storedPreference !== null) {
    try {
      window.localStorage.removeItem(storageKey);
    } catch {
      // Keep the fail-closed system preference when storage is unavailable.
    }
  }

  let prefersDark = true;
  try {
    if (typeof window.matchMedia === 'function') {
      prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
  } catch {
    prefersDark = true;
  }

  const colorMode = preference === 'system'
    ? (prefersDark ? 'dark' : 'light')
    : preference;
  root.dataset.theme = preference;
  root.dataset.colorMode = colorMode;
  root.style.colorScheme = colorMode;
})();
