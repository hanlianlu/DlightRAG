// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {
    parseThemePreference,
    resolveColorMode,
    THEME_STORAGE_KEY,
    type ThemePreference,
} from '../lib/theme.ts';
import {createAutoDismiss} from '../lib/popover.ts';
import {installRovingArrowNavigation} from '../lib/listbox.ts';

type ThemeElements = {
    root: HTMLElement;
    control: HTMLElement;
    trigger: HTMLButtonElement;
    menu: HTMLElement;
    items: HTMLButtonElement[];
};

function getThemeElements(): ThemeElements | null {
    const root = document.documentElement;
    const control = document.getElementById('theme-control');
    const trigger = document.getElementById('theme-trigger');
    const menu = document.getElementById('theme-menu');
    if (
        !(root instanceof HTMLElement) ||
        !(control instanceof HTMLElement) ||
        !(trigger instanceof HTMLButtonElement) ||
        !(menu instanceof HTMLElement)
    ) {
        return null;
    }
    const items = Array.from(menu.querySelectorAll<HTMLButtonElement>('[role="menuitemradio"]'));
    if (items.length === 0) return null;
    return {root, control, trigger, menu, items};
}

function readStoragePreference(): ThemePreference {
    try {
        return parseThemePreference(window.localStorage.getItem(THEME_STORAGE_KEY));
    } catch (_error) {
        return 'system';
    }
}

function writeStoragePreference(preference: ThemePreference): void {
    try {
        if (preference === 'system') {
            window.localStorage.removeItem(THEME_STORAGE_KEY);
        } else {
            window.localStorage.setItem(THEME_STORAGE_KEY, preference);
        }
    } catch (_error) {
        // Ignore unavailable or blocked storage.
    }
}

function currentPrefersDark(mediaQuery: MediaQueryList): boolean {
    return mediaQuery.matches;
}

function applyRootTheme(
    root: HTMLElement,
    preference: ThemePreference,
    prefersDark: boolean,
): void {
    const colorMode = resolveColorMode(preference, prefersDark);
    root.setAttribute('data-theme', preference);
    root.setAttribute('data-color-mode', colorMode);
    root.style.colorScheme = colorMode;
}

function checkedItem(items: HTMLButtonElement[]): HTMLButtonElement {
    return items.find((item) => item.getAttribute('aria-checked') === 'true') || items[0];
}

function syncMenuState(items: HTMLButtonElement[], preference: ThemePreference): void {
    items.forEach((item) => {
        const value = parseThemePreference(item.dataset.themeValue || null);
        const isChecked = value === preference;
        item.setAttribute('aria-checked', isChecked ? 'true' : 'false');
        item.tabIndex = isChecked ? 0 : -1;
    });
}

function focusMenuItem(items: HTMLButtonElement[], preference: ThemePreference): void {
    const target =
        items.find((item) => parseThemePreference(item.dataset.themeValue || null) === preference) ||
        checkedItem(items);
    target.focus();
}

export function setupTheme(): void {
    const elements = getThemeElements();
    if (!elements) return;

    const {root, control, trigger, menu, items} = elements;
    if (control.dataset.themeBound === 'true') return;
    control.dataset.themeBound = 'true';
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    let preference = parseThemePreference(root.getAttribute('data-theme'));
    const storedPreference = readStoragePreference();
    if (storedPreference !== 'system') {
        preference = storedPreference;
    }

    let mediaListening = false;
    let isMenuOpen = false;

    const dismiss = createAutoDismiss({
        getAnchor: () => control,
        isOpen: () => isMenuOpen,
        onDismiss: (reason) => closeMenu(reason === 'escape'),
    });

    const onMediaChange = (): void => {
        if (preference !== 'system') return;
        applyRootTheme(root, preference, currentPrefersDark(mediaQuery));
    };

    function addMediaListener(): void {
        if (mediaListening) return;
        mediaListening = true;
        mediaQuery.addEventListener('change', onMediaChange);
    }

    function removeMediaListener(): void {
        if (!mediaListening) return;
        mediaListening = false;
        mediaQuery.removeEventListener('change', onMediaChange);
    }

    function syncMediaListener(nextPreference: ThemePreference): void {
        if (nextPreference === 'system') {
            addMediaListener();
            return;
        }
        removeMediaListener();
    }

    function openMenu(focusCurrent: boolean): void {
        if (isMenuOpen) return;
        isMenuOpen = true;
        menu.hidden = false;
        trigger.setAttribute('aria-expanded', 'true');
        dismiss.activate();
        if (focusCurrent) focusMenuItem(items, preference);
    }

    function closeMenu(restoreTriggerFocus: boolean): void {
        if (!isMenuOpen) return;
        isMenuOpen = false;
        menu.hidden = true;
        trigger.setAttribute('aria-expanded', 'false');
        dismiss.deactivate();
        if (restoreTriggerFocus) {
            window.requestAnimationFrame(() => trigger.focus());
        }
    }

    function applyPreference(nextPreference: ThemePreference, persist: boolean): void {
        preference = nextPreference;
        applyRootTheme(root, preference, currentPrefersDark(mediaQuery));
        syncMenuState(items, preference);
        syncMediaListener(preference);
        if (persist) writeStoragePreference(preference);
    }

    function selectTheme(nextPreference: ThemePreference): void {
        applyPreference(nextPreference, true);
        closeMenu(true);
    }

    trigger.addEventListener('click', () => {
        if (isMenuOpen) {
            closeMenu(false);
            return;
        }
        openMenu(false);
    });

    trigger.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowDown') {
            event.preventDefault();
            openMenu(true);
            return;
        }
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            if (isMenuOpen) {
                closeMenu(false);
            } else {
                openMenu(true);
            }
        }
    });

    menu.addEventListener('keydown', (event) => {
        const active = document.activeElement;
        const isItem = active instanceof HTMLButtonElement && active.getAttribute('role') === 'menuitemradio';
        if (event.key === 'Escape') {
            event.preventDefault();
            event.stopImmediatePropagation();
            closeMenu(true);
            return;
        }
        if (!isItem) return;
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            selectTheme(parseThemePreference(active.dataset.themeValue || null));
        }
    });

    items.forEach((item) => {
        item.addEventListener('click', () => {
            selectTheme(parseThemePreference(item.dataset.themeValue || null));
        });
    });

    installRovingArrowNavigation(menu, '[role="menuitemradio"]');

    window.addEventListener('storage', (event) => {
        let storageArea: Storage | null = null;
        try {
            storageArea = window.localStorage;
        } catch (_error) {
            storageArea = null;
        }
        if (event.storageArea !== storageArea) return;
        if (event.key !== null && event.key !== THEME_STORAGE_KEY) return;
        applyPreference(parseThemePreference(event.newValue), false);
    });

    applyPreference(preference, false);
}
