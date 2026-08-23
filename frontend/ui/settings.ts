// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Unified settings dialog: appearance, memory, and conversation data. */

import {clearMemory, putMemorySettings} from '../api/memory.ts';
import {refreshMemorySettingsPanel} from './memory.ts';
import {applyThemePreference, parseThemePreference, THEME_STORAGE_KEY} from '../lib/theme.ts';
import {showToast} from './toast.ts';

const SETTINGS_BUTTON_ID = 'settings-btn';
const SETTINGS_DIALOG_ID = 'settings-dialog';
const CLEAR_MEMORY_BUTTON_ID = 'memory-clear-btn';

function syncThemeSegment(): void {
  const stored = parseThemePreference(
    (() => {
      try {
        return window.localStorage.getItem(THEME_STORAGE_KEY);
      } catch {
        return null;
      }
    })(),
  );
  const row = document.getElementById('settings-theme-row');
  row?.querySelectorAll<HTMLButtonElement>('[data-theme-choice]').forEach((button) => {
    const active = parseThemePreference(button.dataset.themeChoice || null) === stored;
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
}

/** Open Settings; resolve with the chosen submit value ('close-settings'|'delete-all'). */
export function openSettings(): Promise<string> {
  const dialog = document.getElementById(SETTINGS_DIALOG_ID) as HTMLDialogElement | null;
  if (!dialog) return Promise.resolve('');
  dialog.returnValue = '';
  dialog.showModal();
  return new Promise(function(resolve) {
    dialog.addEventListener('close', function() {
      resolve(dialog.returnValue);
    }, {once: true});
  });
}

export function setupSettings(onDeleteAll: () => void): void {
  const trigger = document.getElementById(SETTINGS_BUTTON_ID);
  const dialog = document.getElementById(SETTINGS_DIALOG_ID) as HTMLDialogElement | null;
  if (!trigger || !(trigger instanceof HTMLButtonElement) || !dialog) return;

  trigger.addEventListener('click', async () => {
    syncThemeSegment();
    try {
      await refreshMemorySettingsPanel();
    } catch {
      showToast('Could not load memory settings.', 5000);
    }
    const action = await openSettings();
    const toggle = document.getElementById('memory-enabled-toggle') as HTMLInputElement | null;
    if (toggle) {
      try {
        await putMemorySettings(toggle.checked);
      } catch {
        showToast('Could not save memory settings.', 5000);
      }
    }
    if (action === 'delete-all') onDeleteAll();
  });

  dialog.querySelectorAll<HTMLButtonElement>('[data-theme-choice]').forEach((button) => {
    button.addEventListener('click', function() {
      const preference = parseThemePreference(button.dataset.themeChoice || null);
      applyThemePreference(preference);
      syncThemeSegment();
    });
  });

  document.getElementById(CLEAR_MEMORY_BUTTON_ID)?.addEventListener('click', async function() {
    try {
      await clearMemory();
      await refreshMemorySettingsPanel();
      showToast('Memory cleared.', 3000);
    } catch {
      showToast('Could not clear memory.', 5000);
    }
  });
}
