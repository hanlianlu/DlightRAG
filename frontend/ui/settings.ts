// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Unified settings dialog: Profile Memory and Conversation History. */

import {clearMemory, putMemorySettings} from '../api/memory.ts';
import {refreshMemorySettingsPanel} from './memory.ts';
import {showToast} from './toast.ts';

const SETTINGS_BUTTON_ID = 'settings-btn';
const SETTINGS_DIALOG_ID = 'settings-dialog';
const CLEAR_MEMORY_BUTTON_ID = 'memory-clear-btn';

/** Open Settings; resolve with the chosen submit value ('close-settings'|'delete-all'). */
function openSettings(): Promise<string> {
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
