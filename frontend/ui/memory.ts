// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Cross-conversation Memory management panel wiring. */

import {clearMemory, getMemorySettings, putMemorySettings} from '../api/memory.ts';
import {dialogResult} from './conversations.ts';
import {showToast} from './toast.ts';

const SETTINGS_BUTTON_ID = 'memory-settings-btn';
const SETTINGS_DIALOG_ID = 'memory-settings-dialog';
const ENABLED_INPUT_ID = 'memory-enabled-toggle';
const COUNT_TEXT_ID = 'memory-active-count';

function resolveButton(): HTMLButtonElement | null {
  return document.getElementById(SETTINGS_BUTTON_ID) as HTMLButtonElement | null;
}

function resolveDialog(): HTMLDialogElement | null {
  return document.getElementById(SETTINGS_DIALOG_ID) as HTMLDialogElement | null;
}

async function refreshPanel(): Promise<void> {
  const settings = await getMemorySettings();
  const toggle = document.getElementById(ENABLED_INPUT_ID) as HTMLInputElement | null;
  const count = document.getElementById(COUNT_TEXT_ID);
  if (toggle) toggle.checked = settings.enabled;
  if (count) {
    count.textContent =
      settings.active_count === 1 ? '1 stored item' : `${settings.active_count} stored items`;
  }
}

export function setupMemorySettings(): void {
  const trigger = resolveButton();
  const dialog = resolveDialog();
  if (!trigger || !dialog) return;

  trigger.addEventListener('click', async () => {
    try {
      await refreshPanel();
    } catch {
      showToast('Could not load memory settings.', 5000);
    }
    const action = await dialogResult(dialog, () => trigger);
    const toggle = document.getElementById(ENABLED_INPUT_ID) as HTMLInputElement | null;
    if (action === 'save' && toggle) {
      try {
        await putMemorySettings(toggle.checked);
        showToast('Memory settings saved.', 3000);
      } catch {
        showToast('Could not save memory settings.', 5000);
      }
    } else if (action === 'clear') {
      try {
        await clearMemory();
        await refreshPanel();
        showToast('Memory cleared.', 3000);
      } catch {
        showToast('Could not clear memory.', 5000);
      }
    }
  });
}
