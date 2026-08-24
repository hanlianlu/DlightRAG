// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Profile Memory settings projection and interaction inside the settings drawer. */

import {getMemorySettings, putMemorySettings} from '../api/memory.ts';
import type {MemorySettings} from '../api/memory.ts';

const ENABLED_INPUT_ID = 'memory-enabled-toggle';
const COUNT_TEXT_ID = 'memory-active-count';
const CLEAR_BUTTON_ID = 'memory-clear-btn';

export function renderMemorySettingsPanel(settings: MemorySettings): void {
  const toggle = document.getElementById(ENABLED_INPUT_ID) as HTMLInputElement | null;
  const count = document.getElementById(COUNT_TEXT_ID);
  const clear = document.getElementById(CLEAR_BUTTON_ID) as HTMLButtonElement | null;
  if (toggle) toggle.checked = settings.enabled;
  if (count) {
    const active = settings.active_count;
    count.textContent = active === null
      ? ''
      : active === 1
        ? '1 stored item'
        : `${active} stored items`;
    count.hidden = active === null;
  }
  if (clear) clear.hidden = !settings.enabled;
}

export async function prepareMemorySettingsPanel(): Promise<boolean> {
  const toggle = document.getElementById(ENABLED_INPUT_ID) as HTMLInputElement | null;
  try {
    const settings = await getMemorySettings();
    renderMemorySettingsPanel(settings);
    if (toggle) toggle.disabled = false;
    return true;
  } catch {
    renderMemorySettingsPanel({enabled: false, active_count: null});
    if (toggle) toggle.disabled = true;
    return false;
  }
}

export function setupMemorySettings(onError: (message: string) => void): void {
  const toggle = document.getElementById(ENABLED_INPUT_ID) as HTMLInputElement | null;
  toggle?.addEventListener('change', async function() {
    const requested = toggle.checked;
    toggle.disabled = true;
    try {
      renderMemorySettingsPanel(await putMemorySettings(requested));
    } catch {
      toggle.checked = !requested;
      onError('Could not save memory settings.');
    } finally {
      toggle.disabled = false;
    }
  });
}

/** Load and reflect the authoritative setting after a Memory mutation. */
export async function refreshMemorySettingsPanel(): Promise<MemorySettings> {
  const settings = await getMemorySettings();
  renderMemorySettingsPanel(settings);
  return settings;
}
