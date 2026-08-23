// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Cross-conversation Memory facts shown inside the unified settings dialog. */

import {getMemorySettings} from '../api/memory.ts';

const ENABLED_INPUT_ID = 'memory-enabled-toggle';
const COUNT_TEXT_ID = 'memory-active-count';

/** Reflect the current memory settings into the settings dialog controls. */
export async function refreshMemorySettingsPanel(): Promise<void> {
  const settings = await getMemorySettings();
  const toggle = document.getElementById(ENABLED_INPUT_ID) as HTMLInputElement | null;
  const count = document.getElementById(COUNT_TEXT_ID);
  if (toggle) toggle.checked = settings.enabled;
  if (count) {
    count.textContent =
      settings.active_count === 1 ? '1 stored item' : `${settings.active_count} stored items`;
  }
}
