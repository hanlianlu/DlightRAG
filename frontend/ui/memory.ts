// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Profile Memory settings projection and interaction inside the settings drawer. */

import {
  getMemorySettings,
  putMemorySettings,
  undoMemoryChange,
  type MemorySettings,
} from '../api/memory.ts';
import type {ChatMemoryOperationDetail} from './chat_feature.ts';
import {showActionToast, showToast} from './toast.ts';

const ENABLED_INPUT_ID = 'memory-enabled-toggle';
const COUNT_TEXT_ID = 'memory-active-count';
const CLEAR_BUTTON_ID = 'memory-clear-btn';
const seenMemoryOperations = new Set<string>();

function memorySummary(event: ChatMemoryOperationDetail): string {
  const body = String(event.body || '').replace(/\s+/g, ' ').trim();
  const concise = body.length > 120 ? body.slice(0, 117) + '…' : body;
  if (event.outcome === 'unchanged') return 'Already remembered.';
  if (event.outcome === 'conflict') return 'Profile Memory changed; recall it before retrying.';
  if (event.outcome === 'rejected') return 'Profile Memory operation was rejected.';
  if (event.operation === 'forget') return concise ? `Forgot: ${concise}` : 'Profile Memory forgotten.';
  if (event.operation === 'undo') return concise ? `Restored: ${concise}` : 'Profile Memory restored.';
  return concise ? `Remembered: ${concise}` : 'Saved to Profile Memory.';
}

function handleChatMemoryOperation(event: ChatMemoryOperationDetail): void {
  if (!event.live) return;
  const identity = event.change_id || `${event.intent_id || ''}:${event.operation}:${event.outcome}`;
  if (!identity || seenMemoryOperations.has(identity)) return;
  seenMemoryOperations.add(identity);
  const message = memorySummary(event);
  if (event.outcome !== 'changed' || !event.change_id) {
    showToast(message, 5000);
    return;
  }
  const changeId = event.change_id;
  showActionToast(message, {
    actionLabel: 'Undo',
    duration: 12_000,
    onAction: async () => {
      const receipt = await undoMemoryChange(changeId);
      if (receipt.outcome !== 'changed') throw new Error('Memory undo conflicted');
      void refreshMemorySettingsPanel().catch(() => {});
      return 'Profile Memory change undone.';
    },
  });
  void refreshMemorySettingsPanel().catch(() => {});
}

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

/** Milestone 5 Shell adapter for Chat memory facts, Settings, and Toast. */
export function setupChatMemoryOperationAdapter(): void {
  document.querySelector('dl-chat-feature')?.addEventListener(
    'dl-chat-memory-operation',
    (event) => handleChatMemoryOperation(
      (event as CustomEvent<ChatMemoryOperationDetail>).detail,
    ),
  );
}

/** Load and reflect the authoritative setting after a Memory mutation. */
export async function refreshMemorySettingsPanel(): Promise<MemorySettings> {
  const settings = await getMemorySettings();
  renderMemorySettingsPanel(settings);
  return settings;
}
