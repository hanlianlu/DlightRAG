// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Web API client for Cross-conversation Memory management. */

import {csrfHeaders} from './csrf.ts';

export interface MemorySettings {
  enabled: boolean;
  active_count: number;
}

export async function getMemorySettings(signal?: AbortSignal): Promise<MemorySettings> {
  const response = await fetch('/web/api/memory/settings', {signal});
  if (!response.ok) {
    throw new Error(`Failed to load memory settings (${response.status})`);
  }
  return (await response.json()) as MemorySettings;
}

export async function putMemorySettings(
  enabled: boolean,
  signal?: AbortSignal,
): Promise<MemorySettings> {
  const response = await fetch('/web/api/memory/settings', {
    method: 'PUT',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({enabled}),
    signal,
  });
  if (!response.ok) {
    throw new Error(`Failed to update memory settings (${response.status})`);
  }
  return (await response.json()) as MemorySettings;
}

export async function clearMemory(signal?: AbortSignal): Promise<void> {
  const response = await fetch('/web/api/memory/clear', {
    method: 'POST',
    headers: csrfHeaders(),
    signal,
  });
  if (!response.ok) {
    throw new Error(`Failed to clear memory (${response.status})`);
  }
}
