// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Web API client for owner Profile Memory management. */

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {parseWire} from './wire.ts';

const memorySettings = v.pipe(
  v.object({enabled: v.boolean(), active_count: v.nullable(v.number())}),
  v.transform((w) => ({enabled: w.enabled, activeCount: w.active_count})),
);
export type MemorySettings = v.InferOutput<typeof memorySettings>;

const memoryOperationReceipt = v.pipe(
  v.object({
    action: v.picklist(['remember', 'forget', 'undo']),
    outcome: v.picklist(['changed', 'unchanged', 'conflict']),
    change_id: v.string(),
    memory_ids: v.array(v.string()),
    kind: v.optional(v.nullable(v.picklist(['preference', 'fact']))),
    body: v.string(),
    supersedes_id: v.optional(v.nullable(v.string())),
    target_change_id: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({
    action: w.action,
    outcome: w.outcome,
    changeId: w.change_id,
    memoryIds: w.memory_ids,
    kind: w.kind ?? null,
    body: w.body,
    supersedesId: w.supersedes_id ?? null,
    targetChangeId: w.target_change_id ?? null,
  })),
);
export type MemoryOperationReceipt = v.InferOutput<typeof memoryOperationReceipt>;

export async function getMemorySettings(signal?: AbortSignal): Promise<MemorySettings> {
  const response = await fetch('/web/api/memory/settings', {signal});
  return parseWire(
    response,
    memorySettings,
    (status, message) => new Error(`${message} (${status})`),
    'Failed to load memory settings',
  );
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
  return parseWire(
    response,
    memorySettings,
    (status, message) => new Error(`${message} (${status})`),
    'Failed to update memory settings',
  );
}

export async function undoMemoryChange(
  changeId: string,
  signal?: AbortSignal,
): Promise<MemoryOperationReceipt> {
  const response = await fetch(
    `/web/api/memory/changes/${encodeURIComponent(changeId)}/undo`,
    {
      method: 'POST',
      headers: {
        ...csrfHeaders(),
        'Idempotency-Key': crypto.randomUUID(),
      },
      signal,
    },
  );
  return parseWire(
    response,
    memoryOperationReceipt,
    (status, message) => new Error(`${message} (${status})`),
    'Failed to undo memory change',
  );
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
