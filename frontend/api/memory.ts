// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Web API client for owner Profile Memory management. */

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {apiError, parseWire} from './wire.ts';

/** Every kind a memory may have: what the owner wants, what they are like, what was learned. */
const memoryKind = v.picklist(['preference', 'fact', 'experience']);

const memorySettings = v.pipe(
  v.object({enabled: v.boolean(), active_count: v.nullable(v.number())}),
  v.transform((w) => ({enabled: w.enabled, activeCount: w.active_count})),
);
export type MemorySettings = v.InferOutput<typeof memorySettings>;

const memoryOperationReceipt = v.pipe(
  v.object({
    action: v.picklist(['remember', 'forget', 'undo', 'fuse']),
    outcome: v.picklist(['changed', 'unchanged', 'conflict']),
    change_id: v.string(),
    memory_ids: v.array(v.string()),
    kind: v.optional(v.nullable(memoryKind)),
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
type MemoryOperationReceipt = v.InferOutput<typeof memoryOperationReceipt>;

/** A live `memory_operation_settled` Answer event, as the browser projection sends it. */
const memoryOperationEvent = v.pipe(
  v.object({
    operation: v.picklist(['remember', 'forget', 'undo']),
    outcome: v.picklist(['changed', 'unchanged', 'rejected', 'conflict']),
    change_id: v.optional(v.nullable(v.string())),
    intent_id: v.optional(v.nullable(v.string())),
    body: v.optional(v.nullable(v.string())),
    live: v.optional(v.boolean()),
  }),
  v.transform((w) => ({
    operation: w.operation,
    outcome: w.outcome,
    changeId: w.change_id ?? null,
    intentId: w.intent_id ?? null,
    body: w.body ?? '',
    live: w.live ?? false,
  })),
);
export type MemoryOperationEvent = v.InferOutput<typeof memoryOperationEvent>;

/** Return the typed event, or null for a payload this client cannot interpret. */
export function parseMemoryOperationEvent(value: unknown): MemoryOperationEvent | null {
  const parsed = v.safeParse(memoryOperationEvent, value);
  return parsed.success ? parsed.output : null;
}

export async function getMemorySettings(signal?: AbortSignal): Promise<MemorySettings> {
  const response = await fetch('/web/api/memory/settings', {signal});
  return parseWire(response, memorySettings);
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
  return parseWire(response, memorySettings);
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
  return parseWire(response, memoryOperationReceipt);
}

export async function clearMemory(signal?: AbortSignal): Promise<void> {
  const response = await fetch('/web/api/memory/clear', {
    method: 'POST',
    headers: csrfHeaders(),
    signal,
  });
  if (!response.ok) throw await apiError(response);
}

const memoryRecord = v.pipe(
  v.object({
    memory_id: v.string(),
    kind: memoryKind,
    body: v.string(),
  }),
  v.transform((w) => ({
    memoryId: w.memory_id,
    kind: w.kind,
    body: w.body,
  })),
);
export type MemoryRecord = v.InferOutput<typeof memoryRecord>;
const memoryPage = v.pipe(
  v.object({memories: v.array(memoryRecord), next_cursor: v.nullable(v.string())}),
  v.transform((w) => ({items: w.memories, nextCursor: w.next_cursor})),
);
export type MemoryPage = v.InferOutput<typeof memoryPage>;

export async function listMemories(cursor: string | null, signal?: AbortSignal): Promise<MemoryPage> {
  const query = new URLSearchParams({limit: '20'});
  if (cursor) query.set('cursor', cursor);
  const response = await fetch(`/web/api/memory?${query}`, {signal});
  return parseWire(response, memoryPage);
}

/** One record a Run's writers left in the owner's profile that is still active, with the change that made it. */
const runMemoryChange = v.pipe(
  v.object({
    memory_id: v.string(),
    kind: memoryKind,
    body: v.string(),
    // A new statement, the correction of an earlier one, or the merge of several.
    action: v.picklist(['add', 'update', 'fuse']),
    change_id: v.string(),
    // The Agent's own memory tool during the Run, or the pass that read the conversation after it.
    writer: v.picklist(['agent', 'conversation']),
  }),
  v.transform((w) => ({
    memoryId: w.memory_id,
    kind: w.kind,
    body: w.body,
    action: w.action,
    changeId: w.change_id,
    writer: w.writer,
  })),
);
export type RunMemoryChange = v.InferOutput<typeof runMemoryChange>;
const runMemoryChanges = v.pipe(
  v.object({changes: v.array(runMemoryChange)}),
  v.transform((w) => w.changes),
);

/** What one Run's writers left in the owner's Profile Memory, oldest first; none while Memory is off. */
export async function listRunMemoryChanges(
  runId: string,
  signal?: AbortSignal,
): Promise<readonly RunMemoryChange[]> {
  const response = await fetch(`/web/api/memory/runs/${encodeURIComponent(runId)}/changes`, {signal});
  return parseWire(response, runMemoryChanges);
}

export async function forgetMemory(memoryId: string, signal?: AbortSignal): Promise<MemoryOperationReceipt> {
  const response = await fetch(`/web/api/memory/${encodeURIComponent(memoryId)}`, {
    method: 'DELETE',
    headers: {...csrfHeaders(), 'Idempotency-Key': crypto.randomUUID()},
    signal,
  });
  return parseWire(response, memoryOperationReceipt);
}
