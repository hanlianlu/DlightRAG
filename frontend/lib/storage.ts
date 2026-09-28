// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Per-browser preferences in localStorage: an enhancement, never a requirement.

 * Storage can be missing or throw (private windows, hardened settings,
 * sandboxed frames). Every read then reports "unset" and every write is
 * dropped, so the caller's choice still applies for this page.
 */

function localStore(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export function readStored(key: string): string | null {
  try {
    return localStore()?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

/** Store a value; null removes the key. */
export function writeStored(key: string, value: string | null): void {
  try {
    const storage = localStore();
    if (value === null) storage?.removeItem(key);
    else storage?.setItem(key, value);
  } catch {
    // The choice still applies for this page when storage is blocked.
  }
}

/** Whether a `storage` event reports a change to this origin's localStorage. */
export function isLocalStorageEvent(event: StorageEvent): boolean {
  const storage = localStore();
  return storage !== null && event.storageArea === storage;
}
