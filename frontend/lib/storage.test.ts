// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {isLocalStorageEvent, readStored, writeStored} from './storage.ts';

const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');

function useStorage(storage: () => Storage): void {
  Object.defineProperty(globalThis, 'localStorage', {configurable: true, get: storage});
}

function memoryStorage(): Storage {
  const entries = new Map<string, string>();
  return {
    getItem: (key: string) => entries.get(key) ?? null,
    setItem: (key: string, value: string) => { entries.set(key, value); },
    removeItem: (key: string) => { entries.delete(key); },
  } as Storage;
}

test.afterEach(() => {
  if (original) Object.defineProperty(globalThis, 'localStorage', original);
  else delete (globalThis as {localStorage?: Storage}).localStorage;
});

test('preferences round-trip, and null removes one', () => {
  const storage = memoryStorage();
  useStorage(() => storage);

  assert.equal(readStored('dlightrag.mode'), null);
  writeStored('dlightrag.mode', 'research');
  assert.equal(readStored('dlightrag.mode'), 'research');
  writeStored('dlightrag.mode', null);
  assert.equal(readStored('dlightrag.mode'), null);
});

test('blocked storage reads as unset and drops writes without throwing', () => {
  useStorage(() => { throw new DOMException('denied', 'SecurityError'); });
  assert.equal(readStored('dlightrag.mode'), null);
  assert.doesNotThrow(() => { writeStored('dlightrag.mode', 'fast'); });

  const failing = memoryStorage();
  failing.setItem = () => { throw new DOMException('full', 'QuotaExceededError'); };
  failing.getItem = () => { throw new DOMException('denied', 'SecurityError'); };
  useStorage(() => failing);
  assert.equal(readStored('dlightrag.mode'), null);
  assert.doesNotThrow(() => { writeStored('dlightrag.mode', 'fast'); });
});

test('only a change to localStorage on this origin counts as a preference change', () => {
  const storage = memoryStorage();
  useStorage(() => storage);
  assert.equal(isLocalStorageEvent({storageArea: storage} as StorageEvent), true);
  assert.equal(isLocalStorageEvent({storageArea: memoryStorage()} as StorageEvent), false);
  useStorage(() => { throw new DOMException('denied', 'SecurityError'); });
  assert.equal(isLocalStorageEvent({storageArea: storage} as StorageEvent), false);
});
