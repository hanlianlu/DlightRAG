// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** DOM helpers shared by the browser tests. */

// Bound at import, before any test replaces the page's timers with fakes.
const realSetTimeout = globalThis.setTimeout.bind(globalThis);

/** Resolve once `predicate` holds, yielding one macrotask between checks. */
export async function waitFor(predicate: () => boolean, attempts = 100): Promise<void> {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => realSetTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

/** The first button whose accessible name (its aria-label, else its text) is `name`. */
export function buttonNamed<T extends HTMLElement = HTMLElement>(
  root: ParentNode,
  name: string,
): T | null {
  return Array.from(root.querySelectorAll<T>('button, dl-icon-button'))
    .find((button) => (button.getAttribute('aria-label') || button.textContent?.trim()) === name)
    ?? null;
}
