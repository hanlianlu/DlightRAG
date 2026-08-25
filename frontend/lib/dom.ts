// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Shared DOM helpers used across the UI modules. */

/**
 * Wrap Tab at the ends of `focusable`. Returns true when the event was handled.
 *
 * Callers keep their own visibility predicate and their own policy for an empty
 * container or for focus that escaped it — only the wrap itself is shared.
 */
export function wrapTabFocus(focusable: readonly HTMLElement[], event: KeyboardEvent): boolean {
    if (focusable.length === 0) return false;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
        return true;
    }
    if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
        return true;
    }
    return false;
}
