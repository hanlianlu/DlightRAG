// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Shared DOM helpers used across the UI modules. */

export function closestElement<T extends Element = Element>(
    target: EventTarget | null,
    selector: string,
): T | null {
    if (!(target instanceof Element)) return null;
    return target.closest(selector) as T | null;
}

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

const SHELL_SELECTORS = ['.topbar', '.chat-area', '.composer'];

/**
 * Mark the app shell inert while any drawer-style surface is modal.
 *
 * Derived from the body flags rather than from a single caller, so the panel and
 * the conversation sidebar cannot clear each other's inert state.
 */
export function syncShellInert(extraModal = false): void {
    const inert =
        extraModal ||
        document.body.classList.contains('panel-drawer-open') ||
        document.body.classList.contains('conversation-drawer-open');
    for (const selector of SHELL_SELECTORS) {
        const element = document.querySelector<HTMLElement>(selector);
        if (element) element.inert = inert;
    }
}
