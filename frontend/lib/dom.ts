// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Shared DOM helpers used across the UI modules. */

/** What Tab can reach; a <summary> is focusable although it carries no tabindex. */
const TABBABLE = [
    'a[href]',
    'button:not([disabled])',
    'dl-icon-button:not([disabled])',
    'input:not([disabled]):not([type="hidden"])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    'summary',
    'iframe',
    'audio[controls]',
    'video[controls]',
    '[contenteditable]:not([contenteditable="false"])',
    '[tabindex]:not([tabindex="-1"])',
].join(', ');

/** Whether a collapsed <details> hides `element`; its own <summary> stays reachable. */
function collapsedAway(element: HTMLElement): boolean {
    const collapsed = element.closest('details:not([open])');
    return collapsed !== null && !collapsed.querySelector(':scope > summary')?.contains(element);
}

/** Every rendered Tab stop inside `root`, in document order, for a focus trap. */
export function tabbables(root: ParentNode): HTMLElement[] {
    // A collapsed <details> still reports client rects for its content, so the
    // disclosure is checked explicitly: Tab skips that content, and a trap whose
    // last stop is unreachable never wraps.
    return Array.from(root.querySelectorAll<HTMLElement>(TABBABLE))
        .filter((element) => !element.hidden
            && element.getClientRects().length > 0
            && !collapsedAway(element));
}

/**
 * Wrap Tab at the ends of `focusable`. Returns true when the event was handled.
 *
 * Callers choose the elements (usually `tabbables`) and keep their own policy for
 * an empty container or for focus that escaped it — only the wrap itself is shared.
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
