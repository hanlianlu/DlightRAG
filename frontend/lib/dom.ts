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
 * Mark every surface outside a modal Shell pane inert.
 *
 * Derived from body flags so the Inspector, conversation drawer, and Artifact
 * Canvas cannot clear each other's accessibility state.
 */
export function syncShellInert(): void {
    const artifactModal = document.body.classList.contains('artifact-canvas-modal');
    const shellInert =
        artifactModal ||
        document.body.classList.contains('panel-drawer-open') ||
        document.body.classList.contains('conversation-drawer-open');
    for (const selector of SHELL_SELECTORS) {
        const element = document.querySelector<HTMLElement>(selector);
        if (element) element.inert = shellInert;
    }

    const sidebar = document.getElementById('chat-sidebar');
    if (sidebar) {
        const expanded = document.body.classList.contains('conversation-sidebar-open');
        sidebar.inert = artifactModal || !expanded;
    }
    const panel = document.getElementById('panel');
    if (panel) panel.inert = artifactModal || !panel.classList.contains('open');
}
