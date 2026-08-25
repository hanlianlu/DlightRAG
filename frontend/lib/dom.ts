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

const TOPBAR_SIDEBAR = 'dl-conversation-sidebar';

/**
 * Mark every surface outside a modal Shell pane inert.
 *
 * Derived from body flags so the Inspector, conversation drawer, and Artifact
 * Canvas cannot clear each other's accessibility state.
 */
export function syncShellInert(): void {
    const artifactModal = document.body.classList.contains('artifact-canvas-modal');
    const panelDrawer = document.body.classList.contains('panel-drawer-open');
    const conversationDrawer = document.body.classList.contains('conversation-drawer-open');
    const shellInert = artifactModal || panelDrawer || conversationDrawer;
    const chat = document.querySelector<HTMLElement>('dl-chat-feature');
    if (chat) chat.inert = shellInert;
    const notificationOffer = document.getElementById('notify-offer');
    if (notificationOffer) notificationOffer.inert = shellInert;

    const topbar = document.querySelector<HTMLElement>('.topbar');
    if (topbar) {
        topbar.inert = artifactModal || panelDrawer;
        for (const child of topbar.children) {
            if (!(child instanceof HTMLElement) || child.matches(TOPBAR_SIDEBAR)) continue;
            child.inert = shellInert;
        }
    }

    // Milestone 5 deletes these Shell lookups when modal state flows through composition.
    const sidebar = document.querySelector<HTMLElement & {
        setShellInert(inert: boolean): void;
    }>('dl-conversation-sidebar');
    sidebar?.setShellInert(artifactModal);
    const inspector = document.querySelector<HTMLElement & {
        setShellInert(inert: boolean): void;
    }>('dl-inspector');
    inspector?.setShellInert(artifactModal);
}
