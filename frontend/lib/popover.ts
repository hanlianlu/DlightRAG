// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/**
 * Shared outside-click / Escape dismissal for a transient popover.
 *
 * Each popover still builds and mounts its own DOM; this only owns the fiddly
 * lifecycle the workspace-selector and ingest-target popovers had duplicated:
 * a capture-phase Escape handler that yields to any open `<dialog>`, and an
 * outside-click handler armed on the next tick so the opening click does not
 * immediately dismiss it.
 */
export interface AutoDismissOptions {
    /** Element whose subtree counts as "inside"; clicks within it never dismiss. */
    getAnchor: () => Element | null;
    /** Whether the popover is currently open (guards the Escape handler). */
    isOpen: () => boolean;
    /** Invoked when an outside click or Escape should close the popover. */
    onDismiss: () => void;
}

export interface AutoDismiss {
    /** Start listening for outside-click and Escape. Call when the popover opens. */
    activate(): void;
    /** Stop listening. Call when the popover closes. Idempotent. */
    deactivate(): void;
}

export function createAutoDismiss(options: AutoDismissOptions): AutoDismiss {
    const {getAnchor, isOpen, onDismiss} = options;

    function onOutsideClick(event: MouseEvent): void {
        const anchor = getAnchor();
        if (anchor && event.target instanceof Node && !anchor.contains(event.target)) {
            onDismiss();
        }
    }

    function onEscapeKey(event: KeyboardEvent): void {
        if (event.key !== 'Escape' || !isOpen()) return;
        if (document.querySelector('dialog[open]')) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        onDismiss();
    }

    return {
        activate(): void {
            document.addEventListener('keydown', onEscapeKey, true);
            setTimeout(() => {
                document.addEventListener('click', onOutsideClick);
            }, 0);
        },
        deactivate(): void {
            document.removeEventListener('click', onOutsideClick);
            document.removeEventListener('keydown', onEscapeKey, true);
        },
    };
}
