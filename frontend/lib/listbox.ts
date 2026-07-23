// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/**
 * Add ArrowUp/ArrowDown/Home/End roving navigation to a container and a
 * focusable item selector. Wrapping is circular; Home/End jump to the
 * first/last item. Enter/Space activation stays with each item's own handler.
 */
export function installRovingArrowNavigation(container: HTMLElement, itemSelector: string): void {
    container.addEventListener('keydown', (event) => {
        if (
            event.key !== 'ArrowDown' &&
            event.key !== 'ArrowUp' &&
            event.key !== 'Home' &&
            event.key !== 'End'
        ) {
            return;
        }
        const options = Array.from(container.querySelectorAll<HTMLElement>(itemSelector));
        if (options.length === 0) return;
        event.preventDefault();
        const current = options.indexOf(document.activeElement as HTMLElement);
        let next: number;
        if (event.key === 'Home') {
            next = 0;
        } else if (event.key === 'End') {
            next = options.length - 1;
        } else {
            const delta = event.key === 'ArrowDown' ? 1 : -1;
            next =
                current < 0
                    ? delta > 0
                        ? 0
                        : options.length - 1
                    : (current + delta + options.length) % options.length;
        }
        options[next]?.focus();
    });
}
