// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

const PANEL_DISMISS_EXEMPT_SELECTOR = [
    '.panel',
    '#files-btn',
    '#composer',
    '[data-action="filter-source"]',
    '[data-action="open-ref-source"]',
].join(', ');

export function closestElement<T extends Element = Element>(
    target: EventTarget | null,
    selector: string,
): T | null {
    if (!(target instanceof Element)) return null;
    return target.closest(selector) as T | null;
}

function shouldDismissPanelOnOutsideClick(target: EventTarget | null): boolean {
    const panel = document.getElementById('panel');
    if (!panel || !panel.classList.contains('open')) return false;
    if (document.body.hasAttribute('data-resizing')) return false;
    return !closestElement(target, PANEL_DISMISS_EXEMPT_SELECTOR);
}

export function openPanel(title?: string): void {
    const panel = document.getElementById('panel');
    if (!panel) return;
    panel.classList.add('open');
    document.body.classList.add('panel-open');
    if (!title) return;

    const titleEl = document.getElementById('panel-title');
    if (!titleEl) return;
    titleEl.textContent = title;

    if (title !== 'FILES') {
        document.getElementById('ingest-target')?.replaceChildren();
    }
}

export function closePanel(): void {
    const panel = document.getElementById('panel');
    if (panel) panel.classList.remove('open');
    document.body.classList.remove('panel-open');
    document.body.dispatchEvent(new CustomEvent('panelClosed'));
}

export function setupPanel(): void {
    const closeBtn = document.getElementById('panel-close-btn');
    if (closeBtn) closeBtn.addEventListener('click', closePanel);

    document.addEventListener('click', function(e) {
        if (shouldDismissPanelOnOutsideClick(e.target)) closePanel();
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closePanel();
    });
}
