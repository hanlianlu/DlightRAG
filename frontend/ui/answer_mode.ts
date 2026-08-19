// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

const STORAGE_KEY = 'dlightrag.answerMode';
const MODES = ['auto', 'fast', 'research'] as const;
type AnswerMode = (typeof MODES)[number];

const LABELS: Record<AnswerMode, string> = {
    auto: 'Auto',
    fast: 'Fast',
    research: 'Research',
};

export function readStoredAnswerMode(): AnswerMode | null {
    const value = localStorage.getItem(STORAGE_KEY);
    return MODES.includes(value as AnswerMode) ? value as AnswerMode : null;
}

export function setupAnswerModeMenu(): void {
    const trigger = document.getElementById('composer-mode');
    const menu = document.getElementById('composer-mode-menu');
    if (!(trigger instanceof HTMLButtonElement) || !(menu instanceof HTMLElement)) return;

    const apply = (mode: AnswerMode | null): void => {
        trigger.dataset.mode = mode ?? 'auto';
        trigger.textContent = LABELS[mode ?? 'auto'];
        trigger.setAttribute('aria-label', `Answer mode: ${LABELS[mode ?? 'auto']}`);
        for (const button of menu.querySelectorAll<HTMLButtonElement>('[data-mode]')) {
            button.setAttribute(
                'aria-checked',
                button.dataset.mode === (mode ?? 'auto') ? 'true' : 'false',
            );
        }
    };

    apply(readStoredAnswerMode());

    trigger.addEventListener('click', function(event) {
        event.stopPropagation();
        const open = menu.hasAttribute('hidden');
        menu.toggleAttribute('hidden', !open);
        trigger.setAttribute('aria-expanded', open ? 'true' : 'false');
    });

    menu.addEventListener('click', function(event) {
        const button = (event.target as HTMLElement | null)?.closest<HTMLButtonElement>('[data-mode]');
        if (!button || !MODES.includes(button.dataset.mode as AnswerMode)) return;
        const mode = button.dataset.mode as AnswerMode;
        localStorage.setItem(STORAGE_KEY, mode);
        apply(mode);
        menu.setAttribute('hidden', '');
        trigger.setAttribute('aria-expanded', 'false');
    });

    document.addEventListener('click', function() {
        menu.setAttribute('hidden', '');
        trigger.setAttribute('aria-expanded', 'false');
    });
}
