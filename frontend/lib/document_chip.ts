// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import chatStyles from '../styles/chat.module.css';

// Single source of truth for the document-attachment chip, shared by the
// composer strip (ui/attachments.ts) and history rendering (lib/chat_renderer.ts)
// so the two never drift. Built with createElement/textContent only (no
// innerHTML) and marked `data-document-attachment` for E2E/selection.

export function formatDocumentSize(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes < 0) return '';
    if (bytes < 1024) return `${bytes} B`;
    const units = ['KB', 'MB', 'GB'];
    let value = bytes / 1024;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
    }
    const rounded =
        value >= 10 || Number.isInteger(value) ? Math.round(value) : Math.round(value * 10) / 10;
    return `${rounded} ${units[unitIndex]}`;
}

export interface DocumentChipOptions {
    filename: string;
    byteSize: number;
    // Same-origin download link (history chips). Omit for a non-link chip.
    href?: string;
    // Remove affordance (composer chips). Omit for a read-only chip.
    onRemove?: () => void;
}

export function createDocumentChip(options: DocumentChipOptions): HTMLElement {
    const chip = document.createElement(options.href ? 'a' : 'div');
    chip.className = chatStyles.documentChip;
    chip.dataset.documentAttachment = 'true';
    if (chip instanceof HTMLAnchorElement && options.href) {
        chip.href = options.href;
        chip.setAttribute('target', '_blank');
        chip.setAttribute('rel', 'noopener noreferrer');
        chip.setAttribute('aria-label', `Download ${options.filename}`);
    }

    const info = document.createElement('div');
    info.className = chatStyles.documentChipInfo;

    const name = document.createElement('span');
    name.className = chatStyles.documentChipName;
    name.textContent = options.filename;
    name.title = options.filename;

    const meta = document.createElement('span');
    meta.className = chatStyles.documentChipMeta;
    meta.textContent = formatDocumentSize(options.byteSize);

    info.append(name, meta);
    chip.appendChild(info);

    const onRemove = options.onRemove;
    if (onRemove) {
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = chatStyles.documentChipRemove;
        remove.setAttribute('aria-label', `Remove ${options.filename}`);
        remove.textContent = '\u00d7';
        remove.addEventListener('click', function() { onRemove(); });
        chip.appendChild(remove);
    }

    return chip;
}
