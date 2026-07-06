// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {sanitizeHtml} from '../lib/safe_html.ts';
import {renderMath} from './mathjax.ts';

function htmxEvent(event: Event): HTMXEvent {
    return event as HTMXEvent;
}

function isPanelTarget(event: Event): boolean {
    const detail = htmxEvent(event).detail;
    return detail.target?.id === 'panel-content';
}

export function setupHtmxInteractions(): void {
    document.body.addEventListener('htmx:beforeSwap', function(event) {
        const detail = htmxEvent(event).detail;
        if (typeof detail.serverResponse === 'string') {
            detail.serverResponse = sanitizeHtml(detail.serverResponse);
        }
        if (isPanelTarget(event) && detail.xhr.status >= 400) {
            detail.shouldSwap = true;
            detail.isError = false;
        }
    });

    document.body.addEventListener('htmx:afterSettle', function(event) {
        const target = htmxEvent(event).detail.target;
        if (target) renderMath(target);
    });
}
