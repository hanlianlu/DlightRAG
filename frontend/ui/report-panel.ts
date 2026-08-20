// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {llmFragmentFromSanitizedHtml} from '../lib/safe_html.ts';
import {renderMath} from '../lib/math.ts';
import chatStyles from '../styles/chat.module.css';
import {renderDiagrams} from './mermaid.ts';
import {openPanel} from './panel.ts';

function fixExternalLinks(container: ParentNode): void {
    container.querySelectorAll('a[href]').forEach(function(el: Element) {
        const a = el as HTMLAnchorElement;
        if (a.hasAttribute('download')) return;
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener noreferrer');
    });
}

export function bindPrimaryReportControl(
    aiDiv: HTMLElement,
    runId: string,
    handle: string | null | undefined,
): void {
    aiDiv.querySelectorAll('[data-action="open-primary-report"]').forEach((node) => node.remove());
    if (!handle || !runId) return;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = chatStyles.reportControl;
    button.dataset.action = 'open-primary-report';
    button.dataset.runId = runId;
    button.textContent = 'View report';
    aiDiv.appendChild(button);
}

export async function openPrimaryReport(runId: string): Promise<void> {
    const response = await fetch(`/web/answer/${encodeURIComponent(runId)}/report`);
    if (!response.ok) return;
    const html = await response.text();
    const panelContent = document.getElementById('report-panel-content');
    if (!panelContent) return;
    const fragment = llmFragmentFromSanitizedHtml(html);
    const answerContent = fragment.querySelector('#answer-content');
    panelContent.replaceChildren();
    if (answerContent) {
        panelContent.append(
            ...Array.from(answerContent.childNodes).map((node) => node.cloneNode(true)),
        );
    }
    const sourceData = fragment.querySelector('#source-data, .source-data');
    if (sourceData) {
        const copy = sourceData.cloneNode(true) as HTMLElement;
        copy.className = 'source-data hidden';
        copy.removeAttribute('id');
        panelContent.appendChild(copy);
    }
    const references = fragment.querySelector('.answer-references');
    if (references) panelContent.appendChild(references.cloneNode(true));
    renderMath(panelContent);
    renderDiagrams(panelContent);
    fixExternalLinks(panelContent);
    openPanel('REPORT');
}

export function setupReportPanel(): void {
    document.body.addEventListener('click', function(event: MouseEvent) {
        const target = event.target;
        if (!(target instanceof Element)) return;
        const button = target.closest('[data-action="open-primary-report"]');
        if (!(button instanceof HTMLElement)) return;
        const runId = button.dataset.runId;
        if (runId) void openPrimaryReport(runId);
    });
}
