// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {getAnswerReport, type AnswerPresentation} from '../api/conversations.ts';
import chatStyles from '../styles/chat.module.css';
import type {AnswerPresentationElement} from './answer_presentation.ts';
import './answer_presentation.ts';
import {openPanel} from './panel.ts';
import {showToast} from './toast.ts';

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
  let presentation: AnswerPresentation;
  try {
    presentation = await getAnswerReport(runId);
  } catch {
    showToast('Could not open the report.');
    return;
  }
  const panelContent = document.getElementById('report-panel-content');
  if (!panelContent) return;
  const element = document.createElement('answer-presentation') as AnswerPresentationElement;
  element.presentation = presentation;
  panelContent.replaceChildren(element);
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
