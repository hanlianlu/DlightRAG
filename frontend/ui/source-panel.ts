// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import type {AnswerPresentation} from '../api/conversations.ts';
import {openPanel} from './panel.ts';
import type {SourcePanelView} from './source_panel_view.ts';
import './source_panel_view.ts';

function getToggleAllBtn(): HTMLButtonElement | null {
  return document.getElementById('source-toggle-all-btn') as HTMLButtonElement | null;
}

function sourceView(): SourcePanelView | null {
  return document.querySelector('#panel-content > source-panel-view');
}

function updateToggleAllButton(): void {
  const button = getToggleAllBtn();
  const view = sourceView();
  if (!button) return;
  button.hidden = !view || view.sources.length === 0;
  if (button.hidden || !view) return;
  button.textContent = view.fullyExpanded ? 'Collapse all' : 'Show all';
  button.setAttribute('aria-pressed', view.fullyExpanded ? 'true' : 'false');
}

function hideToggleAllButton(): void {
  const button = getToggleAllBtn();
  if (button) button.hidden = true;
}

/** Milestone 4 adapter from AnswerPresentation intent to the legacy Inspector. */
export function openAnswerSources(
  presentation: AnswerPresentation,
  referenceId?: string,
  chunkId?: string,
  returnFocus?: HTMLElement | null,
): void {
  const panelContent = document.getElementById('panel-content');
  if (!panelContent) return;
  const view = document.createElement('source-panel-view');
  view.sources = presentation.sources;
  view.setSelection(referenceId, chunkId);
  panelContent.replaceChildren(view);
  openPanel('SOURCES', returnFocus);
  updateToggleAllButton();
}

function toggleAllSources(): void {
  const view = sourceView();
  if (!view) return;
  if (view.fullyExpanded) view.collapseAll();
  else view.expandAll();
  updateToggleAllButton();
}

export function setupSourcePanel(): void {
  getToggleAllBtn()?.addEventListener('click', toggleAllSources);
  document.addEventListener('source-panel-change', updateToggleAllButton);

  document.body.addEventListener('panelOpening', function(event) {
    const title = (event as CustomEvent<{title?: string}>).detail?.title;
    if (title === 'FILES') hideToggleAllButton();
  });
  document.body.addEventListener('panelClosed', hideToggleAllButton);
}
