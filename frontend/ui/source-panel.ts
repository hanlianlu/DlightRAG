// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {closestElement} from '../lib/dom.ts';
import type {AnswerPresentationElement} from './answer_presentation.ts';
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

function presentationHost(origin: Element): AnswerPresentationElement | null {
  return origin.closest('answer-presentation');
}

function openSources(origin: HTMLElement, ref?: string, chunk?: string): void {
  const presentation = presentationHost(origin)?.presentation;
  const panelContent = document.getElementById('panel-content');
  if (!presentation || !panelContent) return;
  const view = document.createElement('source-panel-view');
  view.sources = presentation.sources;
  view.setSelection(ref, chunk);
  panelContent.replaceChildren(view);
  openPanel('SOURCES');
  updateToggleAllButton();
}

function openRefSource(reference: HTMLElement): void {
  openSources(reference, reference.dataset.ref);
}

export function filterSource(citation: HTMLElement): void {
  openSources(citation, citation.dataset.ref, citation.dataset.chunk);
}

function toggleAllSources(): void {
  const view = sourceView();
  if (!view) return;
  if (view.fullyExpanded) view.collapseAll();
  else view.expandAll();
  updateToggleAllButton();
}

export function setupSourcePanel(): void {
  document.addEventListener('click', function(event) {
    const citation = closestElement<HTMLElement>(event.target, '.citation-badge[data-ref]');
    if (citation) {
      event.preventDefault();
      filterSource(citation);
      return;
    }
    const reference = closestElement<HTMLElement>(event.target, '.answer-ref-item[data-ref]');
    if (reference) {
      event.preventDefault();
      openRefSource(reference);
    }
  });

  document.addEventListener('keydown', function(event) {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const citation = closestElement<HTMLElement>(event.target, '.citation-badge[data-ref]');
    if (citation) {
      event.preventDefault();
      filterSource(citation);
      return;
    }
    const reference = closestElement<HTMLElement>(event.target, '.answer-ref-item[data-ref]');
    if (reference) {
      event.preventDefault();
      openRefSource(reference);
    }
  });

  getToggleAllBtn()?.addEventListener('click', toggleAllSources);
  document.addEventListener('source-panel-change', updateToggleAllButton);

  document.body.addEventListener('panelOpening', function(event) {
    const title = (event as CustomEvent<{title?: string}>).detail?.title;
    if (title === 'FILES') hideToggleAllButton();
  });
  document.body.addEventListener('panelClosed', hideToggleAllButton);
}
