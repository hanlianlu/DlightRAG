// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The box a reader types into: what it reports, and what it leaves alone. */

import {expect} from '@esm-bundle/chai';
import type {DlTraceSearch} from './trace-search.ts';
import './trace-search.ts';

const nothingToDo = {search: async () => ({})};

async function mountSearch(source: unknown): Promise<DlTraceSearch> {
  const element = document.createElement('dl-trace-search') as DlTraceSearch;
  element.source = source as never;
  document.body.append(element);
  await element.updateComplete;
  return element;
}

function boxOf(element: DlTraceSearch): HTMLInputElement | null {
  return element.querySelector('input[type="search"]');
}

const settled = () => new Promise((resolve) => setTimeout(resolve, 320));

it('says what the reader settled on, after they stop typing', async () => {
  const asked: string[] = [];
  const element = await mountSearch(nothingToDo);
  element.addEventListener('dl-trace-query', (event) => asked.push(event.detail.terms));
  const box = boxOf(element)!;
  box.value = 'germany';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await settled();
  expect(asked).to.deep.equal(['germany']);
});

it('keeps the space a reader types between two words', async () => {
  const element = await mountSearch(nothingToDo);
  const box = boxOf(element)!;
  box.value = 'china ';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await element.updateComplete;
  expect(box.value, "the box is the reader's, edge to edge").to.equal('china ');
});

it('reports the words without their edges, and nothing when cleared', async () => {
  const asked: string[] = [];
  const element = await mountSearch(nothingToDo);
  element.addEventListener('dl-trace-query', (event) => asked.push(event.detail.terms));
  const box = boxOf(element)!;
  box.value = '  china  report  ';
  box.dispatchEvent(new Event('input', {bubbles: true}));
  await settled();
  expect(asked).to.deep.equal(['china  report']);
  (element.querySelector('button') as HTMLButtonElement).click();
  await element.updateComplete;
  expect(asked, 'and clearing says the panes are whole again').to.deep.equal(['china  report', '']);
  expect(boxOf(element)!.value, 'with the box empty').to.equal('');
});

it('shows no box for a Run whose words cannot be searched', async () => {
  const element = await mountSearch(null);
  expect(boxOf(element), 'nothing to type into').to.equal(null);
});
