// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import './run_dialogs.ts';
import type {ChildRosterEntry, DlChildrenRoster} from './run_dialogs.ts';

function entry(id: string, status = 'succeeded'): ChildRosterEntry {
  return {child_session_id: id, status, objective: `objective ${id}`};
}

function deferredPage() {
  let resolve!: (page: {children: ChildRosterEntry[]; next_cursor: string | null}) => void;
  const promise = new Promise<{children: ChildRosterEntry[]; next_cursor: string | null}>(
    (done) => { resolve = done; },
  );
  return {promise, resolve};
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

function roster(): DlChildrenRoster {
  const element = document.createElement('dl-children-roster') as DlChildrenRoster;
  document.body.appendChild(element);
  return element;
}

afterEach(() => {
  document.body.replaceChildren();
});

it('legacy fetcher renders every child without a paging control', async () => {
  const panel = roster();
  panel.open(async () => [entry('a'), entry('b')]);
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 2);

  expect([...panel.querySelectorAll('li[role="listitem"]')].map((li) => li.textContent?.trim()))
    .to.deep.equal(['succeeded: objective a', 'succeeded: objective b']);
  expect(panel.querySelector('[data-load-older-children]')).to.equal(null);
});

it('paged roster renders the newest page and appends older pages with dedup', async () => {
  const older = deferredPage();
  let olderRequests = 0;
  const panel = roster();
  panel.open(
    async () => [entry('newest')],
    async (cursor) => {
      if (cursor === null) {
        return {children: [entry('newest')], next_cursor: 'older-1'};
      }
      expect(cursor).to.equal('older-1');
      olderRequests += 1;
      return older.promise;
    },
  );
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 1);

  const button = panel.querySelector<HTMLButtonElement>('[data-load-older-children]')!;
  expect(button.textContent?.trim()).to.equal('Load older children');
  button.click();
  const flight = panel.loadOlderChildren();
  expect(panel.loadOlderChildren()).to.equal(flight);
  await panel.updateComplete;
  expect(button.disabled).to.equal(true);
  expect(button.getAttribute('aria-busy')).to.equal('true');
  expect(olderRequests).to.equal(1);

  older.resolve({
    children: [entry('newest'), entry('older')],
    next_cursor: null,
  });
  await flight;
  await panel.updateComplete;

  expect([...panel.querySelectorAll('li[role="listitem"]')].map((li) => li.textContent?.trim()))
    .to.deep.equal(['succeeded: objective newest', 'succeeded: objective older']);
  expect(panel.querySelector('[data-load-older-children]')).to.equal(null);
  expect(panel.querySelector('[data-roster-status]')?.textContent).to.contain(
    'Loaded 1 older child.',
  );
});

it('older-page failure keeps loaded rows and stays retryable', async () => {
  let attempts = 0;
  const panel = roster();
  panel.open(
    async () => [entry('newest')],
    async (cursor) => {
      if (cursor === null) {
        return {children: [entry('newest')], next_cursor: 'older-1'};
      }
      expect(cursor).to.equal('older-1');
      attempts += 1;
      if (attempts === 1) throw new Error('unavailable');
      return {children: [entry('older')], next_cursor: null};
    },
  );
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 1);

  await panel.loadOlderChildren();
  await panel.updateComplete;
  expect(panel.querySelector('[data-load-older-children]')?.textContent).to.contain(
    'Retry loading older children',
  );

  await panel.loadOlderChildren();
  await panel.updateComplete;
  expect(panel.querySelectorAll('li[role="listitem"]')).to.have.length(2);
  expect(panel.querySelector('[data-load-older-children]')).to.equal(null);
});

it('refresh resets the traversal and rejects a late older response', async () => {
  const older = deferredPage();
  let firstPage = 0;
  const panel = roster();
  panel.open(
    async () => [entry('fresh')],
    async (cursor) => {
      if (cursor === null) {
        firstPage += 1;
        return {children: [entry('fresh')], next_cursor: 'older-1'};
      }
      return older.promise;
    },
  );
  await waitFor(() => firstPage === 1);

  const flight = panel.loadOlderChildren();
  await panel.refresh();
  await waitFor(() => firstPage === 2);
  older.resolve({children: [entry('stale')], next_cursor: null});
  await flight;
  await panel.updateComplete;

  expect([...panel.querySelectorAll('li[role="listitem"]')].map((li) => li.textContent?.trim()))
    .to.deep.equal(['succeeded: objective fresh']);
  expect(panel.querySelector('[data-load-older-children]')?.textContent).to.contain(
    'Load older children',
  );
});

it('closing the dialog aborts in-flight pages and resets paging state', async () => {
  const older = deferredPage();
  const panel = roster();
  panel.open(
    async () => [entry('newest')],
    async (cursor) => (cursor === null
      ? {children: [entry('newest')], next_cursor: 'older-1'}
      : older.promise),
  );
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 1);

  const flight = panel.loadOlderChildren();
  const dialog = panel.querySelector<HTMLDialogElement>('dialog')!;
  dialog.close();
  // The native close event is a queued task; wait until the reset is visible.
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 0);
  older.resolve({children: [entry('stale')], next_cursor: null});
  await flight;
  await panel.updateComplete;

  expect(panel.querySelectorAll('li[role="listitem"]')).to.have.length(0);
  expect(panel.querySelector('[data-load-older-children]')).to.equal(null);
  expect(panel.querySelector('.roster-list')?.textContent).to.contain(
    'No child agents were started.',
  );
});
