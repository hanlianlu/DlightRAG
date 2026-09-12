// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import './run-dialogs.ts';
import type {ChildObservation} from '../api/conversations.ts';
import type {ChildRosterEntry, DlChildrenRoster} from './run-dialogs.ts';

function entry(id: string, status = 'succeeded'): ChildRosterEntry {
  return {childSessionId: id, status, objective: `objective ${id}`};
}

function deferredPage() {
  let resolve!: (page: {children: ChildRosterEntry[]; nextCursor: string | null}) => void;
  const promise = new Promise<{children: ChildRosterEntry[]; nextCursor: string | null}>(
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
        return {children: [entry('newest')], nextCursor: 'older-1'};
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
    nextCursor: null,
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
        return {children: [entry('newest')], nextCursor: 'older-1'};
      }
      expect(cursor).to.equal('older-1');
      attempts += 1;
      if (attempts === 1) throw new Error('unavailable');
      return {children: [entry('older')], nextCursor: null};
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
        return {children: [entry('fresh')], nextCursor: 'older-1'};
      }
      return older.promise;
    },
  );
  await waitFor(() => firstPage === 1);

  const flight = panel.loadOlderChildren();
  await panel.refresh();
  await waitFor(() => firstPage === 2);
  older.resolve({children: [entry('stale')], nextCursor: null});
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
      ? {children: [entry('newest')], nextCursor: 'older-1'}
      : older.promise),
  );
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 1);

  const flight = panel.loadOlderChildren();
  const dialog = panel.querySelector<HTMLDialogElement>('dialog')!;
  dialog.close();
  // The native close event is a queued task; wait until the reset is visible.
  await waitFor(() => panel.querySelectorAll('li[role="listitem"]').length === 0);
  older.resolve({children: [entry('stale')], nextCursor: null});
  await flight;
  await panel.updateComplete;

  expect(panel.querySelectorAll('li[role="listitem"]')).to.have.length(0);
  expect(panel.querySelector('[data-load-older-children]')).to.equal(null);
  expect(panel.querySelector('.roster-list')?.textContent).to.contain(
    'No child agents were started.',
  );
});

function observation(status = 'running'): ChildObservation {
  return {
    runId: 'run-1',
    child: {
      childSessionId: 'newest',
      status,
      objective: 'objective newest',
      modelRole: 'query',
      usage: null,
      operationId: 'op-1',
      operationSequence: 1,
      operationStatus: status,
      cancellationOrigin: status === 'cancelled' ? 'user' : null,
      summary: 'working',
      resultHandles: ['ev-1'],
    },
    transcript: [{role: 'user', content: 'inspect', toolCalls: [], toolCallId: '', name: '', isError: false}],
    controls: [{
      controlSequence: 3, kind: 'steer', content: 'focus', origin: 'user',
      consumed: false, consumedAt: null, createdAt: null, operationId: 'op-1',
    }],
    questions: [{
      requestId: 'req-1', question: 'Which source?', status: 'pending',
      reply: null, replyOrigin: null, expiresAt: null, createdAt: null,
    }],
    result: {status, summary: 'working', handles: ['ev-1'], operationId: 'op-1'},
  };
}

it('selecting a child shows lineage and posts a queued steer', async () => {
  const steered: string[] = [];
  const panel = roster();
  panel.open(
    async () => [entry('newest', 'running')],
    async () => ({children: [entry('newest', 'running')], nextCursor: null}),
    {
      runId: 'run-1',
      observe: async () => observation('running'),
      control: async (_child, action, content) => {
        steered.push(`${action}:${content}`);
        return {
          runId: 'run-1', childSessionId: 'newest', action, outcome: 'queued',
          operationId: 'op-1', operationSequence: 1, controlSequence: 4, consumedAt: null,
          requestId: null,
        };
      },
    },
  );
  await waitFor(() => Boolean(panel.querySelector('[data-child-session="newest"]')));
  panel.querySelector<HTMLButtonElement>('[data-child-session="newest"]')!.click();
  await waitFor(() => Boolean(panel.querySelector('[name="instruction"]')));

  expect(panel.textContent).to.contain('Queued');
  expect(panel.textContent).to.contain('Which source?');
  expect(panel.textContent).to.contain('ev-1');

  const instruction = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  instruction.value = 'focus on dates';
  instruction.closest('form')!.dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));
  await waitFor(() => steered.length === 1);
  expect(steered[0]).to.equal('steer:focus on dates');
  await waitFor(() => (panel.textContent || '').includes('has not necessarily followed'));
});

it('rejected terminal steer stays explicit and does not look like success', async () => {
  const panel = roster();
  panel.open(
    async () => [entry('newest', 'running')],
    async () => ({children: [entry('newest', 'running')], nextCursor: null}),
    {
      runId: 'run-1',
      observe: async () => observation('running'),
      control: async () => {
        const error = new Error('terminal_child') as Error & {status: number; outcome: string};
        error.status = 409;
        error.outcome = 'terminal_child';
        throw error;
      },
    },
  );
  await waitFor(() => Boolean(panel.querySelector('[data-child-session="newest"]')));
  panel.querySelector<HTMLButtonElement>('[data-child-session="newest"]')!.click();
  await waitFor(() => Boolean(panel.querySelector('[name="instruction"]')));
  const instruction = panel.querySelector<HTMLTextAreaElement>('[name="instruction"]')!;
  instruction.value = 'too late';
  instruction.closest('form')!.dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));
  await waitFor(() => (panel.textContent || '').includes('already terminal'));
});
