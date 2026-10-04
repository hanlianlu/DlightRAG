// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {parseMemoryOperationEvent} from '../api/memory.ts';
import {buttonNamed, waitFor} from '../testing/dom.ts';
import {memoryPage, memorySettings, mountSettings, openSettings, wire} from '../testing/settings.ts';
import type {DlSettingsDialog} from './settings.ts';
import type {DlToastRegion} from './toast.ts';

const originalFetch = window.fetch;

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
});

/** The Memory switch, once the page has read the owner's setting. */
function memorySwitch(settings: DlSettingsDialog): HTMLButtonElement {
  return settings.querySelector<HTMLButtonElement>('#memory-enabled-toggle')!;
}

/** The status the navigation shows for Profile Memory: its stored count, or nothing. */
function memoryStatus(settings: DlSettingsDialog): string {
  return settings.querySelector('nav .dl-nav-item[data-section="memory"] .dl-nav-item-status')!
    .textContent!.trim();
}

/** The remembered texts the list shows, in order. */
function remembered(settings: DlSettingsDialog): string[] {
  return [...settings.querySelectorAll('dl-settings-memory li p')].map((body) => body.textContent!.trim());
}

/** The dialog's own notice region, where Settings shows a notice while it is open. */
function notice(settings: DlSettingsDialog): DlToastRegion {
  return settings.querySelector('dl-toast-region')!;
}

function liveFact(changeId: string, body: string, operation: 'remember' | 'forget' = 'remember'): ReturnType<typeof parseMemoryOperationEvent> {
  // The server's own field names, through the same parser the stream uses.
  return parseMemoryOperationEvent({
    live: true,
    intent_id: `intent-${changeId}`,
    operation,
    outcome: 'changed',
    change_id: changeId,
    body,
  });
}

/** Open Settings on Profile Memory and wait until its page has an answer to show, whichever it is. */
async function openMemory(settings: DlSettingsDialog): Promise<HTMLDialogElement> {
  const dialog = await openSettings(settings, 'memory');
  const page = settings.querySelector('dl-settings-memory')!;
  await waitFor(() => page.textContent!.trim() !== '' && !page.textContent!.includes('Loading memory settings'));
  return dialog;
}

it('paints the switch only once the owner\'s setting has been read, never an unread "off"', async () => {
  let release!: (response: Response) => void;
  const pending = new Promise<Response>((resolve) => { release = resolve; });
  window.fetch = wire({
    'GET /web/api/memory/settings': () => pending,
    'GET /web/api/memory': () => memoryPage([{id: 'one', body: 'Use concise answers'}]),
  }).fetch;
  const {settings} = mountSettings();
  await openSettings(settings, 'memory');

  expect(memorySwitch(settings)).to.equal(null);
  expect(settings.querySelector('dl-settings-memory')!.textContent).to.contain('Loading memory settings');

  release(memorySettings(true, 1));
  await waitFor(() => Boolean(memorySwitch(settings)));
  expect(memorySwitch(settings).getAttribute('aria-checked')).to.equal('true');
  expect(memorySwitch(settings).disabled).to.equal(false);
  expect(memoryStatus(settings)).to.equal('1');
});

it('opens fail-closed when the authoritative memory read fails', async () => {
  window.fetch = wire({'GET /web/api/memory/settings': () => new Response('unavailable', {status: 503})}).fetch;
  const {settings} = mountSettings();
  const dialog = await openMemory(settings);

  expect(dialog.open).to.equal(true);
  // A refused read leaves no control at all: an unchecked switch would render "memory is off"
  // for a state nobody read.
  expect(memorySwitch(settings)).to.equal(null);
  expect(settings.querySelector('dl-settings-memory')!.textContent).to.contain('Could not load memory settings.');
  expect(settings.querySelector('#memory-clear-btn')).to.equal(null);
  expect(memoryStatus(settings)).to.equal('');
});

it('shows the stored memories with their kind and their text', async () => {
  const long = 'Reports go to the investment committee and use tables and short bullets, never long paragraphs.';
  window.fetch = wire({
    'GET /web/api/memory/settings': () => memorySettings(true, 2),
    'GET /web/api/memory': () => memoryPage([
      {id: 'one', kind: 'preference', body: long},
      {id: 'two', kind: 'fact', body: '<img src=x> Lives in Sweden'},
    ]),
  }).fetch;
  const {settings} = mountSettings();
  await openMemory(settings);
  await waitFor(() => remembered(settings).length === 2);

  const list = settings.querySelector('dl-settings-memory section')!;
  expect(list.getAttribute('aria-labelledby')).to.equal('memory-list-title');
  expect(document.getElementById('memory-list-title')!.textContent).to.equal('Stored memories');
  expect(list.querySelector('[class*=badge]')!.textContent).to.equal('2');
  const rows = [...list.querySelectorAll('li')];
  expect(rows.map((item) => item.firstElementChild!.textContent)).to.deep.equal(['Preference', 'Fact']);
  expect(remembered(settings)[0]).to.equal(long);
  // The text of a memory is text, never markup.
  expect(list.querySelector('img')).to.equal(null);
  expect(remembered(settings)[1]).to.equal('<img src=x> Lives in Sweden');
});

it('shows a switch that is off with a note in place of the list and the clear action', async () => {
  window.fetch = wire({'GET /web/api/memory/settings': () => memorySettings(false)}).fetch;
  const {settings} = mountSettings();
  await openMemory(settings);

  expect(memorySwitch(settings).getAttribute('aria-checked')).to.equal('false');
  expect(settings.querySelector('dl-settings-memory')!.textContent)
    .to.contain('Off: stored memories are kept, but the agent neither reads nor writes them');
  expect(settings.querySelector('dl-settings-memory')!.textContent)
    .to.contain('Turn it on to view, forget or clear the stored memories.');
  expect(settings.querySelector('dl-settings-memory section')).to.equal(null);
  expect(settings.querySelector('#memory-clear-btn')).to.equal(null);
});

it('turns the switch with one PUT, shows its final state, and lists the memories once it is on', async () => {
  let enabled = false;
  const backend = wire({
    'GET /web/api/memory/settings': () => memorySettings(enabled),
    'PUT /web/api/memory/settings': (request) => {
      enabled = (request.body as {enabled: boolean}).enabled;
      return memorySettings(enabled, 1);
    },
    'GET /web/api/memory': () => memoryPage([{id: 'one', body: 'Use concise answers'}]),
  });
  window.fetch = backend.fetch;
  const {settings} = mountSettings();
  await openMemory(settings);
  await waitFor(() => Boolean(memorySwitch(settings)));
  expect(settings.querySelector('dl-settings-memory section')).to.equal(null);

  memorySwitch(settings).click();
  await waitFor(() => memorySwitch(settings).getAttribute('aria-checked') === 'true' && !memorySwitch(settings).disabled);

  expect(backend.requests.filter((request) => request.method === 'PUT').map((request) => request.body))
    .to.deep.equal([{enabled: true}]);
  await waitFor(() => remembered(settings).length === 1);
  expect(settings.querySelector('#memory-clear-btn')).not.to.equal(null);
  expect(memoryStatus(settings)).to.equal('1');

  // The whole card is the switch's label, so a tap anywhere on it turns it back off.
  settings.querySelector<HTMLElement>('label [id="memory-enabled-toggle-label"]')!.click();
  await waitFor(() => memorySwitch(settings).getAttribute('aria-checked') === 'false' && !memorySwitch(settings).disabled);
  expect(settings.querySelector('#memory-clear-btn')).to.equal(null);
  expect(memoryStatus(settings)).to.equal('');
});

it('keeps the authoritative switch state after a failed mutation, and says so', async () => {
  window.fetch = wire({
    'GET /web/api/memory/settings': () => memorySettings(true, 2),
    'PUT /web/api/memory/settings': () => new Response('unavailable', {status: 503}),
    'GET /web/api/memory': () => memoryPage([]),
  }).fetch;
  const {settings} = mountSettings();
  await openMemory(settings);
  await waitFor(() => Boolean(memorySwitch(settings)));

  memorySwitch(settings).click();
  await waitFor(() => !memorySwitch(settings).disabled
    && (notice(settings).textContent?.includes('Could not save memory settings.') ?? false));

  expect(memorySwitch(settings).getAttribute('aria-checked')).to.equal('true');
});

it('rejects a delayed memory read after a newer toggle mutation settles', async () => {
  let resolveOldRead!: (response: Response) => void;
  const oldRead = new Promise<Response>((resolve) => { resolveOldRead = resolve; });
  let reads = 0;
  const backend = wire({
    'GET /web/api/memory/settings': () => {
      reads += 1;
      return reads === 1 ? oldRead : memorySettings(true, 2);
    },
    'PUT /web/api/memory/settings': () => memorySettings(false),
    'GET /web/api/memory': () => memoryPage([]),
  });
  window.fetch = backend.fetch;
  const {settings} = mountSettings();
  // The first read is still on its way when a live change makes the page read again.
  await openSettings(settings, 'memory');
  await waitFor(() => reads === 1);
  settings.handleMemoryOperation(liveFact('stale-read', 'Remember this')!);
  await waitFor(() => Boolean(memorySwitch(settings)));

  memorySwitch(settings).click();
  await waitFor(() => backend.requests.some((request) => request.method === 'PUT') && !memorySwitch(settings).disabled);
  resolveOldRead(memorySettings(true, 9));
  await new Promise((resolve) => setTimeout(resolve, 0));
  await settings.updateComplete;

  expect(reads).to.equal(2);
  expect(memorySwitch(settings).getAttribute('aria-checked')).to.equal('false');
  expect(memoryStatus(settings)).to.equal('');
});

it('browses paginated memories, retries a page, forgets one item, and restores it with Undo', async () => {
  let forgotten = false;
  let olderAttempts = 0;
  const mutations: Array<{method: string; key: string | null}> = [];
  window.fetch = wire({
    'GET /web/api/memory/settings': () => memorySettings(true, forgotten ? 1 : 2),
    'GET /web/api/memory': (request) => {
      if (new URLSearchParams(request.search).has('cursor')) {
        olderAttempts += 1;
        if (olderAttempts === 1) return new Response('Unavailable', {status: 503});
        return memoryPage([{id: 'two', kind: 'fact', body: '<img src=x> Lives in Sweden'}]);
      }
      return forgotten
        ? memoryPage([{id: 'two', kind: 'fact', body: 'Lives in Sweden'}])
        : memoryPage([{id: 'one', body: 'Use concise answers'}], 'older');
    },
    'DELETE /web/api/memory/one': (request) => {
      mutations.push({method: request.method, key: request.headers.get('Idempotency-Key')});
      forgotten = true;
      return Response.json({action: 'forget', outcome: 'changed', change_id: 'forgot-1', memory_ids: ['one'],
        body: 'Use concise answers'});
    },
    'POST /web/api/memory/changes/forgot-1/undo': (request) => {
      mutations.push({method: request.method, key: request.headers.get('Idempotency-Key')});
      forgotten = false;
      return Response.json({action: 'undo', outcome: 'changed', change_id: 'undo-1', memory_ids: ['one'],
        body: 'Use concise answers'});
    },
  }).fetch;
  const {settings} = mountSettings();
  await openMemory(settings);
  await waitFor(() => remembered(settings).length === 1);

  settings.querySelector<HTMLButtonElement>('[data-load-older="memories"]')!.click();
  await waitFor(() => Boolean(buttonNamed(settings, 'Retry')));
  buttonNamed(settings, 'Retry')!.click();
  await waitFor(() => remembered(settings).length === 2);
  expect(settings.querySelector('dl-settings-memory img')).to.equal(null);
  expect(remembered(settings)[1]).to.equal('<img src=x> Lives in Sweden');
  expect(settings.querySelector('[data-load-older="memories"]')).to.equal(null);

  const forget = settings.querySelector<HTMLElement>('dl-settings-memory li dl-icon-button')!;
  expect(forget.getAttribute('aria-label')).to.equal('Forget this memory');
  forget.click();
  await waitFor(() => remembered(settings).length === 1 && remembered(settings)[0] === 'Lives in Sweden');
  const toast = notice(settings);
  expect(toast.textContent).to.contain('Forgot: Use concise answers');
  const undo = buttonNamed(toast, 'Undo')!;
  // The row the reader was on is gone, so focus lands on the Undo that appeared in its place.
  await waitFor(() => document.activeElement === undo);
  expect(undo.closest('dialog[open]')).to.equal(settings.querySelector('#settings-dialog'));
  expect(toast.inert).to.equal(false);

  undo.click();
  await waitFor(() => remembered(settings).includes('Use concise answers'));
  expect(mutations.map((item) => item.method)).to.deep.equal(['DELETE', 'POST']);
  expect(mutations.every((item) => Boolean(item.key))).to.equal(true);
});

it('retries a failed first memory page from its own note', async () => {
  let firstPages = 0;
  window.fetch = wire({
    'GET /web/api/memory/settings': () => memorySettings(true, 1),
    'GET /web/api/memory': () => {
      firstPages += 1;
      if (firstPages === 1) return new Response('Unavailable', {status: 503});
      return memoryPage([{id: 'one', kind: 'fact', body: 'Lives in Sweden'}]);
    },
  }).fetch;
  const {settings} = mountSettings();
  await openMemory(settings);

  const page = settings.querySelector('dl-settings-memory')!;
  await waitFor(() => page.textContent?.includes('Could not load memories.') ?? false);
  expect(settings.querySelector('[data-load-older="memories"]')).to.equal(null);
  buttonNamed(page, 'Retry')!.click();

  await waitFor(() => remembered(settings).length === 1);
  expect(firstPages).to.equal(2);
  expect(buttonNamed(page, 'Retry')).to.equal(null);
});

it('says so when there is nothing stored', async () => {
  window.fetch = wire({
    'GET /web/api/memory/settings': () => memorySettings(true, 0),
    'GET /web/api/memory': () => memoryPage([]),
  }).fetch;
  const {settings} = mountSettings();
  await openMemory(settings);

  await waitFor(() => settings.querySelector('dl-settings-memory')!.textContent!.includes('No stored memories.'));
  expect(remembered(settings)).to.deep.equal([]);
});

it('rejects a stale list page after memory is disabled, even when transport ignores abort', async () => {
  let releasePage!: (response: Response) => void;
  let pageStarted = false;
  window.fetch = wire({
    'GET /web/api/memory/settings': () => memorySettings(true, 1),
    'PUT /web/api/memory/settings': () => memorySettings(false),
    'GET /web/api/memory': () => {
      pageStarted = true;
      return new Promise<Response>((resolve) => { releasePage = resolve; });
    },
  }).fetch;
  const {settings} = mountSettings();
  await openMemory(settings);
  await waitFor(() => pageStarted);

  memorySwitch(settings).click();
  await waitFor(() => memorySwitch(settings).getAttribute('aria-checked') === 'false');
  releasePage(memoryPage([{id: 'late', kind: 'fact', body: 'Stale private content'}]));
  await new Promise((resolve) => setTimeout(resolve, 0));
  await settings.updateComplete;

  expect(settings.textContent).not.to.contain('Stale private content');
});

it('asks before clearing every memory, sends one clear, and lists what is left', async () => {
  let cleared = false;
  const backend = wire({
    'GET /web/api/memory/settings': () => memorySettings(true, cleared ? 0 : 1),
    'GET /web/api/memory': () => memoryPage(cleared ? [] : [{id: 'one', body: 'Use concise answers'}]),
    'POST /web/api/memory/clear': () => {
      cleared = true;
      return new Response(null, {status: 204});
    },
  });
  window.fetch = backend.fetch;
  const {settings} = mountSettings();
  await openMemory(settings);
  await waitFor(() => remembered(settings).length === 1);
  const clear = settings.querySelector<HTMLButtonElement>('#memory-clear-btn')!;
  expect(clear.getAttribute('aria-label')).to.equal('Clear all memories');
  const dialog = settings.querySelector<HTMLDialogElement>('#clear-memory-dialog')!;

  clear.click();
  await waitFor(() => dialog.open);
  expect(dialog.textContent).to.contain('Clear Profile memory?');
  dialog.querySelector<HTMLButtonElement>('button[value=cancel]')!.click();
  await waitFor(() => !dialog.open && document.activeElement === clear);
  expect(backend.requests.some((request) => request.path.endsWith('/clear'))).to.equal(false);

  clear.click();
  await waitFor(() => dialog.open);
  dialog.querySelector<HTMLButtonElement>('button[value=clear]')!.click();
  await waitFor(() => remembered(settings).length === 0 && notice(settings).textContent!.includes('Memory cleared.'));
  expect(backend.requests.filter((request) => request.path.endsWith('/clear'))).to.have.length(1);
});

it('turns a live fact into a notice with Undo, and refreshes the page after Undo', async () => {
  let undone = false;
  const backend = wire({
    'GET /web/api/memory/settings': () => memorySettings(true, undone ? 0 : 1),
    'GET /web/api/memory': () => memoryPage(undone ? [] : [{id: 'one', body: 'Use concise answers'}]),
    'POST /web/api/memory/changes/change-settings-test/undo': () => {
      undone = true;
      return Response.json({action: 'undo', outcome: 'changed', change_id: 'undo-1', memory_ids: [], body: ''});
    },
  });
  window.fetch = backend.fetch;
  const {settings} = mountSettings();
  await openMemory(settings);
  await waitFor(() => memoryStatus(settings) === '1');

  settings.handleMemoryOperation(liveFact('change-settings-test', 'Use concise answers')!);
  const toast = notice(settings);
  await waitFor(() => toast.textContent!.includes('Remembered: Use concise answers'));
  buttonNamed(toast, 'Undo')!.click();

  await waitFor(() => toast.textContent!.trim() === 'Profile Memory change undone.' && memoryStatus(settings) === '0');
  expect(backend.requests.filter((request) => request.method === 'POST')).to.have.length(1);
});

it('consumes a live fact once, however many times Chat reports it', async () => {
  window.fetch = wire({
    'GET /web/api/memory/settings': () => memorySettings(true, 1),
    'GET /web/api/memory': () => memoryPage([]),
  }).fetch;
  const {settings} = mountSettings();
  await openMemory(settings);
  const toast = notice(settings);
  const say = (message: string): void => {
    settings.querySelector('dl-settings-memory')!.dispatchEvent(
      new CustomEvent('dl-toast-request', {detail: {message}, bubbles: true, composed: true}),
    );
  };

  const fact = liveFact('twice', 'Use concise answers')!;
  settings.handleMemoryOperation(fact);
  await waitFor(() => toast.textContent!.includes('Remembered: Use concise answers'));
  // Something else takes the region; a repeat of the same change must not take it back.
  say('Something else');
  await toast.updateComplete;
  expect(toast.textContent).to.contain('Something else');

  settings.handleMemoryOperation(fact);
  settings.handleMemoryOperation({...fact, live: false, changeId: 'replayed'});
  await toast.updateComplete;
  expect(toast.textContent).to.contain('Something else');
});

it('keeps a live fact\'s Undo while Settings is closed, without reading anything', async () => {
  let undoCalls = 0;
  const backend = wire({
    'POST /web/api/memory/changes/closed-change/undo': () => {
      undoCalls += 1;
      return Response.json({action: 'undo', outcome: 'changed', change_id: 'undo-1', memory_ids: [], body: ''});
    },
  });
  window.fetch = backend.fetch;
  const {settings, toast} = mountSettings();
  await settings.updateComplete;

  settings.handleMemoryOperation(liveFact('closed-change', 'Use concise answers')!);
  await toast.updateComplete;
  expect(toast.textContent).to.contain('Remembered: Use concise answers');
  buttonNamed(toast, 'Undo')!.click();

  await waitFor(() => toast.textContent!.trim() === 'Profile Memory change undone.');
  expect(undoCalls).to.equal(1);
  // Nothing is open, so nothing reads: the page holds no state while its dialog is closed.
  expect(backend.requests.map((request) => `${request.method} ${request.path}`))
    .to.deep.equal(['POST /web/api/memory/changes/closed-change/undo']);
});

it('does not reopen Settings or publish a late Memory read after closing it', async () => {
  let releaseRead!: (response: Response) => void;
  window.fetch = wire({
    'GET /web/api/memory/settings': () => new Promise<Response>((resolve) => { releaseRead = resolve; }),
  }).fetch;
  const {settings} = mountSettings();
  const opened = settings.open();
  await waitFor(() => settings.querySelector<HTMLDialogElement>('#settings-dialog')?.open === true);
  await waitFor(() => typeof releaseRead === 'function');
  settings.querySelector<HTMLDialogElement>('#settings-dialog')!.close();
  await waitFor(() => !document.body.classList.contains('settings-open'));
  releaseRead(memorySettings(true, 1));
  await opened;
  await new Promise((resolve) => setTimeout(resolve, 0));

  expect(memorySwitch(settings)).to.equal(null);
  expect(memoryStatus(settings)).to.equal('');
  expect(settings.querySelector<HTMLDialogElement>('#settings-dialog')!.open).to.equal(false);
});

for (const reopen of [false, true]) it(`settles an in-flight Undo after close (reopen=${reopen}) without a second Undo`, async () => {
  let releaseUndo!: (response: Response) => void;
  let undoCalls = 0;
  window.fetch = wire({
    'GET /web/api/memory/settings': () => memorySettings(true, 1),
    'GET /web/api/memory': () => memoryPage([]),
    'POST /web/api/memory/changes/forgot-slow/undo': () => {
      undoCalls += 1;
      return new Promise<Response>((resolve) => { releaseUndo = resolve; });
    },
  }).fetch;
  const {settings, toast: shellToast} = mountSettings();
  const dialog = await openMemory(settings);
  settings.handleMemoryOperation(liveFact('forgot-slow', 'One item', 'forget')!);
  const localToast = notice(settings);
  await localToast.updateComplete;
  buttonNamed(localToast, 'Undo')!.click();
  await waitFor(() => undoCalls === 1);
  dialog.close();
  await waitFor(() => settings.querySelector('dl-toast-region') === null);
  expect(buttonNamed(shellToast, 'Undo')).to.equal(null);
  if (reopen) await settings.open();
  const visibleToast = reopen ? notice(settings) : shellToast;
  releaseUndo(Response.json({action: 'undo', outcome: 'changed', change_id: 'undo-slow', memory_ids: [], body: ''}));
  await waitFor(() => visibleToast.textContent?.trim() === 'Profile Memory change undone.');

  expect(buttonNamed(shellToast, 'Undo')).to.equal(null);
  expect(buttonNamed(visibleToast, 'Undo')).to.equal(null);
  expect(undoCalls).to.equal(1);
});
