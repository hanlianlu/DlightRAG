// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {DlSettingsDialog} from './settings.ts';
import './settings.ts';
import type {DlToastRegion, ToastRequestDetail} from './toast.ts';
import './toast.ts';

const originalFetch = window.fetch;

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

function buttonNamed(root: ParentNode, name: string): HTMLButtonElement | null {
  return Array.from(root.querySelectorAll<HTMLButtonElement>('button'))
    .find((button) => (button.getAttribute('aria-label') || button.textContent?.trim()) === name)
    ?? null;
}

function mount(): DlSettingsDialog {
  const shell = document.createElement('div');
  const toast = document.createElement('dl-toast-region') as DlToastRegion;
  toast.className = 'toast';
  shell.addEventListener('dl-toast-request', (event: CustomEvent<ToastRequestDetail>) => {
    if (event.detail.action) toast.showAction(event.detail.message, event.detail.action);
    else toast.show(event.detail.message, event.detail.duration);
  });
  const settings = document.createElement('dl-settings-dialog') as DlSettingsDialog;
  shell.append(toast, settings);
  document.body.appendChild(shell);
  return settings;
}

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
});

it('does not expose runtime model catalogue administration in Settings', async () => {
  const settings = mount();
  await settings.updateComplete;

  expect(settings.textContent).not.to.contain('Runtime Model Catalogue');
  expect(Boolean(settings.querySelector('dl-model-catalogue'))).to.equal(false);
  expect(customElements.get('dl-model-catalogue')).to.equal(undefined);
});

it('consumes a typed memory fact through its command and refreshes after Undo', async () => {
  const methods: string[] = [];
  window.fetch = async (_input, init) => {
    const method = init?.method || 'GET';
    methods.push(method);
    const payload = method === 'POST'
      ? {action: 'undo', outcome: 'changed', change_id: 'undo-1', memory_ids: [], body: ''}
      : {enabled: true, active_count: methods.includes('POST') ? 0 : 1};
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: {'Content-Type': 'application/json'},
    });
  };
  const settings = mount();

  settings.handleMemoryOperation({
    live: true,
    intent_id: 'intent-settings-test',
    operation: 'remember',
    outcome: 'changed',
    changeId: 'change-settings-test',
    body: 'Use concise answers',
  });
  await waitFor(() => settings.textContent?.includes('1 stored item') ?? false);

  const toast = document.querySelector('dl-toast-region')!;
  expect(toast.textContent).to.contain('Remembered: Use concise answers');
  toast.querySelector<HTMLButtonElement>('button')?.click();
  await waitFor(() => methods.length === 3
    && toast.textContent?.trim() === 'Profile Memory change undone.'
    && (settings.textContent?.includes('0 stored items') ?? false));
  expect(methods).to.deep.equal(['GET', 'POST', 'GET']);
});

it('opens fail-closed when the authoritative memory read fails', async () => {
  window.fetch = async () => new Response('unavailable', {status: 503});
  const settings = mount();

  await settings.open();
  await waitFor(() => !settings.memoryLoading);
  await settings.updateComplete;

  expect(settings.querySelector<HTMLDialogElement>('dialog[open]')).not.to.equal(null);
  // A refused read leaves no control at all: a disabled unchecked checkbox would
  // render "memory is off" for a state nobody read.
  expect(settings.querySelector('#memory-enabled-toggle')).to.equal(null);
  expect(settings.textContent).to.contain('Could not load memory settings.');
  expect(buttonNamed(settings, 'Clear memory')?.hidden).to.equal(true);
});

it('owns an explicit memory toggle mutation and its final visible state', async () => {
  const methods: string[] = [];
  window.fetch = async (_input, init) => {
    methods.push(init?.method || 'GET');
    const payload = init?.method === 'PUT'
      ? {enabled: false, active_count: null}
      : {enabled: true, active_count: 2};
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: {'Content-Type': 'application/json'},
    });
  };
  const settings = mount();
  await settings.open();
  await waitFor(() => !settings.memoryLoading);
  await settings.updateComplete;
  const toggle = settings.querySelector<HTMLInputElement>('label.dl-dialog-checkbox input')!;

  toggle.checked = false;
  toggle.dispatchEvent(new Event('change'));
  await waitFor(() => buttonNamed(settings, 'Clear memory')?.hidden === true);

  expect(methods).to.deep.equal(['GET', 'PUT']);
  expect(settings.textContent).not.to.contain('2 stored items');
});

it('restores the authoritative checkbox state after a failed toggle mutation', async () => {
  window.fetch = async (_input, init) => {
    if (init?.method === 'PUT') return new Response('unavailable', {status: 503});
    return new Response(JSON.stringify({enabled: true, active_count: 2}), {
      status: 200,
      headers: {'Content-Type': 'application/json'},
    });
  };
  const settings = mount();
  await settings.open();
  await waitFor(() => !settings.memoryLoading);
  await settings.updateComplete;
  const toggle = settings.querySelector<HTMLInputElement>('label.dl-dialog-checkbox input')!;

  toggle.checked = false;
  toggle.dispatchEvent(new Event('change'));
  const toast = settings.querySelector('dl-toast-region')!;
  await waitFor(() => !toggle.disabled
    && (toast.textContent?.includes('Could not save memory settings.') ?? false));

  expect(toggle.checked).to.equal(true);
});

it('rejects a delayed memory read after a newer toggle mutation settles', async () => {
  let resolveOldRead!: (response: Response) => void;
  const oldRead = new Promise<Response>((resolve) => { resolveOldRead = resolve; });
  const methods: string[] = [];
  let reads = 0;
  window.fetch = async (_input, init) => {
    const method = init?.method || 'GET';
    methods.push(method);
    if (method === 'GET') {
      reads += 1;
      if (reads === 1) return await oldRead;
      return new Response(JSON.stringify({enabled: true, active_count: 2}), {
        status: 200,
        headers: {'Content-Type': 'application/json'},
      });
    }
    return new Response(JSON.stringify({enabled: false, active_count: null}), {
      status: 200,
      headers: {'Content-Type': 'application/json'},
    });
  };
  const settings = mount();
  settings.memory = {enabled: true, activeCount: 2};

  settings.handleMemoryOperation({
    live: true,
    intent_id: 'stale-read-intent',
    operation: 'remember',
    outcome: 'changed',
    changeId: 'stale-read-change',
    body: 'Remember this',
  });
  await waitFor(() => reads === 1);
  await settings.open();
  await waitFor(() => !settings.memoryLoading);
  await settings.updateComplete;
  const toggle = settings.querySelector<HTMLInputElement>('label.dl-dialog-checkbox input')!;
  toggle.checked = false;
  toggle.dispatchEvent(new Event('change'));
  await waitFor(() => methods.includes('PUT') && toggle.disabled === false);

  resolveOldRead(new Response(JSON.stringify({enabled: true, active_count: 9}), {
    status: 200,
    headers: {'Content-Type': 'application/json'},
  }));
  await new Promise((resolve) => setTimeout(resolve, 0));
  await settings.updateComplete;

  expect(methods).to.deep.equal(['GET', 'GET', 'PUT']);
  expect(toggle.checked).to.equal(false);
  expect(settings.textContent).not.to.contain('9 stored items');
});

it('renders Connections independently and never paints an unread memory state', async () => {
  let releaseRead!: (response: Response) => void;
  const pendingRead = new Promise<Response>((resolve) => { releaseRead = resolve; });
  window.fetch = async (input) => {
    if (String(input).includes('/memory/settings')) return await pendingRead;
    return Response.json({revision: '1', connections: [], presets: []});
  };
  const settings = mount();

  const opened = settings.open();
  settings.personalMcpConnections = true;
  await waitFor(() => Boolean(settings.querySelector('dl-settings-connections')));

  // The MCP section is on screen while the memory projection is still unknown,
  // and Profile Memory is not painted as "off" in the meantime.
  expect(settings.querySelector<HTMLDialogElement>('#settings-dialog')!.open).to.equal(true);
  expect(settings.querySelector('dl-settings-connections')!.getBoundingClientRect().height).to.be.greaterThan(0);
  expect(settings.querySelector('#memory-enabled-toggle')).to.equal(null);
  expect(settings.textContent).to.contain('Loading memory settings');

  releaseRead(new Response(JSON.stringify({enabled: true, active_count: 3}), {
    status: 200,
    headers: {'Content-Type': 'application/json'},
  }));
  await opened;
  await waitFor(() => Boolean(settings.querySelector('#memory-enabled-toggle')));

  const toggle = settings.querySelector<HTMLInputElement>('#memory-enabled-toggle')!;
  expect(toggle.checked).to.equal(true);
  expect(settings.textContent).to.contain('3 stored items');
});

it('hides personal Connections without capability and tears the Feature down on close', async () => {
  window.fetch = async (url) => Response.json(String(url).includes('connections')
    ? {revision: '0', connections: [], presets: []}
    : {enabled: false, active_count: 0});
  const settings = mount();
  await settings.open();
  await waitFor(() => !settings.memoryLoading);
  await settings.updateComplete;
  expect(settings.querySelector('dl-settings-connections')).to.equal(null);
  settings.personalMcpConnections = true;
  await settings.updateComplete;
  expect(settings.querySelector('dl-settings-connections')).not.to.equal(null);
  settings.querySelector<HTMLDialogElement>('#settings-dialog')!.close();
  await waitFor(() => settings.querySelector('dl-settings-connections') === null);
});

it('browses paginated memories, retries a page, forgets one item and restores it with Undo', async () => {
  let forgotten = false;
  let olderAttempts = 0;
  const mutations: Array<{method: string; key: string | undefined}> = [];
  window.fetch = async (input, init) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname.endsWith('/settings')) return Response.json({enabled: true, active_count: forgotten ? 1 : 2});
    if (init?.method === 'DELETE' || init?.method === 'POST') {
      mutations.push({method: init.method, key: (init.headers as Record<string, string>)['Idempotency-Key']});
      forgotten = init.method === 'DELETE';
      return Response.json({action: forgotten ? 'forget' : 'undo', outcome: 'changed',
        change_id: forgotten ? 'forgot-1' : 'undo-1', memory_ids: ['one'], body: 'Use concise answers'});
    }
    if (url.searchParams.has('cursor')) {
      olderAttempts += 1;
      if (olderAttempts === 1) return new Response('Unavailable', {status: 503});
      return Response.json({memories: [{memory_id: 'two', kind: 'fact', body: '<img src=x> Lives in Sweden'}], next_cursor: null});
    }
    return Response.json({memories: forgotten ? [{memory_id: 'two', kind: 'fact', body: 'Lives in Sweden'}]
      : [{memory_id: 'one', kind: 'preference', body: 'Use concise answers'}], next_cursor: forgotten ? null : 'older'});
  };
  const settings = mount();
  await settings.open();
  await waitFor(() => !settings.memoryLoading);
  await settings.updateComplete;
  expect(settings.querySelector('.memory-list li')).to.equal(null);
  settings.querySelector<HTMLDetailsElement>('.memory-list')!.open = true;
  await waitFor(() => settings.querySelectorAll('.memory-list li').length === 1);
  buttonNamed(settings, 'Load more')!.click();
  await waitFor(() => Boolean(buttonNamed(settings, 'Retry')));
  buttonNamed(settings, 'Retry')!.click();
  await waitFor(() => settings.querySelectorAll('.memory-list li').length === 2);
  expect(settings.querySelector('.memory-list img')).to.equal(null);
  expect(settings.textContent).to.contain('<img src=x> Lives in Sweden');
  expect(buttonNamed(settings, 'Load more')).to.equal(null);

  buttonNamed(settings, 'Forget')!.click();
  await waitFor(() => settings.querySelectorAll('.memory-list li').length === 1 && !settings.memoryPending);
  expect(settings.querySelector('.memory-list')!.textContent).not.to.contain('Use concise answers');
  const toast = settings.querySelector('dl-toast-region')!;
  expect(toast.textContent).to.contain('Forgot: Use concise answers');
  const undo = buttonNamed(toast, 'Undo')!;
  undo.focus();
  expect(document.activeElement).to.equal(undo);
  expect(undo.closest('dialog[open]')).to.equal(settings.querySelector('#settings-dialog'));
  expect(toast.inert).to.equal(false);
  undo.click();
  await waitFor(() => settings.querySelector('.memory-list')!.textContent?.includes('Use concise answers') ?? false);
  expect(mutations.map((item) => item.method)).to.deep.equal(['DELETE', 'POST']);
  expect(mutations.every((item) => Boolean(item.key))).to.equal(true);
});

it('rejects a stale list page after memory is disabled, even when transport ignores abort', async () => {
  let releasePage!: (response: Response) => void;
  let pageStarted = false;
  window.fetch = async (input, init) => {
    if (String(input).includes('/settings')) return Response.json({enabled: init?.method !== 'PUT', active_count: 1});
    pageStarted = true;
    return new Promise<Response>((resolve) => { releasePage = resolve; });
  };
  const settings = mount();
  await settings.open();
  await waitFor(() => !settings.memoryLoading);
  await settings.updateComplete;
  settings.querySelector<HTMLDetailsElement>('.memory-list')!.open = true;
  await waitFor(() => pageStarted);
  settings.querySelector<HTMLInputElement>('#memory-enabled-toggle')!.click();
  await waitFor(() => settings.memory?.enabled === false);
  releasePage(Response.json({memories: [{memory_id: 'late', kind: 'fact', body: 'Stale private content'}], next_cursor: null}));
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(settings.memoryRecords).to.equal(null);
  expect(settings.textContent).not.to.contain('Stale private content');
});

it('does not reopen Settings or publish a late Memory read after closing it', async () => {
  let releaseRead!: (response: Response) => void;
  window.fetch = async () => new Promise<Response>((resolve) => { releaseRead = resolve; });
  const settings = mount();
  const opened = settings.open();
  await waitFor(() => settings.querySelector<HTMLDialogElement>('#settings-dialog')?.open === true);
  settings.querySelector<HTMLDialogElement>('#settings-dialog')!.close();
  await waitFor(() => !document.body.classList.contains('settings-open'));
  releaseRead(Response.json({enabled: true, active_count: 1}));
  await opened;
  expect(settings.memory).to.equal(null);
  expect(settings.querySelector<HTMLDialogElement>('#settings-dialog')!.open).to.equal(false);
});

for (const reopen of [false, true]) it(`settles in-flight Undo after close (reopen=${reopen}) without a second Undo`, async () => {
  let releaseUndo!: (response: Response) => void;
  let undoCalls = 0;
  window.fetch = async (input, init) => {
    if (String(input).includes('/settings')) return Response.json({enabled: true, active_count: 1});
    if (init?.method === 'POST') {
      undoCalls += 1;
      return new Promise<Response>((resolve) => { releaseUndo = resolve; });
    }
    return Response.json({memories: [], next_cursor: null});
  };
  const settings = mount();
  await settings.open();
  await waitFor(() => !settings.memoryLoading);
  settings.handleMemoryOperation({live: true, operation: 'forget', outcome: 'changed', changeId: 'forgot-slow', body: 'One item'});
  const localToast = settings.querySelector('dl-toast-region')!;
  await localToast.updateComplete;
  buttonNamed(localToast, 'Undo')!.click();
  await waitFor(() => undoCalls === 1);
  settings.querySelector<HTMLDialogElement>('#settings-dialog')!.close();
  await waitFor(() => settings.querySelector('dl-toast-region') === null);
  const shellToast = document.querySelector('dl-toast-region')!;
  expect(buttonNamed(shellToast, 'Undo')).to.equal(null);
  if (reopen) await settings.open();
  const visibleToast = reopen ? settings.querySelector('dl-toast-region')! : shellToast;
  releaseUndo(Response.json({action: 'undo', outcome: 'changed', change_id: 'undo-slow', memory_ids: [], body: ''}));
  await waitFor(() => visibleToast.textContent?.trim() === 'Profile Memory change undone.');
  expect(buttonNamed(shellToast, 'Undo')).to.equal(null);
  expect(buttonNamed(visibleToast, 'Undo')).to.equal(null);
  expect(undoCalls).to.equal(1);
});
