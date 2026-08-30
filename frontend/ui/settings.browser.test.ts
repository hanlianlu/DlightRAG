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
    change_id: 'change-settings-test',
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

  expect(settings.querySelector<HTMLDialogElement>('dialog[open]')).not.to.equal(null);
  expect(settings.querySelector<HTMLInputElement>('label.ui-dialog-checkbox input')?.disabled)
    .to.equal(true);
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
  const toggle = settings.querySelector<HTMLInputElement>('label.ui-dialog-checkbox input')!;

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
  const toggle = settings.querySelector<HTMLInputElement>('label.ui-dialog-checkbox input')!;

  toggle.checked = false;
  toggle.dispatchEvent(new Event('change'));
  const toast = document.querySelector('dl-toast-region')!;
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

  settings.handleMemoryOperation({
    live: true,
    intent_id: 'stale-read-intent',
    operation: 'remember',
    outcome: 'changed',
    change_id: 'stale-read-change',
    body: 'Remember this',
  });
  await waitFor(() => reads === 1);
  await settings.open();
  const toggle = settings.querySelector<HTMLInputElement>('label.ui-dialog-checkbox input')!;
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
