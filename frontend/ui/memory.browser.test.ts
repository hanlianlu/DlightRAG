// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {
  prepareMemorySettingsPanel,
  setupChatMemoryOperationAdapter,
  setupMemorySettings,
} from './memory.ts';

const originalFetch = window.fetch;

function mountMemorySettings(): void {
  document.body.innerHTML = `
    <input id="memory-enabled-toggle" type="checkbox" />
    <p id="memory-active-count"></p>
    <button id="memory-clear-btn" type="button">Clear</button>`;
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
});

it('adapts typed Chat memory facts into Toast and authoritative Settings refresh', async () => {
  const methods: string[] = [];
  window.fetch = async (_input, init) => {
    const method = init?.method || 'GET';
    methods.push(method);
    const payload = method === 'POST'
      ? {
          action: 'undo', outcome: 'changed', change_id: 'undo-1', memory_ids: [], body: '',
        }
      : {enabled: true, active_count: methods.filter((item) => item === 'POST').length ? 0 : 1};
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: {'Content-Type': 'application/json'},
    });
  };
  mountMemorySettings();
  const chat = document.createElement('dl-chat-feature');
  const toast = document.createElement('div');
  toast.id = 'toast';
  document.body.append(chat, toast);
  setupChatMemoryOperationAdapter();

  chat.dispatchEvent(new CustomEvent('dl-chat-memory-operation', {
    detail: {
      live: true,
      intent_id: 'intent-adapter-test',
      operation: 'remember',
      outcome: 'changed',
      change_id: 'change-adapter-test',
      body: 'Use concise answers',
    },
  }));
  await waitFor(() => document.getElementById('memory-active-count')?.textContent === '1 stored item');

  expect(methods).to.deep.equal(['GET']);
  expect(toast.textContent).to.contain('Remembered: Use concise answers');
  expect(toast.querySelector('button')?.textContent).to.equal('Undo');
  expect(document.getElementById('memory-active-count')?.textContent).to.equal('1 stored item');

  toast.querySelector<HTMLButtonElement>('button')?.click();
  await waitFor(() =>
    methods.length === 3
    && toast.textContent === 'Profile Memory change undone.'
    && document.getElementById('memory-active-count')?.textContent === '0 stored items',
  );
  expect(methods).to.deep.equal(['GET', 'POST', 'GET']);
});

it('never enables mutation controls after a failed authoritative read', async () => {
  const methods: string[] = [];
  window.fetch = async (_input, init) => {
    methods.push(init?.method || 'GET');
    return new Response('unavailable', {status: 503});
  };
  mountMemorySettings();
  setupMemorySettings(() => {});

  expect(await prepareMemorySettingsPanel()).to.equal(false);

  expect(methods).to.deep.equal(['GET']);
  expect((document.getElementById('memory-enabled-toggle') as HTMLInputElement).disabled)
    .to.equal(true);
  expect((document.getElementById('memory-clear-btn') as HTMLButtonElement).hidden).to.equal(true);
});

it('saves only an explicit toggle and hides data controls when deactivated', async () => {
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
  mountMemorySettings();
  setupMemorySettings(() => {});
  expect(await prepareMemorySettingsPanel()).to.equal(true);

  const toggle = document.getElementById('memory-enabled-toggle') as HTMLInputElement;
  toggle.checked = false;
  toggle.dispatchEvent(new Event('change'));
  await waitFor(() =>
    (document.getElementById('memory-clear-btn') as HTMLButtonElement).hidden === true,
  );

  expect(methods).to.deep.equal(['GET', 'PUT']);
  expect((document.getElementById('memory-clear-btn') as HTMLButtonElement).hidden).to.equal(true);
  expect(document.getElementById('memory-active-count')?.hidden).to.equal(true);
});
