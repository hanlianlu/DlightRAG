// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import './app.ts';
import type {DlApp} from './app.ts';

const bootstrap = {
  contract_version: 1,
  workspaces: [
    {workspace: 'default', display_name: 'Default', embedding_model: 'embed-test'},
  ],
  primary_workspace: 'default',
  active_workspaces: ['default'],
  answer_attachments: {
    count_limit: 6,
    image_max_bytes: 1024,
    document_max_bytes: 2048,
    extensions: ['md', 'pdf'],
    image_capability: 'supported',
    image_limit: 3,
    accept: 'image/*,.md,.pdf',
  },
  active_html_preview_enabled: true,
} as const;

const originalFetch = window.fetch;

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {'Content-Type': 'application/json'},
  });
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

it('renders the application shell from the typed bootstrap before resolving ready', async () => {
  window.fetch = async () => response(bootstrap);
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);

  const loaded = await app.ready;

  expect(loaded).to.deep.equal(bootstrap);
  const shell = app.querySelector<HTMLElement>('#app');
  expect(shell?.inert).to.equal(false);
  expect(shell?.dataset.attachmentCountLimit).to.equal('6');
  expect(shell?.dataset.attachmentDocumentMaxBytes).to.equal('2048');
  expect(app.querySelector('workspace-scope')?.getAttribute('data-primary')).to.equal('default');
  expect(app.querySelector<HTMLInputElement>('#attachment-input')?.accept).to.equal(
    'image/*,.md,.pdf',
  );
});

it('fails closed and resolves the same ready promise after an explicit retry', async () => {
  let attempts = 0;
  window.fetch = async () => {
    attempts += 1;
    return attempts === 1 ? response({detail: 'down'}, 503) : response(bootstrap);
  };
  const app = document.createElement('dl-app') as DlApp;
  document.body.appendChild(app);
  await waitFor(() => app.bootState === 'error');

  expect(app.querySelector<HTMLElement>('#app')?.inert).to.equal(true);
  const retry = app.querySelector<HTMLButtonElement>('.bootstrap-status button');
  expect(retry).not.to.equal(null);
  retry?.click();

  await app.ready;
  expect(attempts).to.equal(2);
  expect(app.bootState).to.equal('ready');
});
