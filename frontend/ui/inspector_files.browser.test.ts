// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {workspaceStore} from '../stores/workspaceStore.ts';
import {ingestStore} from '../stores/ingestStore.ts';
import './inspector_files.ts';
import type {DlInspectorFiles} from './inspector_files.ts';

const originalFetch = window.fetch;

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

beforeEach(() => {
  workspaceStore.init(
    [{workspace: 'default', displayName: 'Default', embeddingModel: 'embed'}],
    ['default'],
    'default',
  );
  ingestStore.resetToPrimary();
});

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
});

it('clears upload chrome when the panel is paused for close or workspace change', () => {
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.uploading = true;

  panel.pause();

  expect(panel.uploading).to.equal(false);
});

it('renders typed file data as escaped Lit text without an HTML fragment sink', async () => {
  window.fetch = async () => new Response(JSON.stringify({
    workspace: 'default',
    files: [{file_name: '<img src=x>', file_path: '/docs/report.pdf'}],
    ingest: {
      busy: false,
      message: '',
      progress_percent: null,
      current_batch: null,
      total_batches: null,
      documents: null,
      pending_enqueues: 0,
    },
  }), {status: 200, headers: {'Content-Type': 'application/json'}});
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);

  expect(panel.querySelector('.file-name')?.textContent).to.equal('<img src=x>');
  expect(panel.querySelector('.file-name img')).to.equal(null);
  expect(panel.querySelector<HTMLButtonElement>('.file-delete')?.ariaLabel).to.equal(
    'Delete <img src=x>',
  );
});
