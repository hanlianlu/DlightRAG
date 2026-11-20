// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {setLanguagePreference} from '../i18n/locale.ts';
import {productionHandles} from '../stores/app-handles.ts';
import {waitFor} from '../testing/dom.ts';
import {DEFAULT_CHANGES} from '../testing/workspaces.ts';
import './inspector-files.ts';
import type {DlInspectorFiles} from './inspector-files.ts';

const {workspaces: workspaceStore, ingest: ingestStore} = productionHandles();

const originalFetch = window.fetch;

function receipt(fileCount: number) {
  return {
    run_id: 'run-accepted',
    run_kind: 'corpus_mutation',
    lane: 'corpus_mutation',
    status: 'running',
    status_url: '/web/api/corpus-runs/run-accepted',
    events_url: '/web/api/corpus-runs/run-accepted/events',
    cancel_url: '/web/api/corpus-runs/run-accepted',
    resume_url: '/web/api/corpus-runs/run-accepted/resume',
    workspace: 'default',
    file_count: fileCount,
  };
}

/** Upload `count` files and resolve with the panel once the accepted-files notice is up. */
async function panelAfterUpload(count: number): Promise<DlInspectorFiles> {
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname.endsWith('/files/upload')) return Response.json(receipt(count), {status: 202});
    if (url.pathname === '/web/api/corpus-runs/run-accepted') return Response.json(receipt(count));
    if (url.pathname.endsWith('/files/failed')) {
      return Response.json({workspace: 'default', failed: [], next_cursor: null});
    }
    return Response.json({workspace: 'default', files: [], next_cursor: null});
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);
  const files = Array.from(
    {length: count},
    (_, index) => new File(['x'], `file-${index}.pdf`, {type: 'application/pdf'}),
  );
  await panel.upload(files);
  await panel.updateComplete;
  return panel;
}

function notice(panel: DlInspectorFiles): string {
  return panel.querySelector('[class*="ingest-queue-notice"]')?.textContent?.trim() ?? '';
}

beforeEach(() => {
  workspaceStore.init(
    [{workspace: 'default', displayName: 'Default', embeddingModel: 'embed', changes: DEFAULT_CHANGES}],
    ['default'],
    'default',
  );
  ingestStore.resetToPrimary();
});

afterEach(async () => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  await setLanguagePreference('auto');
});

it('says one accepted file in the singular and several in the plural', async () => {
  const one = await panelAfterUpload(1);
  await waitFor(() => notice(one) !== '');
  expect(notice(one)).to.equal('1 new file accepted for ingest');
  one.pause();
  document.body.replaceChildren();

  const three = await panelAfterUpload(3);
  await waitFor(() => notice(three) !== '');
  expect(notice(three)).to.equal('3 new files accepted for ingest');
  three.pause();
});

it('says the accepted count in the reader language', async () => {
  await setLanguagePreference('zh');

  const one = await panelAfterUpload(1);
  await waitFor(() => notice(one) !== '');
  expect(notice(one)).to.equal('1 个新文件已接受摄取');
  one.pause();
  document.body.replaceChildren();

  const three = await panelAfterUpload(3);
  await waitFor(() => notice(three) !== '');
  expect(notice(three)).to.equal('3 个新文件已接受摄取');
  three.pause();
});
