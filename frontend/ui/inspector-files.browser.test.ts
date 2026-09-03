// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {workspaceStore} from '../stores/workspace-store.ts';
import {ingestStore} from '../stores/ingest-store.ts';
import './inspector-files.ts';
import type {DlFailedFileRecovery} from './failed-file-recovery.ts';
import type {DlInspectorFiles} from './inspector-files.ts';

const originalFetch = window.fetch;

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

function confirmDeleteDialog(panel: DlInspectorFiles, value: string): void {
  const dialog = panel.querySelector<HTMLDialogElement>('#delete-file-dialog')!;
  dialog.returnValue = value;
  dialog.close();
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
  expect(panel.querySelector<HTMLButtonElement>('[data-file-delete]')?.ariaLabel).to.equal(
    'Delete <img src=x>',
  );
});

function snapshot(
  files: Array<{file_name: string; file_path: string}>,
  nextCursor: string | null,
  workspace = 'default',
) {
  return {
    workspace,
    files,
    ingest: {
      busy: false,
      message: '',
      progress_percent: null,
      current_batch: null,
      total_batches: null,
      documents: null,
      pending_enqueues: 0,
    },
    next_cursor: nextCursor,
  };
}

function deferredResponse() {
  let resolve!: (response: Response) => void;
  const promise = new Promise<Response>((done) => { resolve = done; });
  return {promise, resolve};
}

it('appends older files with coalescing, overlap dedup, and accessible exhaustion focus', async () => {
  const older = deferredResponse();
  let olderRequests = 0;
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (!url.searchParams.has('cursor')) {
      return new Response(JSON.stringify(snapshot([
        {file_name: 'Newest', file_path: '/newest'},
        {file_name: 'Overlap', file_path: '/overlap'},
      ], 'older-1')), {status: 200, headers: {'Content-Type': 'application/json'}});
    }
    expect(url.searchParams.get('cursor')).to.equal('older-1');
    olderRequests += 1;
    return older.promise;
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);

  const button = panel.querySelector<HTMLButtonElement>('[data-load-older-files]')!;
  expect(button.type).to.equal('button');
  expect(button.textContent?.trim()).to.equal('Load older files');
  button.focus();
  button.click();
  const flight = panel.loadOlderFiles();
  expect(panel.loadOlderFiles()).to.equal(flight);
  await panel.updateComplete;
  expect(button.disabled).to.equal(true);
  expect(button.getAttribute('aria-busy')).to.equal('true');
  expect(olderRequests).to.equal(1);

  older.resolve(new Response(JSON.stringify(snapshot([
    {file_name: 'Overlap stale', file_path: '/overlap'},
    {file_name: 'Oldest', file_path: '/oldest'},
  ], null)), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await flight;
  await panel.updateComplete;

  expect([...panel.querySelectorAll('.file-name')].map((item) => item.textContent)).to.deep.equal([
    'Newest', 'Overlap', 'Oldest',
  ]);
  expect(panel.querySelectorAll('[role="listitem"]')).to.have.length(3);
  expect(panel.querySelector('[data-load-older-files]')).to.equal(null);
  expect(document.activeElement).to.equal(panel.querySelector('#file-list'));
  expect(panel.querySelector('[data-older-files-status]')?.textContent).to.contain(
    'Loaded 1 older file.',
  );
});

it('keeps loaded rows and cursor retryable after an older-page failure', async () => {
  let olderAttempts = 0;
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (!url.searchParams.has('cursor')) {
      return new Response(JSON.stringify(snapshot([
        {file_name: 'Newest', file_path: '/newest'},
      ], 'older-1')), {status: 200, headers: {'Content-Type': 'application/json'}});
    }
    olderAttempts += 1;
    return olderAttempts === 1
      ? new Response('unavailable', {status: 503})
      : new Response(JSON.stringify(snapshot([
          {file_name: 'Oldest', file_path: '/oldest'},
        ], null)), {status: 200, headers: {'Content-Type': 'application/json'}});
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);

  await panel.loadOlderFiles();
  await panel.updateComplete;
  expect(panel.querySelectorAll('[data-file-item]')).to.have.length(1);
  expect(panel.querySelector('[data-load-older-files]')?.textContent).to.contain(
    'Retry loading older files',
  );
  expect(panel.error).to.equal(null);

  await panel.loadOlderFiles();
  await panel.updateComplete;
  expect(panel.querySelectorAll('[data-file-item]')).to.have.length(2);
  expect(panel.querySelector('[data-load-older-files]')).to.equal(null);
});

it('rejects a late older page after pause invalidates its generation', async () => {
  const older = deferredResponse();
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (!url.searchParams.has('cursor')) {
      return new Response(JSON.stringify(snapshot([
        {file_name: 'Newest', file_path: '/newest'},
      ], 'older-1')), {status: 200, headers: {'Content-Type': 'application/json'}});
    }
    return older.promise;
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);

  const flight = panel.loadOlderFiles();
  panel.pause();
  older.resolve(new Response(JSON.stringify(snapshot([
    {file_name: 'Stale', file_path: '/stale'},
  ], null)), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await flight;
  await panel.updateComplete;

  expect([...panel.querySelectorAll('.file-name')].map((item) => item.textContent)).to.deep.equal([
    'Newest',
  ]);
  expect(panel.filesLoadMoreState).to.equal('idle');
});

it('preserves loaded files and cursor when an upload only changes ingest status', async () => {
  window.fetch = async (input, init) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname.endsWith('/files/upload')) {
      return new Response(JSON.stringify({
        workspace: 'default',
        file_count: 1,
        queued: false,
        ingest: {...snapshot([], null).ingest, busy: true, message: 'Starting ingest...'},
      }), {status: 200, headers: {'Content-Type': 'application/json'}});
    }
    expect(init?.method).to.equal(undefined);
    return new Response(JSON.stringify(snapshot([
      {file_name: 'Newest', file_path: '/newest'},
    ], 'older-1')), {status: 200, headers: {'Content-Type': 'application/json'}});
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);

  await panel.upload([new File(['report'], 'report.pdf', {type: 'application/pdf'})]);
  await panel.updateComplete;

  expect(panel.snapshot?.files.map((item) => item.filePath)).to.deep.equal(['/newest']);
  expect(panel.snapshot?.nextCursor).to.equal('older-1');
  panel.pause();
});

it('deletion replaces loaded traversal with the returned fresh first page', async () => {
  window.fetch = async (_input, init) => {
    if (init?.method === 'DELETE') {
      return new Response(JSON.stringify(snapshot([
        {file_name: 'Replacement', file_path: '/replacement'},
      ], 'replacement-older')), {status: 200, headers: {'Content-Type': 'application/json'}});
    }
    return new Response(JSON.stringify(snapshot([
      {file_name: 'Delete me', file_path: '/delete'},
      {file_name: 'Loaded older', file_path: '/loaded-older'},
    ], 'old-cursor')), {status: 200, headers: {'Content-Type': 'application/json'}});
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);

  panel.querySelector<HTMLButtonElement>('[data-file-delete]')!.click();
  await panel.updateComplete;
  confirmDeleteDialog(panel, 'confirm');
  await waitFor(() => panel.snapshot?.files[0]?.filePath === '/replacement');
  await panel.updateComplete;

  expect(panel.snapshot?.files.map((item) => item.filePath)).to.deep.equal(['/replacement']);
  expect(panel.snapshot?.nextCursor).to.equal('replacement-older');
  expect(panel.querySelector('[data-load-older-files]')).not.to.equal(null);
});

it('cancelling the delete dialog keeps the file and restores trigger focus', async () => {
  let deleteRequests = 0;
  window.fetch = async (_input, init) => {
    if (init?.method === 'DELETE') {
      deleteRequests += 1;
      return new Response(null, {status: 204});
    }
    return new Response(JSON.stringify(snapshot([
      {file_name: 'Keep me', file_path: '/keep'},
    ], null)), {status: 200, headers: {'Content-Type': 'application/json'}});
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);
  const deleteButton = panel.querySelector<HTMLButtonElement>('[data-file-delete]')!;
  deleteButton.focus();

  deleteButton.click();
  await panel.updateComplete;
  const dialog = panel.querySelector<HTMLDialogElement>('#delete-file-dialog')!;
  expect(dialog.open).to.equal(true);
  expect(panel.querySelector<HTMLElement>('#delete-file-message')?.textContent).to.contain(
    'keep',
  );

  confirmDeleteDialog(panel, 'cancel');
  await panel.updateComplete;

  expect(dialog.open).to.equal(false);
  expect(deleteRequests).to.equal(0);
  expect(panel.querySelector('.file-name')?.textContent).to.equal('Keep me');
  expect(document.activeElement).to.equal(deleteButton);
});

it('clears prior-workspace rows when the selected workspace reload fails', async () => {
  let deleteRequests = 0;
  window.fetch = async (input, init) => {
    if (init?.method === 'DELETE') {
      deleteRequests += 1;
      return new Response(null, {status: 204});
    }
    const url = new URL(String(input), window.location.origin);
    if (url.searchParams.get('workspace') === 'secondary') {
      return new Response('unavailable', {status: 503});
    }
    return new Response(JSON.stringify(snapshot([
      {file_name: 'Default report', file_path: '/default-report'},
    ], 'default-older')), {status: 200, headers: {'Content-Type': 'application/json'}});
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);
  expect(panel.querySelector('[data-file-delete]')).not.to.equal(null);

  ingestStore.set('secondary');
  await panel.updateComplete;
  expect(
    panel.querySelector<DlFailedFileRecovery>('dl-failed-file-recovery')?.workspace,
  ).to.equal('secondary');
  await waitFor(() => panel.loading === false && panel.error !== null);
  await panel.updateComplete;

  expect(panel.snapshot).to.equal(null);
  expect(panel.acceptedFiles).to.equal(0);
  expect(panel.querySelector('.file-name')).to.equal(null);
  expect(panel.querySelector('[data-file-delete]')).to.equal(null);
  expect(panel.querySelector('[data-load-older-files]')).to.equal(null);
  expect(deleteRequests).to.equal(0);
});

it('delete during an older-page flight cannot apply stale rows or latch loading state', async () => {
  const older = deferredResponse();
  const deletion = deferredResponse();
  let olderRequests = 0;
  window.fetch = async (input, init) => {
    const url = new URL(String(input), window.location.origin);
    if (init?.method === 'DELETE') return deletion.promise;
    if (url.searchParams.has('cursor')) {
      olderRequests += 1;
      return older.promise;
    }
    return new Response(JSON.stringify(snapshot([
      {file_name: 'Delete me', file_path: '/delete'},
    ], 'old-cursor')), {status: 200, headers: {'Content-Type': 'application/json'}});
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);

  const olderFlight = panel.loadOlderFiles();
  await panel.updateComplete;
  expect(panel.filesLoadMoreState).to.equal('loading');
  panel.querySelector<HTMLButtonElement>('[data-file-delete]')!.click();
  await panel.updateComplete;
  confirmDeleteDialog(panel, 'confirm');
  await waitFor(() => panel.hasActiveMutation);
  await panel.updateComplete;
  expect(panel.filesLoadMoreState).to.equal('idle');
  await panel.loadOlderFiles();
  expect(olderRequests).to.equal(1);

  deletion.resolve(new Response(JSON.stringify(snapshot([
    {file_name: 'Replacement', file_path: '/replacement'},
  ], 'replacement-older')), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await waitFor(() => panel.snapshot?.files[0]?.filePath === '/replacement');
  older.resolve(new Response(JSON.stringify(snapshot([
    {file_name: 'Stale older', file_path: '/stale'},
  ], null)), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await olderFlight;
  await panel.updateComplete;

  expect(panel.snapshot?.files.map((item) => item.filePath)).to.deep.equal(['/replacement']);
  expect(panel.filesLoadMoreState).to.equal('idle');
  const button = panel.querySelector<HTMLButtonElement>('[data-load-older-files]')!;
  expect(button.disabled).to.equal(false);
  expect(button.getAttribute('aria-busy')).to.equal('false');
});

it('load older is a no-op while a same-workspace first-page reload is active', async () => {
  const reloaded = deferredResponse();
  let firstPageRequests = 0;
  let olderRequests = 0;
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname.endsWith('/files/failed')) {
      return new Response(JSON.stringify({
        workspace: 'default',
        failed: [],
        next_cursor: null,
        active_recovery: null,
      }), {status: 200, headers: {'Content-Type': 'application/json'}});
    }
    if (url.searchParams.has('cursor')) {
      olderRequests += 1;
      return new Response(JSON.stringify(snapshot([], null)), {
        status: 200,
        headers: {'Content-Type': 'application/json'},
      });
    }
    firstPageRequests += 1;
    if (firstPageRequests === 1) {
      return new Response(JSON.stringify(snapshot([
        {file_name: 'Original', file_path: '/original'},
      ], 'original-older')), {status: 200, headers: {'Content-Type': 'application/json'}});
    }
    return reloaded.promise;
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  document.body.appendChild(panel);
  panel.active = true;
  await waitFor(() => panel.loading === false);
  await panel.updateComplete;
  // Let any update-cycle-driven reload start and finish before exercising
  // reload(false), so the test's own first-page request cannot be superseded
  // by a straggler update while it is in flight.
  await new Promise((resolve) => { setTimeout(resolve, 0); });
  await panel.updateComplete;

  const reloadFlight = panel.reload(false);
  await panel.loadOlderFiles();
  expect(olderRequests).to.equal(0);
  expect(panel.filesLoadMoreState).to.equal('idle');

  reloaded.resolve(new Response(JSON.stringify(snapshot([
    {file_name: 'Fresh', file_path: '/fresh'},
  ], 'fresh-older')), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await reloadFlight;
  await panel.updateComplete;

  expect(panel.snapshot?.files.map((item) => item.filePath)).to.deep.equal(['/fresh']);
  expect(panel.filesLoadMoreState).to.equal('idle');
  const button = panel.querySelector<HTMLButtonElement>('[data-load-older-files]')!;
  expect(button.disabled).to.equal(false);
  expect(button.getAttribute('aria-busy')).to.equal('false');
});

it('failed upload settles loading after superseding a pending visible reload', async () => {
  const staleReload = deferredResponse();
  let firstPageRequests = 0;
  window.fetch = async (input, init) => {
    if (init?.method === 'POST') {
      return new Response(JSON.stringify({detail: 'Upload rejected.'}), {
        status: 503,
        headers: {'Content-Type': 'application/json'},
      });
    }
    const url = new URL(String(input), window.location.origin);
    expect(url.pathname).to.equal('/web/api/files');
    firstPageRequests += 1;
    if (firstPageRequests === 1) {
      return new Response(JSON.stringify(snapshot([
        {file_name: 'Original', file_path: '/original'},
      ], null)), {status: 200, headers: {'Content-Type': 'application/json'}});
    }
    return staleReload.promise;
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  document.body.appendChild(panel);
  panel.active = true;
  await waitFor(() => panel.loading === false);

  const reloadFlight = panel.reload(true);
  expect(panel.loading).to.equal(true);
  await panel.upload([new File(['report'], 'report.pdf', {type: 'application/pdf'})]);
  await panel.updateComplete;

  expect(panel.loading).to.equal(false);
  expect(panel.uploading).to.equal(false);
  expect(panel.error).to.equal('Upload rejected.');
  expect(panel.querySelector('.file-error')?.textContent).to.contain('Upload rejected.');
  expect(panel.querySelector('.file-status--loading')).to.equal(null);
  expect(panel.snapshot?.files.map((item) => item.filePath)).to.deep.equal(['/original']);

  staleReload.resolve(new Response(JSON.stringify(snapshot([
    {file_name: 'Stale reload', file_path: '/stale'},
  ], null)), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await reloadFlight;
  await panel.updateComplete;

  expect(panel.loading).to.equal(false);
  expect(panel.error).to.equal('Upload rejected.');
  expect(panel.snapshot?.files.map((item) => item.filePath)).to.deep.equal(['/original']);
});

it('failed deletion settles loading after superseding a pending visible reload', async () => {
  const staleReload = deferredResponse();
  let firstPageRequests = 0;
  window.fetch = async (input, init) => {
    if (init?.method === 'DELETE') {
      return new Response(JSON.stringify({detail: 'Deletion rejected.'}), {
        status: 503,
        headers: {'Content-Type': 'application/json'},
      });
    }
    const url = new URL(String(input), window.location.origin);
    expect(url.pathname).to.equal('/web/api/files');
    firstPageRequests += 1;
    if (firstPageRequests === 1) {
      return new Response(JSON.stringify(snapshot([
        {file_name: 'Original', file_path: '/original'},
      ], null)), {status: 200, headers: {'Content-Type': 'application/json'}});
    }
    return staleReload.promise;
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  document.body.appendChild(panel);
  panel.active = true;
  await waitFor(() => panel.loading === false);
  const deleteButton = panel.querySelector<HTMLButtonElement>('[data-file-delete]')!;

  const reloadFlight = panel.reload(true);
  expect(panel.loading).to.equal(true);
  deleteButton.click();
  await panel.updateComplete;
  confirmDeleteDialog(panel, 'confirm');
  await waitFor(() => panel.error === 'Deletion rejected.');
  await panel.updateComplete;

  expect(panel.loading).to.equal(false);
  expect(panel.error).to.equal('Deletion rejected.');
  expect(panel.querySelector('.file-error')?.textContent).to.contain('Deletion rejected.');
  expect(panel.querySelector('.file-status--loading')).to.equal(null);
  expect(panel.snapshot?.files.map((item) => item.filePath)).to.deep.equal(['/original']);

  staleReload.resolve(new Response(JSON.stringify(snapshot([
    {file_name: 'Stale reload', file_path: '/stale'},
  ], null)), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await reloadFlight;
  await panel.updateComplete;

  expect(panel.loading).to.equal(false);
  expect(panel.error).to.equal('Deletion rejected.');
  expect(panel.snapshot?.files.map((item) => item.filePath)).to.deep.equal(['/original']);
});
