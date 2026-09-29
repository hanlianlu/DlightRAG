// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {productionHandles} from '../stores/app-handles.ts';
import {setLanguagePreference} from '../i18n/locale.ts';
import './inspector-files.ts';
import type {DlFailedFileRecovery} from './failed-file-recovery.ts';
import type {DlInspectorFiles} from './inspector-files.ts';
import {waitFor} from '../testing/dom.ts';

const {workspaces: workspaceStore, ingest: ingestStore} = productionHandles();

const originalFetch = window.fetch;

function confirmDeleteDialog(panel: DlInspectorFiles, value: string): void {
  const dialog = panel.querySelector<HTMLDialogElement>('#delete-file-dialog')!;
  dialog.close(value);
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
    next_cursor: nextCursor,
  };
}

function corpusReceipt(runId: string, status = 'queued') {
  return {
    run_id: runId,
    run_kind: 'corpus_mutation',
    lane: 'corpus_mutation',
    status,
    status_url: `/web/api/corpus-runs/${runId}`,
    events_url: `/web/api/corpus-runs/${runId}/events`,
    cancel_url: `/web/api/corpus-runs/${runId}`,
    resume_url: `/web/api/corpus-runs/${runId}/resume`,
    workspace: 'default',
    file_count: 1,
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

  const button = panel.querySelector<HTMLButtonElement>('[data-load-older="files"]')!;
  expect(button.type).to.equal('button');
  expect(button.textContent?.trim()).to.equal('Load older files');
  button.focus();
  button.click();
  const flight = panel.loadOlderFiles();
  expect(panel.loadOlderFiles()).to.equal(flight);
  await panel.updateComplete;
  expect(button.getAttribute('aria-disabled')).to.equal('true');
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
  expect(panel.querySelector('[data-load-older="files"]')).to.equal(null);
  expect(document.activeElement).to.equal(panel.querySelector('#file-list'));
  expect(panel.querySelector('[data-load-older-status="files"]')?.textContent).to.contain(
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
  expect(panel.querySelector('[data-load-older="files"]')?.textContent).to.contain(
    'Retry loading older files',
  );
  expect(panel.error).to.equal(null);

  await panel.loadOlderFiles();
  await panel.updateComplete;
  expect(panel.querySelectorAll('[data-file-item]')).to.have.length(2);
  expect(panel.querySelector('[data-load-older="files"]')).to.equal(null);
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

it('preserves loaded files and cursor when an upload accepts a durable Run', async () => {
  window.fetch = async (input, init) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname.endsWith('/files/upload')) {
      return new Response(JSON.stringify(corpusReceipt('run-upload')), {
        status: 202,
        headers: {'Content-Type': 'application/json'},
      });
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

it('deletion reloads the first page after its durable Run succeeds', async () => {
  let fileLists = 0;
  window.fetch = async (input, init) => {
    const url = new URL(String(input), window.location.origin);
    if (init?.method === 'DELETE') {
      return new Response(JSON.stringify(corpusReceipt('run-delete')), {
        status: 202,
        headers: {'Content-Type': 'application/json'},
      });
    }
    if (url.pathname === '/web/api/corpus-runs/run-delete') {
      return new Response(JSON.stringify(corpusReceipt('run-delete', 'succeeded')), {
        headers: {'Content-Type': 'application/json'},
      });
    }
    if (url.pathname.endsWith('/files')) {
      fileLists += 1;
      const page = fileLists === 1
        ? snapshot([
          {file_name: 'Delete me', file_path: '/delete'},
          {file_name: 'Loaded older', file_path: '/loaded-older'},
        ], 'old-cursor')
        : snapshot([{file_name: 'Replacement', file_path: '/replacement'}], 'replacement-older');
      return new Response(JSON.stringify(page), {
        status: 200,
        headers: {'Content-Type': 'application/json'},
      });
    }
    return new Response(JSON.stringify({workspace: 'default', failed: [], next_cursor: null}), {
      headers: {'Content-Type': 'application/json'},
    });
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
  expect(panel.querySelector('[data-load-older="files"]')).not.to.equal(null);
});

/** A Files panel whose one deletion's Run status answers `status` for as long as it is read. */
async function refusedDeletion(status: number, body: Record<string, unknown>) {
  const reads = {status: 0, files: 0, failed: 0};
  const toasts: string[] = [];
  window.fetch = async (input, init) => {
    const url = new URL(String(input), window.location.origin);
    if (init?.method === 'DELETE') {
      return new Response(JSON.stringify(corpusReceipt('run-gone')), {
        status: 202,
        headers: {'Content-Type': 'application/json'},
      });
    }
    if (url.pathname === '/web/api/corpus-runs/run-gone') {
      reads.status += 1;
      return new Response(JSON.stringify(body), {
        status,
        headers: {'Content-Type': 'application/json'},
      });
    }
    if (url.pathname.endsWith('/files/failed')) {
      reads.failed += 1;
      return new Response(JSON.stringify({workspace: 'default', failed: [], next_cursor: null}), {
        headers: {'Content-Type': 'application/json'},
      });
    }
    reads.files += 1;
    // The list after the refusal differs, so only a reload can show it.
    const files = reads.status === 0
      ? [{file_name: 'Keep', file_path: '/keep'}]
      : [{file_name: 'Now', file_path: '/now'}];
    return new Response(JSON.stringify(snapshot(files, null)), {
      headers: {'Content-Type': 'application/json'},
    });
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.addEventListener('dl-toast-request', (event) => { toasts.push(event.detail.message); });
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);
  await ticks();
  const listed = reads.files;
  const checked = reads.failed;
  panel.querySelector<HTMLButtonElement>('[data-file-delete]')!.click();
  await panel.updateComplete;
  confirmDeleteDialog(panel, 'confirm');
  return {panel, reads, toasts, listed, checked};
}

async function withImmediateTimers(run: () => Promise<void>): Promise<void> {
  const originalSetTimeout = window.setTimeout;
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  try {
    await run();
  } finally {
    window.setTimeout = originalSetTimeout;
  }
}

async function ticks(count = 20): Promise<void> {
  for (let tick = 0; tick < count; tick += 1) await new Promise((resolve) => setTimeout(resolve, 0));
}

it('stops polling a deletion whose status the Corpus Run API refuses', async () => {
  await withImmediateTimers(async () => {
    // The route answers a bare 404 for a Run this reader no longer sees.
    const {panel, reads, toasts, checked} = await refusedDeletion(404, {detail: 'Corpus Mutation Run not found'});
    await waitFor(() => panel.snapshot?.files[0]?.filePath === '/now' && !panel.loading);
    const statusReads = reads.status;
    await ticks();

    expect(reads.status).to.equal(statusReads);
    expect(panel.mutationRun).to.equal(null);
    expect(toasts.at(-1)).to.equal('Corpus update status is no longer available.');
    await waitFor(() => reads.failed > checked);
    await panel.updateComplete;
    expect(panel.querySelector('#ingest-progress')).to.equal(null);
  });
});

it('tells a reader who lost access why the deletion is no longer followed', async () => {
  await withImmediateTimers(async () => {
    const {panel, toasts} = await refusedDeletion(403, {
      detail: 'Access denied for action=workspace.list_files workspace=default',
      error_type: 'auth',
    });
    await waitFor(() => panel.snapshot?.files[0]?.filePath === '/now' && !panel.loading);

    expect(toasts.at(-1)).to.equal('You do not have permission to do that.');
    expect(toasts.join(' ')).not.to.contain('Access denied');
  });
});

it('keeps polling a status that fails for any reason but a refusal', async () => {
  await withImmediateTimers(async () => {
    const {panel, reads, listed} = await refusedDeletion(503, {detail: 'Service unavailable', error_type: 'unavailable'});
    await waitFor(() => reads.status >= 3);

    expect(panel.mutationRun?.runId).to.equal('run-gone');
    expect(reads.files, 'a retried status never reloads the list').to.equal(listed);
    panel.pause();
  });
});

it('keeps a second upload alive while the Run it replaces settles', async () => {
  await withImmediateTimers(async () => {
    const uploadSignals: AbortSignal[] = [];
    const toasts: string[] = [];
    let statusReads = 0;
    let releaseStatus!: () => void;
    const statusGate = new Promise<void>((resolve) => { releaseStatus = resolve; });
    let releaseUpload!: () => void;
    const uploadGate = new Promise<void>((resolve) => { releaseUpload = resolve; });
    window.fetch = async (input, init) => {
      const url = new URL(String(input), window.location.origin);
      if (url.pathname.endsWith('/files/upload')) {
        uploadSignals.push(init!.signal!);
        if (uploadSignals.length === 1) return Response.json(corpusReceipt('run-1'), {status: 202});
        return new Promise<Response>((resolve, reject) => {
          init!.signal!.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
          void uploadGate.then(() => resolve(Response.json(corpusReceipt('run-2'), {status: 202})));
        });
      }
      if (url.pathname === '/web/api/corpus-runs/run-1') {
        statusReads += 1;
        if (statusReads === 1) return Response.json(corpusReceipt('run-1', 'running'));
        await statusGate;
        return Response.json(corpusReceipt('run-1', 'succeeded'));
      }
      if (url.pathname === '/web/api/corpus-runs/run-2') return Response.json(corpusReceipt('run-2', 'running'));
      if (url.pathname.endsWith('/files/failed')) {
        return Response.json({workspace: 'default', failed: [], next_cursor: null});
      }
      return Response.json(snapshot([], null));
    };
    const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
    panel.addEventListener('dl-toast-request', (event) => { toasts.push(event.detail.message); });
    document.body.appendChild(panel);
    panel.active = true;
    await waitFor(() => panel.loading === false);

    await panel.upload([new File(['one'], 'one.pdf', {type: 'application/pdf'})]);
    await waitFor(() => statusReads === 2);
    const second = panel.upload([new File(['two'], 'two.pdf', {type: 'application/pdf'})]);
    await waitFor(() => uploadSignals.length === 2);
    releaseStatus();
    await ticks();
    releaseUpload();
    await second;
    await ticks();

    expect(uploadSignals[1]!.aborted, 'the settle reload never aborts the upload').to.equal(false);
    expect(panel.mutationRun?.runId).to.equal('run-2');
    expect(toasts).not.to.include('Corpus update finished.');
    panel.pause();
  });
});

it('hands polling back to the followed Run when a later upload is refused', async () => {
  await withImmediateTimers(async () => {
    const toasts: string[] = [];
    let statusReads = 0;
    let uploads = 0;
    let releaseUpload!: () => void;
    const uploadGate = new Promise<void>((resolve) => { releaseUpload = resolve; });
    window.fetch = async (input) => {
      const url = new URL(String(input), window.location.origin);
      if (url.pathname.endsWith('/files/upload')) {
        uploads += 1;
        if (uploads === 1) return Response.json(corpusReceipt('run-1'), {status: 202});
        await uploadGate;
        return Response.json({detail: 'Corpus writes are paused', error_type: 'unavailable'}, {status: 503});
      }
      if (url.pathname === '/web/api/corpus-runs/run-1') {
        statusReads += 1;
        return Response.json(corpusReceipt('run-1', statusReads < 3 ? 'running' : 'succeeded'));
      }
      if (url.pathname.endsWith('/files/failed')) {
        return Response.json({workspace: 'default', failed: [], next_cursor: null});
      }
      return Response.json(snapshot([], null));
    };
    const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
    panel.addEventListener('dl-toast-request', (event) => { toasts.push(event.detail.message); });
    document.body.appendChild(panel);
    panel.active = true;
    await waitFor(() => panel.loading === false);

    await panel.upload([new File(['one'], 'one.pdf', {type: 'application/pdf'})]);
    await waitFor(() => statusReads === 1);
    const refused = panel.upload([new File(['two'], 'two.pdf', {type: 'application/pdf'})]);
    await ticks();
    expect(statusReads, 'the followed Run waits while the upload is pending').to.equal(1);
    releaseUpload();
    await refused;
    await waitFor(() => toasts.includes('Corpus update finished.'));

    expect(toasts).to.include('Corpus writes are paused');
    expect(panel.mutationRun?.status).to.equal('succeeded');
  });
});

for (const mutation of ['upload', 'delete'] as const) {
  it(`keeps an in-flight ${mutation} alive when a document recovery finishes, then reloads`, async () => {
    const signals: AbortSignal[] = [];
    let listReads = 0;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    window.fetch = async (input, init) => {
      const url = new URL(String(input), window.location.origin);
      if (url.pathname.endsWith('/files/upload') || init?.method === 'DELETE') {
        signals.push(init!.signal!);
        return new Promise<Response>((resolve, reject) => {
          init!.signal!.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
          void gate.then(() => resolve(Response.json(corpusReceipt('run-mutation'), {status: 202})));
        });
      }
      if (url.pathname === '/web/api/corpus-runs/run-mutation') {
        return Response.json(corpusReceipt('run-mutation', 'running'));
      }
      if (url.pathname.endsWith('/files/failed')) {
        return Response.json({workspace: 'default', failed: [], next_cursor: null});
      }
      listReads += 1;
      return Response.json(snapshot([{file_name: 'Keep', file_path: '/keep'}], null));
    };
    const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
    panel.active = true;
    document.body.appendChild(panel);
    await waitFor(() => panel.loading === false);
    await ticks();
    const readsBefore = listReads;

    let settled: Promise<void> = Promise.resolve();
    if (mutation === 'upload') {
      settled = panel.upload([new File(['one'], 'one.pdf', {type: 'application/pdf'})]);
    } else {
      panel.querySelector<HTMLButtonElement>('[data-file-delete]')!.click();
      await panel.updateComplete;
      confirmDeleteDialog(panel, 'confirm');
    }
    await waitFor(() => signals.length === 1);
    panel.querySelector('dl-failed-file-recovery')!.dispatchEvent(new CustomEvent(
      'dl-failed-file-recovery-complete',
      {bubbles: true, composed: true},
    ));
    await ticks();
    expect(signals[0]!.aborted, `the recovery's reload leaves the ${mutation} running`).to.equal(false);
    expect(listReads, 'and waits for it').to.equal(readsBefore);

    release();
    await settled;
    await waitFor(() => panel.mutationRun?.runId === 'run-mutation' && listReads === readsBefore + 1);
    expect(signals[0]!.aborted).to.equal(false);
    panel.pause();
  });
}

it('cancelling the delete dialog keeps the file and restores trigger focus', async () => {
  window.fetch = async (_input, init) => {
    if (init?.method === 'DELETE') {
      return new Response(JSON.stringify(corpusReceipt('unexpected-delete')), {
        status: 202,
        headers: {'Content-Type': 'application/json'},
      });
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
  await waitFor(() => !dialog.open && document.activeElement === deleteButton);
  // Let the async click handler consume the modal result before this test
  // replaces the global fetch stub in afterEach.
  await new Promise((resolve) => setTimeout(resolve, 0));

  expect(dialog.open).to.equal(false);
  expect(panel.mutationRun).to.equal(null);
  expect(panel.querySelector('.file-name')?.textContent).to.equal('Keep me');
  expect(document.activeElement).to.equal(deleteButton);
});

it('clears prior-workspace rows when the selected workspace reload fails', async () => {
  const staleDefault = deferredResponse();
  const secondaryFailure = deferredResponse();
  let deferDefault = false;
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (
      url.searchParams.get('workspace') === 'secondary'
      && url.pathname.endsWith('/files')
    ) return secondaryFailure.promise;
    if (url.searchParams.get('workspace') === 'secondary') {
      return new Response('unavailable', {status: 503});
    }
    if (deferDefault && url.pathname.endsWith('/files')) return staleDefault.promise;
    return new Response(JSON.stringify(snapshot([
      {file_name: 'Default report', file_path: '/default-report'},
    ], 'default-older')), {status: 200, headers: {'Content-Type': 'application/json'}});
  };
  const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
  panel.active = true;
  document.body.appendChild(panel);
  await waitFor(() => panel.loading === false);
  expect(panel.querySelector('[data-file-delete]')).not.to.equal(null);

  // Leave a prior-Workspace request unresolved. The transport deliberately
  // ignores AbortSignal so request generation, rather than fetch cooperation,
  // owns the stale-result exclusion.
  deferDefault = true;
  const priorWorkspaceFlight = panel.reload();
  ingestStore.set('secondary');
  await panel.updateComplete;
  expect(panel.snapshot).to.equal(null);
  expect(
    panel.querySelector<DlFailedFileRecovery>('dl-failed-file-recovery')?.workspace,
  ).to.equal('secondary');

  secondaryFailure.resolve(new Response('unavailable', {status: 503}));
  await waitFor(() => panel.loading === false && panel.error !== null);
  staleDefault.resolve(new Response(JSON.stringify(snapshot([
    {file_name: 'Stale default report', file_path: '/stale-default-report'},
  ], null)), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await priorWorkspaceFlight;
  await panel.updateComplete;

  expect(panel.snapshot).to.equal(null);
  expect(panel.acceptedFiles).to.equal(0);
  expect(panel.querySelector('.file-name')).to.equal(null);
  expect(panel.querySelector('[data-file-delete]')).to.equal(null);
  expect(panel.querySelector('[data-load-older="files"]')).to.equal(null);
});

it('delete Run settlement invalidates an older-page flight without latching loading state', async () => {
  const older = deferredResponse();
  const deletion = deferredResponse();
  let olderRequests = 0;
  let fileLists = 0;
  window.fetch = async (input, init) => {
    const url = new URL(String(input), window.location.origin);
    if (init?.method === 'DELETE') return deletion.promise;
    if (url.pathname === '/web/api/corpus-runs/run-delete-race') {
      return new Response(JSON.stringify(corpusReceipt('run-delete-race', 'succeeded')), {
        headers: {'Content-Type': 'application/json'},
      });
    }
    if (url.searchParams.has('cursor')) {
      olderRequests += 1;
      return older.promise;
    }
    if (url.pathname.endsWith('/files')) {
      fileLists += 1;
      const page = fileLists === 1
        ? snapshot([{file_name: 'Delete me', file_path: '/delete'}], 'old-cursor')
        : snapshot([{file_name: 'Replacement', file_path: '/replacement'}], 'replacement-older');
      return new Response(JSON.stringify(page), {
        status: 200,
        headers: {'Content-Type': 'application/json'},
      });
    }
    return new Response(JSON.stringify({workspace: 'default', failed: [], next_cursor: null}), {
      headers: {'Content-Type': 'application/json'},
    });
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

  deletion.resolve(new Response(JSON.stringify(corpusReceipt('run-delete-race')), {
    status: 202,
    headers: {'Content-Type': 'application/json'},
  }));
  await waitFor(() => panel.snapshot?.files[0]?.filePath === '/replacement');
  older.resolve(new Response(JSON.stringify(snapshot([
    {file_name: 'Stale older', file_path: '/stale'},
  ], null)), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await olderFlight;
  await panel.updateComplete;

  expect(panel.snapshot?.files.map((item) => item.filePath)).to.deep.equal(['/replacement']);
  expect(panel.filesLoadMoreState).to.equal('idle');
  const button = panel.querySelector<HTMLButtonElement>('[data-load-older="files"]')!;
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
  const button = panel.querySelector<HTMLButtonElement>('[data-load-older="files"]')!;
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

it('a refused upload without a server reason shows the panel copy in the reader language', async () => {
  const toasts: string[] = [];
  window.fetch = async (_input, init) => {
    if (init?.method === 'POST') return new Response('<html>Bad gateway</html>', {status: 502});
    return new Response(JSON.stringify(snapshot([], null)), {
      status: 200,
      headers: {'Content-Type': 'application/json'},
    });
  };
  await setLanguagePreference('zh');
  try {
    const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
    panel.addEventListener('dl-toast-request', (event) => { toasts.push(event.detail.message); });
    document.body.appendChild(panel);
    panel.active = true;
    await waitFor(() => panel.loading === false);

    await panel.upload([new File(['report'], 'report.pdf', {type: 'application/pdf'})]);
    await panel.updateComplete;

    expect(panel.error).to.equal('上传失败。');
    expect(toasts.at(-1)).to.equal('上传失败。');
  } finally {
    await setLanguagePreference('auto');
  }
});

it('parks an upload Run for operator repair and resumes that same Run', async () => {
  const originalSetTimeout = window.setTimeout;
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  const toasts: string[] = [];
  let statusReads = 0;
  let resumed = false;
  window.fetch = async (input, init) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname.endsWith('/files/upload')) {
      return Response.json(corpusReceipt('run-upload'), {status: 202});
    }
    if (url.pathname === '/web/api/corpus-runs/run-upload/resume' && init?.method === 'POST') {
      resumed = true;
      return Response.json(corpusReceipt('run-upload', 'queued'), {status: 202});
    }
    if (url.pathname === '/web/api/corpus-runs/run-upload') {
      statusReads += 1;
      return Response.json(resumed
        ? corpusReceipt('run-upload', 'succeeded')
        : {...corpusReceipt('run-upload', 'running'), phase: 'waiting_for_repair'});
    }
    if (url.pathname.endsWith('/files/failed')) {
      return Response.json({workspace: 'default', failed: [], next_cursor: null});
    }
    return Response.json(snapshot([], null));
  };
  try {
    const panel = document.createElement('dl-inspector-files') as DlInspectorFiles;
    panel.addEventListener('dl-toast-request', (event) => { toasts.push(event.detail.message); });
    document.body.appendChild(panel);
    panel.active = true;
    await waitFor(() => panel.loading === false);

    await panel.upload([new File(['report'], 'report.pdf', {type: 'application/pdf'})]);
    await waitFor(() => panel.mutationRun?.phase === 'waiting_for_repair');
    await panel.updateComplete;
    const progress = panel.querySelector<HTMLElement>('#ingest-progress')!;
    expect(progress.getAttribute('role')).to.equal('status');
    expect(progress.textContent).to.contain('The Corpus outcome needs operator repair.');
    expect(progress.textContent).to.contain('Repair the Corpus, then resume this same Run.');
    expect(statusReads).to.equal(1);

    progress.querySelector<HTMLButtonElement>('button')!.click();
    await waitFor(() => toasts.includes('Corpus update finished.'));
    expect(toasts.indexOf('Corpus repair resume accepted.'))
      .to.be.lessThan(toasts.indexOf('Corpus update finished.'));
    expect(statusReads).to.equal(2);
    expect(panel.mutationRun?.status).to.equal('succeeded');
  } finally {
    window.setTimeout = originalSetTimeout;
  }
});
