// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import './failed_file_recovery.ts';
import type {DlFailedFileRecovery} from './failed_file_recovery.ts';

const originalFetch = window.fetch;
const originalSetTimeout = window.setTimeout;

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

function failedPage(nextCursor: string | null = null) {
  return {
    workspace: 'personel',
    failed: [
      {
        document_id: 'doc-1',
        file_name: '货币、权力与人.pdf',
        error: 'technical embedding failure',
        updated_at: '2026-08-31T21:36:15',
      },
      {
        document_id: 'doc-2',
        file_name: 'Advanced Macroeconomics.pdf',
        error: 'batch cancelled',
        updated_at: '2026-08-31T21:55:01',
      },
    ],
    next_cursor: nextCursor,
    active_recovery: null,
  };
}

function mount(): DlFailedFileRecovery {
  const recovery = document.createElement('dl-failed-file-recovery') as DlFailedFileRecovery;
  recovery.workspace = 'personel';
  recovery.active = true;
  document.body.appendChild(recovery);
  return recovery;
}

function deferredResponse() {
  let resolve!: (response: Response) => void;
  const promise = new Promise<Response>((done) => { resolve = done; });
  return {promise, resolve};
}

afterEach(() => {
  window.fetch = originalFetch;
  window.setTimeout = originalSetTimeout;
  document.body.replaceChildren();
});

it('keeps Retry all available while failed-document details are collapsed', async () => {
  window.fetch = async () => new Response(JSON.stringify(failedPage()), {
    headers: {'Content-Type': 'application/json'},
  });

  const recovery = mount();
  await waitFor(() => recovery.loading === false && recovery.page !== null);

  const disclosure = recovery.querySelector<HTMLDetailsElement>('.failed-file-recovery')!;
  const retry = recovery.querySelector<HTMLButtonElement>('.failed-file-retry')!;
  expect(disclosure.open).to.equal(false);
  expect(disclosure.textContent).to.contain('2 documents need attention');
  expect(retry.textContent?.trim()).to.equal('Retry all');
  expect(retry.disabled).to.equal(false);
  const failedRow = recovery.querySelector<HTMLDetailsElement>('.failed-file-row');
  expect(failedRow?.open).to.equal(false);
  expect(failedRow?.getAttribute('role')).to.equal(null);
  expect(failedRow?.parentElement?.tagName).to.equal('LI');
  expect(recovery.textContent).to.contain('technical embedding failure');
});

it('clears stale rows and cancels confirmation when the workspace changes', async () => {
  const nextWorkspace = deferredResponse();
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.searchParams.get('workspace') === 'other') return nextWorkspace.promise;
    return new Response(JSON.stringify(failedPage()), {
      headers: {'Content-Type': 'application/json'},
    });
  };

  const recovery = mount();
  await waitFor(() => recovery.loading === false && recovery.page !== null);
  recovery.querySelector<HTMLButtonElement>('.failed-file-retry')?.click();
  const dialog = recovery.querySelector<HTMLDialogElement>('#retry-failed-files-dialog')!;
  await waitFor(() => dialog.open);

  recovery.workspace = 'other';
  await recovery.updateComplete;

  expect(dialog.open).to.equal(false);
  expect(recovery.page).to.equal(null);
  expect(recovery.textContent).not.to.contain('货币、权力与人.pdf');
  expect(recovery.querySelector('.failed-file-recovery')).to.equal(null);

  nextWorkspace.resolve(new Response(JSON.stringify({
    workspace: 'other',
    failed: [],
    next_cursor: null,
    active_recovery: null,
  }), {headers: {'Content-Type': 'application/json'}}));
  await waitFor(() => recovery.loading === false);
});

it('loads failed documents in bounded pages without duplicating rows', async () => {
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (!url.searchParams.has('cursor')) {
      return new Response(JSON.stringify(failedPage('older')), {
        headers: {'Content-Type': 'application/json'},
      });
    }
    return new Response(JSON.stringify({
      workspace: 'personel',
      failed: [
        failedPage().failed[1],
        {
          document_id: 'doc-3',
          file_name: 'Third.pdf',
          error: 'parser failure',
          updated_at: '2026-08-30T12:00:00',
        },
      ],
      next_cursor: null,
      active_recovery: null,
    }), {headers: {'Content-Type': 'application/json'}});
  };

  const recovery = mount();
  await waitFor(() => recovery.loading === false && recovery.page !== null);
  recovery.querySelector<HTMLButtonElement>('.failed-file-more-button')?.click();
  await waitFor(() => recovery.page?.next_cursor === null && recovery.page.failed.length === 3);

  expect(recovery.querySelectorAll('.failed-file-row')).to.have.length(3);
  expect(recovery.querySelector('.failed-file-more-button')).to.equal(null);
});

it('refreshes the parent file list after a recovery job reaches a terminal state', async () => {
  let failedPageRequests = 0;
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname.includes('/files/retry/')) {
      return new Response(JSON.stringify({
        job_id: 'retry-1',
        workspace: 'personel',
        status: 'succeeded',
        retried: 2,
        succeeded: 2,
        failed: 0,
      }), {headers: {'Content-Type': 'application/json'}});
    }
    failedPageRequests += 1;
    const page = failedPageRequests === 1
      ? {
        ...failedPage(),
        active_recovery: {
          job_id: 'retry-1',
          workspace: 'personel',
          status: 'running',
          retried: 0,
          succeeded: 0,
          failed: 0,
        },
      }
      : {...failedPage(), failed: [], active_recovery: null};
    return new Response(JSON.stringify(page), {
      headers: {'Content-Type': 'application/json'},
    });
  };

  const recovery = document.createElement('dl-failed-file-recovery') as DlFailedFileRecovery;
  let completed = false;
  recovery.addEventListener('dl-failed-file-recovery-complete', () => { completed = true; });
  recovery.workspace = 'personel';
  recovery.active = true;
  document.body.appendChild(recovery);
  await waitFor(() => completed);

  expect(failedPageRequests).to.equal(2);
  expect(recovery.page?.failed).to.have.length(0);
});

it('confirms Retry all, starts one durable job, and disables duplicate retry', async () => {
  const requests: string[] = [];
  window.fetch = async (input, init) => {
    const url = String(input);
    requests.push(`${init?.method ?? 'GET'} ${url}`);
    if (init?.method === 'POST') {
      return new Response(JSON.stringify({
        job_id: 'retry-1',
        workspace: 'personel',
        status: 'queued',
        retried: 0,
        succeeded: 0,
        failed: 0,
      }), {headers: {'Content-Type': 'application/json'}});
    }
    return new Response(JSON.stringify(failedPage()), {
      headers: {'Content-Type': 'application/json'},
    });
  };

  const recovery = mount();
  await waitFor(() => recovery.loading === false && recovery.page !== null);
  recovery.querySelector<HTMLButtonElement>('.failed-file-retry')?.click();
  const dialog = recovery.querySelector<HTMLDialogElement>('#retry-failed-files-dialog')!;
  await waitFor(() => dialog.open);
  dialog.returnValue = 'retry';
  dialog.close();
  await waitFor(() => recovery.recovery?.status === 'queued');

  const retry = recovery.querySelector<HTMLButtonElement>('.failed-file-retry')!;
  expect(requests).to.include('POST /web/api/files/retry?workspace=personel');
  expect(retry.disabled).to.equal(true);
  expect(retry.textContent?.trim()).to.equal('Running…');
  expect(recovery.querySelector<HTMLDetailsElement>('.failed-file-recovery')?.open).to.equal(false);
});
