// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import './failed-file-recovery.ts';
import type {DlFailedFileRecovery} from './failed-file-recovery.ts';
import {waitFor} from '../testing/dom.ts';

const originalFetch = window.fetch;
const originalSetTimeout = window.setTimeout;

function failedPage(workspace = 'personel', failed = true) {
  return {
    workspace,
    failed: failed ? [{
      document_id: 'doc-1',
      file_name: '货币、权力与人.pdf',
      error: 'technical embedding failure',
      updated_at: '2026-08-31T21:36:15',
    }] : [],
    next_cursor: null,
  };
}

function receipt() {
  return {
    run_id: 'run-retry-1',
    run_kind: 'corpus_mutation',
    lane: 'corpus_mutation',
    status: 'queued',
    status_url: '/web/api/corpus-runs/run-retry-1',
    events_url: '/web/api/corpus-runs/run-retry-1/events',
    cancel_url: '/web/api/corpus-runs/run-retry-1',
    resume_url: '/web/api/corpus-runs/run-retry-1/resume',
    workspace: 'personel',
  };
}

function terminalRun() {
  return {
    ...receipt(),
    status: 'succeeded',
    result: {action: 'retry', documents: [{document_id: 'doc-1', status: 'succeeded'}]},
  };
}

function mount(): DlFailedFileRecovery {
  const recovery = document.createElement('dl-failed-file-recovery') as DlFailedFileRecovery;
  recovery.workspace = 'personel';
  recovery.active = true;
  document.body.appendChild(recovery);
  return recovery;
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
  expect(disclosure.textContent).to.contain('1 document needs attention');
  expect(retry.textContent?.trim()).to.equal('Retry all');
  expect(retry.disabled).to.equal(false);
  expect(recovery.textContent).to.contain('technical embedding failure');
});

it('accepts one durable retry Run and polls its canonical status URL', async () => {
  const requests: Array<{url: string; method: string}> = [];
  let failedLists = 0;
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  window.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    requests.push({url, method});
    if (method === 'POST') {
      return new Response(JSON.stringify(receipt()), {
        status: 202,
        headers: {'Content-Type': 'application/json'},
      });
    }
    if (url === '/web/api/corpus-runs/run-retry-1') {
      return new Response(JSON.stringify(terminalRun()), {
        headers: {'Content-Type': 'application/json'},
      });
    }
    failedLists += 1;
    return new Response(JSON.stringify(failedPage('personel', failedLists === 1)), {
      headers: {'Content-Type': 'application/json'},
    });
  };

  const recovery = mount();
  let completed = false;
  recovery.addEventListener('dl-failed-file-recovery-complete', () => { completed = true; });
  await waitFor(() => recovery.page?.failed.length === 1);
  recovery.querySelector<HTMLButtonElement>('.failed-file-retry')?.click();
  const dialog = recovery.querySelector<HTMLDialogElement>('#retry-failed-files-dialog')!;
  await waitFor(() => dialog.open);
  dialog.returnValue = 'retry';
  dialog.close();

  await waitFor(() => completed);
  expect(recovery.recovery?.runId).to.equal('run-retry-1');
  expect(recovery.recovery?.status).to.equal('succeeded');
  expect(requests).to.deep.include({url: '/web/api/files/retry?workspace=personel', method: 'POST'});
  expect(requests).to.deep.include({url: '/web/api/corpus-runs/run-retry-1', method: 'GET'});
  expect(requests.some(({url}) => url.includes('/files/retry/run-'))).to.equal(false);
});

it('stops polling a status the Corpus Run API refuses', async () => {
  let statusReads = 0;
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  window.fetch = async (input, init) => {
    const url = String(input);
    if ((init?.method ?? 'GET') === 'POST') {
      return new Response(JSON.stringify(receipt()), {
        status: 202,
        headers: {'Content-Type': 'application/json'},
      });
    }
    if (url === '/web/api/corpus-runs/run-retry-1') {
      statusReads += 1;
      return new Response(JSON.stringify({detail: 'Corpus Mutation Run not found'}), {
        status: 404,
        headers: {'Content-Type': 'application/json'},
      });
    }
    return new Response(JSON.stringify(failedPage()), {
      headers: {'Content-Type': 'application/json'},
    });
  };

  const recovery = mount();
  await waitFor(() => recovery.page?.failed.length === 1);
  recovery.querySelector<HTMLButtonElement>('.failed-file-retry')?.click();
  const dialog = recovery.querySelector<HTMLDialogElement>('#retry-failed-files-dialog')!;
  await waitFor(() => dialog.open);
  dialog.returnValue = 'retry';
  dialog.close();

  await waitFor(() => recovery.error !== null);
  const reads = statusReads;
  for (let tick = 0; tick < 20; tick += 1) {
    await new Promise((resolve) => originalSetTimeout(resolve, 0));
  }
  expect(statusReads).to.equal(reads);
  expect(recovery.recovery).to.equal(null);
  expect(recovery.error).to.equal('Document recovery status is no longer available.');
});

function waitingRun() {
  return {
    ...receipt(),
    status: 'running',
    phase: 'waiting_for_repair',
    repair_reason: 'Inspect upstream state.',
    repair_remedy: 'Repair it, then resume.',
  };
}

async function confirmRetryAll(recovery: DlFailedFileRecovery): Promise<void> {
  recovery.querySelector<HTMLButtonElement>('.failed-file-retry')?.click();
  const dialog = recovery.querySelector<HTMLDialogElement>('#retry-failed-files-dialog')!;
  await waitFor(() => dialog.open);
  dialog.returnValue = 'retry';
  dialog.close();
}

it('offers explicit same-Run resume while waiting for operator repair', async () => {
  const requests: Array<{url: string; method: string}> = [];
  const toasts: string[] = [];
  let resumed = false;
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  window.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    requests.push({url, method});
    if (url === '/web/api/files/retry?workspace=personel') {
      return Response.json(receipt(), {status: 202});
    }
    if (url === '/web/api/corpus-runs/run-retry-1/resume') {
      resumed = true;
      return Response.json({...receipt(), status: 'queued'}, {status: 202});
    }
    if (url === '/web/api/corpus-runs/run-retry-1') {
      return Response.json(resumed ? terminalRun() : waitingRun());
    }
    return Response.json(failedPage());
  };

  const recovery = mount();
  recovery.addEventListener('dl-toast-request', (event) => { toasts.push(event.detail.message); });
  await waitFor(() => recovery.page?.failed.length === 1);
  await confirmRetryAll(recovery);
  await waitFor(() => recovery.textContent?.includes('Inspect upstream state.') ?? false);

  const resume = recovery.querySelector<HTMLButtonElement>('.failed-file-retry')!;
  expect(resume.textContent?.trim()).to.equal('Resume after repair');
  expect(resume.disabled).to.equal(false);
  expect(recovery.textContent).to.contain('Repair it, then resume.');
  const reads = requests.filter(({url}) => url === '/web/api/corpus-runs/run-retry-1').length;
  resume.click();

  await waitFor(() => toasts.includes('Document recovery finished.'));
  expect(requests).to.deep.include({url: '/web/api/corpus-runs/run-retry-1/resume', method: 'POST'});
  expect(toasts.indexOf('Corpus repair resume accepted.'))
    .to.be.lessThan(toasts.indexOf('Document recovery finished.'));
  expect(requests.filter(({url}) => url === '/web/api/corpus-runs/run-retry-1').length)
    .to.equal(reads + 1);
  expect(recovery.recovery?.status).to.equal('succeeded');
});

it('offers Retry all again once a recovery Run settles with documents still failed', async () => {
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  let completed = false;
  window.fetch = async (input) => {
    const url = String(input);
    if (url === '/web/api/files/retry?workspace=personel') {
      return Response.json(receipt(), {status: 202});
    }
    if (url === '/web/api/corpus-runs/run-retry-1') {
      return Response.json({...terminalRun(), status: 'failed'});
    }
    return Response.json(failedPage());
  };

  const recovery = mount();
  recovery.addEventListener('dl-failed-file-recovery-complete', () => { completed = true; });
  await waitFor(() => recovery.page?.failed.length === 1);
  await confirmRetryAll(recovery);
  await waitFor(() => completed);
  await recovery.updateComplete;

  const retry = recovery.querySelector<HTMLButtonElement>('.failed-file-retry')!;
  expect(recovery.recovery?.status).to.equal('failed');
  expect(retry.textContent?.trim()).to.equal('Retry all');
  expect(retry.disabled).to.equal(false);
});

it('keeps Resume available while a repair wait lists no failed rows', async () => {
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  let listed = true;
  window.fetch = async (input) => {
    const url = String(input);
    if (url.startsWith('/web/api/files/retry')) return Response.json(receipt(), {status: 202});
    if (url === '/web/api/corpus-runs/run-retry-1') return Response.json(waitingRun());
    return Response.json(failedPage('personel', listed));
  };
  const recovery = mount();
  await waitFor(() => recovery.page?.failed.length === 1);
  await confirmRetryAll(recovery);
  await waitFor(() => recovery.textContent?.includes('Resume after repair') ?? false);

  listed = false;
  await recovery.refresh(false);
  await recovery.updateComplete;
  const resume = recovery.querySelector<HTMLButtonElement>('.failed-file-retry')!;
  expect(resume.textContent?.trim()).to.equal('Resume after repair');
  expect(resume.disabled).to.equal(false);
});

it('reads the parked Run again after a refused resume and follows where it went', async () => {
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  const toasts: string[] = [];
  let resumeRefused = false;
  window.fetch = async (input) => {
    const url = String(input);
    if (url.startsWith('/web/api/files/retry')) return Response.json(receipt(), {status: 202});
    if (url === '/web/api/corpus-runs/run-retry-1/resume') {
      resumeRefused = true;
      return Response.json(
        {detail: 'Run is not waiting for repair', error_type: 'conflict'},
        {status: 409},
      );
    }
    if (url === '/web/api/corpus-runs/run-retry-1') {
      // Another operator resumed it; this reader only learns so by asking.
      return Response.json(resumeRefused ? terminalRun() : waitingRun());
    }
    return Response.json(failedPage());
  };
  const recovery = mount();
  recovery.addEventListener('dl-toast-request', (event) => { toasts.push(event.detail.message); });
  await waitFor(() => recovery.page?.failed.length === 1);
  await confirmRetryAll(recovery);
  await waitFor(() => recovery.textContent?.includes('Resume after repair') ?? false);

  recovery.querySelector<HTMLButtonElement>('.failed-file-retry')!.click();
  await waitFor(() => toasts.includes('Document recovery finished.'));

  expect(toasts.indexOf('Corpus repair resume failed.'))
    .to.be.lessThan(toasts.indexOf('Document recovery finished.'));
  expect(recovery.recovery?.status).to.equal('succeeded');
});

it('clears workspace-scoped state before loading the next workspace', async () => {
  let resolveOther!: (response: Response) => void;
  const other = new Promise<Response>((resolve) => { resolveOther = resolve; });
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.searchParams.get('workspace') === 'other') return other;
    return new Response(JSON.stringify(failedPage()), {
      headers: {'Content-Type': 'application/json'},
    });
  };

  const recovery = mount();
  await waitFor(() => recovery.page?.failed.length === 1);
  recovery.workspace = 'other';
  await recovery.updateComplete;
  expect(recovery.page).to.equal(null);
  expect(recovery.textContent).not.to.contain('货币、权力与人.pdf');

  resolveOther(new Response(JSON.stringify(failedPage('other', false)), {
    headers: {'Content-Type': 'application/json'},
  }));
  await waitFor(() => recovery.loading === false);
});

it('pages more failed documents through the shared control and announces each page', async () => {
  let moreAttempts = 0;
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (!url.searchParams.has('cursor')) {
      return Response.json({...failedPage(), next_cursor: 'more-1'});
    }
    moreAttempts += 1;
    if (moreAttempts === 1) return new Response('unavailable', {status: 503});
    return Response.json({
      workspace: 'personel',
      failed: [
        failedPage().failed[0],
        {document_id: 'doc-2', file_name: 'later.pdf', error: 'parse failure', updated_at: ''},
      ],
      next_cursor: null,
    });
  };

  const recovery = mount();
  await waitFor(() => recovery.page?.failed.length === 1);
  expect(recovery.textContent).to.contain('1+ documents need attention');
  const more = recovery.querySelector<HTMLButtonElement>('[data-load-older="failed-documents"]')!;
  expect(more.textContent?.trim()).to.equal('Load more failed documents');

  more.click();
  await waitFor(() => recovery.querySelector('[data-load-older="failed-documents"]')
    ?.textContent?.includes('Retry') ?? false);
  const status = () => recovery.querySelector('[data-load-older-status="failed-documents"]')
    ?.textContent?.trim();
  expect(status()).to.equal('More failed documents could not be loaded.');
  expect(recovery.page?.failed).to.have.length(1);

  recovery.querySelector<HTMLButtonElement>('[data-load-older="failed-documents"]')!.click();
  await waitFor(() => recovery.page?.failed.length === 2);
  await recovery.updateComplete;
  expect(recovery.querySelector('[data-load-older="failed-documents"]')).to.equal(null);
  expect(status()).to.equal('Loaded 1 more failed document.');
  expect(recovery.textContent).to.contain('2 documents need attention');
});

it('refreshes in place without dropping Load more, its count, or the reader\'s focus', async () => {
  let lists = 0;
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  window.fetch = async () => {
    lists += 1;
    if (lists > 1) await gate;
    return Response.json({...failedPage(), next_cursor: 'more-1'});
  };
  const recovery = mount();
  await waitFor(() => recovery.page !== null);
  await recovery.updateComplete;
  recovery.querySelector<HTMLDetailsElement>('details')!.open = true;
  const more = () => recovery.querySelector<HTMLButtonElement>('[data-load-older="failed-documents"]');
  const status = () => recovery.querySelector('[data-load-older-status="failed-documents"]')
    ?.textContent?.trim();
  more()!.focus();

  const refreshing = recovery.refresh(false);
  await recovery.updateComplete;
  // Compare identities: a failing DOM-node equality would stall the reporter.
  expect(more() !== null && more() === document.activeElement, 'the list keeps its next page and focus')
    .to.equal(true);
  expect(more()!.getAttribute('aria-busy')).to.equal('true');
  expect(more()!.getAttribute('aria-disabled')).to.equal('true');
  expect(status(), 'a first page is not announced as a next page').to.equal('');
  expect(recovery.textContent).to.contain('1+ documents need attention');

  release();
  await refreshing;
  await recovery.updateComplete;
  expect(lists).to.equal(2);
  expect(more() !== null && more() === document.activeElement, 'focus stays after the page lands')
    .to.equal(true);
  expect(more()!.hasAttribute('aria-disabled')).to.equal(false);
});
