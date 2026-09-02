// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import './failed-file-recovery.ts';
import type {DlFailedFileRecovery} from './failed-file-recovery.ts';

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
  await waitFor(() => recovery.page?.nextCursor === null && recovery.page.failed.length === 3);

  expect(recovery.querySelectorAll('.failed-file-row')).to.have.length(3);
  expect(recovery.querySelector('.failed-file-more-button')).to.equal(null);
});

for (const status of [401, 403, 409]) {
  it(`routes load-more ${status} through localized fatal handling`, async () => {
    window.fetch = async (input) => {
      const url = new URL(String(input), window.location.origin);
      if (url.searchParams.has('cursor')) {
        return new Response(JSON.stringify({detail: 'fatal'}), {
          status,
          headers: {'Content-Type': 'application/json'},
        });
      }
      return new Response(JSON.stringify(failedPage('older')), {
        headers: {'Content-Type': 'application/json'},
      });
    };

    const recovery = mount();
    await waitFor(() => recovery.loading === false && recovery.page !== null);
    recovery.querySelector<HTMLButtonElement>('.failed-file-more-button')?.click();
    await waitFor(() => recovery.page === null && recovery.error !== null);

    expect(recovery.querySelector('.failed-file-row')).to.equal(null);
    expect(recovery.loadMoreState).to.equal('idle');
    expect(recovery.querySelector('.failed-file-more-button')).to.equal(null);
    expect(recovery.error).to.contain(status === 409 ? 'no longer available' : 'permission');
  });
}

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

it('ignores a stale poll response after its request controller is cancelled', async () => {
  const poll = deferredResponse();
  let pollStarted = false;
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname.includes('/files/retry/')) {
      pollStarted = true;
      return poll.promise;
    }
    return new Response(JSON.stringify({
      ...failedPage(),
      active_recovery: {
        job_id: 'retry-stale', workspace: 'personel', status: 'running',
        retried: 0, succeeded: 0, failed: 0,
      },
    }), {headers: {'Content-Type': 'application/json'}});
  };

  const recovery = mount();
  await waitFor(() => pollStarted);
  // Supersede the poll controller without deactivating the component; the
  // stale response must be rejected by controller identity alone.
  await recovery.refresh(false);
  expect(recovery.active).to.equal(true);
  poll.resolve(new Response(JSON.stringify({
    job_id: 'retry-stale', workspace: 'personel', status: 'succeeded',
    retried: 1, succeeded: 1, failed: 0,
  }), {headers: {'Content-Type': 'application/json'}}));
  await new Promise((resolve) => originalSetTimeout(resolve, 0));

  expect(recovery.recovery?.status).to.equal('running');
});

it('keeps a successful retry POST across a concurrent same-workspace refresh', async () => {
  const post = deferredResponse();
  let postStarted = false;
  window.fetch = async (_input, init) => {
    if (init?.method === 'POST') {
      postStarted = true;
      return post.promise;
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
  await waitFor(() => postStarted);

  await recovery.refresh(false);
  post.resolve(new Response(JSON.stringify({
    job_id: 'retry-race',
    workspace: 'personel',
    status: 'queued',
    retried: 0,
    succeeded: 0,
    failed: 0,
  }), {headers: {'Content-Type': 'application/json'}}));
  await waitFor(() => recovery.recoveryPending === false);

  expect(recovery.recovery?.jobId).to.equal('retry-race');
  expect(recovery.querySelector<HTMLButtonElement>('.failed-file-retry')?.disabled).to.equal(true);
});

it('does not POST when an open confirmation discovers an existing recovery', async () => {
  let failedRequests = 0;
  let postRequests = 0;
  window.fetch = async (_input, init) => {
    if (init?.method === 'POST') {
      postRequests += 1;
      return new Response(null, {status: 500});
    }
    failedRequests += 1;
    return new Response(JSON.stringify(failedRequests === 1 ? failedPage() : {
      ...failedPage(),
      active_recovery: {
        job_id: 'retry-discovered', workspace: 'personel', status: 'running',
        retried: 0, succeeded: 0, failed: 0,
      },
    }), {headers: {'Content-Type': 'application/json'}});
  };

  const recovery = mount();
  await waitFor(() => recovery.loading === false && recovery.page !== null);
  recovery.querySelector<HTMLButtonElement>('.failed-file-retry')?.click();
  const dialog = recovery.querySelector<HTMLDialogElement>('#retry-failed-files-dialog')!;
  await waitFor(() => dialog.open);
  await recovery.refresh(false);
  expect(recovery.recovery?.jobId).to.equal('retry-discovered');
  dialog.returnValue = 'retry';
  dialog.close();
  await new Promise((resolve) => originalSetTimeout(resolve, 0));

  expect(postRequests).to.equal(0);
});

it('preserves a POST-created recovery when the older refresh resolves afterwards', async () => {
  const post = deferredResponse();
  const refresh = deferredResponse();
  let postStarted = false;
  let refreshStarted = false;
  window.fetch = async (_input, init) => {
    if (init?.method === 'POST') {
      postStarted = true;
      return post.promise;
    }
    if (postStarted) {
      refreshStarted = true;
      return refresh.promise;
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
  await waitFor(() => postStarted);
  const refreshFlight = recovery.refresh(false);
  await waitFor(() => refreshStarted);

  post.resolve(new Response(JSON.stringify({
    job_id: 'retry-post-first', workspace: 'personel', status: 'queued',
    retried: 0, succeeded: 0, failed: 0,
  }), {headers: {'Content-Type': 'application/json'}}));
  await waitFor(() => recovery.recovery?.jobId === 'retry-post-first');
  refresh.resolve(new Response(JSON.stringify({
    ...failedPage(),
    active_recovery: {
      job_id: 'retry-stale-page', workspace: 'personel', status: 'running',
      retried: 0, succeeded: 0, failed: 0,
    },
  }), {headers: {'Content-Type': 'application/json'}}));
  await refreshFlight;

  expect(recovery.recovery?.jobId).to.equal('retry-post-first');
});

it('preserves a POST-created recovery when delayed load-more reports an older job', async () => {
  const older = deferredResponse();
  let getRequests = 0;
  window.fetch = async (_input, init) => {
    if (init?.method === 'POST') {
      return new Response(JSON.stringify({
        job_id: 'retry-post-load', workspace: 'personel', status: 'queued',
        retried: 0, succeeded: 0, failed: 0,
      }), {headers: {'Content-Type': 'application/json'}});
    }
    getRequests += 1;
    if (getRequests === 1) {
      return new Response(JSON.stringify(failedPage('older-cursor')), {
        headers: {'Content-Type': 'application/json'},
      });
    }
    return older.promise;
  };

  const recovery = mount();
  await waitFor(() => recovery.page?.nextCursor === 'older-cursor');
  recovery.querySelector<HTMLButtonElement>('.failed-file-more-button')?.click();
  await waitFor(() => recovery.loadMoreState === 'loading');
  recovery.querySelector<HTMLButtonElement>('.failed-file-retry')?.click();
  const dialog = recovery.querySelector<HTMLDialogElement>('#retry-failed-files-dialog')!;
  await waitFor(() => dialog.open);
  dialog.returnValue = 'retry';
  dialog.close();
  await waitFor(() => recovery.recovery?.jobId === 'retry-post-load');

  older.resolve(new Response(JSON.stringify({
    ...failedPage(),
    active_recovery: {
      job_id: 'retry-old-load', workspace: 'personel', status: 'running',
      retried: 0, succeeded: 0, failed: 0,
    },
  }), {headers: {'Content-Type': 'application/json'}}));
  await waitFor(() => recovery.loadMoreState === 'idle');

  expect(recovery.recovery?.jobId).to.equal('retry-post-load');
});

it('keeps polling a known active job after refresh omits it, then observes terminal', async () => {
  let failedRequests = 0;
  let statusRequests = 0;
  let poll: (() => void) | null = null;
  window.setTimeout = ((handler: TimerHandler, delay?: number) => {
    if (delay === 2000) {
      poll = () => { if (typeof handler === 'function') handler(); };
      return 1;
    }
    return originalSetTimeout(handler, delay);
  }) as typeof window.setTimeout;
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname.includes('/files/retry/')) {
      statusRequests += 1;
      return new Response(JSON.stringify({
        job_id: 'retry-finished-during-refresh', workspace: 'personel', status: 'succeeded',
        retried: 2, succeeded: 2, failed: 0,
      }), {headers: {'Content-Type': 'application/json'}});
    }
    failedRequests += 1;
    return new Response(JSON.stringify(failedRequests === 1 ? {
      ...failedPage(),
      active_recovery: {
        job_id: 'retry-finished-during-refresh', workspace: 'personel', status: 'running',
        retried: 0, succeeded: 0, failed: 0,
      },
    } : {...failedPage(), failed: [], active_recovery: null}), {
      headers: {'Content-Type': 'application/json'},
    });
  };

  const recovery = mount();
  await waitFor(() => recovery.recovery?.status === 'running' && poll !== null);
  await recovery.refresh(false);
  expect(recovery.recovery?.status).to.equal('running');
  expect(statusRequests).to.equal(0);
  (poll as unknown as () => void)();
  await waitFor(() => recovery.recovery?.status === 'succeeded');
  await waitFor(() => recovery.page?.failed.length === 0);

  expect(statusRequests).to.equal(1);
});

it('keeps Retry all disabled while terminal recovery settlement refreshes stale rows', async () => {
  const settlement = deferredResponse();
  let failedRequests = 0;
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname.includes('/files/retry/')) {
      return new Response(JSON.stringify({
        job_id: 'retry-settle', workspace: 'personel', status: 'succeeded',
        retried: 2, succeeded: 2, failed: 0,
      }), {headers: {'Content-Type': 'application/json'}});
    }
    failedRequests += 1;
    if (failedRequests > 1) return settlement.promise;
    return new Response(JSON.stringify({
      ...failedPage(),
      active_recovery: {
        job_id: 'retry-settle', workspace: 'personel', status: 'running',
        retried: 0, succeeded: 0, failed: 0,
      },
    }), {headers: {'Content-Type': 'application/json'}});
  };

  const recovery = mount();
  await waitFor(() => recovery.recovery?.status === 'succeeded');
  await recovery.updateComplete;

  expect(recovery.querySelector<HTMLButtonElement>('.failed-file-retry')?.disabled).to.equal(true);
  settlement.resolve(new Response(JSON.stringify({...failedPage(), failed: []}), {
    headers: {'Content-Type': 'application/json'},
  }));
  await waitFor(() => recovery.page?.failed.length === 0);
});

it('suppresses terminal completion after switching workspace during settlement', async () => {
  const settlement = deferredResponse();
  let failedRequests = 0;
  let completed = 0;
  let toasts = 0;
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.searchParams.get('workspace') === 'other') {
      return new Response(JSON.stringify({
        workspace: 'other', failed: [], next_cursor: null, active_recovery: null,
      }), {headers: {'Content-Type': 'application/json'}});
    }
    if (url.pathname.includes('/files/retry/')) {
      return new Response(JSON.stringify({
        job_id: 'retry-old-workspace', workspace: 'personel', status: 'succeeded',
        retried: 1, succeeded: 1, failed: 0,
      }), {headers: {'Content-Type': 'application/json'}});
    }
    failedRequests += 1;
    if (failedRequests > 1) return settlement.promise;
    return new Response(JSON.stringify({
      ...failedPage(),
      active_recovery: {
        job_id: 'retry-old-workspace', workspace: 'personel', status: 'running',
        retried: 0, succeeded: 0, failed: 0,
      },
    }), {headers: {'Content-Type': 'application/json'}});
  };

  const recovery = mount();
  recovery.addEventListener('dl-failed-file-recovery-complete', () => { completed += 1; });
  recovery.addEventListener('dl-toast-request', () => { toasts += 1; });
  await waitFor(() => recovery.recovery?.status === 'succeeded');
  recovery.workspace = 'other';
  await recovery.updateComplete;
  settlement.resolve(new Response(JSON.stringify({...failedPage(), failed: []}), {
    headers: {'Content-Type': 'application/json'},
  }));
  await new Promise((resolve) => originalSetTimeout(resolve, 0));

  expect(completed).to.equal(0);
  expect(toasts).to.equal(0);
  expect(recovery.workspace).to.equal('other');
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
