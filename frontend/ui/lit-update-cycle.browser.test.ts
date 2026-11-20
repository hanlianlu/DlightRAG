// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** A panel that opens or moves changes what it shows at once, and decides that before the render that
 *  draws it: setting state from `updated` draws the panel twice, and Lit says so in development.
 *
 * Lit says it once for each element, so this file is the first to open each of these three, and a warning
 * that comes back is a panel that went back to deciding its state after it had drawn. */

import {expect} from '@esm-bundle/chai';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {productionHandles} from '../stores/app-handles.ts';
import {waitFor} from '../testing/dom.ts';
import {DEFAULT_CHANGES} from '../testing/workspaces.ts';
import type {DlFailedFileRecovery} from './failed-file-recovery.ts';
import './failed-file-recovery.ts';
import type {DlInspectorFiles} from './inspector-files.ts';
import './inspector-files.ts';
import type {DlInspectorLive} from './inspector-live.ts';
import './inspector-live.ts';

defineDesignSystemElements();

const originalFetch = window.fetch;
const originalWebSocket = window.WebSocket;
const originalWarn = console.warn;
let warnings: string[] = [];

/** A frames socket that never connects: the dock opens it, and nothing else is asked of it. */
class IdleSocket {
  binaryType = 'blob';
  onmessage: unknown = null;
  onclose: ((event: {code: number; reason: string}) => void) | null = null;
  onerror: unknown = null;
  close(code = 1000, reason = ''): void { this.onclose?.({code, reason}); }
}

beforeEach(() => {
  warnings = [];
  console.warn = (...args: unknown[]) => { warnings.push(args.map(String).join(' ')); };
  (window as unknown as {WebSocket: unknown}).WebSocket = IdleSocket;
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    const workspace = url.searchParams.get('workspace');
    return Response.json(url.pathname === '/web/api/files/failed'
      ? {workspace, failed: []}
      : {workspace, files: []});
  };
  productionHandles().workspaces.init(
    ['alpha', 'beta'].map((workspace) => ({
      workspace, displayName: workspace, embeddingModel: 'embed', changes: DEFAULT_CHANGES,
    })),
    ['alpha'],
    'alpha',
  );
  productionHandles().ingest.resetToPrimary();
});

afterEach(() => {
  console.warn = originalWarn;
  window.fetch = originalFetch;
  (window as unknown as {WebSocket: unknown}).WebSocket = originalWebSocket;
  document.body.replaceChildren();
});

const settled = (): Promise<unknown> => new Promise((resolve) => setTimeout(resolve, 30));

function updateWarnings(): string[] {
  return warnings.filter((warning) => warning.includes('change-in-update'));
}

describe('opening a panel and moving it', () => {
  it('draws the live dock once for opening, moving to another Run, and closing', async () => {
    const dock = document.createElement('dl-inspector-live') as DlInspectorLive;
    document.body.append(dock);
    dock.runId = 'run-1';
    dock.active = true;
    await dock.updateComplete;
    await settled();
    dock.runId = 'run-2';
    await dock.updateComplete;
    await settled();
    dock.active = false;
    await dock.updateComplete;
    await settled();

    expect(updateWarnings()).to.deep.equal([]);
  });

  it('draws failed-document recovery once for opening, moving to another workspace, and closing', async () => {
    const recovery = document.createElement('dl-failed-file-recovery') as DlFailedFileRecovery;
    document.body.append(recovery);
    recovery.workspace = 'alpha';
    recovery.active = true;
    await recovery.updateComplete;
    // The first render already says it is checking: the state was set before it, not after.
    expect(recovery.loading).to.equal(true);
    expect(recovery.textContent).to.contain('Checking document status');
    await waitFor(() => !recovery.loading);
    recovery.workspace = 'beta';
    await recovery.updateComplete;
    await settled();
    recovery.active = false;
    await recovery.updateComplete;
    await settled();

    expect(updateWarnings()).to.deep.equal([]);
  });

  it('draws the Files panel once for opening, moving to another workspace, and closing', async () => {
    const files = document.createElement('dl-inspector-files') as DlInspectorFiles;
    document.body.append(files);
    files.active = true;
    await files.updateComplete;
    // The first render already says it is loading.
    expect(files.textContent).to.contain('Loading files');
    await waitFor(() => !files.loading);
    productionHandles().ingest.set('beta');
    await files.updateComplete;
    await settled();
    files.active = false;
    await files.updateComplete;
    await settled();

    expect(updateWarnings()).to.deep.equal([]);
  });
});
