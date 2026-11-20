// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The browser channel is off the render path: an SDK that cannot start is silent. */

import {expect} from '@esm-bundle/chai';
import {initializeObservability, type ObservabilityConfig} from './observability.ts';

const CONFIG: ObservabilityConfig = {
  url: 'http://127.0.0.1:4318',
  apiKey: 'ingestion-key',
  service: 'dlightrag-web',
  replaySampleRate: 1,
};

it('leaves the page rendered and quiet when the SDK cannot start', async () => {
  const page = document.createElement('div');
  page.textContent = 'DlightRAG';
  document.body.appendChild(page);
  const logged: unknown[][] = [];
  const originalConsole = {error: console.error, warn: console.warn};
  console.error = (...args: unknown[]) => void logged.push(args);
  console.warn = (...args: unknown[]) => void logged.push(args);
  const failures: unknown[] = [];
  const onError = (event: ErrorEvent): void => void failures.push(event.error ?? event.message);
  const onRejection = (event: PromiseRejectionEvent): void => void failures.push(event.reason);
  window.addEventListener('error', onError);
  window.addEventListener('unhandledrejection', onRejection);
  try {
    initializeObservability(CONFIG, () => {
      throw new Error('the SDK chunk was blocked');
    });
    initializeObservability(CONFIG, async () => {
      throw new Error('the SDK chunk was blocked');
    });
    initializeObservability(CONFIG, () =>
      Promise.resolve({default: {init: () => {
        throw new Error('the collector refused the session');
      }}}),
    );

    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(failures).to.deep.equal([]);
    expect(logged).to.deep.equal([]);
    expect(page.textContent).to.equal('DlightRAG');
  } finally {
    console.error = originalConsole.error;
    console.warn = originalConsole.warn;
    window.removeEventListener('error', onError);
    window.removeEventListener('unhandledrejection', onRejection);
    page.remove();
  }
});
