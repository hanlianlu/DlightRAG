// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {initializeBrowserAdapters} from './browser_adapters.ts';
import type {DlApp} from './app.ts';

interface TestSplitPanel extends HTMLElement {
  readonly updateComplete: Promise<boolean>;
  readonly divider: HTMLElement;
  positionInPixels: number;
}

describe('cold browser adapter initialization', () => {
  it('initializes both split panels and schedules same-origin MathJax once app rendering is ready', async () => {
    let runIdle = (): void => {};
    const originalIdle = window.requestIdleCallback;
    window.requestIdleCallback = ((callback: IdleRequestCallback) => {
      runIdle = () => callback({didTimeout: false, timeRemaining: () => 10});
      return 1;
    }) as typeof window.requestIdleCallback;
    document.documentElement.style.setProperty('--panel-width', '420px');
    document.documentElement.style.setProperty('--artifact-canvas-width', '420px');
    document.body.innerHTML = `
      <wa-split-panel id="panel-split" primary="end" position-in-pixels="0">
        <div slot="end" id="inspector" class="open"></div>
      </wa-split-panel>
      <wa-split-panel id="artifact-canvas-split" primary="end" position-in-pixels="0">
        <div slot="end" id="artifact-canvas" class="open"></div>
      </wa-split-panel>
    `;
    const app = {ready: Promise.resolve()} as unknown as DlApp;

    await initializeBrowserAdapters(app);
    const files = document.querySelector<TestSplitPanel>('#panel-split')!;
    const artifact = document.querySelector<TestSplitPanel>('#artifact-canvas-split')!;
    await Promise.all([files.updateComplete, artifact.updateComplete]);
    await new Promise((resolve) => requestAnimationFrame(resolve));

    expect(files.positionInPixels).to.equal(420);
    expect(artifact.positionInPixels).to.equal(420);
    expect(files.divider.getAttribute('aria-label')).to.equal('Resize Files or Sources');
    runIdle();
    expect(window.MathJax).to.exist;
    expect(document.querySelector<HTMLScriptElement>('script[src*="/static/vendor/mathjax/"]')).to.exist;
    window.requestIdleCallback = originalIdle;
  });
});
