// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {defineDesignSystemElements, type DlSplitLayout} from '../design-system/index.ts';
import {initializeBrowserAdapters} from './browser_adapters.ts';
import type {DlApp} from './app.ts';

defineDesignSystemElements();

describe('cold browser adapter initialization', () => {
  it('initializes both split layouts and schedules same-origin MathJax once app rendering is ready', async () => {
    let runIdle = (): void => {};
    const originalIdle = window.requestIdleCallback;
    const originalMatchMedia = window.matchMedia;
    window.matchMedia = ((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    })) as typeof window.matchMedia;
    window.requestIdleCallback = ((callback: IdleRequestCallback) => {
      runIdle = () => callback({didTimeout: false, timeRemaining: () => 10});
      return 1;
    }) as typeof window.requestIdleCallback;
    document.documentElement.style.setProperty('--panel-width', '420px');
    document.documentElement.style.setProperty('--artifact-canvas-width', '420px');
    document.body.classList.add('artifact-canvas-open');
    document.body.innerHTML = `
      <dl-split-layout id="panel-split" primary="end" size="0"
        style="display:block;width:1400px;height:500px">
        <dl-split-layout id="artifact-canvas-split" primary="end" size="0" slot="start">
          <div slot="start"></div>
          <div slot="end" id="artifact-canvas" class="open"></div>
        </dl-split-layout>
        <div slot="end" id="inspector" class="open"></div>
      </dl-split-layout>
    `;
    const app = {ready: Promise.resolve()} as unknown as DlApp;

    await initializeBrowserAdapters(app);
    const files = document.querySelector<DlSplitLayout>('#panel-split')!;
    const artifact = document.querySelector<DlSplitLayout>('#artifact-canvas-split')!;
    await new Promise((resolve) => requestAnimationFrame(resolve));

    expect(files.size).to.equal(420);
    expect(artifact.size).to.equal(420);
    expect(files.max).to.be.within(458, 460);
    expect(files.divider.getAttribute('aria-label')).to.equal('Resize Files or Sources');

    files.size = 320;
    files.dispatchEvent(new CustomEvent('dl-split-input', {
      bubbles: true,
      composed: true,
      detail: {position: 320},
    }));
    expect(artifact.max).to.be.within(558, 560);
    artifact.size = 550;
    artifact.dispatchEvent(new CustomEvent('dl-split-input', {
      bubbles: true,
      composed: true,
      detail: {position: 550},
    }));
    expect(files.max).to.be.within(328, 330);
    artifact.size = 320;
    artifact.dispatchEvent(new CustomEvent('dl-split-input', {
      bubbles: true,
      composed: true,
      detail: {position: 320},
    }));
    files.size = 500;
    expect(files.size).to.equal(500);

    runIdle();
    expect(window.MathJax).to.exist;
    expect(document.querySelector<HTMLScriptElement>('script[src*="/static/vendor/mathjax/"]')).to.exist;
    window.requestIdleCallback = originalIdle;
    window.matchMedia = originalMatchMedia;
  });
});
