// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {type DlSplitLayout, defineDesignSystemElements } from '../design-system/index.ts';
import {setupPanelSplits} from './split-panel.ts';

defineDesignSystemElements();

const originalMatchMedia = window.matchMedia;

afterEach(() => {
  document.body.replaceChildren();
  document.body.className = '';
  window.matchMedia = originalMatchMedia;
});

it('desktop conversation split is enabled and keyboard-resizable', async () => {
  window.matchMedia = ((query: string) => ({
    matches: query === '(width < 1200px)' ? false : query === '(min-width: 1200px)',
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  })) as typeof window.matchMedia;
  document.documentElement.style.setProperty('--layout-chat-sidebar-width', '260px');
  document.body.innerHTML = `
    <dl-split-layout id="conversation-split" primary="start" size="0" min="240"
      style="display:block;width:1440px;height:500px">
      <div id="conversation-sidebar" class="open" slot="start"></div>
      <dl-split-layout id="panel-split" slot="end" primary="end" size="0" min="320">
        <div slot="start"></div>
        <div slot="end" id="inspector"></div>
      </dl-split-layout>
    </dl-split-layout>
  `;

  setupPanelSplits();
  await new Promise((resolve) => requestAnimationFrame(resolve));

  const split = document.querySelector<DlSplitLayout>('#conversation-split')!;
  expect(split.disabled, 'conversation split should not be disabled on desktop').to.equal(false);
  expect(split.hasAttribute('data-collapsed')).to.equal(false);
  expect(split.size).to.be.at.least(240);
  expect(split.divider.getAttribute('aria-label')).to.equal('Resize conversations');
  expect(getComputedStyle(split.divider).pointerEvents).to.not.equal('none');

  const before = split.size;
  split.divider.focus();
  split.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}));
  expect(split.size).to.be.greaterThan(before);
});
