// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {type DlSplitLayout, defineDesignSystemElements } from '../design-system/index.ts';
import './app.ts';
import type {DlApp} from './app.ts';
import {initializeBrowserAdapters} from './browser-adapters.ts';

defineDesignSystemElements();

const bootstrap = {
  contract_version: 1,
  workspaces: [
    {workspace: 'default', display_name: 'Default', embedding_model: 'embed-test'},
  ],
  workspaces_next_cursor: null,
  primary_workspace: 'default',
  active_workspaces: ['default'],
  known_workspaces: ['default'],
  answer_attachments: {
    count_limit: 6,
    image_max_bytes: 1024,
    document_max_bytes: 2048,
    extensions: ['md', 'pdf'],
    image_capability: 'supported',
    image_limit: 3,
    accept: 'image/*,.md,.pdf',
  },
  active_html_preview_enabled: true,
} as const;

afterEach(() => {
  document.body.replaceChildren();
  document.body.className = '';
});

it('composed desktop shell exposes a hittable conversation divider', async () => {
  window.matchMedia = ((query: string) => ({
    matches: query === '(min-width: 1200px)',
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  })) as typeof window.matchMedia;
  window.fetch = async (input) => new Response(
    JSON.stringify(String(input) === '/web/api/conversations'
      ? {items: [], next_cursor: null}
      : bootstrap),
    {headers: {'Content-Type': 'application/json'}},
  );

  const app = document.createElement('dl-app') as DlApp;
  app.style.display = 'block';
  app.style.width = '1440px';
  app.style.height = '900px';
  document.body.append(app);
  await app.ready;
  await initializeBrowserAdapters(app);
  await app.querySelector('dl-conversation-sidebar')?.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));

  const split = app.querySelector<DlSplitLayout>('#conversation-split')!;
  const sidebar = app.querySelector('#conversation-sidebar')!;
  const dump = {
    disabled: split.disabled,
    collapsed: split.hasAttribute('data-collapsed'),
    size: split.size,
    openClass: sidebar.classList.contains('open'),
    pointerEvents: getComputedStyle(split.divider).pointerEvents,
    label: split.divider.getAttribute('aria-label'),
    bodyOpen: document.body.classList.contains('conversation-sidebar-open'),
  };
  expect(dump, JSON.stringify(dump)).to.include({
    disabled: false,
    collapsed: false,
    openClass: true,
  });
  expect(split.size).to.be.at.least(240);
  expect(getComputedStyle(split.divider).pointerEvents).to.not.equal('none');

  const before = split.size;
  split.divider.focus();
  split.divider.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}));
  expect(split.size).to.be.greaterThan(before);
});
