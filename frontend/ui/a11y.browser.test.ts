// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {waitFor} from '../testing/dom.ts';
import {
  memoryPage,
  memorySettings,
  mountSettings,
  openSettings,
  wire,
} from '../testing/settings.ts';
type Axe = {
  run: (
    root: HTMLElement,
    options: {runOnly: {type: string; values: string[]}},
  ) => Promise<{violations: {id: string; impact?: string | null}[]}>;
};

async function loadAxe(): Promise<Axe> {
  const existing = (window as unknown as {axe?: Axe}).axe;
  if (existing?.run) return existing;
  await new Promise<void>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = new URL('../node_modules/axe-core/axe.min.js', import.meta.url).href;
    script.addEventListener('load', () => resolve(), {once: true});
    script.addEventListener('error', () => reject(new Error('axe-core failed to load')), {once: true});
    document.head.append(script);
  });
  const loaded = (window as unknown as {axe?: Axe}).axe;
  if (!loaded?.run) throw new Error('axe-core run() is unavailable');
  return loaded;
}
import {defineDesignSystemElements} from '../design-system/index.ts';
import './chat-message-list.ts';
import type {DlChatMessageList} from './chat-message-list.ts';
import './inspector.ts';
import type {DlInspector} from './inspector.ts';
import type {ChatTurnView} from '../lib/chat-views.ts';

defineDesignSystemElements();

/** Known historical issues; new serious/critical ids must not be added. */
const ALLOWED_SERIOUS = new Set<string>([]);

async function seriousIds(root: HTMLElement): Promise<string[]> {
  const axe = await loadAxe();
  const results = await axe.run(root, {
    runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa']},
  });
  return results.violations
    .filter((item) => item.impact === 'serious' || item.impact === 'critical')
    .map((item) => item.id)
    .filter((id) => !ALLOWED_SERIOUS.has(id));
}

afterEach(() => {
  document.body.replaceChildren();
});

it('chat message list has no new serious axe violations', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const turn: ChatTurnView = {
    id: 'turn-a11y',
    userText: 'What is retrieval?',
    userAttachments: [],
    runId: 'run-a11y',
    state: 'succeeded',
    streamText: '',
    presentation: {
      answerText: 'Retrieval finds passages.',
      parts: [{
        type: 'markdown',
        text: 'Retrieval finds passages.',
        html: '<p>Retrieval finds passages.</p>',
        artifact: null,
        evidenceImage: null,
        inline: false,
      }],
      sources: [],
      evidenceImages: [], linkCards: [],
      artifacts: [],
      artifactOutcome: {status: 'complete', issues: []},
    },
    usage: {},
    error: '',
    progress: '',
    liveStatus: '',
    sawChildren: false,
    cancelRequested: false,
    steeringMessages: [],
    toolRows: [],
  };
  list.turns = [turn];
  document.body.append(list);
  await list.updateComplete;
  expect(await seriousIds(list)).to.deep.equal([]);
});

it('inspector has no new serious axe violations when closed', async () => {
  const inspector = document.createElement('dl-inspector') as DlInspector;
  document.body.append(inspector);
  await inspector.updateComplete;
  expect(await seriousIds(inspector)).to.deep.equal([]);
});

it('Settings MCP consent and credential forms have no serious accessible-name or contrast violations', async () => {
  await import('./settings-connections.ts');
  const originalFetch = window.fetch;
  window.fetch = async () => Response.json({revision: '1', connections: [{
    connection_id: 'a', label: 'Personal tools', endpoint: 'https://fixture.example/mcp',
    enabled: false, activation_epoch: 1, generation: 1, authentication: 'oauth',
    status: 'needs-auth', authorization_status: 'failed',
  }], presets: [{preset_id: 'notion', label: 'Notion', endpoint: 'https://mcp.notion.com/mcp', default_authentication: 'oauth'}]});
  try {
    const feature = document.createElement('dl-settings-connections');
    document.body.append(feature);
    await waitFor(() => Boolean(feature.view));
    await feature.updateComplete;
    feature.querySelector<HTMLButtonElement>('[data-switch="a"]')!.click();
    await feature.updateComplete;
    expect(await seriousIds(feature)).to.deep.equal([]);
  } finally {window.fetch = originalFetch;}
});

it('every page of the Settings dialog has no serious accessible-name or structure violations', async () => {
  const originalFetch = window.fetch;
  window.fetch = wire({
    'GET /web/api/connections/mcp': () => Response.json({revision: '1', presets: [], connections: [{
      connection_id: 'a', label: 'Personal tools', endpoint: 'https://fixture.example/mcp',
      enabled: true, activation_epoch: 1, generation: 1, authentication: 'bearer',
      status: 'degraded', authorization_status: null,
    }]}),
    'GET /web/api/memory/settings': () => memorySettings(true, 2),
    'GET /web/api/memory': () => memoryPage([
      {id: 'one', body: 'Use concise answers'}, {id: 'two', kind: 'fact', body: 'Lives in Sweden'},
    ]),
  }).fetch;
  try {
    const {settings} = mountSettings();
    const dialog = await openSettings(settings);
    for (const section of ['connections', 'memory', 'conversations', 'language']) {
      const row = settings.querySelector<HTMLButtonElement>(`nav [data-section="${section}"]`)!;
      row.click();
      await settings.updateComplete;
      if (section === 'connections') {
        const feature = settings.querySelector('dl-settings-connections')!;
        await waitFor(() => Boolean(feature.view));
        await feature.updateComplete;
        feature.querySelector<HTMLButtonElement>('[data-card="a"]')!.click();
        await feature.updateComplete;
      } else if (section === 'memory') {
        await waitFor(() => settings.querySelectorAll('dl-settings-memory li').length === 2);
      }
      expect(await seriousIds(dialog), section).to.deep.equal([]);
    }
  } finally {window.fetch = originalFetch;}
});
