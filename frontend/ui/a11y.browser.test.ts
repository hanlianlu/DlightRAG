// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {waitFor} from '../testing/dom.ts';
import {
  agentAccountsView,
  memoryPage,
  memorySettings,
  mountSettings,
  openSettings,
  wire,
  wireAccount,
} from '../testing/settings.ts';
type AxeNode = {
  target: string[];
  any: {data?: {fgColor?: string; bgColor?: string; contrastRatio?: number} | null}[];
};
type Axe = {
  run: (
    root: HTMLElement,
    options: {runOnly: {type: string; values: string[]}},
  ) => Promise<{violations: {id: string; impact?: string | null; nodes: AxeNode[]}[]}>;
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
import './settings.ts';
import type {ChatTurnView} from '../lib/chat-views.ts';

defineDesignSystemElements();

// The product-styles page (web-test-runner.config.mjs) links the shipped
// stylesheet, and components render the class names it was built with.
before(() => {
  const token = getComputedStyle(document.documentElement).getPropertyValue('--color-bg-base');
  expect(token, 'the shipped stylesheet is applied').to.not.equal('');
});

/** Finish running transitions, so axe reads each color at its settled value. */
function settleTransitions(): void {
  for (const animation of document.getAnimations()) {
    if (animation instanceof CSSTransition) animation.finish();
  }
}

function describeViolation(colorMode: string, rule: string, node: AxeNode): string {
  const contrast = node.any.find((check) => check.data?.contrastRatio)?.data;
  const measured = contrast ? ` (${contrast.fgColor} on ${contrast.bgColor}, ${contrast.contrastRatio}:1)` : '';
  return `${colorMode}: ${rule} ${node.target.join(' ')}${measured}`;
}

/** Serious and critical WCAG A/AA violations, judged in each color mode. */
async function seriousViolations(root: HTMLElement): Promise<string[]> {
  const axe = await loadAxe();
  const found: string[] = [];
  for (const colorMode of ['dark', 'light']) {
    document.documentElement.dataset.colorMode = colorMode;
    settleTransitions();
    const results = await axe.run(root, {
      runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa']},
    });
    for (const violation of results.violations) {
      if (violation.impact !== 'serious' && violation.impact !== 'critical') continue;
      found.push(...violation.nodes.map((node) => describeViolation(colorMode, violation.id, node)));
    }
  }
  return found;
}

/** Mount `element` in the app container, which sizes and clips every product surface. */
function mountInApp<T extends HTMLElement>(element: T): T {
  const app = document.createElement('div');
  app.className = 'app';
  app.append(element);
  document.body.append(app);
  return element;
}

afterEach(() => {
  document.body.replaceChildren();
  document.documentElement.dataset.colorMode = 'dark';
});

it('chat message list has no serious axe violations', async () => {
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
  mountInApp(list);
  await list.updateComplete;
  expect(await seriousViolations(list)).to.deep.equal([]);
});

it('inspector has no serious axe violations when closed', async () => {
  const inspector = document.createElement('dl-inspector') as DlInspector;
  mountInApp(inspector);
  await inspector.updateComplete;
  expect(await seriousViolations(inspector)).to.deep.equal([]);
});

it('Settings MCP consent and credential forms have no serious accessible-name or contrast violations', async () => {
  const originalFetch = window.fetch;
  window.fetch = async (input) => String(input).includes('/connections/')
    ? Response.json({revision: '1', connections: [{
      connection_id: 'a', label: 'Personal tools', endpoint: 'https://fixture.example/mcp',
      enabled: false, activation_epoch: 1, generation: 1, authentication: 'oauth',
      status: 'needs-auth', authorization_status: 'failed',
    }], presets: [{preset_id: 'notion', label: 'Notion', endpoint: 'https://mcp.notion.com/mcp', default_authentication: 'oauth'}]})
    : Response.json({enabled: true, active_count: 0});
  try {
    // Connections renders on the Settings dialog surface, so it is judged there.
    const settings = mountInApp(document.createElement('dl-settings-dialog'));
    await settings.open();
    const feature = settings.querySelector('dl-settings-connections')!;
    await waitFor(() => Boolean(feature.view));
    await feature.updateComplete;
    feature.querySelector<HTMLButtonElement>('[data-card="a"]')!.click();
    await feature.updateComplete;
    const credentialForm = await seriousViolations(feature);
    // The consent modal makes the rest of Settings inert, so it is judged on its own.
    feature.querySelector<HTMLButtonElement>('[data-switch="a"]')!.click();
    await feature.updateComplete;
    const consent = await seriousViolations(feature);
    expect([...credentialForm, ...consent]).to.deep.equal([]);
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
    'GET /web/api/agent-accounts': () => Response.json(agentAccountsView([
      wireAccount('discourse.org', {last_used_at: '2026-10-04T15:13:01Z'}),
      wireAccount('ycombinator.com', {email: null, username: null}),
    ])),
    'GET /web/api/memory/settings': () => memorySettings(true, 2),
    'GET /web/api/memory': () => memoryPage([
      {id: 'one', body: 'Use concise answers'}, {id: 'two', kind: 'fact', body: 'Lives in Sweden'},
    ]),
  }).fetch;
  try {
    const {settings} = mountSettings();
    const dialog = await openSettings(settings);
    for (const section of ['connections', 'agent-accounts', 'memory', 'conversations', 'language']) {
      const row = settings.querySelector<HTMLButtonElement>(`nav [data-section="${section}"]`)!;
      row.click();
      await settings.updateComplete;
      if (section === 'connections') {
        const feature = settings.querySelector('dl-settings-connections')!;
        await waitFor(() => Boolean(feature.view));
        await feature.updateComplete;
        feature.querySelector<HTMLButtonElement>('[data-card="a"]')!.click();
        await feature.updateComplete;
      } else if (section === 'agent-accounts') {
        const feature = settings.querySelector('dl-settings-agent-accounts')!;
        await waitFor(() => Boolean(feature.view));
        await feature.updateComplete;
      } else if (section === 'memory') {
        await waitFor(() => settings.querySelectorAll('dl-settings-memory li').length === 2);
      }
      expect(await seriousViolations(dialog), section).to.deep.equal([]);
    }
  } finally {window.fetch = originalFetch;}
});
