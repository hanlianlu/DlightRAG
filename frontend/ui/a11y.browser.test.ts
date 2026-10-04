// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {setViewport} from '@web/test-runner-commands';
import {buttonNamed, waitFor} from '../testing/dom.ts';
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
    options: {
      runOnly: {type: string; values: string[]};
      checks: Record<string, {options: Record<string, unknown>}>;
    },
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
import {getArtifactPresentationAt, type AnswerArtifact} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {productionHandles} from '../stores/app-handles.ts';
import {DEFAULT_CHANGES} from '../testing/workspaces.ts';
import './artifact-canvas.ts';
import './chat-message-list.ts';
import type {DlChatMessageList} from './chat-message-list.ts';
import './conversation-sidebar.ts';
import './inspector.ts';
import type {DlInspector} from './inspector.ts';
import './settings.ts';
import './workspace-scope.ts';
import type {ChatTurnView} from '../lib/chat-views.ts';

defineDesignSystemElements();

const SAFE_PNG =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=';

// Every kind of gold text an answer draws: a link in prose and in a table, a
// citation, an evidence image's Source button, and enough references to fold.
const answerWire = {
  answer_text: 'Retrieval finds passages.',
  parts: [{
    type: 'markdown', text: 'Retrieval finds passages.', artifact: null, evidence_image: null, inline: false,
    html: '<p>Retrieval finds passages, as <a href="https://example.com/survey">the survey</a> shows '
      + '<cite class="citation-badge" data-ref="1" role="button" tabindex="0" aria-label="Reference 1">1</cite>.</p>'
      + '<table><thead><tr><th><a href="https://example.com/methods">Method</a></th></tr></thead>'
      + '<tbody><tr><td><a href="https://example.com/bm25">BM25</a></td></tr>'
      + '<tr><td><a href="https://example.com/dense">Dense</a></td></tr></tbody></table>',
  }],
  sources: Array.from({length: 6}, (_, index) => ({
    id: String(index + 1), title: `Source ${index + 1}`, source_url: null, download_url: null, chunks: [],
  })),
  evidence_images: [{
    id: 'image-1', chunk_id: 'chunk-1', source_ref: '2', url: SAFE_PNG, thumbnail_url: SAFE_PNG,
    label: 'Figure 1', answer_image_sent: true,
  }],
  artifacts: [],
  artifact_outcome: {status: 'complete', issues: []},
};

const workspaceRecords = ['default', 'research'].map((workspace) => ({
  workspace, displayName: workspace, embeddingModel: 'embed', changes: DEFAULT_CHANGES,
}));

/** Answer every fetch with the JSON `route` returns for its URL while `run` runs. */
async function withFetch<T>(route: (url: string) => unknown, run: () => Promise<T>): Promise<T> {
  const originalFetch = window.fetch;
  window.fetch = async (input) => Response.json(route(String(input)));
  try {
    return await run();
  } finally {
    window.fetch = originalFetch;
  }
}

// The product-styles page (web-test-runner.config.mjs) links the application's
// shipped stylesheets, and components render the class names they were built with.
before(async () => {
  const token = getComputedStyle(document.documentElement).getPropertyValue('--color-bg-base');
  expect(token, 'the shipped stylesheets are applied').to.not.equal('');
  // Axe judges only what is on screen, so each fixture renders whole in a tall
  // viewport. The default width keeps the compact shell.
  await setViewport({width: 800, height: 1600});
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
      // By default axe leaves failing one-character text, such as a reference
      // id, incomplete. It is text, so it is held to the same contrast.
      checks: {'color-contrast': {options: {ignoreLength: true}}},
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
  document.body.className = '';
  document.documentElement.dataset.colorMode = 'dark';
});

it('chat message list has no serious axe violations', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  // The list holds the answer as the product parses it off the wire.
  const presentation = await withFetch(() => answerWire, () => getArtifactPresentationAt('/presentation'));
  const turn: ChatTurnView = {
    id: 'turn-a11y',
    userText: 'What is retrieval?',
    userAttachments: [],
    runId: 'run-a11y',
    state: 'succeeded',
    streamText: '',
    presentation,
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
  await waitFor(() => Boolean(list.querySelector('.citation-badge')));
  expect(await seriousViolations(list)).to.deep.equal([]);
});

it('a Markdown Artifact in the Canvas has no serious axe violations', async () => {
  const artifact: AnswerArtifact = {
    resourceId: 'report', mediaType: 'text/markdown', label: 'Report', filename: 'report.md',
    byteSize: 20, digest: 'a'.repeat(64), presentation: 'markdown', status: 'available',
    uri: 'dlightrag://answer/run-a11y/artifacts/report', width: null, height: null,
    dataUrl: '/web/api/answer/run-a11y/artifacts/report',
    downloadUrl: '/web/api/answer/run-a11y/artifacts/report?download=1',
    presentationUrl: '/web/api/answer/run-a11y/artifacts/report/presentation', issue: null,
  };
  await withFetch(() => answerWire, async () => {
    const canvas = mountInApp(document.createElement('dl-artifact-canvas'));
    canvas.className = 'panel';
    await canvas.open(artifact);
    await waitFor(() => Boolean(canvas.querySelector('.citation-badge')));
    expect(await seriousViolations(canvas)).to.deep.equal([]);
  });
});

it('conversation list has no serious axe violations with older conversations to load', async () => {
  await withFetch(() => ({
    items: ['a', 'b'].map((id) => ({
      conversation_id: id, title: `Conversation ${id}`,
      created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
    })),
    next_cursor: 'older',
  }), async () => {
    const sidebar = mountInApp(document.createElement('dl-conversation-sidebar'));
    sidebar.enabled = true;
    await productionHandles().conversations.loadList();
    await sidebar.open();
    // The shell slides the compact drawer in by classing the body.
    document.body.classList.add('conversation-drawer-open');
    await waitFor(() => Boolean(sidebar.querySelector('[data-load-older]')));
    // The list is judged on the sidebar surface; the sidebar's own Settings button
    // controls a dialog only the full shell renders.
    expect(await seriousViolations(sidebar.querySelector('dl-conversation-list')!)).to.deep.equal([]);
  });
});

it('workspace picker has no serious axe violations while open', async () => {
  const {ingest, workspaces} = productionHandles();
  workspaces.init(workspaceRecords, ['default'], 'default');
  ingest.resetToPrimary();
  const scope = mountInApp(document.createElement('dl-workspace-scope'));
  scope.className = 'workspace-selector';
  await scope.updateComplete;
  scope.querySelector<HTMLButtonElement>('#workspace-trigger')!.click();
  await scope.updateComplete;
  expect(await seriousViolations(scope)).to.deep.equal([]);
});

it('inspector has no serious axe violations when closed', async () => {
  const inspector = document.createElement('dl-inspector') as DlInspector;
  mountInApp(inspector);
  await inspector.updateComplete;
  expect(await seriousViolations(inspector)).to.deep.equal([]);
});

it('inspector Files panel has no serious axe violations', async () => {
  const {ingest, workspaces} = productionHandles();
  workspaces.init(workspaceRecords, ['default'], 'default');
  ingest.resetToPrimary();
  await withFetch((url) => (url.includes('/files/failed')
    ? {workspace: 'default', next_cursor: 'more', failed: [{
      document_id: 'scan', file_name: 'scan.pdf', error: 'Parsing failed', updated_at: '2026-01-01T00:00:00Z',
    }]}
    : {workspace: 'default', next_cursor: null, files: [{file_name: 'notes.md', file_path: 'notes.md'}]}
  ), async () => {
    const inspector = mountInApp(document.createElement('dl-inspector') as DlInspector);
    await inspector.openFiles();
    await waitFor(() => Boolean(inspector.querySelector('#ingest-target-trigger')
      && inspector.querySelector('[data-load-older]')));
    expect(await seriousViolations(inspector)).to.deep.equal([]);
  });
});

it('Settings MCP consent, credential, and create forms have no serious accessible-name or contrast violations', async () => {
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
    buttonNamed(feature, 'Add MCP connection')!.click();
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
