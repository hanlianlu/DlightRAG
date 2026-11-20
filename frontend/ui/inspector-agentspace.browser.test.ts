// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Agentspace Inspector: the Run timeline of saved copies, what each copy changed, one file's
 *  versions as a second grain, and the quiet reset handshake. */

import {expect} from '@esm-bundle/chai';
import {waitFor} from '../testing/dom.ts';
import type {DlInspectorAgentspace} from './inspector-agentspace.ts';
import './inspector-agentspace.ts';

const originalFetch = window.fetch;

const handles = () => ({
  conversations: {
    activeConversationId: 'c-1',
    subscribe: () => () => {},
  },
}) as unknown as DlInspectorAgentspace['handles'];

const RUN_ONE = 'run-one-00000000000000000000000000';
const RUN_TWO = 'run-two-00000000000000000000000000';

function serve(): string[] {
  const requests: string[] = [];
  window.fetch = async (input, init) => {
    const request = new Request(new URL(String(input), 'http://localhost'), init);
    const url = new URL(request.url);
    requests.push(`${request.method} ${url.pathname}${url.search}`);
    if (request.method === 'POST') return new Response(null, {status: 204});
    if (url.pathname.includes('/workspace/snapshots/')) {
      return Response.json({
        snapshot: {
          run_id: RUN_TWO,
          parent_run_id: RUN_ONE,
          manifest_digest: 'b'.repeat(64),
          file_count: 2,
          created_at: '2026-08-02T10:00:00Z',
          changes: {added: 1, changed: 0, removed: 0},
        },
        files: [
          {
            relative_path: 'scripts/model.py',
            content_digest: 'c'.repeat(64),
            size_bytes: 11,
            change: 'unchanged',
          },
          {
            relative_path: 'artifacts/report.html',
            content_digest: 'd'.repeat(64),
            size_bytes: 2048,
            change: 'added',
          },
        ],
      });
    }
    if (url.pathname.endsWith('/workspace/files')) {
      return Response.json({
        relative_path: 'artifacts/report.html',
        versions: [
          {
            run_id: RUN_TWO,
            content_digest: 'd'.repeat(64),
            size_bytes: 2048,
            created_at: '2026-08-02T10:00:00Z',
            change: 'added',
          },
        ],
      });
    }
    return Response.json({
      heads: [
        {
          lane_id: 'main',
          head_run_id: RUN_TWO,
          manifest_digest: 'b'.repeat(64),
          revision: 2,
          committed_at: '2026-08-02T10:00:00Z',
        },
      ],
      snapshots: [
        {
          run_id: RUN_TWO,
          parent_run_id: RUN_ONE,
          manifest_digest: 'b'.repeat(64),
          file_count: 2,
          created_at: '2026-08-02T10:00:00Z',
          changes: {added: 1, changed: 0, removed: 0},
        },
        {
          run_id: RUN_ONE,
          parent_run_id: null,
          manifest_digest: 'a'.repeat(64),
          file_count: 1,
          created_at: '2026-08-01T10:00:00Z',
          changes: {added: 1, changed: 0, removed: 0},
        },
      ],
    });
  };
  return requests;
}

async function mount(): Promise<DlInspectorAgentspace> {
  const panel = document.createElement('dl-inspector-agentspace') as DlInspectorAgentspace;
  panel.handles = handles();
  panel.active = true;
  document.body.append(panel);
  await waitFor(() => panel.textContent?.includes('2 files') === true);
  return panel;
}

function buttonWith(panel: ParentNode, text: string): HTMLButtonElement {
  const button = [...panel.querySelectorAll('button')]
    .find((item) => item.textContent?.includes(text));
  if (!(button instanceof HTMLButtonElement)) throw new Error(`no button holds ${text}`);
  return button;
}

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
});

it('draws the saved copies as one timeline with what each copy changed', async () => {
  serve();
  const panel = await mount();

  expect(panel.textContent).to.contain('Run run-two');
  expect(panel.textContent).to.contain('Run run-one');
  expect(panel.textContent).to.contain('2 files');
  expect(panel.textContent).to.contain('+1');
  // One line of work: the branch heads are not a reader's question, so no Branches section.
  expect(panel.textContent).to.not.contain('Branches');
});

it('shows the branches only when the Session carries more than the main line', async () => {
  serve();
  const panel = await mount();
  window.fetch = async () => Response.json({
    heads: [
      {
        lane_id: 'main',
        head_run_id: RUN_TWO,
        manifest_digest: 'b'.repeat(64),
        revision: 2,
        committed_at: '2026-08-02T10:00:00Z',
      },
      {
        lane_id: 'fork-1',
        head_run_id: RUN_ONE,
        manifest_digest: 'a'.repeat(64),
        revision: 1,
        committed_at: '2026-08-01T10:00:00Z',
      },
    ],
    snapshots: [],
  });
  panel.active = false;
  panel.active = true;

  await waitFor(() => panel.textContent?.includes('Branches') === true);
  expect(panel.textContent).to.contain('fork-1');
});

it('opens one copy in place, separates changed files from the ones it carried, and backs out', async () => {
  const requests = serve();
  const panel = await mount();

  const row = buttonWith(panel, '2 files');
  row.click();
  await waitFor(() => panel.textContent?.includes('artifacts/report.html') === true);
  expect(row.getAttribute('aria-expanded')).to.equal('true');
  expect(panel.textContent).to.contain('Changed in this Run');
  expect(panel.querySelector('[data-change="added"]')).to.not.equal(null);
  // The carried files wait behind their own disclosure.
  expect(panel.querySelector('details')).to.not.equal(null);
  expect(panel.textContent).to.contain('1 unchanged file');

  buttonWith(panel, 'artifacts/report.html').click();
  await waitFor(() => panel.textContent?.includes('Every saved copy that carried this file') === true);
  expect(panel.textContent).to.contain('Run run-two');
  expect(requests.some((request) => request.includes('/workspace/files?path='))).to.equal(true);

  buttonWith(panel, 'Back to saved copies').click();
  await waitFor(() => panel.textContent?.includes('2 files') === true);
  expect(panel.textContent).to.contain('Run run-one');
  expect(requests.some((request) => request.includes(`/workspace/snapshots/${RUN_TWO}`)))
    .to.equal(true);
});

it('asks the Shell to open the Run whose copy is open', async () => {
  serve();
  const panel = await mount();
  const opened: string[] = [];
  panel.addEventListener('dl-agentspace-open-run', (event) => {
    opened.push((event as CustomEvent<{runId: string}>).detail.runId);
  });

  buttonWith(panel, '2 files').click();
  await waitFor(() => panel.textContent?.includes('Open this Run’s agent traces') === true);
  buttonWith(panel, 'Open this Run’s agent traces').click();

  expect(opened).to.deep.equal([RUN_TWO]);
});

it('resets the agentspace after an explicit confirmation', async () => {
  const requests = serve();
  const panel = await mount();

  buttonWith(panel, 'Reset Agentspace…').click();
  await waitFor(() => panel.textContent?.includes('This deletes every saved copy') === true);
  buttonWith(panel, 'Reset Agentspace').click();

  await waitFor(() => requests.some((request) => (
    request === 'POST /web/api/conversations/c-1/workspace/reset'
  )));
});

it('follows the conversation the reader switches to', async () => {
  const requests = serve();
  const conversation = {activeConversationId: 'c-1', subscribe: () => () => {}};
  const panel = document.createElement('dl-inspector-agentspace') as DlInspectorAgentspace;
  panel.handles = {conversations: conversation} as unknown as DlInspectorAgentspace['handles'];
  panel.active = true;
  document.body.append(panel);
  await waitFor(() => panel.textContent?.includes('2 files') === true);

  conversation.activeConversationId = 'c-2';
  panel.requestUpdate();

  await waitFor(() => requests.some((request) => (
    request === 'GET /web/api/conversations/c-2/workspace'
  )));
});

it('says a conversation is needed when none is open', async () => {
  serve();
  const panel = document.createElement('dl-inspector-agentspace') as DlInspectorAgentspace;
  panel.handles = {
    conversations: {activeConversationId: null, subscribe: () => () => {}},
  } as unknown as DlInspectorAgentspace['handles'];
  document.body.append(panel);
  await panel.updateComplete;

  expect(panel.textContent).to.contain('Open a conversation');
});
