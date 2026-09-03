// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
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
      evidenceImages: [],
      artifacts: [],
      artifactOutcome: {status: 'complete', issues: []},
    },
    usage: {},
    evidence: {},
    error: '',
    progress: '',
    liveStatus: '',
    sawChildren: false,
    cancelRequested: false,
    steeringMessages: [],
    toolRows: [],
    toolTotal: 0,
    toolExpanded: false,
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
