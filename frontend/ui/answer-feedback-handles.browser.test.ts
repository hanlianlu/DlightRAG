// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The thumbs under an answer read the bag of the Feature that draws them, wherever they sit: in the Chat and
 *  in a child's page in the Inspector's dock. The Shell hands the bag to the Feature, and each element between
 *  the Feature and the thumbs hands it on. */

import {expect} from '@esm-bundle/chai';
import type {ConversationTurn} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {createAppHandles, productionHandles} from '../stores/app-handles.ts';
import {judgementKey} from '../stores/kept-judgements.ts';
import {waitFor} from '../testing/dom.ts';
import {observation, roster, row, serve, sourceFor} from '../testing/traces.ts';
import type {DlChatFeature} from './chat-feature.ts';
import './chat-feature.ts';
import './inspector.ts';
import type {DlInspector} from './inspector.ts';

defineDesignSystemElements();

const originalFetch = window.fetch;

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  productionHandles().judgements.forget();
});

/** The thumb-up's state on the page `root` draws, once it has drawn. */
const thumbUpPressed = (root: ParentNode): string | null => (
  root.querySelector('[data-thumb="up"]')?.getAttribute('aria-pressed') ?? null
);

const answered: ConversationTurn = {
  turnId: 'turn-1', turnNumber: 1, answerRunId: 'run-1', submissionId: 'submission-1', status: 'succeeded',
  cancelRequested: false, userText: 'Question', assistantText: 'Answer', userAttachments: [], steeringMessages: [],
  presentation: {
    answerText: 'Answer',
    parts: [{
      type: 'markdown', text: 'Answer', html: '<p>Answer</p>', artifact: null, evidenceImage: null, inline: false,
    }],
    sources: [], evidenceImages: [], linkCards: [], artifacts: [], artifactOutcome: {status: 'complete', issues: []},
  },
  usage: {}, errorKind: null, errorMessage: null, childCount: 0, answerSequence: 7,
  feedback: null, createdAt: '2026-01-01T00:00:00Z',
};

it('shows, under an answer in the Chat, the judgement its bag kept and not the page\'s', async () => {
  const bag = createAppHandles();
  const key = judgementKey('run-1', 'main', 7);
  bag.judgements.keep(key, {value: 'up', reasons: [], note: null});
  productionHandles().judgements.keep(key, {value: 'down', reasons: ['off_topic'], note: null});
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.handles = bag;
  feature.answerFeedbackEnabled = true;
  feature.view = {kind: 'ready', conversationId: 'conversation-1', lineage: null, history: [answered]};
  document.body.append(feature);

  await waitFor(() => thumbUpPressed(feature) !== null);
  expect(thumbUpPressed(feature)).to.equal('true');
});

it('shows, on a child\'s page in the dock, the judgement its bag kept and not the page\'s', async () => {
  const bag = createAppHandles();
  const key = judgementKey('run-1', 'a', 12);
  bag.judgements.keep(key, {value: 'up', reasons: [], note: null});
  productionHandles().judgements.keep(key, {value: 'down', reasons: ['off_topic'], note: null});
  const answered = row('a', 'succeeded', {result_sequence: 12});
  serve({page: () => roster([answered]), observe: () => observation(answered)});
  const inspector = document.createElement('dl-inspector') as DlInspector;
  inspector.handles = bag;
  inspector.answerFeedbackEnabled = true;
  document.body.append(inspector);
  await inspector.openTraces(sourceFor().source);

  await waitFor(() => inspector.querySelector('[data-agent-session="a"]') !== null);
  inspector.querySelector<HTMLElement>('[data-agent-session="a"]')!.click();
  await waitFor(() => thumbUpPressed(inspector) !== null);
  expect(thumbUpPressed(inspector)).to.equal('true');
});
