// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** A Feature handed a bag of stores of its own uses that bag and no other.
 *
 * `productionHandles()` is what a Feature starts with, and the Shell hands it the same bag. A test, or a
 * second Shell, hands a Feature another one by property, after the Feature was built: what it reads, what it
 * follows for change, and what it writes must all be that bag's, or the Feature would draw one set of stores
 * and answer to another.
 */

import {expect} from '@esm-bundle/chai';
import type {LitElement} from 'lit';
import type {ConversationSummary, ConversationTurn} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {type AppHandles, createAppHandles, productionHandles} from '../stores/app-handles.ts';
import {judgementKey} from '../stores/kept-judgements.ts';
import {waitFor} from '../testing/dom.ts';
import {DEFAULT_CHANGES} from '../testing/workspaces.ts';
import type {DlAnswerFeedback} from './answer-feedback.ts';
import './answer-feedback.ts';
import type {DlChatFeature} from './chat-feature.ts';
import './chat-feature.ts';
import './conversation-list.ts';
import './conversation-sidebar.ts';
import './ingest-target.ts';
import './inspector-files.ts';
import './settings-conversations.ts';
import './workspace-scope.ts';

defineDesignSystemElements();

const originalFetch = window.fetch;

function summary(id: string, title: string): ConversationSummary {
  return {
    conversationId: id,
    title,
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-02T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
    schedule: null,
  };
}

function workspaces(bag: AppHandles, ...names: string[]): void {
  bag.workspaces.init(
    names.map((workspace) => ({
      workspace, displayName: workspace.toUpperCase(), embeddingModel: 'embed', changes: DEFAULT_CHANGES,
    })),
    [names[0]!],
    names[0],
  );
}

/** How many times a store asked `element` to draw again: the one call a store follower makes. */
function redrawsOf(element: LitElement): () => number {
  const request = element.requestUpdate.bind(element);
  let asked = 0;
  element.requestUpdate = ((...args: Parameters<LitElement['requestUpdate']>) => {
    if (args.length === 0) asked += 1;
    return request(...args);
  }) as LitElement['requestUpdate'];
  return () => asked;
}

type Handed = LitElement & {handles: AppHandles};

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  productionHandles().attachments.clear();
});

describe('the Features that follow stores', () => {
  /** What changes in each Feature's stores, in the way that bag publishes it. */
  const FEATURES: Array<{tag: string; change: (bag: AppHandles) => void}> = [
    {tag: 'dl-conversation-sidebar', change: (bag) => bag.conversations.upsertSummary(summary('c', 'C'))},
    {tag: 'dl-conversation-list', change: (bag) => bag.conversations.upsertSummary(summary('c', 'C'))},
    {tag: 'dl-settings-conversations', change: (bag) => bag.conversations.upsertSummary(summary('c', 'C'))},
    {tag: 'dl-workspace-scope', change: (bag) => bag.workspaces.select('beta')},
    {tag: 'dl-ingest-target', change: (bag) => bag.ingest.set('beta')},
    {tag: 'dl-inspector-files', change: (bag) => bag.ingest.set('beta')},
    {tag: 'dl-chat-composer', change: (bag) => { bag.attachments.add(new File(['notes'], 'notes.md'), 'document'); }},
  ];

  for (const {tag, change} of FEATURES) {
    it(`${tag} draws again when the bag it was handed changes, and not when the page's own does`, async () => {
      const bag = createAppHandles();
      workspaces(bag, 'alpha', 'beta');
      workspaces(productionHandles(), 'gamma', 'beta');
      const element = document.createElement(tag) as Handed;
      element.handles = bag;
      document.body.append(element);
      await element.updateComplete;
      const redraws = redrawsOf(element);

      change(productionHandles());
      await element.updateComplete;
      expect(redraws(), 'the page\'s own stores were followed').to.equal(0);

      change(bag);
      await element.updateComplete;
      expect(redraws(), 'the stores it was handed were not').to.equal(1);
    });
  }

  it('files lists the workspace its bag points at, and reloads when that bag points elsewhere', async () => {
    const bag = createAppHandles();
    workspaces(bag, 'alpha', 'beta');
    workspaces(productionHandles(), 'gamma', 'beta');
    const asked: string[] = [];
    window.fetch = async (input) => {
      const url = new URL(String(input), window.location.origin);
      if (url.pathname === '/web/api/files') asked.push(url.searchParams.get('workspace') ?? '');
      return Response.json({workspace: url.searchParams.get('workspace'), files: [], failed: []});
    };
    const workspacesAsked = (): string[] => [...new Set(asked)];
    const files = document.createElement('dl-inspector-files') as unknown as Handed & {active: boolean};
    files.handles = bag;
    files.active = true;
    document.body.append(files);
    await waitFor(() => asked.length > 0);
    expect(workspacesAsked(), 'it read the workspace its bag points at').to.deep.equal(['alpha']);

    productionHandles().ingest.set('delta');
    await files.updateComplete;
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(workspacesAsked(), 'and not the page\'s').to.deep.equal(['alpha']);

    bag.ingest.set('beta');
    await waitFor(() => asked.includes('beta'));
    expect(workspacesAsked()).to.deep.equal(['alpha', 'beta']);
  });
});

describe('the Features that keep what the page learned', () => {
  it('an answer\'s thumbs show the judgement its bag kept, not the page\'s', async () => {
    const bag = createAppHandles();
    const key = judgementKey('run-1', 'main', 7);
    bag.judgements.keep(key, {value: 'up', reasons: [], note: null});
    productionHandles().judgements.keep(key, {value: 'down', reasons: ['off_topic'], note: null});
    const feedback = document.createElement('dl-answer-feedback') as DlAnswerFeedback;
    feedback.handles = bag;
    feedback.runId = 'run-1';
    feedback.agent = 'main';
    feedback.sequence = 7;
    feedback.enabled = true;
    document.body.append(feedback);
    await feedback.updateComplete;

    expect(feedback.kept?.value).to.equal('up');
    expect(feedback.querySelector('[data-thumb="up"]')?.getAttribute('aria-pressed')).to.equal('true');
    expect(feedback.querySelector('[data-thumb="down"]')?.getAttribute('aria-pressed')).to.equal('false');

    productionHandles().judgements.forget();
  });

  it('the Chat follows a Run with the cursors of its bag, and starts submissions in its bag\'s registry', async () => {
    const bag = createAppHandles();
    const page = productionHandles();
    const encoder = new TextEncoder();
    let streamClosed = false;
    window.fetch = async (input, init) => {
      if (String(input) === '/web/api/runs/run-bag/events') {
        // The Run says one thing and goes on: its stream stays open until the Chat lets it go.
        return new Response(new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(encoder.encode('id: 3\nevent: token\ndata: "hello"\n\n'));
            init?.signal?.addEventListener('abort', () => {
              streamClosed = true;
              controller.error(new DOMException('Aborted', 'AbortError'));
            }, {once: true});
          },
        }), {status: 200, headers: {'Content-Type': 'text/event-stream'}});
      }
      if (String(input) === '/web/api/answer') return await new Promise<Response>(() => undefined);
      return Response.json({detail: 'not found'}, {status: 404});
    };
    const running: ConversationTurn = {
      turnId: 'turn-bag', turnNumber: 1, answerRunId: 'run-bag', submissionId: 'submission-bag',
      status: 'running', cancelRequested: false, userText: 'Question', assistantText: '',
      userAttachments: [], steeringMessages: [], presentation: null, usage: {}, errorKind: null,
      errorMessage: null, childCount: 0, answerSequence: null, feedback: null, createdAt: '2026-01-01T00:00:00Z',
    };
    const feature = document.createElement('dl-chat-feature') as DlChatFeature;
    feature.handles = bag;
    feature.view = {kind: 'ready', conversationId: 'conversation-bag', lineage: null, history: [running]};
    document.body.append(feature);

    await waitFor(() => bag.answerEventCursors.lastSequence('conversation-bag', 'run-bag') === 3);
    expect(
      page.answerEventCursors.lastSequence('conversation-bag', 'run-bag'),
      'the position was written to the page\'s cursors, and a reconnect would replay from the start',
    ).to.equal(0);

    // A question sent from it is under way in its bag, which is the bag whose state the Chat draws.
    feature.detachRun();
    await waitFor(() => streamClosed);
    feature.view = {kind: 'new'};
    await feature.updateComplete;
    const input = feature.querySelector<HTMLTextAreaElement>('.composer-input')!;
    input.value = 'Another question';
    input.dispatchEvent(new Event('input', {bubbles: true}));
    await feature.querySelector('dl-chat-composer')?.updateComplete;
    feature.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();

    await waitFor(() => bag.submissions.list().length === 1);
    expect(page.submissions.list(), 'the submission went to the page\'s registry').to.have.length(0);
    expect(feature.submissionPending).to.equal(true);

    bag.submissions.dispose();
    page.answerEventCursors.clear('conversation-bag');
  });
});
