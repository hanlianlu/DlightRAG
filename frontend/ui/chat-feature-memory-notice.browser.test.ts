// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The chat feature and what a Run leaves in Profile Memory: when it looks, and when it says so under the answer.
 *
 * The pass that writes the conversation's memories runs after the Run is done and takes as long as its model
 * does, so the chat looks again after a short backoff; a Run whose own Agent wrote is looked at as soon as it
 * settles; and a conversation opened later is read once, turn by turn. */

import {expect} from '@esm-bundle/chai';
import type {RunMemoryChange} from '../api/memory.ts';
import {type AppHandles, productionHandles} from '../stores/app-handles.ts';
import {MemoryNotices} from '../stores/memory-notices.ts';
import {conversationStore, restoreChatFeatureFixtures, settle, storedTurn} from '../testing/chat-feature.ts';
import {waitFor} from '../testing/dom.ts';
import './chat-composer.ts';
import './chat-feature.ts';
import type {DlChatFeature} from './chat-feature.ts';

restoreChatFeatureFixtures();

const change = (id: string, writer: RunMemoryChange['writer']): RunMemoryChange => ({
  memoryId: `memory-${id}`, kind: 'preference', body: `Body ${id}`, action: 'add', changeId: `change-${id}`, writer,
});

/** What the server lists for each Run, and what the chat has asked for. */
function server() {
  const listed = new Map<string, readonly RunMemoryChange[]>();
  const asked: string[] = [];
  return {
    listed,
    asked,
    read: async (runId: string): Promise<readonly RunMemoryChange[]> => {
      asked.push(runId);
      return listed.get(runId) ?? [];
    },
  };
}

const bagOf = (remote: ReturnType<typeof server>, delays = [60, 120, 180]): AppHandles => ({
  ...productionHandles(),
  memoryNotices: new MemoryNotices({read: remote.read, delays}),
});

const SUCCEEDED = {
  status: 'succeeded',
  presentation: {
    answer_text: 'Done.',
    parts: [{type: 'markdown', text: 'Done.', html: '<p>Done.</p>', artifact: null, evidence_image: null, inline: false}],
    sources: [], evidence_images: [], artifacts: [], artifact_outcome: {status: 'complete', issues: []},
  },
  usage: {},
};

/** An event stream that says `events` and then that the Run is done. */
function answering(events: string[]): void {
  const encoder = new TextEncoder();
  let sequence = 0;
  const frames = [...events.map((event) => `event: ${event.split(' ')[0]}\ndata: ${event.slice(event.indexOf(' ') + 1)}\n\n`),
    `event: done\ndata: ${JSON.stringify(SUCCEEDED)}\n\n`]
    .map((frame) => { sequence += 1; return `id: ${sequence}\n${frame}`; });
  window.fetch = ((input: RequestInfo | URL) => {
    if (String(input).endsWith('/events')) {
      return Promise.resolve(new Response(new ReadableStream<Uint8Array>({
        start(controller): void {
          for (const frame of frames) controller.enqueue(encoder.encode(frame));
          controller.close();
        },
      }), {status: 200, headers: {'Content-Type': 'text/event-stream'}}));
    }
    return Promise.resolve(new Response('{}', {status: 503}));
  }) as typeof fetch;
}

/** A conversation whose one turn is still running, followed by a chat that uses `handles`. */
async function followingARun(handles: AppHandles, runId: string): Promise<DlChatFeature> {
  conversationStore.adoptCreatedConversation({
    conversationId: `conversation-${runId}`, title: 'Memory', createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z', forkedFromConversationId: null, forkedFromTitle: null,
  });
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.handles = handles;
  feature.view = {
    kind: 'ready',
    conversationId: `conversation-${runId}`,
    lineage: null,
    history: [{...storedTurn(), answerRunId: runId, turnId: `turn-${runId}`, status: 'running', presentation: null}],
  };
  document.body.appendChild(feature);
  return feature;
}

const trigger = (feature: DlChatFeature): HTMLButtonElement | null => (
  feature.querySelector<HTMLButtonElement>('dl-memory-notice [aria-haspopup]')
);

it('looks for what the pass learned only after the Run is done, then shows it under the answer', async () => {
  const remote = server();
  answering([]);
  const feature = await followingARun(bagOf(remote), 'run-pass');

  await waitFor(() => feature.turns[0]?.state === 'succeeded');
  await settle(feature);
  expect(remote.asked, 'the Run is done, and the pass has had no time').to.deep.equal([]);
  expect(trigger(feature)).to.equal(null);

  remote.listed.set('run-pass', [change('1', 'conversation'), change('2', 'conversation')]);
  await waitFor(() => trigger(feature) !== null);

  expect(remote.asked).to.deep.equal(['run-pass']);
  expect(trigger(feature)!.textContent?.trim()).to.equal('Remembered 2');
  // It is the last part of the action row under the answer.
  const row = trigger(feature)!.closest('.dl-action-row')!;
  expect(row.lastElementChild?.tagName.toLowerCase()).to.equal('dl-memory-notice');
  expect(row.textContent).to.contain('Agent traces');
});

it('stops looking once the pass has reported, and after the last try when it never does', async () => {
  const remote = server();
  answering([]);
  const feature = await followingARun(bagOf(remote, [30, 60, 90]), 'run-never');

  await waitFor(() => feature.turns[0]?.state === 'succeeded');
  await new Promise((resolve) => setTimeout(resolve, 300));

  expect(remote.asked, 'three tries and no more').to.deep.equal(['run-never', 'run-never', 'run-never']);
  expect(trigger(feature)).to.equal(null);

  const reported = server();
  reported.listed.set('run-yes', [change('1', 'conversation')]);
  answering([]);
  const second = await followingARun(bagOf(reported, [30, 60, 90]), 'run-yes');
  await waitFor(() => trigger(second) !== null);
  await new Promise((resolve) => setTimeout(resolve, 200));

  expect(reported.asked, 'one try found it').to.deep.equal(['run-yes']);
});

it('reads a Run whose own Agent wrote as soon as it is done, without waiting for the pass', async () => {
  const remote = server();
  remote.listed.set('run-agent', [change('1', 'agent')]);
  answering([
    'memory_operation_settled {"operation":"remember","outcome":"changed","change_id":"change-1","intent_id":"i1","live":true}',
  ]);
  const feature = await followingARun(bagOf(remote, [10_000, 20_000, 30_000]), 'run-agent');

  await waitFor(() => trigger(feature) !== null);

  expect(remote.asked).to.deep.equal(['run-agent']);
  expect(trigger(feature)!.textContent?.trim()).to.equal('Remembered 1');
});

it('reads each answered turn of a conversation opened later once, and only the ones that were answered', async () => {
  const remote = server();
  remote.listed.set('run-a', [change('1', 'conversation')]);
  window.fetch = (() => Promise.resolve(new Response('{}', {status: 503}))) as typeof fetch;
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.handles = bagOf(remote);
  feature.view = {
    kind: 'ready',
    conversationId: 'conversation-history',
    lineage: null,
    history: [
      {...storedTurn(), answerRunId: 'run-a', turnId: 'turn-a'},
      {...storedTurn(), answerRunId: 'run-b', turnId: 'turn-b', turnNumber: 2},
      {...storedTurn(), answerRunId: 'run-failed', turnId: 'turn-failed', turnNumber: 3, status: 'failed', presentation: null},
    ],
  };
  document.body.appendChild(feature);

  await waitFor(() => trigger(feature) !== null);
  await settle(feature);
  feature.requestUpdate();
  await settle(feature);

  expect(remote.asked.sort()).to.deep.equal(['run-a', 'run-b']);
  expect(feature.querySelectorAll('dl-memory-notice [aria-haspopup]')).to.have.length(1);
  expect(feature.querySelector('[data-turn-id="turn-a"]')).to.not.equal(null);
});
