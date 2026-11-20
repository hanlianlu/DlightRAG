// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The message list in the page: its history, anchoring and paging, its announcements, and its attachment URLs. */

import {expect} from '@esm-bundle/chai';
import type {ChatTurnView} from '../lib/chat-views.ts';
import {presentation, restoreChatFeatureFixtures, settle, storedTurn} from '../testing/chat-feature.ts';
import './chat-composer.ts';
import './chat-feature.ts';
import type {DlChatFeature} from './chat-feature.ts';
import {type ChatRunActionDetail, type DlChatMessageList, storedTurnView} from './chat-message-list.ts';

restoreChatFeatureFixtures();

/** A followed Run that has shown a child, with its stream held open and its intervals recorded. */
it('Message List anchors the completed turn at its latest user question', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const earlier: ChatTurnView = {
    id: 'turn-earlier', userText: 'Earlier question', runId: 'run-earlier', state: 'succeeded',
    userAttachments: [], streamText: presentation.answerText, processEntries: [],
    presentation, usage: {},
    error: null, activity: null, cancelRequested: false,
    steeringMessages: [],
    answerSequence: null, feedback: null,
    toolRows: [],

  };
  const latest: ChatTurnView = {
    ...earlier,
    id: 'turn-latest',
    userText: 'Latest question',
    runId: 'run-latest',
    state: 'streaming',
    streamText: 'Working',
    processEntries: [],
    presentation: null,
  };
  list.turns = [earlier, latest];
  document.body.appendChild(list);
  await list.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));

  const area = list.querySelector<HTMLElement>('main[aria-label="Chat"]')!;
  const latestQuestion = Array.from(list.querySelectorAll<HTMLElement>('div')).find(
    (element) => element.textContent === 'Latest question',
  )?.parentElement as HTMLElement;
  let scrollTop = 0;
  Object.defineProperties(area, {
    scrollHeight: {configurable: true, value: 2000},
    clientHeight: {configurable: true, value: 300},
    scrollTop: {
      configurable: true,
      get: () => scrollTop,
      set: (value: number) => { scrollTop = value; },
    },
    getBoundingClientRect: {
      configurable: true,
      value: () => ({top: 100}),
    },
  });
  Object.defineProperty(latestQuestion, 'getBoundingClientRect', {
    configurable: true,
    value: () => ({top: 600 - scrollTop}),
  });

  list.turns = [earlier, {...latest, state: 'succeeded', presentation}];
  await list.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));

  expect(scrollTop).to.equal(500);
});

it('Message List bounds steering wrappers within one retained turn', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const turn: ChatTurnView = {
    id: 'turn-steered', userText: 'Steer repeatedly', runId: 'run-steered', state: 'streaming',
    userAttachments: [], streamText: 'Working', processEntries: [],
    presentation: null, usage: {},
    error: null, activity: null, cancelRequested: false,
    steeringMessages: Array.from({length: 51}, (_, index) => `Steering ${index}`),
    answerSequence: null, feedback: null,
    toolRows: [],
  };
  list.turns = [turn];
  document.body.appendChild(list);
  await list.updateComplete;

  expect(list.textContent?.match(/Steering \d+/g)?.length).to.equal(50);
  expect(list.textContent).not.to.contain('Steering 0');
  expect(list.textContent).to.contain('Steering 50');
  const lastSteer = [...list.querySelectorAll('[data-steer]')].at(-1)!;
  const answer = list.querySelector('article[data-run-id="run-steered"]')!;
  expect(lastSteer.compareDocumentPosition(answer) & Node.DOCUMENT_POSITION_FOLLOWING).to.be.greaterThan(0);
});

it('a stored turn shows the steering its run received, in order, above the answer', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  list.turns = [storedTurnView({...storedTurn(), steeringMessages: ['shorter', 'in Chinese']})];
  document.body.appendChild(list);
  await list.updateComplete;

  const steers = [...list.querySelectorAll('[data-steer]')];
  expect(steers.map((steer) => steer.textContent?.trim())).to.deep.equal(['shorter', 'in Chinese']);
  const answer = list.querySelector('article[data-run-id="run-1"]')!;
  expect(steers[1]!.compareDocumentPosition(answer) & Node.DOCUMENT_POSITION_FOLLOWING).to.be.greaterThan(0);
});

it('Message List exposes child-agent progress and roster intent through public state', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const childRow: ChatTurnView['toolRows'][number] = {
    callId: 'child-call', label: 'Tool working...', name: 'spawn_agent',
    verb: 'Dispatching a child agent', verbId: 'chatFeature.tool.spawn_agent',
    object: '', state: 'running', startedAt: 1000, durationMs: null,
  };
  const turn: ChatTurnView = {
    id: 'turn-running',
    userText: 'Delegate this',
    userAttachments: [],
    runId: 'run-with-child',
    state: 'streaming',
    streamText: 'Working',
    processEntries: [{kind: 'tool', row: childRow}],
    presentation: null,
    usage: {},
    error: null,
    activity: {kind: 'tool'},
    cancelRequested: false,
    steeringMessages: [],
    answerSequence: null, feedback: null,
    toolRows: [childRow],

  };
  list.turns = [turn];
  document.body.appendChild(list);
  await list.updateComplete;

  expect(list.querySelector('[role="log"]')?.getAttribute('aria-label')).to.equal(
    'Conversation messages',
  );
  const liveStatus = Array.from(list.querySelectorAll<HTMLElement>('[role="status"]')).find(
    (element) => element.textContent?.includes('Tool working'),
  );
  expect(liveStatus).not.to.equal(undefined);
  let action: ChatRunActionDetail | null = null;
  list.addEventListener('dl-chat-run-action', (event) => {
    action = (event as CustomEvent<ChatRunActionDetail>).detail;
  });
  const actions = Array.from(list.querySelectorAll('button')).filter(
    (button) => button.textContent?.trim() === 'Agent traces',
  );
  expect(actions.length).to.equal(1);
  // The live Run's entry carries the dot that its settled form does not.
  expect(actions[0].querySelectorAll('svg')).to.have.length(1);
  actions[0].click();
  expect(action).to.deep.equal({action: 'traces', runId: 'run-with-child'});
});

it('Message List announces image state and prunes it with the owning turns', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const imageTurn: ChatTurnView = {
    id: 'turn-image',
    userText: 'Image',
    userAttachments: [{
      attachmentId: 'image-1', ordinal: 1, kind: 'image', filename: 'chart.png',
      mimeType: 'image/png', byteSize: 10,
      url: '/images/chart.png', thumbnailUrl: '/images/chart-thumb.png', label: 'Chart',
    }],
    runId: 'run-image',
    state: 'succeeded',
    streamText: '',
    processEntries: [],
    presentation,
    usage: {},
    error: null,
    activity: null,
    cancelRequested: false,
    steeringMessages: [],
    answerSequence: null, feedback: null,
    toolRows: [],

  };
  list.turns = [imageTurn];
  document.body.appendChild(list);
  await list.updateComplete;

  const imageStatus = Array.from(list.querySelectorAll<HTMLElement>('[role="status"]')).find(
    (element) => element.textContent?.includes('Loading Chart'),
  );
  expect(imageStatus).not.to.equal(undefined);
  expect(imageStatus?.getAttribute('role')).to.equal('status');

  list.querySelector<HTMLImageElement>('img[alt="Chart"]')?.dispatchEvent(new Event('error'));
  await list.updateComplete;
  expect(list.querySelector('[role="alert"]')?.textContent).to.contain(
    'History image failed to load: Chart',
  );

  list.turns = [];
  await list.updateComplete;
  list.turns = [imageTurn];
  await list.updateComplete;
  expect(list.querySelector('[role="alert"]')).to.equal(null);
  expect(Array.from(list.querySelectorAll<HTMLElement>('[role="status"]')).some(
    (element) => element.textContent?.includes('Loading Chart'),
  )).to.equal(true);
});

it('Message List exposes an accessible retryable Load older messages control', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  list.view = {
    kind: 'ready', conversationId: 'paged', history: [storedTurn()], lineage: null,
    olderMessages: {state: 'idle', starting: false, hasOlder: true, outcome: null},
  };
  list.turns = [];
  document.body.appendChild(list);
  await list.updateComplete;
  const button = list.querySelector<HTMLButtonElement>('[data-load-older="messages"]')!;
  let requests = 0;
  list.addEventListener('dl-chat-load-older', () => { requests += 1; });

  button.focus();
  button.click();

  expect(button.type).to.equal('button');
  expect(button.textContent).to.contain('Load older messages');
  expect(button.getAttribute('aria-busy')).to.equal('false');
  expect(requests).to.equal(1);

  list.view = {...list.view, olderMessages: {state: 'error', starting: false, hasOlder: true, outcome: 'failed'}};
  await list.updateComplete;
  expect(list.querySelector('[data-load-older="messages"]')?.textContent).to.contain('Retry');
  expect(list.querySelector('[data-load-older-status="messages"]')?.textContent).to.contain(
    'could not be loaded',
  );
  expect(button.closest('[role="log"]')).to.equal(null);
});

it('Message List keeps the final older-page announcement and moves focus into the log', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  list.view = {
    kind: 'ready', conversationId: 'last-page', history: [storedTurn()], lineage: null,
    olderMessages: {state: 'idle', starting: false, hasOlder: true, outcome: null},
  };
  list.turns = [storedTurnView(storedTurn())];
  document.body.appendChild(list);
  await list.updateComplete;
  const button = list.querySelector<HTMLButtonElement>('[data-load-older="messages"]')!;
  button.focus();
  button.click();

  list.view = {...list.view, olderMessages: {state: 'loading', starting: false, hasOlder: true, outcome: null}};
  await list.updateComplete;
  list.view = {...list.view, olderMessages: {state: 'idle', starting: false, hasOlder: false, outcome: 'loaded'}};
  await list.updateComplete;
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

  expect(list.querySelector('[data-load-older="messages"]')).to.equal(null);
  expect(list.querySelector('[data-load-older-status="messages"]')?.textContent).to.contain(
    'Loaded older messages',
  );
  expect(document.activeElement).to.equal(list.querySelector('#chat-messages'));

  list.view = {
    kind: 'ready', conversationId: 'another-page', history: [storedTurn()], lineage: null,
    olderMessages: {state: 'idle', starting: false, hasOlder: true, outcome: null},
  };
  await list.updateComplete;
  expect(list.querySelector('[data-load-older-status="messages"]')?.textContent?.trim()).to.equal('');
});

it('Message List anchors the existing viewport when an older page is prepended', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const existing = {
    ...storedTurn(), turnId: 'turn-2', turnNumber: 2, answerRunId: 'run-2',
  };
  list.view = {
    kind: 'ready', conversationId: 'anchor', history: [existing], lineage: null,
    olderMessages: {state: 'idle', starting: false, hasOlder: true, outcome: null},
  };
  list.turns = [storedTurnView(existing)];
  document.body.appendChild(list);
  await list.updateComplete;
  const area = list.querySelector<HTMLElement>('#chat-area')!;
  area.style.height = '48px';
  area.style.overflow = 'auto';
  const existingElement = list.querySelector<HTMLElement>('[data-turn-id="turn-2"]')!;
  const before = existingElement.getBoundingClientRect().top - area.getBoundingClientRect().top;

  list.querySelector<HTMLButtonElement>('[data-load-older="messages"]')!.click();
  list.view = {...list.view, olderMessages: {state: 'loading', starting: false, hasOlder: true, outcome: null}};
  await list.updateComplete;
  const older = {
    ...storedTurn(), turnId: 'turn-1', turnNumber: 1, answerRunId: 'run-1',
  };
  list.turns = [storedTurnView(older), storedTurnView(existing)];
  list.view = {...list.view, olderMessages: {state: 'idle', starting: false, hasOlder: true, outcome: 'loaded'}};
  await list.updateComplete;
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

  const after = list.querySelector<HTMLElement>('[data-turn-id="turn-2"]')!
    .getBoundingClientRect().top - area.getBoundingClientRect().top;
  expect(Math.abs(after - before)).to.be.lessThan(1);
});

it('Message List revokes live attachment URLs when a turn is evicted', async () => {
  const revoked: string[] = [];
  const originalRevoke = URL.revokeObjectURL;
  URL.revokeObjectURL = (url) => { revoked.push(url); };
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const turn: ChatTurnView = {
    id: 'turn-blob', userText: 'Blob', runId: 'run-blob', state: 'cancelled',
    userAttachments: [{
      attachmentId: 'blob-1', ordinal: 1, kind: 'document', filename: 'notes.md',
      mimeType: 'text/markdown', byteSize: 2, url: 'blob:http://localhost/live-turn',
      thumbnailUrl: null, label: 'notes.md',
    }],
    streamText: '',
    processEntries: [],
    presentation: null, usage: {}, error: null, activity: null,
    cancelRequested: false, steeringMessages: [],
    answerSequence: null, feedback: null,
    toolRows: [],

  };
  try {
    list.turns = [turn];
    document.body.appendChild(list);
    await list.updateComplete;
    list.turns = [];
    await list.updateComplete;
    expect(revoked).to.deep.equal(['blob:http://localhost/live-turn']);
  } finally {
    URL.revokeObjectURL = originalRevoke;
  }
});

it('Message List exposes loading and retry through visible ARIA state and typed intent', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {kind: 'error'};
  document.body.appendChild(feature);
  await settle(feature);

  expect(feature.querySelector('[role="alert"]')?.textContent).to.contain(
    'Conversation history is unavailable.',
  );
  let action = '';
  feature.addEventListener('dl-chat-view-action', (event) => {
    action = (event as CustomEvent).detail.action;
  });
  feature.querySelector<HTMLButtonElement>('[aria-label="Retry loading conversation history"]')?.click();
  expect(action).to.equal('retry');
});
