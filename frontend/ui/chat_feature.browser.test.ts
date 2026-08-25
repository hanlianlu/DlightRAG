// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {
  AnswerPresentation,
  AnswerRunDescriptor,
  ConversationTurn,
} from '../api/conversations.ts';
import {attachmentStore} from '../stores/attachmentStore.ts';
import {conversationStore} from '../stores/conversationStore.ts';
import type {DlChatComposer} from './chat_composer.ts';
import './chat_composer.ts';
import type {DlChatFeature} from './chat_feature.ts';
import './chat_feature.ts';
import type {
  ChatRunActionDetail,
  ChatTurnView,
  DlChatMessageList,
} from './chat_message_list.ts';
import {webRouter} from './router.ts';

const originalFetch = window.fetch;

const policy = {
  countLimit: 6,
  imageMaxBytes: 1024,
  documentMaxBytes: 2048,
  extensions: new Set(['md', 'pdf']),
  imageCapability: 'supported' as const,
  imageLimit: 3,
};

const presentation: AnswerPresentation = {
  answer_text: 'A stored answer.',
  parts: [{
    type: 'markdown',
    text: 'A stored answer.',
    html: '<p>A stored answer.</p>',
    artifact: null,
    evidence_image: null,
    inline: false,
  }],
  sources: [],
  evidence_images: [],
  artifacts: [],
  artifact_outcome: {status: 'complete', issues: []},
};

function storedTurn(): ConversationTurn {
  return {
    turn_id: 'turn-1',
    turn_number: 1,
    answer_run_id: 'run-1',
    submission_id: 'submission-1',
    status: 'succeeded',
    cancel_requested: false,
    user_text: 'Question',
    assistant_text: presentation.answer_text,
    user_attachments: [],
    presentation,
    usage: {},
    evidence: {},
    error_kind: null,
    error_message: null,
    created_at: '2026-01-01T00:00:00Z',
  };
}

function continuationDescriptor(conversationId: string): AnswerRunDescriptor {
  return {
    run_id: `run-${conversationId}`,
    status: 'queued',
    cancel_requested: false,
    turn_id: `turn-${conversationId}`,
    turn_number: 1,
    submission_id: `submission-${conversationId}`,
    events_url: `/web/api/answer/run-${conversationId}/events`,
    status_url: `/web/api/answer/run-${conversationId}`,
    cancel_url: `/web/api/answer/run-${conversationId}`,
    conversation: {
      conversation_id: conversationId,
      title: 'Continuation',
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
    },
  };
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

async function settle(element: DlChatFeature): Promise<void> {
  await element.updateComplete;
  await element.querySelector('dl-chat-message-list')?.updateComplete;
  await element.querySelector('dl-chat-composer')?.updateComplete;
  await element.querySelector('dl-answer-presentation')?.updateComplete;
}

afterEach(() => {
  window.fetch = originalFetch;
  attachmentStore.clear();
  conversationStore.openNew();
  document.body.replaceChildren();
  localStorage.removeItem('dlightrag.answerMode');
});

it('composes stored history through public properties and AnswerPresentation properties', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.attachmentPolicy = policy;
  feature.attachmentAccept = 'image/*,.md,.pdf';
  feature.view = {
    kind: 'ready',
    conversationId: 'conversation-1',
    history: [storedTurn()],
    lineage: 'Original thread',
  };
  document.body.appendChild(feature);
  await settle(feature);

  const answer = feature.querySelector('dl-answer-presentation');
  expect(answer?.presentation).to.equal(presentation);
  expect(answer?.textContent).to.contain('A stored answer.');
  expect(feature.querySelector('[role="log"]')?.getAttribute('aria-label')).to.equal(
    'Conversation messages',
  );
  expect(feature.textContent).to.contain('Forked from Original thread');

  let action: ChatRunActionDetail | null = null;
  feature.addEventListener('dl-chat-run-action', (event) => {
    action = (event as CustomEvent<ChatRunActionDetail>).detail;
  });
  feature.querySelector<HTMLButtonElement>('button')?.focus();
  const followUp = Array.from(feature.querySelectorAll('button')).find(
    (button) => button.textContent?.trim() === 'Follow up',
  );
  followUp?.click();
  expect(action).to.deep.equal({action: 'follow-up', runId: 'run-1'});
});

it('raises background intent without treating interactive message controls as background', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'background-intent',
    history: [storedTurn()],
    lineage: null,
  };
  let backgroundIntents = 0;
  feature.addEventListener('dl-chat-background-click', () => { backgroundIntents += 1; });
  document.body.appendChild(feature);
  await settle(feature);

  feature.querySelector<HTMLElement>('main[aria-label="Chat"]')?.click();
  expect(backgroundIntents).to.equal(1);
  const followUp = Array.from(feature.querySelectorAll<HTMLButtonElement>('button')).find(
    (button) => button.textContent?.trim() === 'Follow up',
  );
  followUp?.click();
  expect(backgroundIntents).to.equal(1);
});

it('renders cancelled history without a simultaneous stopping phase', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'cancelled-history',
    lineage: null,
    history: [{
      ...storedTurn(),
      status: 'cancelled',
      cancel_requested: true,
      presentation: null,
    }],
  };
  document.body.appendChild(feature);
  await settle(feature);

  expect(feature.textContent).to.contain('Stopped');
  expect(feature.turns[0].progress).to.equal('');
});

it('forces a submitted turn into view without forcing later stream updates', async () => {
  window.fetch = async () => new Response('{}', {
    status: 503,
    headers: {'Content-Type': 'application/json'},
  });
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'submit-scroll',
    lineage: null,
    history: [storedTurn()],
  };
  document.body.appendChild(feature);
  await settle(feature);

  const area = feature.querySelector<HTMLElement>('main[aria-label="Chat"]')!;
  let scrollTop = 0;
  Object.defineProperties(area, {
    scrollHeight: {configurable: true, value: 1000},
    clientHeight: {configurable: true, value: 100},
    scrollTop: {
      configurable: true,
      get: () => scrollTop,
      set: (value: number) => { scrollTop = value; },
    },
  });
  const input = feature.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = 'A new question';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.querySelector('dl-chat-composer')?.updateComplete;
  feature.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await waitFor(() => feature.turns.length === 2);
  await feature.querySelector('dl-chat-message-list')?.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(scrollTop).to.equal(1000);

  await waitFor(() => feature.turns.at(-1)?.state === 'failed');
  scrollTop = 0;
  feature.turns = feature.turns.map((turn, index) => index === feature.turns.length - 1
    ? {...turn, state: 'streaming', streamText: 'Later token'}
    : turn);
  await feature.querySelector('dl-chat-message-list')?.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(scrollTop).to.equal(0);
});

it('Composer owns draft, attachment, mode, and typed submission intent', async () => {
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  composer.attachmentAccept = 'image/*,.md,.pdf';
  document.body.appendChild(composer);
  await composer.updateComplete;

  composer.addFiles([new File(['notes'], 'notes.md', {type: 'text/markdown'})]);
  await composer.updateComplete;
  expect(composer.hasDraft).to.equal(true);
  expect(composer.textContent).to.contain('notes.md');

  const input = composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = '  Explain this  ';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;

  let submitted: {query: string; mode: string | null} | null = null;
  composer.addEventListener('dl-composer-submit', (event) => {
    submitted = (event as CustomEvent).detail;
  });
  composer.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await composer.updateComplete;

  expect(submitted).to.deep.equal({query: 'Explain this', mode: null});
  expect(composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')?.value).to.equal('');
  expect(composer.hasDraft).to.equal(true);

  composer.clearDraft();
  await composer.updateComplete;
  expect(composer.hasDraft).to.equal(false);
  composer.focusInput();
  await composer.updateComplete;
  expect(document.activeElement).to.equal(composer.querySelector('[aria-label="Message"]'));
});

it('Composer clears steering text without discarding attachments and detects wrapped drafts', async () => {
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  document.body.appendChild(composer);
  await composer.updateComplete;
  composer.addFiles([new File(['notes'], 'notes.md', {type: 'text/markdown'})]);
  await composer.updateComplete;
  const draft = composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  draft.value = 'temporary steering text';
  draft.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;

  composer.clearText();
  await composer.updateComplete;

  expect(composer.hasDraft).to.equal(true);
  expect(composer.textContent).to.contain('notes.md');
  expect(composer.querySelector<HTMLInputElement>('[aria-label="Message"]')?.value).to.equal('');

  const input = composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.style.width = '60px';
  input.style.lineHeight = '20px';
  input.value = 'one very long single line that wraps over several visible rows';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(composer.querySelector('form')?.classList.contains('multiline')).to.equal(true);
});

it('delayed steering preserves newer text and is aborted when the run detaches', async () => {
  interface DeferredSteer {
    resolve: (response: Response) => void;
    signal: AbortSignal;
  }
  const steeringRequests: DeferredSteer[] = [];
  window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const signal = init?.signal as AbortSignal;
    if (url.endsWith('/steer')) {
      return new Promise<Response>((resolve, reject) => {
        steeringRequests.push({resolve, signal});
        signal.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          {once: true},
        );
      });
    }
    if (url.endsWith('/events')) {
      return new Promise<Response>((_resolve, reject) => {
        signal.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          {once: true},
        );
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  }) as typeof fetch;

  const mountRunningFeature = async (runId: string): Promise<DlChatFeature> => {
    const feature = document.createElement('dl-chat-feature') as DlChatFeature;
    feature.view = {
      kind: 'ready',
      conversationId: `conversation-${runId}`,
      lineage: null,
      history: [{
        ...storedTurn(),
        answer_run_id: runId,
        turn_id: `turn-${runId}`,
        status: 'running',
        presentation: null,
      }],
    };
    document.body.appendChild(feature);
    await waitFor(() => feature.querySelector('dl-chat-composer')?.running === true);
    return feature;
  };

  const first = await mountRunningFeature('run-steer-first');
  const firstInput = first.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  firstInput.value = 'original steering';
  firstInput.dispatchEvent(new Event('input', {bubbles: true}));
  await first.querySelector('dl-chat-composer')?.updateComplete;
  first.querySelector<HTMLButtonElement>('[aria-label="Steer"]')?.click();
  await waitFor(() => steeringRequests.length === 1);
  firstInput.value = 'newer draft';
  firstInput.dispatchEvent(new Event('input', {bubbles: true}));
  steeringRequests[0].resolve(new Response('{}', {
    status: 200,
    headers: {'Content-Type': 'application/json'},
  }));
  await waitFor(() => first.textContent?.includes('original steering') === true);
  expect(firstInput.value).to.equal('newer draft');
  first.detachRun();
  first.remove();

  const second = await mountRunningFeature('run-steer-second');
  const secondInput = second.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  secondInput.value = 'detached steering';
  secondInput.dispatchEvent(new Event('input', {bubbles: true}));
  await second.querySelector('dl-chat-composer')?.updateComplete;
  second.querySelector<HTMLButtonElement>('[aria-label="Steer"]')?.click();
  await waitFor(() => steeringRequests.length === 2);
  secondInput.value = 'draft in another conversation';
  secondInput.dispatchEvent(new Event('input', {bubbles: true}));
  second.detachRun();
  second.view = {kind: 'new'};
  await settle(second);

  expect(steeringRequests[1].signal.aborted).to.equal(true);
  expect(secondInput.value).to.equal('draft in another conversation');
  expect(second.textContent).not.to.contain('detached steering');
});

it('detaching invalidates delayed follow-up and fork continuations', async () => {
  const continuationRequests: Array<{
    resolve: (response: Response) => void;
    signal: AbortSignal;
  }> = [];
  window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (init?.method === 'POST' && (url.endsWith('/follow-up') || url.endsWith('/fork'))) {
      return new Promise<Response>((resolve) => {
        continuationRequests.push({resolve, signal: init.signal as AbortSignal});
      });
    }
    const conversationId = decodeURIComponent(url.split('/').pop() || '');
    const descriptor = continuationDescriptor(conversationId);
    return Promise.resolve(new Response(JSON.stringify({
      conversation: descriptor.conversation,
      turns: [],
    }), {status: 200, headers: {'Content-Type': 'application/json'}}));
  }) as typeof fetch;

  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  document.body.appendChild(feature);
  await settle(feature);
  const originalRoute = webRouter.current;

  for (const kind of ['follow-up', 'fork'] as const) {
    const conversationId = `stale-${kind}`;
    const continuation = feature.continueRun(kind, 'run-old', 'Continue');
    await waitFor(() => continuationRequests.length > 0);
    const request = continuationRequests.shift()!;
    feature.detachRun();
    expect(request.signal.aborted).to.equal(true);
    request.resolve(new Response(JSON.stringify(continuationDescriptor(conversationId)), {
      status: 200,
      headers: {'Content-Type': 'application/json'},
    }));
    await continuation;

    expect(conversationStore.conversations.some(
      (conversation) => conversation.conversation_id === conversationId,
    )).to.equal(false);
    expect(conversationStore.activeConversationId).not.to.equal(conversationId);
    expect(webRouter.current).to.deep.equal(originalRoute);
  }
});

it('shows cancel-aware reconnect state and preserves stopping when reconnecting', async () => {
  const conversationId = 'conversation-stopping';
  const runId = 'run-stopping';
  let eventRequests = 0;
  window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith('/events')) {
      eventRequests += 1;
      if (eventRequests === 1) return Promise.resolve(new Response('', {status: 410}));
      return new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        signal?.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          {once: true},
        );
      });
    }
    if (url.endsWith(`/answer/${runId}`)) {
      return Promise.resolve(new Response(JSON.stringify({
        ...storedTurn(),
        answer_run_id: runId,
        turn_id: 'turn-stopping',
        status: 'running',
        cancel_requested: true,
        presentation: null,
      }), {status: 200, headers: {'Content-Type': 'application/json'}}));
    }
    throw new Error(`unexpected fetch: ${url}`);
  }) as typeof fetch;
  conversationStore.adoptCreatedConversation({
    conversation_id: conversationId,
    title: 'Stopping',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  });

  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId,
    lineage: null,
    history: [{
      ...storedTurn(),
      answer_run_id: runId,
      turn_id: 'turn-stopping',
      status: 'running',
      cancel_requested: true,
      presentation: null,
    }],
  };
  document.body.appendChild(feature);

  await waitFor(() => feature.textContent?.includes(
    'Connection lost. This answer is still stopping.',
  ) === true);
  feature.querySelector<HTMLButtonElement>('[aria-label="Reconnect to this answer"]')?.click();
  await waitFor(() => eventRequests === 2);

  expect(feature.querySelector('dl-chat-composer')?.stopping).to.equal(true);
  expect(feature.turns[0].progress).to.equal('Stopping...');
});

it('Message List bounds steering wrappers within one retained turn', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const turn: ChatTurnView = {
    id: 'turn-steered', userText: 'Steer repeatedly', runId: 'run-steered', state: 'streaming',
    userAttachments: [], streamText: 'Working', presentation: null, usage: {}, evidence: {},
    error: '', progress: '', liveStatus: '', sawChildren: false, cancelRequested: false,
    steeringMessages: Array.from({length: 51}, (_, index) => `Steering ${index}`),
  };
  list.turns = [turn];
  document.body.appendChild(list);
  await list.updateComplete;

  expect(list.textContent?.match(/Steering \d+/g)?.length).to.equal(50);
  expect(list.textContent).not.to.contain('Steering 0');
  expect(list.textContent).to.contain('Steering 50');
});

it('Message List exposes child-agent progress and roster intent through public state', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const turn: ChatTurnView = {
    id: 'turn-running',
    userText: 'Delegate this',
    userAttachments: [],
    runId: 'run-with-child',
    state: 'streaming',
    streamText: 'Working',
    presentation: null,
    usage: {},
    evidence: {},
    error: '',
    progress: 'spawn_agent · running',
    liveStatus: 'spawn_agent · running',
    sawChildren: true,
    cancelRequested: false,
    steeringMessages: [],
  };
  list.turns = [turn];
  document.body.appendChild(list);
  await list.updateComplete;

  expect(list.querySelector('[role="log"]')?.getAttribute('aria-label')).to.equal(
    'Conversation messages',
  );
  expect(list.querySelector('[role="status"]')?.textContent).to.contain('spawn_agent');
  let action: ChatRunActionDetail | null = null;
  list.addEventListener('dl-chat-run-action', (event) => {
    action = (event as CustomEvent<ChatRunActionDetail>).detail;
  });
  Array.from(list.querySelectorAll('button')).find(
    (button) => button.textContent?.includes('Child agents working'),
  )?.click();
  expect(action).to.deep.equal({action: 'children', runId: 'run-with-child'});
});

it('Message List announces image state and prunes it with the owning turns', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const imageTurn: ChatTurnView = {
    id: 'turn-image',
    userText: 'Image',
    userAttachments: [{
      attachment_id: 'image-1', ordinal: 1, kind: 'image', filename: 'chart.png',
      mime_type: 'image/png', byte_size: 10,
      url: '/images/chart.png', thumbnail_url: '/images/chart-thumb.png', label: 'Chart',
    }],
    runId: 'run-image',
    state: 'succeeded',
    streamText: '',
    presentation,
    usage: {},
    evidence: {},
    error: '',
    progress: '',
    liveStatus: '',
    sawChildren: false,
    cancelRequested: false,
    steeringMessages: [],
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
  expect(list.querySelector('[role="status"]')?.textContent).to.contain('Loading Chart');
});

it('Chat Feature bounds rendered history to the owned Message List capacity', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'bounded-history',
    lineage: null,
    history: Array.from({length: 101}, (_, index) => ({
      ...storedTurn(),
      turn_id: `turn-${index}`,
      answer_run_id: `run-${index}`,
      turn_number: index + 1,
      user_text: `Question ${index}`,
    })),
  };
  document.body.appendChild(feature);
  await settle(feature);

  expect(feature.querySelector('[role="log"]')?.querySelectorAll('article').length).to.equal(100);
  expect(feature.textContent).not.to.contain('Question 0');
  expect(feature.textContent).to.contain('Question 100');
});

it('Message List revokes live attachment URLs when a turn is evicted', async () => {
  const revoked: string[] = [];
  const originalRevoke = URL.revokeObjectURL;
  URL.revokeObjectURL = (url) => { revoked.push(url); };
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const turn: ChatTurnView = {
    id: 'turn-blob', userText: 'Blob', runId: 'run-blob', state: 'cancelled',
    userAttachments: [{
      attachment_id: 'blob-1', ordinal: 1, kind: 'document', filename: 'notes.md',
      mime_type: 'text/markdown', byte_size: 2, url: 'blob:http://localhost/live-turn',
      thumbnail_url: null, label: 'notes.md',
    }],
    streamText: '', presentation: null, usage: {}, evidence: {}, error: '', progress: '',
    liveStatus: '', sawChildren: false, cancelRequested: false, steeringMessages: [],
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
