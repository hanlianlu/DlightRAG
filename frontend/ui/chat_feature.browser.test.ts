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
import type {DlConversationSidebar} from './conversation_sidebar.ts';
import './conversation_sidebar.ts';
import {
  ANSWER_PHASE_LABELS,
  ANSWER_TOOL_EVENT_LABELS,
  answerPhaseLabel,
  answerToolEventLabel,
  type DlChatFeature,
} from './chat_feature.ts';
import {
  ANSWER_RECONNECT_COPY,
  answerReconnectState,
  storedTurnView,
  type ChatRunActionDetail,
  type ChatTurnView,
  type DlChatMessageList,
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

it('maps every server answer phase and tool event to qualitative deterministic copy', () => {
  expect(ANSWER_PHASE_LABELS).to.deep.equal({
    routing: 'Routing answer...',
    planning: 'Planning answer...',
    searching: 'Searching knowledge base...',
    researching: 'Researching sources...',
    generating: 'Generating answer...',
  });
  expect(ANSWER_TOOL_EVENT_LABELS).to.deep.equal({
    tool_start: 'Tool started...',
    tool_progress: 'Tool working...',
    tool_end: 'Tool finished...',
  });
  for (const [phase, label] of Object.entries(ANSWER_PHASE_LABELS)) {
    expect(answerPhaseLabel(phase)).to.equal(label);
  }
  for (const [eventType, label] of Object.entries(ANSWER_TOOL_EVENT_LABELS)) {
    expect(answerToolEventLabel(eventType)).to.equal(label);
  }
  for (const unknown of ['provider-secret-phase', 'toString', '__proto__']) {
    expect(answerPhaseLabel(unknown)).to.equal(null);
  }
  for (const unknown of ['tool_telemetry', 'toString', 'constructor']) {
    expect(answerToolEventLabel(unknown)).to.equal(null);
  }
  expect(Object.values({...ANSWER_PHASE_LABELS, ...ANSWER_TOOL_EVENT_LABELS}).join(' '))
    .not.to.match(/\b(?:bytes?|elapsed|model|remaining)\b|\d+\s*(?:ms|%)/i);
});

it('maps and renders every reconnect state with one visible status and action', async () => {
  expect(answerReconnectState(false)).to.equal('running');
  expect(answerReconnectState(true)).to.equal('stopping');
  expect(ANSWER_RECONNECT_COPY).to.deep.equal({
    running: {
      status: 'Connection lost while this answer is running.',
      action: 'Reconnect',
    },
    stopping: {
      status: 'Connection lost while this answer is stopping.',
      action: 'Reconnect',
    },
  });

  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const retryable = (cancelRequested: boolean): ChatTurnView => ({
    id: `turn-${cancelRequested ? 'stopping' : 'running'}`,
    userText: 'Question',
    userAttachments: [],
    runId: `run-${cancelRequested ? 'stopping' : 'running'}`,
    state: 'retryable',
    streamText: '',
    presentation: null,
    usage: {},
    evidence: {},
    error: '',
    progress: '',
    liveStatus: '',
    sawChildren: false,
    cancelRequested,
    steeringMessages: [],
  });
  list.turns = [retryable(false), retryable(true)];
  document.body.appendChild(list);
  await list.updateComplete;

  for (const state of ['running', 'stopping'] as const) {
    const notice = list.querySelector<HTMLElement>(`[data-reconnect-state="${state}"]`)!;
    expect(notice.querySelector('[role="status"]')?.textContent?.trim())
      .to.equal(ANSWER_RECONNECT_COPY[state].status);
    expect(notice.querySelector('button')?.textContent?.trim())
      .to.equal(ANSWER_RECONNECT_COPY[state].action);
    expect(notice.querySelector('button')?.getAttribute('aria-label'))
      .to.equal('Reconnect to this answer');
  }
});

it('forces a submitted turn into view without forcing later stream updates', async () => {
  let finishRequest!: () => void;
  window.fetch = () => new Promise<Response>((resolve) => {
    finishRequest = () => resolve(new Response('{}', {
      status: 503,
      headers: {'Content-Type': 'application/json'},
    }));
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

  finishRequest();
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

it('continuation start failure raises a toast intent instead of a blocking alert', async () => {
  const alerts: string[] = [];
  const originalAlert = window.alert;
  window.alert = (message?: string) => { alerts.push(message ?? ''); };
  const toasts: Array<{message?: string}> = [];
  const onToast = (event: Event): void => {
    toasts.push((event as CustomEvent<{message?: string}>).detail);
  };
  document.addEventListener('dl-toast-request', onToast);
  window.fetch = (() => Promise.resolve(new Response('unavailable', {status: 503}))) as typeof fetch;

  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  document.body.appendChild(feature);
  await settle(feature);

  try {
    await feature.continueRun('follow-up', 'run-old', 'Continue');
    await waitFor(() => toasts.length === 1);
    expect(toasts[0].message).to.equal('The continuation could not be started.');
    expect(alerts).to.deep.equal([]);
  } finally {
    document.removeEventListener('dl-toast-request', onToast);
    window.alert = originalAlert;
  }
});

it('failed steering raises a toast intent instead of a blocking alert', async () => {
  const alerts: string[] = [];
  const originalAlert = window.alert;
  window.alert = (message?: string) => { alerts.push(message ?? ''); };
  const toasts: Array<{message?: string}> = [];
  const onToast = (event: Event): void => {
    toasts.push((event as CustomEvent<{message?: string}>).detail);
  };
  document.addEventListener('dl-toast-request', onToast);
  window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith('/steer')) {
      return Promise.resolve(new Response('gone', {status: 410}));
    }
    if (url.endsWith('/events')) {
      return new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        signal?.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          {once: true},
        );
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  }) as typeof fetch;

  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'conversation-steer-failure',
    lineage: null,
    history: [{
      ...storedTurn(),
      answer_run_id: 'run-steer-failure',
      turn_id: 'turn-steer-failure',
      status: 'running',
      presentation: null,
    }],
  };
  document.body.appendChild(feature);
  await waitFor(() => feature.querySelector('dl-chat-composer')?.running === true);
  const steerInput = feature.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  steerInput.value = 'steer instruction';
  steerInput.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.querySelector('dl-chat-composer')?.updateComplete;
  feature.querySelector<HTMLButtonElement>('[aria-label="Steer"]')?.click();

  try {
    await waitFor(() => toasts.length === 1);
    expect(toasts[0].message).to.equal('This run can no longer be steered.');
    expect(alerts).to.deep.equal([]);
  } finally {
    document.removeEventListener('dl-toast-request', onToast);
    window.alert = originalAlert;
  }
});

it('shows cancel-aware reconnect state and preserves stopping when reconnecting', async () => {
  const conversationId = 'conversation-stopping';
  const runId = 'run-stopping';
  let eventRequests = 0;
  let finishFirstEvents!: (response: Response) => void;
  const firstEvents = new Promise<Response>((resolve) => { finishFirstEvents = resolve; });
  window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith('/events')) {
      eventRequests += 1;
      if (eventRequests === 1) return firstEvents;
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

  await waitFor(() => eventRequests === 1 && feature.turns.length === 1);
  feature.turns = feature.turns.map((turn) => ({
    ...turn,
    liveStatus: 'Planning answer...',
  }));
  finishFirstEvents(new Response('', {status: 410}));

  await waitFor(() => feature.textContent?.includes(
    'Connection lost while this answer is stopping.',
  ) === true);
  expect(feature.turns[0].liveStatus).to.equal('');
  expect(feature.querySelectorAll('[data-reconnect-state="stopping"] [role="status"]'))
    .to.have.length(1);
  feature.querySelector<HTMLButtonElement>('[aria-label="Reconnect to this answer"]')?.click();
  await waitFor(() => eventRequests === 2);

  expect(feature.querySelector('dl-chat-composer')?.stopping).to.equal(true);
  expect(feature.turns[0].progress).to.equal('Stopping...');
});

it('frame-batches 2,000 streamed tokens into bounded Chat and Message List updates', async () => {
  const originalRequestFrame = window.requestAnimationFrame;
  const originalCancelFrame = window.cancelAnimationFrame;
  let nextFrame = 1;
  const frames = new Map<number, FrameRequestCallback>();
  window.requestAnimationFrame = (callback: FrameRequestCallback): number => {
    const id = nextFrame;
    nextFrame += 1;
    frames.set(id, callback);
    return id;
  };
  window.cancelAnimationFrame = (id: number): void => {
    frames.delete(id);
  };
  const runFrames = (): void => {
    const pending = [...frames.values()];
    frames.clear();
    pending.forEach((callback) => callback(performance.now()));
  };

  const conversationId = 'conversation-frame-batching';
  const runId = 'run-frame-batching';
  const expected = Array.from(
    {length: 2_000},
    (_, index) => String.fromCharCode(97 + (index % 26)),
  ).join('');
  const chunks: string[] = [];
  let sequence = 1;
  chunks.push(`id: ${sequence}\nevent: progress\ndata: {"phase":"planning"}\n\n`);
  sequence += 1;
  for (let index = 0; index < expected.length; index += 1) {
    chunks.push(
      `id: ${sequence}\nevent: token\ndata: ${JSON.stringify(expected[index])}\n\n`,
    );
    sequence += 1;
    if (index === 499 || index === 1_499) {
      chunks.push(
        `id: ${sequence}\nevent: memory_operation_settled\n`
        + `data: {"operation":"remember","intent_id":"memory-${index}"}\n\n`,
      );
      sequence += 1;
    }
  }
  chunks.push(
    `id: ${sequence}\nevent: done\n`
    + 'data: {"status":"cancelled","presentation":null}\n\n',
  );
  const encoder = new TextEncoder();
  let chunkIndex = 0;
  let releaseEvents!: (response: Response) => void;
  const eventResponse = new Promise<Response>((resolve) => { releaseEvents = resolve; });
  window.fetch = ((input: RequestInfo | URL) => {
    if (String(input).endsWith('/events')) return eventResponse;
    return Promise.resolve(new Response('{}', {status: 503}));
  }) as typeof fetch;
  conversationStore.adoptCreatedConversation({
    conversation_id: conversationId,
    title: 'Frame batching',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  });

  try {
    const feature = document.createElement('dl-chat-feature') as DlChatFeature;
    feature.view = {
      kind: 'ready',
      conversationId,
      lineage: null,
      history: [{
        ...storedTurn(),
        answer_run_id: runId,
        turn_id: 'turn-frame-batching',
        status: 'running',
        presentation: null,
      }],
    };
    const memoryOperations: string[] = [];
    feature.addEventListener('dl-chat-memory-operation', (event) => {
      const detail = (event as CustomEvent<{intent_id?: string}>).detail;
      memoryOperations.push(detail.intent_id ?? '');
    });
    document.body.appendChild(feature);
    await waitFor(() => feature.querySelector('dl-chat-message-list') !== null);
    await settle(feature);
    runFrames();

    const turnsProperty = Object.getOwnPropertyDescriptor(
      Object.getPrototypeOf(feature) as object,
      'turns',
    );
    if (!turnsProperty?.get || !turnsProperty.set) throw new Error('turns accessor unavailable');
    let turnAssignments = 0;
    Object.defineProperty(feature, 'turns', {
      configurable: true,
      get: () => turnsProperty.get!.call(feature) as readonly ChatTurnView[],
      set: (value: readonly ChatTurnView[]) => {
        turnAssignments += 1;
        turnsProperty.set!.call(feature, value);
      },
    });
    const list = feature.querySelector('dl-chat-message-list') as DlChatMessageList;
    const instrumentedList = list as unknown as {
      updated: (changed: Map<PropertyKey, unknown>) => void;
    };
    const originalUpdated = instrumentedList.updated.bind(list);
    let listTurnsUpdates = 0;
    instrumentedList.updated = (changed) => {
      if (changed.has('turns')) listTurnsUpdates += 1;
      originalUpdated(changed);
    };

    releaseEvents(new Response(new ReadableStream<Uint8Array>({
      pull(controller): void {
        if (chunkIndex === chunks.length) {
          controller.close();
          return;
        }
        if (chunkIndex > 0 && chunkIndex % 500 === 0) runFrames();
        controller.enqueue(encoder.encode(chunks[chunkIndex]));
        chunkIndex += 1;
      },
    }), {status: 200, headers: {'Content-Type': 'text/event-stream'}}));

    await waitFor(() => feature.turns[0]?.state === 'cancelled');
    await settle(feature);
    runFrames();

    expect(feature.turns[0].streamText).to.equal(expected);
    expect(feature.turns[0].liveStatus).to.equal('Answer stopped');
    expect(memoryOperations).to.deep.equal(['memory-499', 'memory-1499']);
    expect(turnAssignments).to.equal(6);
    expect(listTurnsUpdates).to.equal(5);
    expect(feature.querySelectorAll('[role="status"]:not([data-older-status])')).to.have.length(1);
    expect(feature.textContent).to.contain('Stopped');
  } finally {
    window.requestAnimationFrame = originalRequestFrame;
    window.cancelAnimationFrame = originalCancelFrame;
  }
});

it('Message List anchors the completed turn at its latest user question', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const earlier: ChatTurnView = {
    id: 'turn-earlier', userText: 'Earlier question', runId: 'run-earlier', state: 'succeeded',
    userAttachments: [], streamText: presentation.answer_text, presentation, usage: {}, evidence: {},
    error: '', progress: '', liveStatus: '', sawChildren: false, cancelRequested: false,
    steeringMessages: [],
  };
  const latest: ChatTurnView = {
    ...earlier,
    id: 'turn-latest',
    userText: 'Latest question',
    runId: 'run-latest',
    state: 'streaming',
    streamText: 'Working',
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
    progress: 'Tool working...',
    liveStatus: 'Tool working...',
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
  const liveStatus = Array.from(list.querySelectorAll<HTMLElement>('[role="status"]')).find(
    (element) => element.textContent?.includes('Tool working'),
  );
  expect(liveStatus).not.to.equal(undefined);
  let action: ChatRunActionDetail | null = null;
  list.addEventListener('dl-chat-run-action', (event) => {
    action = (event as CustomEvent<ChatRunActionDetail>).detail;
  });
  Array.from(list.querySelectorAll('button')).find(
    (button) => button.textContent?.includes('View child agents'),
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
  expect(Array.from(list.querySelectorAll<HTMLElement>('[role="status"]')).some(
    (element) => element.textContent?.includes('Loading Chart'),
  )).to.equal(true);
});

it('Message List exposes an accessible retryable Load older messages control', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  list.view = {
    kind: 'ready', conversationId: 'paged', history: [storedTurn()], lineage: null,
    hasOlderMessages: true, olderMessagesState: 'idle',
  };
  list.turns = [];
  document.body.appendChild(list);
  await list.updateComplete;
  const button = list.querySelector<HTMLButtonElement>('[data-load-older]')!;
  let requests = 0;
  list.addEventListener('dl-chat-load-older', () => { requests += 1; });

  button.focus();
  button.click();

  expect(button.type).to.equal('button');
  expect(button.textContent).to.contain('Load older messages');
  expect(button.getAttribute('aria-busy')).to.equal('false');
  expect(requests).to.equal(1);

  list.view = {...list.view, olderMessagesState: 'error'};
  await list.updateComplete;
  expect(list.querySelector('[data-load-older]')?.textContent).to.contain('Retry');
  expect(list.querySelector('[data-older-status]')?.textContent).to.contain(
    'could not be loaded',
  );
  expect(button.closest('[role="log"]')).to.equal(null);
});

it('Message List keeps the final older-page announcement and moves focus into the log', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  list.view = {
    kind: 'ready', conversationId: 'last-page', history: [storedTurn()], lineage: null,
    hasOlderMessages: true, olderMessagesState: 'idle',
  };
  list.turns = [storedTurnView(storedTurn())];
  document.body.appendChild(list);
  await list.updateComplete;
  const button = list.querySelector<HTMLButtonElement>('[data-load-older]')!;
  button.focus();
  button.click();

  list.view = {...list.view, olderMessagesState: 'loading'};
  await list.updateComplete;
  list.view = {...list.view, hasOlderMessages: false, olderMessagesState: 'idle'};
  await list.updateComplete;
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

  expect(list.querySelector('[data-load-older]')).to.equal(null);
  expect(list.querySelector('[data-older-status]')?.textContent).to.contain(
    'Loaded older messages',
  );
  expect(document.activeElement).to.equal(list.querySelector('#chat-messages'));

  list.view = {
    kind: 'ready', conversationId: 'another-page', history: [storedTurn()], lineage: null,
    hasOlderMessages: true, olderMessagesState: 'idle',
  };
  await list.updateComplete;
  expect(list.querySelector('[data-older-status]')?.textContent?.trim()).to.equal('');
});

it('Message List anchors the existing viewport when an older page is prepended', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const existing = {
    ...storedTurn(), turn_id: 'turn-2', turn_number: 2, answer_run_id: 'run-2',
  };
  list.view = {
    kind: 'ready', conversationId: 'anchor', history: [existing], lineage: null,
    hasOlderMessages: true, olderMessagesState: 'idle',
  };
  list.turns = [storedTurnView(existing)];
  document.body.appendChild(list);
  await list.updateComplete;
  const area = list.querySelector<HTMLElement>('#chat-area')!;
  area.style.height = '48px';
  area.style.overflow = 'auto';
  const existingElement = list.querySelector<HTMLElement>('[data-turn-id="turn-2"]')!;
  const before = existingElement.getBoundingClientRect().top - area.getBoundingClientRect().top;

  list.querySelector<HTMLButtonElement>('[data-load-older]')!.click();
  list.view = {...list.view, olderMessagesState: 'loading'};
  await list.updateComplete;
  const older = {
    ...storedTurn(), turn_id: 'turn-1', turn_number: 1, answer_run_id: 'run-1',
  };
  list.turns = [storedTurnView(older), storedTurnView(existing)];
  list.view = {...list.view, olderMessagesState: 'idle'};
  await list.updateComplete;
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

  const after = list.querySelector<HTMLElement>('[data-turn-id="turn-2"]')!
    .getBoundingClientRect().top - area.getBoundingClientRect().top;
  expect(Math.abs(after - before)).to.be.lessThan(1);
});

it('Chat Feature deduplicates terminal live turns by authoritative run identity', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready', conversationId: 'same', history: [], lineage: null,
  };
  document.body.appendChild(feature);
  await settle(feature);
  feature.turns = [{
    ...storedTurnView(storedTurn()),
    id: 'local-terminal',
  }];

  feature.view = {
    kind: 'ready', conversationId: 'same', history: [storedTurn()], lineage: null,
  };
  await settle(feature);

  expect(feature.turns).to.have.length(1);
  expect(feature.turns[0]?.id).to.equal('turn-1');
  expect(feature.querySelectorAll('[data-run-id="run-1"]')).to.have.length(1);
});

it('Chat Feature lets an authoritative terminal stored turn replace local retryable state', async () => {
  const running = {...storedTurn(), status: 'running' as const, presentation: null};
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready', conversationId: 'terminal-wins', history: [running], lineage: null,
  };
  document.body.appendChild(feature);
  await settle(feature);
  feature.turns = feature.turns.map((turn) => ({
    ...turn,
    state: 'retryable' as const,
    streamText: 'Partial local answer',
    error: 'Reconnect',
  }));

  feature.view = {
    kind: 'ready', conversationId: 'terminal-wins', history: [storedTurn()], lineage: null,
  };
  await settle(feature);

  expect(feature.turns).to.have.length(1);
  expect(feature.turns[0]?.id).to.equal('turn-1');
  expect(feature.turns[0]?.state).to.equal('succeeded');
  expect(feature.turns[0]?.streamText).to.equal('');
  expect(feature.turns[0]?.presentation).to.deep.equal(presentation);
  expect(feature.querySelectorAll('[data-run-id="run-1"]')).to.have.length(1);
});

it('Chat Feature preserves a non-terminal live projection across history republish', async () => {
  const running = {...storedTurn(), status: 'running' as const, presentation: null};
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready', conversationId: 'same-running', history: [running], lineage: null,
    hasOlderMessages: true, olderMessagesState: 'idle',
  };
  document.body.appendChild(feature);
  await settle(feature);
  feature.turns = feature.turns.map((turn) => ({
    ...turn,
    state: 'streaming' as const,
    streamText: 'Live answer',
  }));

  feature.view = {
    ...feature.view,
    olderMessagesState: 'loading',
  };
  await settle(feature);

  expect(feature.turns).to.have.length(1);
  expect(feature.turns[0]?.state).to.equal('streaming');
  expect(feature.turns[0]?.streamText).to.equal('Live answer');
});

it('Chat Feature drops stale terminal ranges while preserving a pending optimistic turn', async () => {
  const oldHistory = Array.from({length: 40}, (_, index) => ({
    ...storedTurn(),
    turn_id: `turn-${index + 1}`,
    turn_number: index + 1,
    answer_run_id: `run-${index + 1}`,
    submission_id: `submission-${index + 1}`,
  }));
  const recentHistory = Array.from({length: 40}, (_, index) => ({
    ...storedTurn(),
    turn_id: `turn-${index + 51}`,
    turn_number: index + 51,
    answer_run_id: `run-${index + 51}`,
    submission_id: `submission-${index + 51}`,
  }));
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready', conversationId: 'gap-replaced', history: oldHistory, lineage: null,
  };
  document.body.appendChild(feature);
  await settle(feature);
  feature.turns = [...feature.turns, {
    ...storedTurnView(storedTurn()),
    id: 'local-pending',
    runId: '',
    state: 'pending',
    userText: 'Pending optimistic question',
    streamText: '',
    presentation: null,
  }];

  feature.view = {
    kind: 'ready', conversationId: 'gap-replaced', history: recentHistory, lineage: null,
  };
  await settle(feature);

  expect(feature.turns).to.have.length(41);
  expect(feature.turns.some((turn) => turn.id === 'turn-1')).to.equal(false);
  expect(feature.turns.filter((turn) => turn.runId).map((turn) => turn.id)).to.deep.equal(
    Array.from({length: 40}, (_, index) => `turn-${index + 51}`),
  );
  expect(feature.turns.at(-1)?.id).to.equal('local-pending');
  expect(feature.turns.at(-1)?.state).to.equal('pending');
});

it('Chat Feature renders every explicitly loaded history page without a product cap', async () => {
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

  expect(feature.querySelector('[role="log"]')?.querySelectorAll('article').length).to.equal(101);
  expect(feature.textContent).to.contain('Question 0');
  expect(feature.textContent).to.contain('Question 100');
});

it('pages more than 40 turns through the wired store, sidebar, and Load older control', async () => {
  const recent = Array.from({length: 40}, (_, index) => ({
    ...storedTurn(),
    turn_id: `turn-${index + 41}`,
    turn_number: index + 41,
    answer_run_id: `run-${index + 41}`,
    submission_id: `submission-${index + 41}`,
    user_text: `Question ${index + 41}`,
  }));
  const older = Array.from({length: 40}, (_, index) => ({
    ...storedTurn(),
    turn_id: `turn-${index + 1}`,
    turn_number: index + 1,
    answer_run_id: `run-${index + 1}`,
    submission_id: `submission-${index + 1}`,
    user_text: `Question ${index + 1}`,
  }));
  window.fetch = (async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname !== '/web/api/conversations/wired/history') {
      throw new Error(`unexpected fetch: ${url}`);
    }
    const isOlder = url.searchParams.get('cursor') === 'before-41';
    return new Response(JSON.stringify({
      conversation: {
        conversation_id: 'wired', title: 'Wired',
        created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
      },
      turns: isOlder ? older : recent,
      next_cursor: isOlder ? null : 'before-41',
    }), {status: 200, headers: {'Content-Type': 'application/json'}});
  }) as typeof fetch;
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.chatFeature = feature;
  document.body.append(feature, sidebar);

  await conversationStore.open('wired');
  await sidebar.updateComplete;
  await settle(feature);
  feature.querySelector<DlChatMessageList>('dl-chat-message-list')
    ?.querySelector<HTMLButtonElement>('[data-load-older]')?.click();
  await waitFor(() => conversationStore.history?.turns.length === 80);
  await sidebar.updateComplete;
  await settle(feature);

  expect(feature.querySelectorAll('[data-turn-id]')).to.have.length(80);
  expect(feature.textContent).to.contain('Question 1');
  expect(feature.textContent).to.contain('Question 80');
  expect(feature.querySelector('[data-load-older]')).to.equal(null);
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
